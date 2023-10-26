"""Resonance Control Chassis Data Acquisition Tool

This script facilitates the passive acquisition of Resonance Control (Res Ctrl)
piezo waveforms. It can either be used standalone or as part of other scripts
that perform automated measurements such as `chirp_collect.py`. It does not
require that the piezos be in any particular configuration and will work even
if one or both stacks are turned off.

Each Res Ctrl chassis acquires waveforms from four piezo controllers, each
connected to one cavity in a 8-cavity Cryomodule (CM). These waveforms are always
guaranteed to be time-synchronous, as they share the same triggering mechanism
and are handed-off to the Host in an atomic manner. Thus, this script is
designed to communicate with on Res Ctrl chassis at a time and will error out
if the user attempts to simultaneously acquire data from cavities that are not
controlled by the same chassis.

Communication with a Res Ctrl chassis requires specifying either a Channel-Access
address (`ca://`) or Leep address (`leep://`), if bypassing the IOC.

The cavities/controllers to acquire data from are CM-referred, that is, they're
identified by their position within the CM, e.g. 1, 2, 7, 8. This means that to
acquire data from cavities 5 through 8, the user would communicate with Res Ctrl
chassis B and set the `-acav` argument to `5 6 7 8`.

Each of the four piezo controllers in a Res Ctrl chassis can collect 10-channel
waveforms, corresponding to the following quantities:

`DAC, DF, AINEG, AVDIFF, AIPOS, ADRV, BINEG, BVDIFF, BIPOS, BDRV.`

The `DAC` and `DF` (detune) channels are entirely digital and do not depend on
the state of piezo stacks directly. `DAC` records the firmware-computed
piezo drive, just before it is sent to the DAC on the Resonance Controller board.
`DF` records cavity detune, as calculated in real-time by the RF station and sent
to the Res Ctrl chassis over fiber. Since detune is calculated based on RF
waveforms, it requires that RF be on and driving the cavity being measured at a
moderate gradient. All other eight quantities are obtained directly from ADCs on
the Resonance Controller board.

Since the buffer where the waveforms are held is of fixed size (16384 points),
reducing the number of channels acquired by one piezo controller increases
the number of points collected for each remaining channel. The user may provide
a list of channels to acquire through the `-ch` switch.
This list of channels applies to all piezo controllers in a Res Ctrl chassis.

The waveform acquisition system in each Res Ctrl chassis runs on a default sample rate
of 2 kS/s. The sampling rate can be reduced by increasing the waveform
decimation factor, specified through the `-wsp` argument. Together with the number
of channels selected for acquisition and the total number of points in a buffer, the
total time span covered by a channel can be calculated. For instance, a 4-channel
acquisition and a decimation of 4, each channel would cover 8.192 seconds.

The output of this script is a space-separated text file where each column corresponds
to a waveform channel. The mapping of columns to channels is established by the header
line that precedes it.

In addition to the per-channel data, it is possible to collect 'context' at the
beginning of the run that is included in the header of the output text file.
This 'context' can take the form of PVs or firmware registers and
is specified by a JSON file that is loaded through the `-j` argument. The JSON is
expected to contain two keys, "REGS and "PVS", each pointing to a string array.
`res_microphon.json` is an example of such a JSON, which covers the most typical context,
such as RF mode and operating gradient.

Typical usage examples:

```python
# Collect 65 waveform buffers containing DAC and Detune (DF) from cavities 1, 2, 3 and 4.
# Data sample rate is 2 kS/s (-wsp 1).

python res_data_acq.py -D ~/<DATA DIR> -a ca://ACCL:L1B:0300:RESA: \
                       -wsp 1 -acav 1 2 3 4 -ch DAC DF -c 65 -j res_microphon.json

# Collect 65 waveform buffers containing DAC and Detune (DF) from cavities 5, 6, 7 and 8.
# Data sample rate is 1 kS/s (-wsp 2).

python res_data_acq.py -D ~/<DATA DIR> -a ca://ACCL:L1B:0300:RESB: \
                       -wsp 2 -acav 5 6 7 8 -ch DAC DF -c 65
```
"""

import datetime
import os
from res_ctl import c_res_ctl
from res_waves import waves_run, wave_collect
import numpy as np
import signal

try:
    input = raw_input
except NameError:
    pass

try:
    import cothread.catools as ca
    from cothread import Event
except ModuleNotFoundError:
    print("Warning: cothread/catools not available")


def zeroidx(c):
    return (c-1) % 4  # Mod to stay within 4 channels in a chassis


class data_acq():

    WVF_LABELS = ["DAC", "DF",
                  "AINEG", "AVDIFF", "AIPOS", "ADRV",
                  "BINEG", "BVDIFF", "BIPOS", "BDRV"]

    WVF_PV = "%(PREF)s%(C)s0:PZT:%(WVF)s:WF"
    CAV_PV = "%(PREF)s%(C)s0:%(PV)s"

    def __init__(self, res_ctl, log_dir, wvf_cnt=1, ch_sel=["CTRL", "DF"],
                 cav_list=[1], wsp=1, json_f=None, verbose=False):
        self.abort = False  # Interlock for Ctrl-C

        self.rcc = res_ctl
        self.log_dir = log_dir
        self.wvf_cnt = wvf_cnt
        self.ch_sel = ch_sel
        self.wsp = wsp
        self.verbose = verbose

        # Maintain absolute (1-indexed) and relative (0-indexed) cavity numbers
        self.cav_list1 = cav_list
        self.cav_list0 = [zeroidx(x) for x in cav_list]

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.reg_context = []
        self.pv_context = []
        if json_f:
            import json
            j_dict = json.load(open(json_f, 'r'))
            if "REGS" in j_dict.keys():
                self.reg_context = j_dict["REGS"]
            if "PVS" in j_dict.keys():
                self.pv_context = j_dict["PVS"]

        # Check if we're running alongside an EPICS IOC
        self.epics = self.rcc.ip.startswith("ca://")

        if self.epics:
            print("Running in EPICS mode")
            # Construct CA PV base address
            self.ca_base = self.rcc.ip.replace("ca://", "").split(':')
            self.ca_base = ":".join(self.ca_base[:2]) + ':' + self.ca_base[2][:2]

        # Check cavity selection; ensure it doesn't cross half-CM
        self.CM_low = False  # 1, 2, 3, 4
        self.CM_high = False  # 5, 6, 7, 8
        for cav in self.cav_list1:
            if cav <= 4:
                self.CM_low = True
            else:
                self.CM_high = True
        if (self.CM_low and self.CM_high):
            print("ERROR: Cavity selection crosses half-CM")
            exit(-1)

        # Convert channel selection to channel mask
        ch_mask = []
        ch_list = []
        for idx, wvf in enumerate(self.WVF_LABELS):
            if True in [(ch.upper() in wvf.upper()) for ch in self.ch_sel]:
                ch_mask.append(idx)  # NOTE: only _older_ leep API expected 12-bit mask
                ch_list.append(wvf.upper())
        if not ch_mask:
            print("ERROR: No waveform channels selected")
            exit(-1)

        self.ch_mask = ch_mask
        self.ch_list = ch_list
        self.ch_count = len(ch_list)

        # Set channel mask and decimation
        self._read_apply_settings()

    def _read_apply_settings(self):
        # Bypass leep channel mask API call if running alongside EPICS
        if self.epics:
            self.prev_ch_mask = []
            for ch in self.WVF_LABELS:
                pv = self.rcc.ip.replace("ca://", "") + ch + ":ENABLE"
                self.prev_ch_mask.append((pv, ca.caget(str(pv))))  # Record previous state

                toggle = "Enable" if ch in self.ch_list else "Disable"
                print("Setting channel %s to: %s" % (ch, toggle))
                ca.caput(pv, toggle)
        else:
            self.prev_ch_mask = self.rcc.leep.get_channel_mask()  # Record previous state
            # Apply channel mask here so it immediately propagates to waveform acquisition
            self.rcc.leep.set_channel_mask(chans=self.ch_mask)

        # Set waveform decimation factor
        self.prev_wsp = self.rcc.leep.get_decimate()[0]
        self.rcc.leep.set_decimate(self.wsp)

    def _restore_settings(self):
        print("Restoring acquisition settings...")
        if self.epics:
            for (pv, v) in self.prev_ch_mask:
                ca.caput(pv, v)
        else:
            self.rcc.leep.set_channel_mask(chans=self.prev_ch_mask)

        self.rcc.leep.set_decimate(self.prev_wsp)
        print("Done")

    def get_context(self):
        # Fetch all context parameters and construct header
        p_header = []
        for cav in self.cav_list1:
            p_header.append("## Cavity %d" % (cav))

            # Global params
            for p in ["wave_samp_per", "wave_shift", "chan_keep", "chirp_en", "chirp_acq_per"]:
                try:
                    v = self.rcc.leep.reg_read([p])[0]
                    p_header.append("%s : %x" % (p, v))
                except Exception:
                    print("Could not retrieve %s. Skipping..." % p)

            # Per-cavity registers
            for p in self.reg_context:
                try:
                    v = self.rcc.read_leaf(p, zeroidx(cav))
                    p_header.append("%s : %x" % (p, v))
                except Exception:
                    print("Could not retrieve %s. Skipping..." % p)

            # Per-cavity PVs
            if not self.epics:
                continue
            for p in self.pv_context:
                try:
                    pv = self.CAV_PV % ({"PREF": self.ca_base, "C": cav, "PV": p})
                    v = ca.caget(str(pv))
                    p_header.append("%s : %s" % (p, v))
                except Exception:
                    print("Could not retrieve %s. Skipping..." % p)

        p_header.append("\n")

        return p_header

    def _capture_leep(self, fname, discard_cnt, header_top=""):
        header = [header_top]
        header += self.get_context()

        # Ordered channel scaling list
        ch_scale_l = []
        for ch in self.ch_list:
            scale = self.rcc.get_adc_scaling(ch)
            ch_scale_l.append(scale[0])
            header.append("%s %f %s" % (ch, scale[0], scale[1]))

        # Channel list
        header.append(" ".join(self.ch_list))
        header = "\n# ".join(header) + "\n"

        # Scale timeout with wsp
        t_out = 5.0*self.wsp

        print("Discarding the first %d buffers" % discard_cnt)
        for cnt in range(discard_cnt):
            wave_collect(self.rcc.leep, self.ch_mask, zone=self.cav_list0, timeout=t_out, verbose=self.verbose)

        waves_run(self.rcc.leep, self.ch_mask, ch_scale=ch_scale_l, count=self.wvf_cnt, header_id=header,
                  zone=self.cav_list0, fname=fname, fpath=self.log_dir, timeout=t_out, verbose=self.verbose)

    def _capture_epics(self, fname, discard_cnt, header_top=""):
        fname = self.log_dir + '/' + fname
        acq_done = Event()  # Synchronization event

        # Construct waveform PV list
        pv_list = []
        for cav in self.cav_list1:
            pv_list += [self.WVF_PV % ({"PREF": self.ca_base, "C": cav, "WVF": x}) for x in self.ch_list]
        print("Monitoring the following PVs:", pv_list)

        # Construct header
        header = [header_top]
        header += self.get_context()
        header.append(" ".join(pv_list))
        header = "\n# ".join(header) + "\n"

        with open(fname, 'w') as FH:
            FH.write(header)

        print("Discarding the first %d buffers" % discard_cnt)
        self.acq_cnt = -discard_cnt

        # Prepare numpy buffer
        nch = len(self.ch_list)
        BUF_LEN = 2**14
        self.npt = (BUF_LEN)//nch  # Discard remainder

        self.npv = len(pv_list)
        self.np_buf = np.zeros((self.npt, self.npv))
        self.pv_cnt = 0
        self.cur_time = 0.0

        def _camonitor_cb(value, index):
            if self.pv_cnt == 0:
                self.cur_time = value.timestamp
            elif (value.timestamp - self.cur_time) > 0.1:  # Possibly over-tight spec
                print("Warning: Data is not time-aligned", index, value.timestamp, self.cur_time)
            self.pv_cnt += 1
            self.np_buf[:, index] = value

            if self.pv_cnt == self.npv:
                if self.abort:
                    acq_done.Signal()
                    return

                self.acq_cnt += 1
                self.pv_cnt = 0
                if self.acq_cnt < 1:
                    return  # Discard

                with open(fname, 'ab') as FH:
                    header = ""
                    if self.acq_cnt == 1:
                        first_ts = datetime.datetime.fromtimestamp(self.cur_time).isoformat()
                        header = "First buffer EPICS timestamp %s\n" % first_ts
                    np.savetxt(FH, self.np_buf, fmt="%9.5f", header=header)

                if (self.acq_cnt == self.wvf_cnt):
                    acq_done.Signal()
                time = datetime.datetime.fromtimestamp(self.cur_time).isoformat()
                print("Acquired %d / %d buffers. %s" % (self.acq_cnt, self.wvf_cnt, time))

        subs = ca.camonitor(pv_list, _camonitor_cb, datatype=float, count=self.npt, format=ca.FORMAT_TIME)

        try:
            acq_done.Wait()
        finally:
            print("Ending PV subscription")
            for s in subs:
                s.close()

    def capture(self, fname, discard_cnt=3):
        try:
            time = "# " + datetime.datetime.now().isoformat()
            if self.epics:
                self._capture_epics(fname, discard_cnt=discard_cnt, header_top=time)
            else:
                self._capture_leep(fname, discard_cnt=discard_cnt, header_top=time)

            # Restore settings prior to acquisition
            self._restore_settings()

        except Exception as e:
            # Attempt to recover original state
            print("Acquisition exception found! Trying to recover original state")
            self._restore_settings()
            raise(e)


def res_data_acq_cli(ap):
    ap.add_argument('-a', '--address', dest="addr", required=True, help="Device address")
    ap.add_argument('-c',  '--count',  dest="count", type=int,
                    help="Number of waveform acquisitions", default=1)
    ap.add_argument('-acav',  '--acq_cav',  dest="acav_list", type=int,
                    choices=list(range(1, 9)), nargs="+", action="store",
                    help="List of cavities in half-CM to acquire data from", default=[1, 2, 3, 4])
    ap.add_argument('-wsp',  '--wave_samp_per',  dest="wsp", type=int,
                    help="Waveform decimation factor", default=1)
    ap.add_argument('-D', '--dir', dest='log_dir',
                    help='Log/data directory prefix (can include path)')
    ap.add_argument('-F', '--fname', dest='fname',
                    help='Optional data file name')
    ap.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                    help='Verbose mode')
    ap.add_argument('-ch', '--ch_list', dest='ch_list', nargs="+", action='store',
                    default=["DAC", "DF"], help='Override channel acq list. ALL to enable all')
    ap.add_argument('-j', '--json', dest='json_file', help='JSON listing context PVs/Registers to acquire')


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Resonance Control Data Acquisition Tool.")
    res_data_acq_cli(parser)
    args = parser.parse_args()

    if len(args.ch_list) == 1 and args.ch_list[0].upper() == 'ALL':
        ch_sel = data_acq.WVF_LABELS
    else:
        ch_sel = args.ch_list

    # Validate inputs
    if args.count < 1:
        print("ERROR: Invalid count")
        exit(-1)

    log_dir = args.log_dir
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not log_dir:
        log_dir = "res_data_acq_" + time_str

    # RCC communication
    res_ctl = c_res_ctl(args.addr)

    # Setup acquisition class
    daq = data_acq(res_ctl, log_dir=log_dir, wvf_cnt=args.count,
                   ch_sel=ch_sel, cav_list=args.acav_list, wsp=args.wsp,
                   json_f=args.json_file, verbose=args.verbose)

    def hSIGINT(signum, frame):
        print("Intercepted signal (%s). Aborting acquisition" % signum)
        print("WARNING: Next CTRL-C will kill immediately.")
        daq.abort = True
        signal.signal(signal.SIGINT, signal.default_int_handler)  # Re-enable SIGINT

    # Intercept Ctrl+C and tear down gracefully
    signal.signal(signal.SIGINT, hSIGINT)

    fname = "res_cav%s_c%d_%s" % ("".join(str(x) for x in args.acav_list), args.count, time_str)
    if args.fname:
        fname = args.fname
    daq.capture(fname=fname)
