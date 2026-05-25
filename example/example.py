import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/evan/Projects/sequencer')

from sequencer.sequencer import *
from sequencer.filters import *

if __name__ == "__main__":
    wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
    lp = HighPassFilter(20000, .25, .5)
    wf.apply_filter(lp)
    wf.plot_fft()
    wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
    lp = HighPassFilter(20000, 2, .5)
    wf.apply_filter(lp)
    wf.plot_fft()
    exit()
    #
    write_sweep_wav(1000, sqrt(sqrt(sqrt(2))), .5, "sweep2.wav")
    p = "/home/evan/Documents/reaper_media/freq_sweeps"
    sweeps = [
        f"{p}/e_flat.wav",
        f"{p}/e_bass_up.wav", 
        f"{p}/e_bass_down.wav",
        f"{p}/e_treble_up.wav",
        f"{p}/e_treble_down.wav"]
    plot_sweeps(sweeps, 3)