from sequencer.sequencer import *
from sequencer.filters import *

if __name__ == "__main__":
    # generate noise samples with various filters and plot their frequencies
    wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
    lp = HighPassFilter(20000, .25, .5)
    wf.apply_filter(lp)
    wf.plot_fft()

    wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
    lp = HighPassFilter(20000, 2, .5)
    wf.apply_filter(lp)
    wf.plot_fft()

    # write a sweep file for analyzing audio systems
    write_sweep_wav(1000, sqrt(sqrt(sqrt(2))), .5, "sweep.wav")

    # plot the frequency response of the system
    # typically you would first play the sweep.wav through an amplifier 
    # and record the output
    # example output here: http://evanmoring.com/amplifier_measurements/
    sweeps = ["sweep.wav"]
    plot_sweeps(sweeps, 3)
