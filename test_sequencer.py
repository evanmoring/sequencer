#!/usr/bin/python3

from sequencer import *
import unittest
import os

SAMPLE_RATE = 48000 # samples / second
BIT_DEPTH = 16

class TestSequencer(unittest.TestCase):
    def test_samples_to_seconds(self):
        wf = Waveform(np.array([]))
        seconds = wf.samples_to_seconds(SAMPLE_RATE*2)
        self.assertEqual(seconds, 2)

    def test_seconds_to_samples(self):
        wf = Waveform(np.array([]))
        samples = wf.seconds_to_samples(2)
        self.assertEqual(samples, 2*SAMPLE_RATE)

    def test_oscillator(self):
        osc = Sine(200,2 * SAMPLE_RATE)
        self.assertEqual(osc.array.size, SAMPLE_RATE * 2)
        self.assertEqual(osc.array[0], 0) 

    def test_rest(self):
        r = Rest(1)
        self.assertEqual(r.array, np.array([0]))

    def test_add_remove_waveforms(self):
        wf_a = Waveform(np.array([0,.5,-1]))
        wf_b = Waveform(np.array([1,.5,0]))
        wf_o = add_waveforms(wf_a,wf_b)
        s = Sequencer(1/60,1,1)
        s.place_waveform(0, wf_o)
        self.assertTrue((s.array == np.array([.5,.5,-.5])).all())
        s.apply_gain(2)
        s.remove_waveform(0,wf_a)
        self.assertTrue((s.array == wf_b.array).all())

    def test_samples_from_beats(self):
        s = Sequencer(8, 60)
        res = s.samples_from_beats(2)
        self.assertEqual(res,96000)

    def test_beats_from_samples(self):
        s = Sequencer(8, 60)
        res = s.beats_from_samples(s.seconds_to_samples(2))
        self.assertEqual(res,2)

    def test_wf_normalize(self):
        wf = Waveform(np.array([.25,0,-.5]))
        wf.normalize()
        self.assertTrue(np.array_equal(wf.array, np.array([.5,0,-1])))

    def test_sine(self):
        s = Sine(200, 2 * DEFAULT_SAMPLE_RATE)
        test_array = s.array
        self.assertEqual(test_array[0],0)
        self.assertEqual(test_array.size, s.seconds_to_samples(2))

    def test_fade(self):
        fade = LinearFadeIn(10)
        self.assertEqual(fade.envelope[0],0)
        self.assertEqual(fade.envelope.size, 10)

    def test_fade_out(self):
        fade_array = LinearFadeOut(2).fill(3)
        ex_array = np.array([1., 0.5, 0.])
        self.assertTrue(np.array_equal(fade_array, ex_array))

    def test_smooth(self):
        ar = np.array([1,3,2,1,5,1,1,1])
        smoothed = (smooth(ar,2))
        ex_array = np.array([1,3,3,3,5,5,5,1])
        self.assertTrue(np.array_equal(smoothed, ex_array))

    def test_filter(self):
        wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
        f = wf.fft()
        f = np.full(f.size, 10)
        wf.ifft(f)
        hp = HighPassFilter(1000, 0, .5)
        lp = LowPassFilter(2000, 0, .5)
        wf.apply_filter(hp)
        wf.apply_filter(lp)
        f = abs(wf.fft())
        self.assertTrue(f[5] < .1)
        self.assertTrue(f[1500] > .1)
        self.assertTrue(f[2500] < .1)

        wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
        hp = HighPassFilter(1000, .25, .4)
        lp = LowPassFilter(2000, 2, .25)
        wf.apply_filter(hp)
        wf.apply_filter(lp)

    def test_drums(self):
        import drums
        drums.Snare()
        drums.Kick()
        drums.Hihat()
        drums.Cymbal()

    def test_sweep(self):
        write_sweep_wav(1000, sqrt(sqrt(sqrt(2))), .5, "sweep_example.wav")
        analyze_sweep("sweep_example.wav")
        os.remove("sweep_example.wav")

    def test_load_csv(self):
        import drums
        load_csv("example.csv", drums.signal_map)

if __name__ == "__main__":
    unittest.main(verbosity=2)
