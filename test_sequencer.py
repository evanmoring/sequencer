#!/usr/bin/python3

from sequencer import *
import unittest

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

    def test_add_waveforms(self):
        test_array = add_waveforms(Waveform(np.array([0,.5,-1])), Waveform(np.array([1,.5,0]))).array
        self.assertTrue((test_array == np.array([.5,.5,-.5])).all())

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
        # TODO test that these are working properly, not just that they don't crash
        wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
        hp = HighPassFilter(1000, 0, .5)
        lp = LowPassFilter(2000, 0, .5)
        wf.apply_filter(hp)
        wf.apply_filter(lp)
        wf = WhiteNoise(DEFAULT_SAMPLE_RATE)
        hp = HighPassFilter(1000, .25, .4)
        lp = LowPassFilter(2000, 2, .25)
        wf.apply_filter(hp)
        wf.apply_filter(lp)

    def test_drums(self):
        # TODO improve this test
        import drums

    def test_sweep(self):
        write_sweep_wav(1000, sqrt(sqrt(sqrt(2))), .5, "sweep2.wav")
        analyze_sweep("sweep2.wav")

#TODO add load_csv test

if __name__ == "__main__":
    unittest.main(verbosity=2)
