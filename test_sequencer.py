from sequencer import *
import unittest

SAMPLE_RATE = 48000# samples / second
BIT_DEPTH = 16

class TestSequencer(unittest.TestCase):
    def test_samples_to_seconds(self):
        seconds = samples_to_seconds(SAMPLE_RATE*2)
        self.assertEqual(seconds, 2)

    def test_seconds_to_samples(self):
        samples = seconds_to_samples(2)
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
        res = samples_from_beats(60,2)
        self.assertEqual(res,96000)

    def test_beats_from_samples(self):
        res = beats_from_samples(60, seconds_to_samples(2))
        self.assertEqual(res,2)

    def test_wf_normalize(self):
        wf = Waveform(np.array([.25,0,-.5]))
        wf.normalize()
        self.assertTrue(np.array_equal(wf.array, np.array([.5,0,-1])))

    def test_sine(self):
        test_array = Sine(200, seconds_to_samples(2)).array
        self.assertEqual(test_array[0],0)
        self.assertEqual(test_array.size, seconds_to_samples(2))

    def test_fade(self):
        fade = LinearFadeIn(10)
        self.assertEqual(fade.envelope[0],0)
        self.assertEqual(fade.envelope.size, 10)

    def test_fade_out(self):
        fade_array = LinearFadeOut(2).fill(3)
        ex_array = np.array([1., 0.5, 0.])
        self.assertTrue(np.array_equal(fade_array, ex_array))

if __name__ == "__main__":
    unittest.main()
