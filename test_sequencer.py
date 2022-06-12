from sequencer import *
import unittest

SAMPLE_RATE = 48000# samples / second
BIT_DEPTH = 16

class TestSequencer(unittest.TestCase):
    def test_oscillator(self):
        osc = Sine(200,2 * SAMPLE_RATE)
        self.assertEqual(osc.waveform.size, SAMPLE_RATE * 2)
        self.assertEqual(osc.waveform[200], -0.8660254037844386)
    def test_add_waveforms(self):
        test_array = add_waveforms(np.array([0,.5,1]), np.array([1,.5,0]))
        self.assertTrue((test_array == np.array([.5,.5,.5])).all())
    def test_samples_from_beats(self):
        res = samples_from_beats(60,2)
        self.assertEqual(res,96000)
    def test_beats_from_samples(self):
        res = beats_from_samples(60, 96000)
        self.assertEqual(res,2)

if __name__ == "__main__":
    unittest.main()
