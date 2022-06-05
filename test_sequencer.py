from sequencer import *
import unittest

SAMPLE_RATE = 48000# samples / second
BIT_DEPTH = 16

class TestOscillator(unittest.TestCase):
    def test_oscillator(self):
        osc = Oscillator(200,2 * SAMPLE_RATE)
        wf = osc.generate_waveform("sine")
        self.assertEqual(wf.size, SAMPLE_RATE * 2)
        self.assertEqual(wf[200], -0.8660254037844386)

class TestSequencer(unittest.TestCase):
    def test_add_waveforms(self):
        test_seq = Sequencer(16,120)
        test_array = test_seq.add_waveforms(np.array([0,.5,1]), np.array([1,.5,0]))
        self.assertTrue((test_array == np.array([.5,.5,.5])).all())
    def test_beat_start(self):
        test_seq = Sequencer(16,120)
        self.assertEqual(test_seq.beat_start(5),120000)

if __name__ == "__main__":
    unittest.main()
