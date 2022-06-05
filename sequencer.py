import numpy as np
from numpy import pi 
from math import floor
import scipy.io.wavfile
rng = np.random.default_rng(12345)

SAMPLE_RATE = 48000# samples / second
BIT_DEPTH = 16

def samples_to_seconds (samples):
    return samples / SAMPLE_RATE

def seconds_to_samples (seconds):
    return int(seconds * SAMPLE_RATE)

def freq_to_wavelength (frequency):
    return floor(SAMPLE_RATE / frequency)

class Sequencer:
    def __init__(self, beats, bpm):
        self.bpm = bpm
        self.beats = beats 
        self.bps = self.bpm / 60
        self.beat_length = (SAMPLE_RATE / self.bps)
        self.sequence = np.empty(int(self.beats * self.beat_length))

    def beat_start(self, beat):
        return int(beat * self.beat_length)

    def fill_seq(self):
        scale = Scale(200)
        # Write sine waves with frequency incrementing by 50hz per beat
        for i in range(self.beats):
            osc = Oscillator(25 * i + 1, SAMPLE_RATE / self.bps)
            osc = Oscillator(np.random.choice(scale.scale), int(SAMPLE_RATE / self.bps))
            wf = osc.generate_waveform("square")
            last_sample = self.beat_start(i+1)
            sample_size = self.sequence[self.beat_start(i) : last_sample].size
            self.sequence[int(i * self.beat_length) : last_sample] = wf[:sample_size]
        return self.sequence

    def add_waveforms(self, wf_a, wf_b):
        new_wf = np.add(wf_a, wf_b)
        new_wf -= .5
        return new_wf

    def write_sequence(self, filename):
        scipy.io.wavfile.write(filename, SAMPLE_RATE, self.sequence) 

class Oscillator:
    def __init__(self, freq, samples, gain = 1):
        self.freq = freq
        self.samples = samples
        self.seconds = samples_to_seconds(samples)
        self.gain = gain

    def generate_waveform (self, shape): # length in seconds
        if shape == "sine":
            return self.sine_wave()
        elif shape == "square":
            return self.square_wave()
        else:
            return False
    # TODO, sawtooth, triangle wave

    def sine_wave(self):
        end_angle = self.freq * 2 * np.pi * self.seconds 
        step_angle = end_angle / self.samples
        sample_array = np.arange(0, end_angle , step_angle)
        wave_array = np.sin(sample_array)
        wave_array *= self.gain
        return wave_array

    def square_wave(self):
        wavelength = freq_to_wavelength(self.freq)
        wave_array = np.empty(self.samples)
        sample_count = 0
        value = 1
        end_flag = False
        while True:
            value *= -1
            for i in range(wavelength):
                if sample_count >= self.samples:
                    wave_array *= self.gain
                    return wave_array
                wave_array[sample_count] = value
                sample_count += 1

    def visualize_waveform(self, waveform):
        for i in waveform:
            length = 100
            num = floor(i * length) + length
            print("#" * num)

class Sine(Oscillator):
    def generate_waveform(self):
        end_angle = self.freq * 2 * np.pi * self.seconds 
        step_angle = end_angle / self.samples
        sample_array = np.arange(0, end_angle , step_angle)
        wave_array = np.sin(sample_array)
        wave_array *= self.gain
        return wave_array

class Square(Oscillator):        
    def generate_waveform(self):
        wavelength = freq_to_wavelength(self.freq)
        wave_array = np.empty(self.samples)
        sample_count = 0
        value = 1
        end_flag = False
        while True:
            value *= -1
            for i in range(wavelength):
                if sample_count >= self.samples:
                    wave_array *= self.gain
                    return wave_array
                wave_array[sample_count] = value
                sample_count += 1

class Scale():
    def __init__(self, unison):
        self.unison = unison 
        self.intervals = np.array([1, 16/15,  9/8, 6/5, 5/4, 4/3, 25/18, 3/2, 8/5, 5/3, 9/5, 15/8, 2])
        self.scale = self.intervals * self.unison
        self.mapping = ["1", "m2", "M2", "m3", "M3", "4", "tri", "5", "m6", "M6", "m7", "M7", "o"]

class WhiteNoise:
    def __init__(self, samples):
        self.noise = True
        self.samples = samples
    def generate_noise (self):
        return np.random.rand(self.samples)

class Envelope:
    def __init__(self, wave, shape):
        self.shape = shape
        self.wave = wave
    pass
    # TODO take a np.array waveform as input, shape based on math to process it, and return
class Filter:
    pass
    # TODO take a np.array waveform as input, run a fourier transform on it, apply appropriate filtering, and return


if __name__ == "__main__":
    osc = Oscillator(200, 2 * SAMPLE_RATE)
    wf = osc.generate_waveform("sine") 
    osc2 = Oscillator(400, 2 * SAMPLE_RATE)
    wf2 = osc2.generate_waveform("sine") 
    wf3 = np.concatenate([wf, wf2])
    #osc.visualize_waveform(wf)
    seq = Sequencer(16, 120)
    wf3 = seq.fill_seq()
    seq.write_sequence("random.wav")
    scipy.io.wavfile.write("karplus.wav", SAMPLE_RATE, wf3)       
    wn = WhiteNoise(1 * SAMPLE_RATE).generate_noise()
    scipy.io.wavfile.write("noise.wav", SAMPLE_RATE, wn)       

