import numpy as np
from numpy import pi 
from math import floor
from scipy.io import wavfile
from scipy.fftpack import  rfftfreq, rfft, irfft
import csv
import pandas as pd
import pyaudio
import time
from threading import Thread, Lock
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)
seq_bpm = 250 

SAMPLE_RATE = 48000# samples / second
BIT_DEPTH = 16
intervals = {
    "1":    1,
    "m2":   16/15, 
    "M2":   9/8,
    "m3":   6/5, 
    "M3":   5/4, 
    "4":    4/3, 
    "tri":  25/18,
    "5":    3/2,
    "m6":   8/5,
    "M6":   5/3,
    "m7":   9/5, 
    "M7":   15/8,
    "o":    2}

def samples_to_seconds (samples):
    return samples / SAMPLE_RATE

def seconds_to_samples (seconds):
    return int(seconds * SAMPLE_RATE)

def freq_to_wavelength (frequency):
    return floor(SAMPLE_RATE / frequency)

def add_waveforms(wf_a, wf_b):
    new_array = np.add(wf_a.array, wf_b.array)
    new_wf = Waveform(new_array)
    new_wf.apply_gain(.5)
    return new_wf

def csv_bpm(filename):
    return load_csv(filename, seq_bpm)
    
def load_csv(filename, overwrite_bpm = 0):
    csv_dicts = []
    waveform_list = []

    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        temp_list = []
        for i in reader:
            temp_list.append(i)

    for i in temp_list[1:]:
        new_dict = {}
        for ii, value in enumerate(i):
            if len(value):
                new_dict[temp_list[0][ii]] = value
        csv_dicts.append(new_dict)
    for i in csv_dicts:
        if 'bpm' in i:
            bpm = float(i['bpm'])
            if overwrite_bpm:
                bpm = overwrite_bpm
            length = float(i['seq_length'])
            seq = Sequencer(length, bpm)
        else:
            wf, start = process_csv_row(i, bpm)
            seq.place_waveform(start, wf) 
    return seq

def process_csv_row(row, bpm):
    signal_type = signal_map[row['signal_type']] 
    start = float(row['start'])

    arg_dict = {}
    if 'gain' in row:
        arg_dict['gain'] = float(row['gain'])
    if 'length' in row:
        length = float(row['length'])
        arg_dict['samples'] = samples_from_beats(bpm, length)
    if 'freq' in row:
        arg_dict['freq'] = float(row['freq'])
    waveform = signal_type(**arg_dict)

    return waveform, start

def samples_from_beats(bpm, beat):
    samples = int((beat * 60 * SAMPLE_RATE) / bpm)
    return samples

def beats_from_samples(bpm, samples):
    beats = (samples * bpm) / (60 * SAMPLE_RATE)
    return beats

def LoadWav(filename):
    r = wavfile.read(filename)
    return Waveform(r[1])

class Waveform():
    def __init__(self, array,):
        self.array = array 
        self.mutex = Lock()

    def apply_gain(self, gain):
        self.array *= gain 

    def apply_filter(self, filt):
        filtered_fft = filt.apply_filter(self)
        self.array = self.ifft(filtered_fft)

    def apply_envelope(self, env):
        a = self.array
        env_array = env.fill(a.size)
        a *= env_array 
        self.array = a

    def write(self, filename):
        wavfile.write(filename, SAMPLE_RATE, self.array) 

    def normalize(self):
        a = self.array
        a_max = a.max()
        a_min = a.min()
        if a_max > abs(a_min):
            gain_factor = a_max
        else:
            gain_factor = abs(a_min)
        gain = 1/gain_factor
        self.apply_gain(gain)
        pass

    def play(self):
        # TODO write audio in chunks
        P = pyaudio.PyAudio()
        CHUNK = 1024
        stream = P.open(rate=SAMPLE_RATE, format=pyaudio.paFloat32, channels=1, output=True)
        data = self.array.astype(np.float32).tostring()
        while True:
            stream.write(data)
            time.sleep(1)
        #stream.close() # this blocks until sound finishes playing

        P.terminate()
    
    def plot(self):
        plt.figure(figsize = (20, 10))
        x = np.arange(0, self.array.size, 1)
        plt.subplot(211)
        plt.plot(x, self.array)
        plt.title("Generated Signal")
        plt.show()

    def fft(self):
        return rfft(self.array)

    def ifft(self, fft):
        return np.fft.irfft(fft, n=self.array.size)

    def plot_fft(self):
        fig,ax = plt.subplots()
        yf = self.fft()
        xf = rfftfreq(self.array.size, 1 / SAMPLE_RATE)
        ax.set_yscale('log')
        ax.set_yscale('log')

        plt.plot(xf[0:3000], np.abs(yf[0:3000]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()

class Sequencer(Waveform):
    def __init__(self, beats, bpm):
        self.bpm = bpm
        self.beats = beats 
        self.bps = self.bpm / 60
        self.beat_length = int(SAMPLE_RATE / self.bps) 
        array = Rest(self.beats * self.beat_length).array
        super().__init__(array)

    def place_waveform(self, beat, waveform):
        first_sample = samples_from_beats(self.bpm, beat)
        last_sample = first_sample + waveform.array.size
        overflow_size = last_sample - self.array.size
        if overflow_size > 0:
            self.array = np.pad(self.array, (0, overflow_size))
        new_array = np.zeros(self.array.size)
        new_array[first_sample: last_sample] = waveform.array
        self.array = np.add(self.array, new_array)

    def overwrite(self, beat, waveform):
        first_sample = samples_from_beats(self.bpm, beat)
        last_sample = first_sample + waveform.array.size
        overflow_size = last_sample - self.array.size
        if overflow_size > 0:
            self.array = np.pad(self.array, (0, overflow_size))
        new_array = np.zeros(self.array.size)
        self.array[first_sample: last_sample] = waveform.array

    def example(self):
        for i in range(self.beats):
            if i % 2:
                osc = Sine(np.random.choice(major_pentatonic.scale), self.beat_length / 2)
                wf = osc
            else:
                wf = WhiteNoise(self.beat_length / 2 ,gain=.25)
            wf = LinearFadeIn(wf, self.beat_length / 8).wave  
            wf = LinearFadeOut(wf, self.beat_length / 8).wave  
            self.place_waveform(i, wf)
        self.write_sequence("example.wav")

class SignalGenerator(Waveform):
    def __init__(self, samples, gain = 1):
        self.samples = int(samples)
        self.seconds = samples_to_seconds(samples)
        self.gain = gain
        array = self.generate_waveform()
        super().__init__(array)
        self.apply_gain(self.gain)

    def generate_waveform(self):
        self.waveform = None

    def visualize_waveform(self, waveform):
        for i in waveform:
            length = 100
            num = floor(i * length) + length
            print("#" * num)

class Oscillator(SignalGenerator):
    def __init__(self, freq, samples, gain = 1):
        self.freq = freq
        super().__init__(samples, gain)

class Sine(Oscillator):
    def generate_waveform(self):
        end_angle = self.freq * 2 * np.pi * self.seconds 
        step_angle = end_angle / self.samples
        sample_array = np.arange(0, end_angle , step_angle)
        wave_array = np.sin(sample_array)
        return wave_array 

class BentSine(Oscillator):
    def __init__(self, freq, freq2, samples, gain = 1):
        self.freq2 = freq2
        super().__init__(freq, samples, gain)
    def generate_waveform(self):
        a = np.full((self.samples),.5)
        freq_step = (self.freq2 - self.freq)/self.samples
        freq_array = np.arange(self.freq, self.freq2 , freq_step)
        for i in range(self.samples):
            angle = np.sin(freq_array[i] * samples_to_seconds(i) *  2 * np.pi)
            a[i] = angle
        return a 

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
                    return wave_array
                wave_array[sample_count] = value
                sample_count += 1

class WhiteNoise(SignalGenerator):
    def generate_waveform(self):
        rand = np.random.rand(self.samples)
        rand -= .5
        return rand 

class Rest(SignalGenerator):
    def generate_waveform(self):
        return np.full((self.samples),0.0)

class Kick(Waveform):
    def __init__(self, gain=1):
        length = seconds_to_samples(.5)
        peak = 200
        k = WhiteNoise(length, gain = 1)
        lp = LowPassFilter(250)
        k.apply_filter(lp)
        k.normalize()
        s = BentSine(150, 100, length, gain=.5)
        k = add_waveforms(k, s)
        fade_in = QuadraticFadeIn(peak)
        k.apply_envelope(fade_in)
        fade_out = QuadraticFadeOut( length - peak)
        k.apply_envelope(fade_out)
        k.normalize()
        k.apply_gain(gain)
        super().__init__(k.array)

class Snare(Waveform):
    def __init__(self, gain=1):
        length = seconds_to_samples(.35)
        peak = 200
        k = WhiteNoise(length, gain = .25)
        s = Sine(120, length)
        fade_in = QuadraticFadeIn(peak)
        k.apply_envelope(fade_in)
        fade_out = QuadraticFadeOut( length - peak)
        k.apply_envelope(fade_out)
        k.normalize()
        k.apply_gain(.5)
        k.apply_gain(gain)
        
        super().__init__(k.array)

class Cymbal(Waveform):
    def __init__(self, gain=1):
        length = seconds_to_samples(1)
        peak = 200
        k = WhiteNoise(length, gain = .25)
        hp = HighPassFilter(500)
        k.apply_filter(hp)
        fade_in = QuadraticFadeIn(peak)
        k.apply_envelope(fade_in)
        fade_out = QuadraticFadeOut( length - peak)
        k.apply_envelope(fade_out)
        k.normalize()
        k.apply_gain(.25)
        k.apply_gain(gain)
        
        super().__init__(k.array)

class Hihat(Waveform):
    def __init__(self, gain=1):
        length = seconds_to_samples(.25)
        peak = 200
        k = WhiteNoise(length, gain = .25)
        hp = HighPassFilter(1000)
        k.apply_filter(hp)
        fade_in = QuadraticFadeIn(peak)
        k.apply_envelope(fade_in)
        fade_out = QuadraticFadeOut( length - peak)
        k.apply_envelope(fade_out)
        k.normalize()
        k.apply_gain(.25)
        k.apply_gain(gain)
        
        super().__init__(k.array)

class Scale():
    def __init__(self, unison, notes):
        self.unison = unison 
        self.intervals = []
        for i in notes:
            self.intervals.append(intervals[i])
        self.intervals = np.array(self.intervals)
        self.scale = self.intervals * self.unison

class Envelope:
    def __init__(self):
        self.generate_envelope()

    def flip(self):
        self.envelope = np.flip(self.envelope)

    def fill(self,new_size):
        larger_envelope = np.full((new_size),1)
        env_array = np.full(new_size, 1.0)
        env_array[:self.envelope.size] = self.envelope
        return env_array

class ReverseEnvelope(Envelope):
    def fill(self,new_size):
        self.envelope = np.flip(self.envelope)
        env_array = np.flip(super().fill(new_size))
        self.envelope = np.flip(self.envelope)
        return env_array

class LinearFadeIn(Envelope):
    def __init__(self, size):
        self.size = size
        super().__init__()
    def generate_envelope(self):
        self.envelope = np.arange(0, 1, 1/self.size)

class LinearFadeOut(LinearFadeIn, ReverseEnvelope):
    def generate_envelope(self):
        super().generate_envelope()
        self.flip()

class QuadraticFadeIn(Envelope):
    def __init__(self, size):
        self.size = size
        super().__init__()
    def generate_envelope(self):
        P = lambda t: t**2
        self.envelope = np.array([P(t) for t in np.linspace(0, 1, self.size)])

class QuadraticFadeOut(QuadraticFadeIn, ReverseEnvelope):
    def generate_envelope(self):
        super().generate_envelope()
        self.flip()

class IQuadraticFadeIn(Envelope):
    def __init__(self, size):
        self.size = size
        super().__init__()
    def generate_envelope(self):
        P = lambda t: 1- t**2
        self.envelope = np.array([P(t) for t in np.linspace(0, 1, self.size)])
        self.flip()

class IQuadraticFadeOut(IQuadraticFadeIn, ReverseEnvelope):
    def generate_envelope(self):
        super().generate_envelope()
        self.flip()

signal_map = {
    "sine": Sine,
    "square": Square,
    "white_noise": WhiteNoise,
    "snare": Snare,
    "kick": Kick,
    "cymbal": Cymbal,
    "hihat": Hihat,
}

class Filter:
    def apply_filter(waveform):
        return waveform

class HighPassFilter(Filter):
    def __init__(self, freq):
        self.freq = freq
        
    def apply_filter(self, waveform):
        N = waveform.array.size
        yf = rfft(waveform.array)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)

        points_per_freq = len(xf) / (SAMPLE_RATE / 2)

        target_idx = int(points_per_freq * self.freq)
        yf = rfft(waveform.array)
        yf[0: target_idx + 2] = 0
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        return yf

class LowPassFilter(Filter):
    def __init__(self, freq):
        self.freq = freq
        
    def apply_filter(self, waveform):
        N = waveform.array.size
        yf = rfft(waveform.array)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)

        points_per_freq = len(xf) / (SAMPLE_RATE / 2)

        target_idx = int(points_per_freq * self.freq)
        yf = rfft(waveform.array)
        yf[target_idx + 2: ] = 0
        return yf

if __name__ == "__main__":
    major_pentatonic = Scale(200,["1","M2","M3","5","M6","o"])


#    verse_a = csv_bpm("drums2.csv")
#    verse_a.write("verse.wav")
#    verse_b = csv_bpm("drums_verse_2.csv")
#    verse_seq = Sequencer(16, seq_bpm)
#    verse_seq.place_waveform(0, verse_a)
#    verse_seq.place_waveform(8, verse_b)
#    verse_fill = csv_bpm("verse_fill.csv")
#    verse_fill_seq = Sequencer(16,seq_bpm)
#    verse_fill_seq.place_waveform(0, verse_a)
#    verse_fill_seq.place_waveform(8, verse_fill)
#    full_verse = Sequencer(65,seq_bpm)
#    full_verse.place_waveform(0, verse_seq)
#    full_verse.place_waveform(16, verse_seq)
#    full_verse.place_waveform(32, verse_seq)
#    full_verse.place_waveform(48, verse_fill_seq)
#    full_verse.write("verse.wav")
#
#    chorus_seq_a = Sequencer(16,seq_bpm)
#    chorus_b = csv_bpm("chorus_b.csv")
#    chorus_b.write("chorus_b.wav")
#    chorus_seq_a.place_waveform(0, verse_a)
#    chorus_seq_a.place_waveform(8, chorus_b)
#    chorus_seq_a.write("chorus_b.wav")
#
#    chorus_seq_b = Sequencer(32,seq_bpm)
#    chorus_fill = csv_bpm("chorus_fill.csv")
#    chorus_seq_b.place_waveform(0, verse_a)
#    chorus_seq_b.place_waveform(8, chorus_fill)
#
#    full_chorus = Sequencer(80, seq_bpm)
#    full_chorus.place_waveform(0, chorus_seq_a)
#    full_chorus.place_waveform(16, chorus_seq_a)
#    full_chorus.place_waveform(32, chorus_seq_a)
#    full_chorus.place_waveform(48, chorus_seq_b)
#    full_chorus.write("full_chorus.wav")
#
#    loaded = load_csv("drums_verse_2.csv")
#    #loaded.play()
#    loaded.plot()
#    loaded.write("drums.wav")
