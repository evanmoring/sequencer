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
from copy import deepcopy

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

class Waveform():
    def __init__(self, array: np.array,):
        self.array = array 
        self.mutex = Lock()

    def clear(self):
        self.array = np.zeros(self.array.size)

    def apply_gain(self, gain: float):
        self.array *= gain 

    def apply_filter(self, filt : "Filter"):
        filtered_fft = filt.apply_filter(self)
        self.array = self.ifft(filtered_fft)

    def apply_envelope(self, env : "Envelope"):
        a = self.array
        env_array = env.fill(a.size)
        a *= env_array 
        self.array = a

    def write(self, filename: str):
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
        self.play_flag = True
        self.player = pyaudio.PyAudio()
        self.CHUNK = 1024
        data = self.array.astype(np.float32).tostring()
        self.count = 0


        stream = self.player.open(
            rate=SAMPLE_RATE, 
            format=pyaudio.paFloat32, 
            frames_per_buffer=self.CHUNK,
            channels=1, 
            output=True,
            stream_callback=self.play_callback)

    def play_callback(self, in_data, frame_count, time_info, status):
        if not self.play_flag:
            self.player.terminate()
            return
        if self.count + self.CHUNK < self.array.size:
            c = self.array[self.count: self.count + self.CHUNK]
            self.count += self.CHUNK
        else:
            diff = (self.count + self.CHUNK ) % self.array.size
            c1 = self.array[self.count :]
            c2 = self.array[ : diff]
            self.count = diff
            c = np.concatenate((c1, c2))
        return (c.astype(np.float32).tostring()  , pyaudio.paContinue)

    def stop(self):
        self.play_flag = False
    
    def plot(self):
        plt.figure(figsize = (20, 10))
        x = np.arange(0, self.array.size, 1)
        plt.subplot(211)
        plt.plot(x, self.array)
        plt.title("Generated Signal")
        plt.show()

    def fft(self) -> np.ndarray:
        return rfft(self.array)

    def ifft(self, fft: np.ndarray) -> np.ndarray:
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
    def __init__(self, beats: float, bpm: float):
        self.bpm = bpm
        self.beats = beats 
        self.bps = self.bpm / 60
        self.beat_length = int(SAMPLE_RATE / self.bps) 
        array = Rest(self.beats * self.beat_length).array
        super().__init__(array)

    def place_waveform(self, beat: float, waveform : Waveform):
        first_sample = samples_from_beats(self.bpm, beat)
        last_sample = first_sample + waveform.array.size
        overflow_size = last_sample - self.array.size
        if overflow_size > 0:
            self.array = np.pad(self.array, (0, overflow_size))
        new_array = np.zeros(self.array.size)
        new_array[first_sample: last_sample] = waveform.array
        self.array = np.add(self.array, new_array)

    def remove_waveform(self, beat: float, waveform : Waveform):
        print("WF Removed")
        waveform.apply_gain(-1)
        self.place_waveform(beat,waveform)

    def overwrite(self, beat: float, waveform : Waveform):
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
    def __init__(self, samples: int, gain: float = 1):
        self.samples = int(samples)
        self.seconds = samples_to_seconds(samples)
        self.gain = gain
        array = self.generate_waveform()
        super().__init__(array)
        self.apply_gain(self.gain)

    def generate_waveform(self):
        return None

    def visualize_waveform(self, waveform: np.array):
        for i in waveform:
            length = 100
            num = floor(i * length) + length
            print("#" * num)

class Oscillator(SignalGenerator):
    def __init__(self, freq: float, samples: int, gain: float = 1):
        self.freq = freq
        super().__init__(samples, gain)

class Sine(Oscillator):
    def generate_waveform(self) -> np.array:
        end_angle = self.freq * 2 * np.pi * self.seconds 
        step_angle = end_angle / self.samples
        sample_array = np.arange(0, end_angle , step_angle)
        wave_array = np.sin(sample_array)
        return wave_array 

def Pentatonic(note, samples, gain = 1) -> Sine:
    freq = major_pentatonic.scale[note]
    return Sine(freq, samples, gain)

class BentSine(Oscillator):
    def __init__(self, freq: float, freq2: float, samples: int, gain: float = 1):
        self.freq2 = freq2
        super().__init__(freq, samples, gain)
    def generate_waveform(self) -> np.array:
        a = np.full((self.samples),.5)
        freq_step = (self.freq2 - self.freq)/self.samples
        freq_array = np.arange(self.freq, self.freq2 , freq_step)
        for i in range(self.samples):
            angle = np.sin(freq_array[i] * samples_to_seconds(i) *  2 * np.pi)
            a[i] = angle
        return a 

class Square(Oscillator):        
    def generate_waveform(self) -> np.array:
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
    def generate_waveform(self) -> np.array:
        rand = np.random.rand(self.samples)
        rand -= .5
        return rand 

class Rest(SignalGenerator):
    def generate_waveform(self):
        return np.full((self.samples),0.0)

class Kick(Waveform):
    def __init__(self, gain: float=1):
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

snare_array = None
class Snare(Waveform):
    def __init__(self, gain: float=1):
        global snare_array
        if type(snare_array) == type(None):
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
            snare_array = k.array
        
        super().__init__(deepcopy(snare_array))
        print(self.array)

class Cymbal(Waveform):
    def __init__(self, gain: float=1):
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
    def __init__(self, gain: float=1):
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
    def __init__(self, unison: float, notes: list):
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

    def fill(self, new_size: int) -> np.array:
        larger_envelope = np.full((new_size),1)
        env_array = np.full(new_size, 1.0)
        env_array[:self.envelope.size] = self.envelope
        return env_array

    def generate_envelope(self):
        return None

    def plot(self):
        plt.figure(figsize = (20, 10))
        x = np.arange(0, self.envelope.size, 1)
        plt.subplot(211)
        plt.plot(x, self.envelope)
        plt.title("Generated Signal")
        plt.show()

class ReverseEnvelope(Envelope):
    def fill(self, new_size: int) -> np.array:
        self.envelope = np.flip(self.envelope)
        env_array = np.flip(super().fill(new_size))
        self.envelope = np.flip(self.envelope)
        return env_array

class LinearFadeIn(Envelope):
    def __init__(self, size: int):
        self.size = size
        super().__init__()
    def generate_envelope(self):
        self.envelope = np.arange(0, 1, 1/self.size)

class LinearFadeOut(LinearFadeIn, ReverseEnvelope):
    def generate_envelope(self):
        super().generate_envelope()
        self.flip()

class QuadraticFadeIn(Envelope):
    def __init__(self, size: int):
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
    def __init__(self, size: int):
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
    "pent": Pentatonic,
}

class Filter:
    def apply_filter(waveform: Waveform) -> Waveform:
        return waveform

class HighPassFilter(Filter):
    def __init__(self, freq: float):
        self.freq = freq
        
    def apply_filter(self, waveform : Waveform) -> np.ndarray:
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
    def __init__(self, freq: float) -> np.ndarray:
        self.freq = freq
        
    def apply_filter(self, waveform : Waveform):
        N = waveform.array.size
        yf = rfft(waveform.array)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)

        points_per_freq = len(xf) / (SAMPLE_RATE / 2)

        target_idx = int(points_per_freq * self.freq)
        yf = rfft(waveform.array)
        yf[target_idx + 2: ] = 0
        return yf

def write_csv(file):
    wf = csv_bpm(f"{file}.csv")
    wf.write(f"{file}.wav")

def add_waveforms(wf_a : Waveform, wf_b: Waveform) -> Waveform:
    new_array = np.add(wf_a.array, wf_b.array)
    new_wf = Waveform(new_array)
    new_wf.apply_gain(.5)
    return new_wf

def samples_to_seconds (samples: float) -> float:
    return samples / SAMPLE_RATE

def seconds_to_samples (seconds : float) -> int:
    return int(seconds * SAMPLE_RATE)

def freq_to_wavelength (frequency : float) -> int:
    return floor(SAMPLE_RATE / frequency)



def csv_bpm(filename: str) -> Sequencer:
    return load_csv(filename, seq_bpm)
    
def load_csv(filename: str, overwrite_bpm: float = 0) -> Sequencer:
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

def process_csv_row(row: dict, bpm: float) -> tuple:
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
    if 'note' in row:
        arg_dict['note'] = int(row['note'])
    waveform = signal_type(**arg_dict)

    return waveform, start

def samples_from_beats(bpm: float, beat: float) -> int:
    samples = int((beat * 60 * SAMPLE_RATE) / bpm)
    return samples

def beats_from_samples(bpm: float, samples: int) -> float:
    beats = (samples * bpm) / (60 * SAMPLE_RATE)
    return beats

def LoadWav(filename: str) -> Waveform:
    r = wavfile.read(filename)
    return Waveform(r[1])

def wf_to_envelope(wf: Waveform, samples_smoothed: int = freq_to_wavelength(20)) -> Envelope:
    wf = deepcopy(wf)
    wf.normalize()
    wf.apply_gain(.5)
    wf.array += .5
    smoothed_array = smooth(wf.array, samples_smoothed)
        
    output = Envelope()
    output.envelope = smoothed_array
    return output

def smooth(array: np.array, smoothing_size: int) -> np.array:
    output_array = deepcopy(array)
    for i in range(1,array.size):
        if i > smoothing_size:
            origin = i - smoothing_size
        else: 
            origin = 0
        output_array[i] = np.max(array[origin:i+1])
    return(output_array)

major_pentatonic = Scale(200,["1","M2","M3","5","M6","o"])

if __name__ == "__main__":
    s = Cymbal()
    print("MAIN")
