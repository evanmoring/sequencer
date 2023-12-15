#!/usr/bin/python3

import numpy as np
from numpy import pi 
from math import floor, sqrt
from scipy.io import wavfile
from scipy.fft import  rfftfreq, rfft, irfft
from scipy.signal.windows import hamming
import csv
import pyaudio
import time
from threading import Thread, Lock
from copy import deepcopy
# matplotlib is slow, only import it if its needed
plt = None
ScalarFormatter = None

def import_plt():
    global plt, ScalarFormatter
    if plt == None:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter

rng = np.random.default_rng(12345)
seq_bpm = 250 

DEFAULT_SAMPLE_RATE = 48000# samples / second
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

def add_waveforms(wf_a : 'Waveform', wf_b: 'Waveform') -> 'Waveform':
    new_array = np.add(wf_a.array, wf_b.array)
    new_wf = Waveform(new_array)
    new_wf.apply_gain(.5)
    return new_wf

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
        import_plt()
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
        l = np.linspace(0, 1, int(self.size))
        l = l ** 2
        self.envelope = l

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

class Filter:
    def apply_filter(waveform: 'Waveform') -> 'Waveform':
        return waveform

class HighPassFilter(Filter):
    def __init__(self, freq: int, period: float = 1, gain: float = .5):
        # period in octaves, gain of .5 would half the signal per period 
        self.freq = freq
        assert period >= 0
        self.period = period
        self.gain = gain
        
    def apply_filter(self, waveform : 'Waveform') -> np.ndarray:
        N = waveform.array.size
        yf = rfft(waveform.array)
        xf = rfftfreq(N, 1 / waveform.sample_rate)
        points_per_freq = len(xf) / (waveform.sample_rate / 2)
        freq_idx = int(self.freq * points_per_freq)

        if self.period:
            # take log of integers corresponding to frequencies to apply gain to
            freq_gains = np.arange(freq_idx, dtype = np.float32)
            freq_gains[0] = 1
            freq_gains = np.emath.logn(2 , freq_gains)
            freq_gains *= 1/self.period
            g = np.full(len(freq_gains), self.gain)
            # raise the gain to the power of the number of periods
            freq_gains = g ** freq_gains
            freq_gains[0] = 1
            # gain for knee frequency should be 1. Scale other freqs to match
            freq_gains /= freq_gains[freq_idx-1]
            freq_gains = np.reciprocal(freq_gains)
        else:
            # handle period of 0
            freq_gains = np.zeros(freq_idx)

        yf[:freq_idx] *= freq_gains
        return yf

class LowPassFilter(Filter):
    def __init__(self, freq: int, period: int = 1, gain: float = .5):
        # period in octaves, gain of .5 would half the signal per period 
        self.freq = freq
        assert period >= 0
        self.period = period
        self.gain = gain
        
    def apply_filter(self, waveform : 'Waveform'):
        N = waveform.array.size
        yf = rfft(waveform.array)
        xf = rfftfreq(N, 1 / waveform.sample_rate)
        freq_idx = int(self.freq)
        if self.period:
            freq_gains = np.arange(1,len(xf), dtype = np.float16)
            # scale frequencies to knee
            freq_gains = freq_gains / self.freq
            freq_gains = np.emath.logn(2, freq_gains)
            freq_gains *= 1/self.period
            g = np.full(len(freq_gains[freq_idx:]), self.gain)
            # raise gain to the log2 value of scaled frequencies
            freq_gains[freq_idx:] =  g ** freq_gains[freq_idx:]
        else:
            # handle period 0
            freq_gains = np.zeros(len(xf) - 1)
        yf[freq_idx+1:] *= freq_gains[freq_idx:]
        return yf

class BandPassFilter(Filter):
    def __init__(self, freq: int, period: int = 0, gain: float = 0, width: float = 0) -> np.ndarray:
        # period in octaves, gain of .5 would half the signal per frequency, width in octaves
        self.freq = freq
        self.period = period
        self.gain = gain
        self.width = width
        
    def apply_filter(self, waveform : 'Waveform') -> np.ndarray:
        right_knee = self.freq * (2**(.5 * self.width)) 
        left_knee = self.freq * (1/(2**(.5 * self.width))) 
        wf = waveform
        lp = LowPassFilter(left_knee, period = self.period, gain = self.gain)
        hp = HighPassFilter(right_knee, period = self.period, gain = self.gain)
        a = lp.apply_filter(wf)
        b = hp.apply_filter(wf)
        a *= b
        a /= max(b)
        return a

class Waveform():
    def __init__(self, 
            array: np.array, 
            sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.array = array 
        self.sample_rate = sample_rate

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
        wavfile.write(filename, self.sample_rate, self.array) 

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

    def play(self):
        self.play_flag = True
        self.player = pyaudio.PyAudio()
        self.CHUNK = 1024
        data = self.array.astype(np.float32).tostring()
        self.count = 0

        stream = self.player.open(
            rate=self.sample_rate, 
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
        import_plt()
        plt.figure(figsize = (20, 10))
        x = np.arange(0, self.array.size, 1)
        plt.subplot(211)
        plt.plot(x, self.array)
        plt.show()

    def fft(self) -> np.ndarray:
        return rfft(self.array)

    def ifft(self, fft: np.ndarray) -> np.ndarray:
        return np.fft.irfft(fft, n=self.array.size)

    def plot_fft(self, log_x: bool = False, log_y: bool = False, band: list  = [20,20000]):
        import_plt()
        fig,ax = plt.subplots()
        yf = self.fft()
        xf = rfftfreq(self.array.size, 1 / self.sample_rate)

        #plt.plot(xf[band[0]:band[1]], np.abs(yf[band[0]:band[1]]))
        plt.plot(xf, np.abs(yf))
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        plt.show()

    def freq_to_wavelength (self, frequency : float) -> int:
        return floor(self.sample_rate / frequency)

    def samples_to_seconds (self, samples: float) -> float:
        return samples / self.sample_rate

    def seconds_to_samples (self, seconds : float) -> int:
        return int(seconds * self.sample_rate)

class Sequencer(Waveform):
    def __init__(self, 
            beats: float, 
            bpm: float, 
            sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.bpm = bpm
        self.beats = beats 
        self.bps = self.bpm / 60
        self.beat_length = int(sample_rate / self.bps) 
        array = Rest(self.beats * self.beat_length).array
        super().__init__(array, sample_rate)

    def place_waveform(self, beat: float, waveform : Waveform):
        first_sample = self.samples_from_beats(beat)
        last_sample = first_sample + waveform.array.size
        overflow_size = last_sample - self.array.size
        if overflow_size > 0:
            self.array = np.pad(self.array, (0, overflow_size))
        new_array = np.zeros(self.array.size)
        new_array[first_sample: last_sample] = waveform.array
        self.array = np.add(self.array, new_array)

    def remove_waveform(self, beat: float, waveform : Waveform):
        waveform.apply_gain(-1)
        self.place_waveform(beat,waveform)

    def overwrite(self, beat: float, waveform : Waveform):
        first_sample = self.samples_from_beats(beat)
        last_sample = first_sample + waveform.array.size
        overflow_size = last_sample - self.array.size
        if overflow_size > 0:
            self.array = np.pad(self.array, (0, overflow_size))
        new_array = np.zeros(self.array.size)
        self.array[first_sample: last_sample] = waveform.array

    def beats_from_samples(self, samples: int) -> float:
        beats = (samples * self.bpm) / (60 * self.sample_rate)
        return beats

    def samples_from_beats(self, beat: float) -> int:
        samples = int((beat * 60 * self.sample_rate) / self.bpm)
        return samples

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
    def __init__(self, 
            samples: int, 
            gain: float = 1, 
            sample_rate: int = DEFAULT_SAMPLE_RATE):

        self.samples = int(samples)
        self.sample_rate = sample_rate
        self.seconds = samples / sample_rate
        self.gain = gain
        self.array = self.generate_waveform()
        self.apply_gain(self.gain)

    def generate_waveform(self):
        return None

    def visualize_waveform(self, waveform: np.array):
        for i in waveform:
            length = 100
            num = floor(i * length) + length
            print("#" * num)

class Oscillator(SignalGenerator):
    def __init__(self, 
            freq: float, 
            samples: int, 
            gain: float = 1,
            sample_rate = DEFAULT_SAMPLE_RATE):
        self.freq = freq
        super().__init__(samples, gain, sample_rate)

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
    def __init__(self, 
            freq: float, 
            freq2: float, 
            samples: int, 
            gain: float = 1,
            sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.freq2 = freq2
        super().__init__(freq, samples, gain, sample_rate)
    def generate_waveform(self) -> np.array:
        a = np.full((self.samples),.5)
        freq_step = (self.freq2 - self.freq)/self.samples
        freq_array = np.arange(self.freq, self.freq2 , freq_step)
        for i in range(self.samples):
            angle = np.sin(freq_array[i] * self.samples_to_seconds(i) *  2 * np.pi)
            a[i] = angle
        return a 

class Square(Oscillator):        
    def generate_waveform(self) -> np.array:
        wavelength = self.freq_to_wavelength(self.freq)
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

class Scale():
    def __init__(self, unison: float, notes: list):
        self.unison = unison 
        self.intervals = []
        for i in notes:
            self.intervals.append(intervals[i])
        self.intervals = np.array(self.intervals)
        self.scale = self.intervals * self.unison

def write_csv(file):
    wf = csv_bpm(f"{file}.csv")
    wf.write(f"{file}.wav")

def csv_bpm(filename: str) -> Sequencer:
    return load_csv(filename, seq_bpm)
    
def load_csv(filename: str, signal_map: dict, overwrite_bpm: float = 0) -> Sequencer:
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
            if 'sample_rate' in i.keys():
                sr = int(i['sample_rate'])
            else:
                sr = DEFAULT_SAMPLE_RATE
            seq = Sequencer(length, bpm, sample_rate = sr)
        else:
            wf, start = process_csv_row(i, bpm, signal_map, seq.sample_rate)
            seq.place_waveform(start, wf) 
    return seq

def process_csv_row(row: dict, 
        bpm: float, 
        signal_map: dict,
        sample_rate: float) -> tuple:

    signal_type = signal_map[row['signal_type']] 
    start = float(row['start'])
    arg_dict = {}
    if 'gain' in row:
        arg_dict['gain'] = float(row['gain'])
    if 'length' in row:
        length = float(row['length'])
        s = int((length * 60 * sample_rate) / bpm) # beats_to_samples from Sequencer
        arg_dict['samples'] = s
    if 'freq' in row:
        arg_dict['freq'] = float(row['freq'])
    if 'note' in row:
        arg_dict['note'] = int(row['note'])
    waveform = signal_type(**arg_dict)
    return waveform, start

def load_wav(filename: str) -> list:
    # return a Waveform for each channel
    rate, data = wavfile.read(filename)
    wf = []
    if len(data.shape) > 1:
        for i in range(len(data.shape[1])):
            wf.append = Waveform(data[:, i], rate)
    else:
        wf.append(Waveform(data,rate))
    return wf

def wf_to_envelope(wf: Waveform, samples_smoothed: int) -> Envelope:
    wf = deepcopy(wf)
    wf.normalize()
    wf.apply_gain(.5)
    wf.array += .5
    smoothed_array = smooth(wf.array, samples_smoothed)
        
    output = Envelope()
    output.envelope = smoothed_array
    return output

def smooth(array: np.array, smoothing_size: int) -> np.array:
    # smoothing size in samples
    output_array = deepcopy(array)
    for i in range(1,array.size):
        if i > smoothing_size:
            origin = i - smoothing_size
        else: 
            origin = 0
        output_array[i] = np.max(array[origin:i+1])
    return(output_array)

def sigmoid_compressor(wf: Waveform, threshold: float, peak: float):
    diff = peak - threshold
    a = np.copy(wf.array)
    for i, val in enumerate(a):
        if abs(val) > threshold:
            neg = 1
            if (val < 0):
                neg = -1
            n = abs(val) - threshold
            n = n / diff
            # sigmoid function
            n = (1 / (1 + np.exp(-n)))
            n -= .5 
            n = n * diff * 2
            n += threshold
            n *= neg
            a[i] = n
    return a

major_pentatonic = Scale(200,["1","M2","M3","5","M6","o"])

def formant(samples: int, formant_a: int, formant_b: int):
    s = samples
    a = WhiteNoise(s)
    b = WhiteNoise(s)
    c = WhiteNoise(s)
    s_a = Sine(formant_a,s)
    s_b = Sine(formant_b,s)
    s_b.apply_gain(.2)
    sc = add_waveforms(s_a, s_b)
    sc.apply_gain(.05)

    lp_a = BandPassFilter(formant_a, period = 1, gain = .001)
    bp_a = BandPassFilter(formant_b, period = 1, gain = .001)
    a.apply_filter(lp_a)
    a.apply_gain(3)
    b.apply_filter(bp_a)
    b.apply_gain(1)
    b.apply_gain(1)
    c.apply_gain(.1)
    o = add_waveforms(a,b)
    o = add_waveforms(o,sc)
    o.apply_gain(10)
    #o = add_waveforms(o,c)
    return o

def gen_sweep_freqs(center_freq: float,
        factor: float) -> list:

    l = [center_freq]
    lower = center_freq
    upper = center_freq
    while lower / factor > 20:
        lower /= factor
        l.append(round(lower))
    while upper * factor < 20000:
        upper *= factor
        l.append(round(upper))
    l.sort()
    return l

def write_sweep_wav(center_freq: float, 
                   factor: float = sqrt(sqrt(2)), 
                   seconds_per_freq: float = 1,
                   filename: str = "sweep.wav") -> 'Waveform': 

    l = gen_sweep_freqs(center_freq, factor)

    s = Sequencer(len(l) * seconds_per_freq, 60)
    for i, freq in enumerate(l):
        sin = Sine(freq, DEFAULT_SAMPLE_RATE * seconds_per_freq)
        s.place_waveform(i * seconds_per_freq, sin)
    s.write(filename)
    return s

def analyze_sweep(filename: str, seconds_per_division: float = 1) -> np.array:

    # returns a list of average values for each division
    # to wit, if you have a 2 second wav file with .5 seconds_per_division
    # you will get the average values of each 4th of the file 
    # (after a hamming window is applied)

    wf = load_wav(filename)[0]
    # split file into chunks
    seconds = int(wf.samples_to_seconds(wf.array.size))
    division = int(wf.sample_rate * seconds_per_division)
    freq_list = []
    start = 0
    end = division
    for i in range(seconds * seconds_per_division):
        freq_list.append(wf.array[start:end])
        start += division
        end += division
    # apply hamming window to deal with spikes from transitions
    # get RMS
    h = np.array(hamming(freq_list[0].size))
    m = []
    for i, a in enumerate(freq_list):
        freq_list[i] = freq_list[i] * h
        freq_list[i] = freq_list[i] ** 2
        a = np.average(freq_list[i])
        a = sqrt(a)
        m.append(a)
    m = np.array(m)
    return m

def plot_sweeps(sweeps: list, 
        offset: float = 0, 
        seconds_per_division = 1, 
        frequencies: list = None):

    # takes a list of wav files and a description of the files 
    # and plots their frequency response
    # see write_sweep_wav for expected files
    # files should be trimmed to the start and end of the wavs

    if frequencies == None:
        frequencies = gen_sweep_freqs(1000, sqrt(sqrt(2)))

    s = {}
    m = 0

    for i in sweeps:
        s[i] = analyze_sweep(i, seconds_per_division)
        m = max(m, max(s[i]))

    for i in sweeps:
    # scale to max value then convert to dBV
        s[i] = convert_to_dBV(s[i] / m)
        s[i] += offset

    # plot
    import_plt()
    fig,ax = plt.subplots()
    plt.xlabel("hz")
    plt.ylabel("dBV")
    # gridlines
    plt.grid()
    ax.set_xscale('log')
    # 3dB per gridline
    major_y_ticks = np.arange(100, -100, -3)
    ax.set_yticks(major_y_ticks)
    # frequency label gets busy if you plot all of them
    # use every 4th frequency instead
    fewer_freqs = [x for i, x in enumerate(frequencies) if i % 4 == 2]
    ax.set_xticks(fewer_freqs)
    # we don't want hz in scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter())
    point_size_factor = 5
    for k, v in s.items():
        plt.scatter(frequencies, v, label=k, s=point_size_factor * np.ones_like(frequencies))
        plt.plot(frequencies, v[:len(frequencies)], label = k, alpha = .5)
    # the legend ended up pretty noisy
    #plt.legend()
    plt.title("Frequency Response")
    plt.show()

def convert_to_dBV(a: float) -> float:
    a = np.abs(a)
    a = np.log10(a)
    a = 20* a
    return a

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
