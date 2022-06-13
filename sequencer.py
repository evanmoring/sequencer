import numpy as np
from numpy import pi 
from math import floor
import scipy.io.wavfile
import csv
import pandas as pd
import pyaudio
import time
from threading import Thread, Lock

rng = np.random.default_rng(12345)

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
    new_wf = np.add(wf_a, wf_b)
    new_wf -= .5
    return new_wf

def load_csv(filename):
    csv_dicts = []
    waveform_list = []

    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
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
            length = float(i['seq_length'])
            seq = Sequencer(length, bpm)
        else:
            wf, start = process_csv_row(i, bpm)
            seq.place_waveform(start, wf.waveform) 
    return seq

def process_csv_row(row, bpm):
    signal_type = signal_map[row['signal_type']] 
    length = float(row['length'])
    start = float(row['start'])
    gain = float(row['gain'])
    samples = samples_from_beats(bpm, length)
    if 'freq' in row:
        freq = float(row['freq'])
        waveform = signal_type(freq, samples, gain = gain)
    else:
        waveform = signal_type(samples, gain = gain)
    return waveform, start

def samples_from_beats(bpm, beat):
    samples = int((beat * 60 * SAMPLE_RATE) / bpm)
    return samples

def beats_from_samples(bpm, samples):
    beats = (samples * bpm) / (60 * SAMPLE_RATE)
    return beats

class Sequencer:
    def __init__(self, beats, bpm):
        self.mutex = Lock()
        self.bpm = bpm
        self.beats = beats 
        self.bps = self.bpm / 60
        self.beat_length = int(SAMPLE_RATE / self.bps)
        self.sequence = Rest(self.beats * self.beat_length).waveform

    def place_waveform(self, beat, waveform):
        first_sample = samples_from_beats(self.bpm, beat)
        try:
            self.sequence[first_sample: first_sample + waveform.size] = waveform
        except ValueError as e:
            print("ERROR! sequence not big enought")

    def example(self):
        for i in range(self.beats):
            if i % 2:
                osc = Sine(np.random.choice(major_pentatonic.scale), self.beat_length / 2)
                wf = osc.waveform
            else:
                wf = WhiteNoise(self.beat_length / 2 ,gain=.25).waveform
            wf = LinearFadeIn(wf, self.beat_length / 8).wave  
            wf = LinearFadeOut(wf, self.beat_length / 8).wave  
            self.place_waveform(i, wf)
        self.write_sequence("example.wav")

    def write_sequence(self, filename):
        scipy.io.wavfile.write(filename, SAMPLE_RATE, self.sequence) 

    def play_sequence(self):
        # TODO write audio in chunks
        P = pyaudio.PyAudio()
        CHUNK = 1024
        stream = P.open(rate=SAMPLE_RATE, format=pyaudio.paFloat32, channels=1, output=True)
        data = self.sequence.astype(np.float32).tostring()
        while True:
            stream.write(data)
            time.sleep(1)

        #stream.close() # this blocks until sound finishes playing

        P.terminate()

class SignalGenerator:
    def __init__(self, samples, gain = 1):
        self.samples = int(samples)
        self.seconds = samples_to_seconds(samples)
        self.gain = gain
        self.generate_waveform()

    def generate_waveform(self):
        self.waveform = None

    def visualize_waveform(self, waveform):
        for i in waveform:
            length = 100
            num = floor(i * length) + length
            print("#" * num)

    def write_waveform(self, filename):
        scipy.io.wavfile.write(filename, SAMPLE_RATE, self.waveform) 

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
        wave_array -= .5
        wave_array *= self.gain
        wave_array += .5
        self.waveform = wave_array

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
                    wave_array -= .5
                    wave_array *= self.gain
                    wave_array += .5
                    self.waveform = wave_array
                    return
                wave_array[sample_count] = value
                sample_count += 1

class WhiteNoise(SignalGenerator):
    def generate_waveform(self):
        self.waveform = np.random.rand(self.samples)
        self.waveform -= .5
        self.waveform *= self.gain
        self.waveform += .5

class Rest(SignalGenerator):
    def generate_waveform(self):
        self.waveform = np.full((self.samples),.5)

class Scale():
    def __init__(self, unison, notes):
        self.unison = unison 
        self.intervals = []
        for i in notes:
            self.intervals.append(intervals[i])
        self.intervals = np.array(self.intervals)
        self.scale = self.intervals * self.unison

class Envelope:
    def __init__(self, wave):
        self.wave = wave
        self.generate_envelope()
        self.apply_envelope()

class LinearFadeIn(Envelope):
    def __init__(self, wave, samples):
        self.samples = samples
        super().__init__(wave)

    def generate_envelope(self):
        self.envelope = np.full((self.wave.size),1)
        fade = np.arange(0, 1, 1/self.samples)
        self.envelope = np.concatenate((fade, self.envelope[fade.size:]))

    def apply_envelope(self):
        self.wave = self.wave- .5
        self.wave *= self.envelope
        self.wave = self.wave + .5

class LinearFadeOut(LinearFadeIn):
    def generate_envelope(self):
        super().generate_envelope()
        self.envelope = np.flip(self.envelope)

signal_map = {
    "sine": Sine,
    "square": Square,
    "white_noise": WhiteNoise,
}

class Filter:
    pass
    # TODO take a np.array waveform as input, run a fourier transform on it, apply appropriate filtering, and return

def test():
    while True:
        print("thread")
        time.sleep(2)

if __name__ == "__main__":
    major_pentatonic = Scale(200,["1","M2","M3","5","M6","o"])
    seq = Sequencer(16, 60)
    csv_seq = load_csv("example.csv")
    csv_seq.write_sequence("csv_seq.wav")
    seq_thread = Thread(target=csv_seq.play_sequence)
    seq_thread.daemon = True
    #seq_thread = Thread(target=test)
    seq_thread.start()
    while True:
        print("print")
        time.sleep(2)
    #csv_seq.play_sequence()
    wf3 = seq.example()
    wn = WhiteNoise(6 * SAMPLE_RATE).waveform
    wn = LinearFadeIn(wn, 2 * SAMPLE_RATE).wave
    wn = LinearFadeOut(wn, 2 * SAMPLE_RATE).wave

    scipy.io.wavfile.write("noise.wav", SAMPLE_RATE, wn)       

