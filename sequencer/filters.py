#!/usr/bin/python3
import numpy as np
from scipy.fft import  rfftfreq, rfft, irfft

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
