import sequencer

class Hihat(Waveform):
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
    cached_array = k.array

    def __init__(self, gain: float=1):
        super().__init__(deepcopy(self.cached_array))
        self.apply_gain(gain)

class Cymbal(Waveform):
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
    cached_array = k.array
        
    def __init__(self, gain: float=1):
        super().__init__(self.cached_array)
        self.apply_gain(gain)

class Kick(Waveform):
    length = seconds_to_samples(.5)
    peak = 200
    k = WhiteNoise(length, gain = 1)
    lp = LowPassFilter(250, gain = .5)
    k.apply_filter(lp)
    k.normalize()
    s = BentSine(150, 100, length, gain=.5)
    k = add_waveforms(k, s)
    fade_in = QuadraticFadeIn(peak)
    k.apply_envelope(fade_in)
    fade_out = QuadraticFadeOut( length - peak)
    k.apply_envelope(fade_out)
    k.normalize()
    cached_array = k.array        

    def __init__(self, gain: float=1):
        super().__init__(deepcopy(self.cached_array))
        self.apply_gain(gain)  

class Snare(Waveform):
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
    cached_array = k.array

    def __init__(self, gain: float=1):
        super().__init__(deepcopy(self.cached_array))
        self.apply_gain(gain)

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
