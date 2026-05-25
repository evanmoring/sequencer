from sequencer.sequencer import *

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

def wf_to_envelope(wf: Waveform, samples_smoothed: int) -> Envelope:
    wf = deepcopy(wf)
    wf.normalize()
    wf.apply_gain(.5)
    wf.array += .5
    smoothed_array = smooth(wf.array, samples_smoothed)
        
    output = Envelope()
    output.envelope = smoothed_array
    return output
