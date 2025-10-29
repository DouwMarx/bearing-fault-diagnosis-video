import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter, spectrogram


class LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive():
    def __init__(self,
                 peak_fault=2,
                 peak_ref=5,
                 reference_band=[0.1, 0.5],  # Band as fraction of nyquist
                 faulty_band=[0.4, 0.7],  # Band as fraction of nyquist (7000Hz is around 7000/(20480/2) = 0.68)
                 f_fault=108.1,  # Hz (Bearing fault frequency, BPFO)
                 f_ref=273.7,  # Hz (Gear mesh frequency) @ 1500 RPM
                 sig_duration=1,  # seconds
                 fs = 20480
                 ):

        self.f_ref = f_ref
        self.f_fault = f_fault
        self.peak_fault = peak_fault
        self.peak_ref = peak_ref
        self.reference_band = reference_band
        self.faulty_band = faulty_band
        self.sig_duration = sig_duration

        self.time = np.linspace(0, sig_duration, int(sig_duration*fs))

        self.signal_length = len(self.time)


    def get_bandpass_noise(self, lowcut, highcut, order=5):
        """

        :param lowcut: low-cut frequency as a fraction of the Nyquist frequency
        :param highcut: high cut frequency as a fraction of the Nyquist frequency
        :param order:  order of the filter
        :return:
        """
        b, a = butter(order, [lowcut, highcut], btype='bandpass')
        x = filtfilt(b, a, np.random.randn(self.signal_length))
        x = x / np.std(x)  # Standardize the signal to have unit variance
        return x

    def get_impulses(self, amplitude, frequency, epsilon=1e-6):

        s = 0.5 * np.cos(2 * np.pi * frequency * self.time) + 0.5
        return amplitude * epsilon * s / (1 + epsilon - s)

    def get_reference_component(self):
        return self.get_impulses(self.peak_ref, self.f_ref, epsilon=1e-1) * self.get_bandpass_noise(
            self.reference_band[0], self.reference_band[1])

    def get_faulty_component(self):
        return self.get_impulses(self.peak_fault, self.f_fault, epsilon=1e-2) * self.get_bandpass_noise(
            self.faulty_band[0], self.faulty_band[1])


f_fault = 108.1  # Hz (Bearing fault frequency, BPFO)

fs = 40960
sig_data = LowFreqBandpassImpulsiveVSHighFreqBandpassImpulsive(f_fault=f_fault, sig_duration=0.5, fs=fs)
# Get fautly signal
faulty_signal = sig_data.get_faulty_component()

# Plot the faulty signal
plt.figure(figsize=(10, 6))
plt.plot(sig_data.time, faulty_signal)
plt.title("Faulty Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Compute spectogram
# make window length to be half of fault period
fault_period = 1/f_fault
window_length = int(fault_period*fs/4)
f, t, Sxx = spectrogram(faulty_signal, fs=fs, nperseg= window_length, noverlap=int(0.9*window_length))

# Plot the spectogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# Do DMD on the spectogram
# Compute the correlation matrix
correlation_matrix = np.matmul(Sxx, Sxx.conj().T) / Sxx.shape[1]

# Eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors =  np.linalg.eig(correlation_matrix)
