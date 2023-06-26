import numpy as np
import matplotlib.pyplot as plt
from music import music_algorithm
from joblib import Parallel, delayed

# Assumption is that wave numbers or frequency is not a function of space

class LocalLinearSpeedEstimator(object):
    def __init__(self, wave_data,fs=2000):
        # Wave data has shape (n_time_steps, n_locations)

        # Remove the mean
        self.music_spectrum = None
        self.wave_data = wave_data - np.mean(wave_data, axis=0)
        self.n_time_steps = wave_data.shape[0]
        self.n_locations = wave_data.shape[1]

        self.fs = fs

    def get_rps_estimate(self):
        thetas, music_spectrum = music_algorithm(self.wave_data, 1)
        theta_max = thetas[np.argmax(music_spectrum)]
        self.music_spectrum = music_spectrum

        rps = theta_max * self.fs / (2 * np.pi)
        return rps

class TimeVaryingSpeedEstimator():
    def __init__(self, wave_data, window_length = 254, overlap = 0.5,fs=2000,n_jobs=6):
        self.rps_estimates = None
        self.wave_data = wave_data
        self.window_length = window_length
        self.overlap = overlap
        self.fs = fs

        self.njobs = n_jobs

        # Make a list of the indices of the start of each window
        self.window_start_indices = np.arange(0, self.wave_data.shape[0] - self.window_length, int(self.window_length * (1 - self.overlap)))

    def get_time_varying_rps_estimate(self):
        # Compute the speed estimate for each window
        def process(wave_data):
            estimator = LocalLinearSpeedEstimator(wave_data,fs=self.fs)
            return estimator.get_rps_estimate()

        rps_estimates = Parallel(n_jobs=self.njobs)(delayed(process)(self.wave_data[window_start_index:window_start_index + self.window_length, :]) for window_start_index in self.window_start_indices)

        self.rps_estimates = rps_estimates
        return rps_estimates

    def show_wave_velocity_estimate(self):
        plt.figure()
        plt.plot(self.window_start_indices, self.rps_estimates)
        plt.xlabel("Time")
        plt.ylabel("rps")
        # plt.show()




