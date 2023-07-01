import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from music import music_algorithm, estimate_wave_velocity


class LocalLinearSpeedEstimator(object):
    def __init__(self, wave_data,fs=2000,max_expected_rps=40):
        # Wave data has shape (n_time_steps, n_locations)

        # Remove the mean
        self.music_spectrum = None
        self.wave_data = wave_data - np.mean(wave_data, axis=0)
        self.n_time_steps = wave_data.shape[0]
        self.n_locations = wave_data.shape[1]
        self.max_expected_rps = max_expected_rps

        self.fs = fs

    def get_rps_estimate(self):
        rps_values, music_spectrum_score = music_algorithm(self.wave_data, 1,
                                                           n_thetas=2000,
                                                           max_expected_rps=self.max_expected_rps,fs=self.fs)
        rps_opt = rps_values[np.argmax(music_spectrum_score)]
        self.music_spectrum = music_spectrum_score
        return rps_opt

        # # Compute the velocity from the cross correlation
        # corr_mat = self.wave_data @ self.wave_data.T # time x time
        # corr_mat = np.abs(corr_mat)
        # corr_mat = corr_mat / np.max(corr_mat)
        # corr_mat = corr_mat - np.eye(corr_mat.shape[0])
        #
        # # Get the lag value for each time step
        # lag_values = np.argmax(corr_mat, axis=1)
        # # Get the relative lag values
        # lag_values = lag_values - np.arange(lag_values.shape[0])
        # # Get the velocity
        # velocity = np.median(lag_values) / self.n_time_steps








class TimeVaryingSpeedEstimator():
    def __init__(self, wave_data, window_length = 254, overlap = 0.5,fs=2000,n_jobs=8,max_expected_rps=40):
        self.rps_estimates = None

        # Pad the wave data with start and end values
        n_padding = window_length // 2
        wave_data = np.pad(wave_data, ((n_padding, n_padding), (0, 0)), 'constant', constant_values=0)

        self.wave_data = wave_data
        self.window_length = window_length
        self.max_expected_rps = max_expected_rps
        self.overlap = overlap
        self.fs = fs

        self.njobs = n_jobs

        # Make a list of the indices of the start of each window

        if overlap<1:
            self.window_start_indices = np.arange(0, self.wave_data.shape[0] - self.window_length, int(self.window_length * (1 - self.overlap)))
        elif overlap == 1:
            self.window_start_indices = np.arange(0, self.wave_data.shape[0] - self.window_length, 1)


    def get_time_varying_rps_estimate(self):
        # Compute the speed estimate for each window
        def process(wave_data):
            estimator = LocalLinearSpeedEstimator(wave_data,fs=self.fs,max_expected_rps=self.max_expected_rps)
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




