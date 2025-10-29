import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import hilbert

from music import music_algorithm, estimate_phase_velocity


# Assumption is that wave numbers or frequency is not a function of space


class WaveGenerator(object):
    def __init__(
        self, wave_number_as_function_of_t, frequency_as_function_of_t, component_phases
    ):
        self.wave_number_as_function_of_t = wave_number_as_function_of_t  # spatial oscillations / distance between spatial locations
        self.frequency_as_function_of_t = frequency_as_function_of_t  # temporal oscillations / time between time steps
        self.component_phases = component_phases  # phase of each wave component

        # Check that the functions return arrays of the same length (Each wave component has a wave number and frequency)
        if len(self.wave_number_as_function_of_t(0)) != len(
            self.frequency_as_function_of_t(0)
        ):
            raise ValueError(
                "The wave number and frequency functions must return arrays of the same length"
            )

    def generate_wave_data(self, n_timesteps, n_locations):
        space = np.arange(
            n_locations
        )  # This discretization means that the distance between spatial locations is 1 and the distance the time steps is also 1
        time = np.arange(n_timesteps)

        wave_data = np.zeros((n_timesteps, n_locations))
        for t_index in range(n_timesteps):  # Loop through all time steps
            for wave_number, frequency, phase in zip(
                self.wave_number_as_function_of_t(time[t_index]),
                self.frequency_as_function_of_t(time[t_index]),
                self.component_phases,
            ):  # Add each frequency component together
                wave_data[t_index, :] += np.sin(
                    wave_number * space * 2 * np.pi / n_locations
                    + frequency * time[t_index] * 2 * np.pi / n_timesteps
                    + phase
                )

        # wave_number*space*2*np.pi  oscillations/(distance between spatial locations) * (number of spatial samples) * 2*pi radians = n_oscillations * 2*pi radians
        return wave_data

    def show_wave_data_in_2D(self, wave_data):
        plt.figure()
        plt.imshow(wave_data)
        # plt.show()

    def animate_wave(self, wave_data):
        # Show scatter plot of wave that evolves in time
        fig = go.Figure(
            data=[go.Scatter(x=np.arange(wave_data.shape[1]), y=wave_data[0, :])],
            layout=go.Layout(
                # xaxis=dict(range=[0, 2*np.pi], autorange=False),
                # yaxis=dict(range= [ np.min(wave_data), np.max(wave_data)], autorange=False),
                title="Time = 0",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {
                                            "duration": 100,  # 100*wave_data.shape[0],
                                            "redraw": True,
                                        },
                                        "transition": {"duration": 0},
                                        "fromcurrent": True,
                                    },
                                ],
                            )
                        ],
                    ),
                ],
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Scatter(x=np.arange(wave_data.shape[1]), y=wave_data[k, :])
                    ],
                    layout=go.Layout(
                        title_text="Time = " + str(k),
                        # xaxis=dict(range=
                        #            autorange=False),
                        # yaxis=dict(range= [ np.min(wave_data), np.max(wave_data)], autorange=False),
                    ),
                )
                for k in range(wave_data.shape[0])
            ],
        )
        fig.show()


class LocalLinearSpeedEstimator(object):
    def __init__(self, wave_data):
        self.wave_data = wave_data
        self.n_time_steps = wave_data.shape[0]
        self.n_locations = wave_data.shape[1]

    # def spatial_amplitude_independent_fft2d(self):
    #     # Compute the 2D FFT separately so that the amplitudes can be modified and the signals windowed
    #     # Window the wave data over space
    #     wave_data =  self.wave_data * np.hanning(self.wave_data.shape[1]) # TODO: might not be necessary for periodic data
    #     fft_space = np.fft.rfft(wave_data, axis=1) # Compute the 1D FFT over space
    #     # Rescale such that all spatial frequencies have amplitude 1
    #     fft_space = fft_space / np.abs(fft_space[:,0])[:,None]
    #
    #     # Window the wave data over time
    #     wave_data =  self.wave_data * np.hanning(self.wave_data.shape[0])
    #     fft_2D = np.fft.rfft(fft_space, axis=0) # Compute the 1D FFT over time
    #     fft_2D = np.abs(fft_2D) # Take the magnitude of the complex numbers
    #
    #     wave_numbers = np.fft.rfftfreq(self.wave_data.shape[1], d=1/self.wave_data.shape[1]) # Compute the wave numbers
    #     frequencies = np.fft.rfftfreq(self.wave_data.shape[0], d=1/self.wave_data.shape[0]) # Compute the frequencies

    def get_fft_2D_magnitude(self, window_time=True, window_space=True):
        wave_data = self.wave_data

        # Window over the time and space dimensions if desired
        if window_time:
            wave_data = wave_data * np.tile(
                np.hanning(self.n_time_steps)[:, None], (1, self.n_locations)
            )
        if window_space:
            wave_data = wave_data * np.tile(
                np.hanning(self.n_locations), (self.n_time_steps, 1)
            )

        # Remove the DC component
        wave_data = wave_data - np.mean(wave_data)

        fft_2D = np.fft.fft2(wave_data)  # Compute the 2D FFT
        fft_2D = fft_2D  # Keeping both positive and negative frequencies since we want to be able to have negative phase speeds
        fft_2D = np.abs(fft_2D)  # Take the magnitude of the complex numbers

        # # Now compute the wave numbers and frequency for each element in the 2D FFT using fftfreq (Up to Nyquist)
        wave_numbers = np.fft.fftfreq(
            self.wave_data.shape[0], d=1 / self.wave_data.shape[0]
        )
        frequencies = np.fft.fftfreq(
            self.wave_data.shape[1], d=1 / self.wave_data.shape[1]
        )

        # Discard the high frequencies above the Nyquist frequency: numpy does [0, 1, 2, ..., n/2, -(n/2-1), ..., -1]

        return fft_2D, wave_numbers, frequencies

    def show_2D_fft(self):
        plt.figure()
        fft, wave_numbers, frequencies = self.get_fft_2D_magnitude()

        plt.imshow(
            fft,
            aspect="auto",
            origin="lower",
            cmap="jet",
        )
        plt.xlabel("Frequency [oscillations / time]")
        plt.ylabel("Wave number [oscillations / space]")

    def get_phase_velocity_prominence(self):
        # Generate an array that has [wave number, frequency, phase velocity, amplitude] a columns from the 2D FFT
        fft_2D, wave_numbers, frequencies = self.get_fft_2D_magnitude()

        # Compute the phase velocity
        wave_numbers, frequencies = np.meshgrid(wave_numbers, frequencies)
        phase_velocity = frequencies / wave_numbers

        # Create a dataframe with the wave number, frequency, phase velocity, and amplitude
        df = pd.DataFrame(
            {
                "wave number": wave_numbers.flatten(),
                "frequency": frequencies.flatten(),
                "phase velocity": phase_velocity.flatten(),
                "amplitude": fft_2D.flatten(),
            }
        )

        # Drop all the rows with wave numbers that would imply less than 1 cycle in the window
        # We expect all spatial patterns to appear at least once in the window but not less than that
        # df = df[df["wave number"] > 1/self.wave_data.shape[1]] # Do both negative and positive wave numbers?

        # Drop all rows where the spatial frequency is higher than the Nyquist frequency
        nyquist_space = df["wave number"].max() / 2
        nyquist_frequency = df["frequency"].max() / 2
        df = df[df["wave number"] < nyquist_space]
        df = df[df["wave number"] > -nyquist_space]
        df = df[df["frequency"] < nyquist_frequency]
        df = df[df["frequency"] > -nyquist_frequency]

        # Make a new dataframe where the amplitudes in df with the same phase velocity are summed
        # Discard the wave number and frequency columns
        df = (
            df.groupby(["phase velocity"])
            .sum()
            .reset_index()[["phase velocity", "amplitude"]]
        )

        # Remove columns with NaNs or infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # Weight the phase velocity by the amplitude
        df["weighted phase velocity"] = df["phase velocity"] * (
            df["amplitude"] ** 2 / (df["amplitude"] ** 2).sum()
        )

        return df

    def get_wave_velocity_estimate(self):
        df = self.get_phase_velocity_prominence()
        # return df["weighted phase velocity"].sum()

        # Return phase velocity with the highest amplitude
        return df.loc[df["amplitude"].idxmax()]["phase velocity"]

    def show_phase_velocity_prominence(self):
        df = self.get_phase_velocity_prominence()
        plt.figure()
        plt.scatter(df["phase velocity"], df["amplitude"])
        plt.xlabel("Phase velocity")
        plt.ylabel("Amplitude")

        estimate = self.get_wave_velocity_estimate()

        # Show vertical lines at the average weighted phase velocity
        plt.axvline(estimate, color="r", linestyle="--")
        # plt.show()


class TimeVaryingSpeedEstimator:
    def __init__(self, wave_data, window_length=254, overlap=0.5):
        self.wave_data = wave_data
        self.window_length = window_length
        self.overlap = overlap

        # Make a list of the indices of the start of each window
        self.window_start_indices = np.arange(
            0,
            self.wave_data.shape[0] - self.window_length,
            int(self.window_length * (1 - self.overlap)),
        )

    def get_wave_velocity_estimate(self):
        # Compute the speed estimate for each window
        speed_estimates = []
        for window_start_index in self.window_start_indices:
            estimator = LocalLinearSpeedEstimator(
                self.wave_data[
                    window_start_index : window_start_index + self.window_length, :
                ]
            )
            speed_estimates.append(estimator.get_wave_velocity_estimate())
        return speed_estimates

    def show_wave_velocity_estimate(self):
        plt.figure()
        plt.plot(self.window_start_indices, self.get_wave_velocity_estimate())
        plt.xlabel("Time")
        plt.ylabel("Wave velocity estimate")
        # plt.show()


if __name__ == "__main__":
    n_space = 100
    # n_time = 50# 1100
    n_time = 1100

    np.random.seed(0)

    n_components = 5

    # constant_wave_numbers = 1/n_space + 0.4 * np.random.rand(n_components)#  np.array([1/n_space]) #0.1+0.2 * np.random.rand(n_components) #  np.array([1/100]) # Oscillations/spatial sample
    constant_wave_numbers = (
        (0.01 + 0.01 * np.random.rand(n_components)) * n_space
    )  # np.array([1/n_space]) #0.1+0.2 * np.random.rand(n_components) #  np.array([1/100]) # Oscillations/spatial sample
    # i.e 1/n_space would be 1 oscillation over the entire space (min we expect to see)
    # 1/2 would be 1 oscillation every 2 spatial samples: Nyquist frequency
    random_phases = 2 * np.pi * np.random.rand(n_components)

    phase_velocity_as_function_of_t = (
        lambda t: 1
    )  # 1 # Spatial samples traversed per temporal sample
    print("wave numbers: ", constant_wave_numbers)
    print("true phase velocity at t=0: ", phase_velocity_as_function_of_t(0))

    wave_number_as_function_of_t = (
        lambda t: constant_wave_numbers
    )  # Spatial oscillations per spatial sample

    # v = omega/k, so omega = v*k
    frequency_as_function_of_t = (
        lambda t: constant_wave_numbers * phase_velocity_as_function_of_t(t)
    )  # Temporal oscillations per temporal sample
    wave_generator = WaveGenerator(
        wave_number_as_function_of_t, frequency_as_function_of_t, random_phases
    )
    wave_data = wave_generator.generate_wave_data(n_time, n_space)
    # Add a small amount of noise
    wave_data += 0.1 * np.random.randn(n_time, n_space)

    # wave_generator.animate_wave(wave_data)

    # estimator = LocalLinearSpeedEstimator(wave_data)

    # print("wave velocity estimate: ", estimator.get_wave_velocity_estimate())

    # estimator.show_2D_fft()
    # estimator.show_phase_velocity_prominence()

    # print("Estimation error: ", estimator.get_wave_velocity_estimate() - phase_velocity_as_function_of_t(0))

    # time_varying_estimator = TimeVaryingSpeedEstimator(wave_data,window_length=n_time//10,overlap=0.5)
    # time_varying_estimator.show_wave_velocity_estimate()

    # Convert the spatial signal into an analytical signal
    # wave_data = hilbert(wave_data, axis=1)
    # wave_data = hilbert(wave_data, axis=0)
    # X = np.fft.fft(wave_data)*np.hanning(wave_data.shape[0])
    # X = np.fft.fft2(wave_data)
    X = wave_data
    # X = wave_data*np.hanning(wave_data.shape[0])
    # X = hilbert(wave_data, axis=0)

    thetas, phase_velocities, music_spectrum = music_algorithm(X.transpose(), 1)
    # thetas, phase_velocities,music_spectrum = music_algorithm(wave_data.transpose(), 1)
    # thetas, phase_velocities,music_spectrum = music_algorithm(wave_data, 1)

    # k_values, music_spectrum,estimate= estimate_phase_velocity(wave_data,1,1)

    plt.figure()
    # plt.plot(k_values,music_spectrum)
    plt.plot(phase_velocities, music_spectrum)
    plt.xlabel("theta")
    plt.ylabel("Phase velocity")

    plt.show()
