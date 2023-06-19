import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Assumption is that wave numbers or frequency is not a function of space

class WaveGenerator(object):
    def __init__(self, wave_number_as_function_of_t, frequency_as_function_of_t):
        self.wave_number_as_function_of_t = wave_number_as_function_of_t # spatial oscillations / distance between spatial locations
        self.frequency_as_function_of_t = frequency_as_function_of_t # temporal oscillations / time between time steps

        # Check that the functions return arrays of the same length (Each wave component has a wave number and frequency)
        if len(self.wave_number_as_function_of_t(0)) != len(self.frequency_as_function_of_t(0)):
            raise ValueError("The wave number and frequency functions must return arrays of the same length")

    def generate_wave_data(self,n_timesteps, n_locations):
        space = np.arange(n_locations) # This discretization means that the distance between spatial locations is 1 and the distance the time steps is also 1
        time = np.arange(n_timesteps)

        wave_data = np.zeros((n_timesteps, n_locations))
        for t in range(n_timesteps): # Loop through all time steps
            for wave_number, frequency in zip(self.wave_number_as_function_of_t(time[t]), self.frequency_as_function_of_t(time[t])): # Add each frequency component together
                wave_data[t,:] += np.sin(wave_number*space*2*np.pi + frequency*time[t]*2*np.pi)
        return wave_data


    def show_wave_data_in_2D(self, wave_data):
        plt.figure()
        plt.imshow(wave_data)
        plt.show()

    def animate_wave(self,wave_data):
        # Show scatter plot of wave that evolves in time
        fig = go.Figure(
            data=[go.Scatter(x=np.arange(wave_data.shape[1]), y=wave_data[0,:])],
            layout=go.Layout(
                xaxis=dict(range=[0, 2*np.pi], autorange=False),
                # yaxis=dict(range=[-5, 5], autorange=False),
                title="Time = 0",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                    method="animate",
                                    args=[ None, {"frame": {"duration": 10, "redraw": True},
                                                  "transition": {"duration":0},
                                                  "fromcurrent": True}],
                                  )]),
                ]
                ),
            frames=[go.Frame(
                data=[go.Scatter(
                    x=np.linspace(0, 2*np.pi, wave_data.shape[1]),
                    y=wave_data[k,:]
                )],
                layout=go.Layout(
                    title_text="Time = " + str(k),
                    xaxis=dict(range=[0, 2*np.pi], autorange=False),
                    yaxis=dict(range=[-5, 5], autorange=False)
                    )
                )
                for k in range(wave_data.shape[0])]
            )
        fig.show()

class LocalLinearSpeedEstimator(object):
    def __init__(self, wave_data):
        self.wave_data = wave_data

    def get_fft_2D_magnitude(self):
        fft_2D = np.fft.fft2(self.wave_data) # Compute the 2D FFT
        fft_2D = np.fft.fftshift(fft_2D) # Shift the zero frequency to the center
        fft_2D = fft_2D[self.wave_data.shape[0] // 2:, self.wave_data.shape[1] // 2:] # Only keep the positive frequencies (below the Nyquist)
        fft_2D = np.abs(fft_2D) # Take the magnitude of the complex numbers

        # Compute the 2D FFT separately so that the amplitudes can be modified and the signals windowed
        # Window the wave data over space
        # wave_data =  self.wave_data * np.hanning(self.wave_data.shape[1]) # TODO: might not be necessary for periodic data
        # fft_space = np.fft.rfft(wave_data, axis=1) # Compute the 1D FFT over space
        # # Rescale such that all spatial frequencies have amplitude 1
        # fft_space = fft_space / np.abs(fft_space[:,0])[:,None]
        #
        # # Window the wave data over time
        # wave_data =  self.wave_data * np.hanning(self.wave_data.shape[0])
        # fft_2D = np.fft.rfft(fft_space, axis=0) # Compute the 1D FFT over time
        # fft_2D = np.abs(fft_2D) # Take the magnitude of the complex numbers

        # wave_numbers = np.fft.rfftfreq(self.wave_data.shape[1], d=1/self.wave_data.shape[1]) # Compute the wave numbers
        # frequencies = np.fft.rfftfreq(self.wave_data.shape[0], d=1/self.wave_data.shape[0]) # Compute the frequencies

        # # Now compute the wave numbers and frequency for each element in the 2D FFT using fftfreq (Up to Nyquist)
        # wave_numbers = np.fft.fftfreq(self.wave_data.shape[0], d=1/self.wave_data.shape[0])[:self.wave_data.shape[0] // 2]
        # frequencies = np.fft.fftfreq(self.wave_data.shape[1], d=1/self.wave_data.shape[1])[:self.wave_data.shape[1] // 2]
        wave_numbers = np.fft.fftfreq(self.wave_data.shape[0], d=1)[:self.wave_data.shape[0] // 2]
        frequencies = np.fft.fftfreq(self.wave_data.shape[1], d=1)[:self.wave_data.shape[1] // 2]

        # The units are: wave_numbers = number of spatial cycles / length between spatial samples
        #                frequencies  = number of temporal cycles / length between temporal samples

        return fft_2D, wave_numbers, frequencies

    def show_2D_fft(self):
        # Show 2D FFT of wave data. Show only the positive frequencies
        plt.figure()
        # plt.contourf(
        #     self.get_fft_2D_magnitude())

        fft, wave_numbers, frequencies = self.get_fft_2D_magnitude()
        plt.imshow(fft,
                   extent = [frequencies[0], frequencies[-1], wave_numbers[0], wave_numbers[-1]],
                   aspect='auto', origin='lower', cmap='jet')
        plt.xlabel("Frequency")
        plt.ylabel("Wave number")
        plt.show()

    def get_phase_velocity_prominence(self):

        # Generate an array that has [wave number, frequency, phase velocity, amplitude] a columns from the 2D FFT
        fft_2D_shifted, wave_numbers, frequencies = self.get_fft_2D_magnitude()

        # Compute the phase velocity
        wave_numbers, frequencies = np.meshgrid(wave_numbers, frequencies)
        phase_velocity = frequencies / wave_numbers

        # Create a dataframe with the wave number, frequency, phase velocity, and amplitude
        df = pd.DataFrame({
            "wave number": wave_numbers.flatten(),
            "frequency": frequencies.flatten(),
            "phase velocity": phase_velocity.flatten(),
            "amplitude": fft_2D_shifted.flatten()
        })

        # Remove columns with NaNs or infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # Weight the phase velocity by the amplitude
        df["weighted phase velocity"] = df["phase velocity"] * (df["amplitude"]**2 / (df["amplitude"]**2).sum())

        # Drop all the rows with wave numbers that would imply less than 1 cycle in the window
        df = df[df["wave number"]  > 1/self.wave_data.shape[1]]

        return df

    def get_wave_velocity_estimate(self):
        df = self.get_phase_velocity_prominence()
        return df["weighted phase velocity"].sum()

    def show_phase_velocity_prominence(self):
        df = self.get_phase_velocity_prominence()
        plt.figure()
        plt.scatter(df["phase velocity"], df["amplitude"])
        plt.xlabel("Phase velocity")
        plt.ylabel("Amplitude")

        # Show vertical lines at the average weighted phase velocity
        plt.axvline(df["weighted phase velocity"].sum(), color='r', linestyle='--')
        plt.show()

class TimeVaryingSpeedEstimator():
    def __init__(self, wave_data, window_length = 254, overlap = 0.5):
        self.wave_data = wave_data
        self.window_length = window_length
        self.overlap = overlap

        # Make a list of the indices of the start of each window
        self.window_start_indices = np.arange(0, self.wave_data.shape[0] - self.window_length, int(self.window_length * (1 - self.overlap)))

    def get_wave_velocity_estimate(self):
        # Compute the speed estimate for each window
        speed_estimates = []
        for window_start_index in self.window_start_indices:
            estimator = LocalLinearSpeedEstimator(self.wave_data[window_start_index:window_start_index + self.window_length, :])
            speed_estimates.append(estimator.get_wave_velocity_estimate())
        return speed_estimates

    def show_wave_velocity_estimate(self):
        plt.figure()
        plt.plot(self.window_start_indices, self.get_wave_velocity_estimate())
        plt.xlabel("Time")
        plt.ylabel("Wave velocity estimate")
        plt.show()


if __name__ == "__main__":
    n_space = 100
    n_time = 100

    rand_freqs = 0.3*np.random.rand(5) #  np.array([1/100]) # Oscillations/spatial sample
                                   # i.e 1/n_space would be 1 oscillation over the entire space (min we expect to see)
                                   # 1/2 would be 1 oscillation every 2 spatial samples: Nyquist frequency

    phase_velocity_as_function_of_t = lambda t: 0.5#np.sin(10*t)+ 1.5# 0.5 # Spatial samples per temporal sample
    print("wave numbers: ", rand_freqs)
    print("phase velocity at t=0: ", phase_velocity_as_function_of_t(0))

    # wave_number_as_function_of_t = lambda t: np.array([1,1.5])*10
    # frequency_as_function_of_t = lambda t: np.array([1/1,1.5/1])*10

    wave_number_as_function_of_t = lambda t: rand_freqs
    frequency_as_function_of_t = lambda t:  rand_freqs * phase_velocity_as_function_of_t(t)

    wave_generator = WaveGenerator(wave_number_as_function_of_t, frequency_as_function_of_t)
    wave_data = wave_generator.generate_wave_data(n_time, n_space)

    wave_generator.animate_wave(wave_data)

    estimator = LocalLinearSpeedEstimator(wave_data)
    estimator.show_2D_fft()
    estimator.show_phase_velocity_prominence()

    time_varying_estimator = TimeVaryingSpeedEstimator(wave_data,window_length=n_time//10,overlap=0.5)
    time_varying_estimator.show_wave_velocity_estimate()





