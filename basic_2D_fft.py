import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Assumption is that wave numbers or frequency is not a function of space

class WaveGenerator(object):
    def __init__(self, wave_number_as_function_of_t, frequency_as_function_of_t):
        self.wave_number_as_function_of_t = wave_number_as_function_of_t
        self.frequency_as_function_of_t = frequency_as_function_of_t

        # Check that the functions return arrays of the same length
        if len(self.wave_number_as_function_of_t(0)) != len(self.frequency_as_function_of_t(0)):
            raise ValueError("The wave number and frequency functions must return arrays of the same length")

    def generate_wave_data(self,n_timesteps, n_locations):
        space = np.linspace(0, 2*np.pi, n_locations)
        time = np.linspace(0, 2*np.pi, n_timesteps)
        # Generate wave data

        wave_data = np.zeros((n_timesteps, n_locations))
        for t in range(n_timesteps):
            for wave_number, frequency in zip(self.wave_number_as_function_of_t(time[t]), self.frequency_as_function_of_t(time[t])):
                wave_data[t,:] += np.sin(wave_number*space + frequency*time[t])
        return wave_data


    def show_wave_data_in_2D(self, wave_data):
        plt.figure()
        plt.imshow(wave_data)
        plt.show()

    def animate_wave(self,wave_data):
        # Show scatter plot of wave the evolves in time
        fig = go.Figure(
            data=[go.Scatter(x=np.linspace(0, 2*np.pi, wave_data.shape[1]), y=wave_data[0,:])],
            layout=go.Layout(
                xaxis=dict(range=[0, 2*np.pi], autorange=False),
                yaxis=dict(range=[-5, 5], autorange=False),
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
        fft_2D_shifted = np.fft.fftshift(fft_2D) # Shift the zero frequency to the center
        fft_2D_shifted = fft_2D_shifted[self.wave_data.shape[0] // 2:, self.wave_data.shape[1] // 2:] # Only keep the positive frequencies (below the Nyquist)
        fft_2D_shifted = np.abs(fft_2D_shifted) # Take the magnitude of the complex numbers


        # Now compute the wave numbers and frequency for each element in the 2D FFT using fftfreq (Up to Nyquist)
        wave_numbers = np.fft.fftfreq(self.wave_data.shape[0], d=1/self.wave_data.shape[0])[:self.wave_data.shape[0] // 2]
        frequencies = np.fft.fftfreq(self.wave_data.shape[1], d=1/self.wave_data.shape[1])[:self.wave_data.shape[1] // 2]

        # The units are: wave_numbers = number of spatial cycles / length between spatial samples
        #                frequencies  = number of temporal cycles / length between temporal samples

        return fft_2D_shifted, wave_numbers, frequencies

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

        return









if __name__ == "__main__":
    r = np.random.rand(3)*2*np.pi*0.1
    r = np.concatenate((r, 7.329*r))
    # wave_number_as_function_of_t = lambda t: np.array([1,1.5])*10
    # frequency_as_function_of_t = lambda t: np.array([1/1,1.5/1])*10
    wave_number_as_function_of_t = lambda t: r
    frequency_as_function_of_t = lambda t: 10*r # 2*r # (t+1)*r
    frequency_as_function_of_t_2 = lambda t: 2*r # 2*r # (t+1)*r

    wave_generator = WaveGenerator(wave_number_as_function_of_t, frequency_as_function_of_t)
    # wave_generator_2 = WaveGenerator(wave_number_as_function_of_t, frequency_as_function_of_t_2)
    wave_data = wave_generator.generate_wave_data(300, 300)
    # wave_data_2 = wave_generator_2.generate_wave_data(300, 300)


    # wave_generator.show_wave_data_in_2D(wave_generator.generate_wave_data(100, 100))
    wave_generator.animate_wave(wave_data)
    # wave_generator.animate_wave(wave_data_2)

    estimator = LocalLinearSpeedEstimator(wave_data)
    estimator.show_2D_fft()


