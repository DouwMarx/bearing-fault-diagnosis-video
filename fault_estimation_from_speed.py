# Load the signal
import numpy as np
import scipy
import plotly.graph_objects as go

fs = 2000
cage_signal = np.load("cage_speed_profile.npy")
inner_signal = np.load("inner_speed_profile.npy")


d_over_D = 1/4.5#0.22
alpha = 20

# for name, signal in zip(["cage", "inner", "difference"], [cage_signal, inner_signal, inner_signal - cage_signal]):
name = "inner race relative to cage"
signal = inner_signal - cage_signal
# Integrate the signal to get the relative angle
relative_displacement = scipy.integrate.cumtrapz(signal, dx=1/fs, initial=0) # In relative revolutions : revs/sec becomes revs

# Get a mapping between angle (revs) and the response signal
signal_as_function_of_angle = scipy.interpolate.interp1d(relative_displacement, signal,kind="linear")

# Define a constant relative displacement angle
constant_relative_displacement_angle = np.linspace(relative_displacement[0], relative_displacement[-1],len(signal)) # Also relative revolutions

angle_sample_rate =  1 / ( np.mean(np.diff(relative_displacement)))

resampled_signal = signal_as_function_of_angle(constant_relative_displacement_angle) # Resample the signal to have constant angle

# Detrend signal using savgol filter
resampled_signal = resampled_signal - scipy.signal.savgol_filter(resampled_signal, 501, 3)

# Plot the resampled signal using plotly

fig = go.Figure(data=go.Scatter(
                        y=resampled_signal,
                        x=constant_relative_displacement_angle,
                        mode='lines',
                    ),
                    layout=go.Layout(
                        title="Resampled signal for {}".format(name),
                        xaxis=dict(
                            title="Number of revolutions"
                        ),
                        yaxis=dict(
                            title="De-trended, order tracked angular velocity (rps)"
                        )
                    ))
# fig.show()
fig.write_image("reports/resampled_{}.png".format(name))

# Show the spectrum of the resampled signal
resampled_fft = np.fft.rfft(resampled_signal-np.mean(resampled_signal))
fft_freqs = np.fft.rfftfreq(resampled_signal.shape[0], d=1/angle_sample_rate)

fig = go.Figure(data=go.Scatter(
                        y=np.abs(resampled_fft),
                        x=fft_freqs,
                        mode='lines',
                        name="Resampled signal spectrum"
                    ),
                    layout=go.Layout(
                        title="Resampled signal spectrum for {}".format(name),
                        xaxis=dict(
                            title="Angular frequency (events/relative revolution)",
                          range=[0, 10]
                        ),
                        yaxis=dict(
                            title="Magnitude"
                        )
                    ))


bpfi = 8
bpfo = (8/2)*(1-d_over_D*np.cos(np.deg2rad(alpha)))
bsf = (1/d_over_D - np.cos(np.deg2rad(alpha)))

freqs = { "BPFI": bpfi, "BPFO": bpfo, "BSF": bsf}

# Plot a labeled vertical line at each fault frequency
for fault_name,freq in freqs.items():
    fig.add_trace(go.Scatter(
                            x=[freq,freq],
                            y=[0,np.max(np.abs(resampled_fft))],
                            mode='lines',
                            name=fault_name,
    ))


fig.write_image("reports/diagnosis_{}.png".format(name))
fig.show()








