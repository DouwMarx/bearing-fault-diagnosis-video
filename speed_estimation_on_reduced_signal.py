import numpy as np
import plotly.graph_objects as go
from music import music_algorithm
from estimators import LocalLinearSpeedEstimator, TimeVaryingSpeedEstimator

# Load the .npy dict with the data


# signals = np.load("10000_frames_pca.npy" , allow_pickle=True).item()
signals = np.load("pca_rpm_variable_23721_frames_0.2_channels_1_space_polar.npy.npy" , allow_pickle=True).item()

# component = "outer"
# component = "cage"
component = "inner"
wave_data = signals[component]

# Remove the mean
wave_data = wave_data - np.mean(wave_data, axis=0)

window_size = 50# $int(wave_data.shape[1] *0.5)

# estimator = TimeVaryingSpeedEstimator(wave_data, window_length=window_size, overlap=0.8, fs=2000)
estimator = TimeVaryingSpeedEstimator(wave_data, window_length=window_size, overlap=1, fs=2000)
rps = estimator.get_time_varying_rps_estimate()

# Save the speed profile
np.save("{}_speed_profile.npy".format(component), rps)

# Plot a heatmap of how the spectrum changes over time
# spectra = np.array(spectra)
# print("Spectra shape: ", spectra.shape)
# fig = go.Figure(data=go.Heatmap(
#                         z=spectra,
#                         colorscale='Viridis'),
#                     layout=go.Layout(
#                         title="Spectra over time",
#                         xaxis=dict(
#                             title="Time (frames)"
#                         ),
#                         yaxis=dict(
#                             title="Angle (degrees)"
#                         )
#                     ))
# fig.show()

# Plot the optimal theta over time
fig = go.Figure(data=go.Scatter(
                        y=rps,
                        mode='lines',
                    ),
                    layout=go.Layout(
                        title="Optimal theta over time",
                        xaxis=dict(
                            title="Time (frames)"
                        ),
                        yaxis=dict(
                            title="Angular velocity (degrees/frame)"
                        )
                    ))
fig.show()






