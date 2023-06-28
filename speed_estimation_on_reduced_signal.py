import numpy as np
import plotly.graph_objects as go
from estimators import LocalLinearSpeedEstimator, TimeVaryingSpeedEstimator

# Load the .npy dict with the data
signals = np.load("pca_rpm_variable_23723_frames_0.1_channels_1_space_polar.npy.npy" , allow_pickle=True).item()
print(signals.keys())

# component = "outer"
# component = "inner"
component = "cage"
wave_data = signals[component]

if component == "cage":
    max_expected_rps = 35
elif component == "inner":
    max_expected_rps = 15
elif component == "outer":
    max_expected_rps = 4


# Remove the mean
wave_data = wave_data - np.mean(wave_data, axis=0)

window_size = 100

estimator = TimeVaryingSpeedEstimator(wave_data,
                                      window_length=window_size,
                                      overlap=1,
                                      fs=2000,
                                      max_expected_rps=max_expected_rps
                                      )
rps = estimator.get_time_varying_rps_estimate()

# Save the speed profile
np.save("{}_speed_profile.npy".format(component), rps)

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
                            title="rps"
                        )
                    ))
fig.show()
# Save the figure in reports directory
fig.write_image("reports/speed_estimate_{}.png".format(component))






