import numpy as np
import plotly.graph_objects as go
from estimators import LocalLinearSpeedEstimator, TimeVaryingSpeedEstimator

# Load the .npy dict with the data

roi_data_dict = (
    "reduced_roi_rpm_variable_23723_frames_0.15_channels_1_space_polar.npy.npy"
)

# Load the dictionary containing the numpy array data
signals = np.load(roi_data_dict, allow_pickle=True)[()]

# component = "inner"
component = "cage"

wave_data = signals[component]


if component == "inner":
    max_expected_rps = 35
elif component == "cage":
    max_expected_rps = 35
elif component == "outer":
    max_expected_rps = 4

# Swap the names of the components because they are incorrectly named
if component == "inner":
    component = "cage"
elif component == "cage":
    component = "inner"

# Remove the mean

window_size = 100

estimator = TimeVaryingSpeedEstimator(
    wave_data,
    window_length=window_size,
    overlap=1,
    # overlap=0.1,
    fs=2000,
    max_expected_rps=max_expected_rps,
)
rps = estimator.get_time_varying_rps_estimate()

print("length of rps: ", len(rps))

# Save the speed profile
np.save("{}_speed_profile.npy".format(component), rps)
# Save the speed profile as a .csv file
np.savetxt("{}_speed_profile.csv".format(component), rps, delimiter=",")

# Save the speed profile as a .txt file in prescribe name
if component == "cage":
    np.savetxt("train.txt", rps, delimiter=",")
elif component == "inner":
    np.savetxt("shaft.txt", rps, delimiter=",")

# Plot the optimal theta over time
fig = go.Figure(
    data=go.Scatter(
        y=rps,
        mode="lines",
    ),
    layout=go.Layout(
        title="RPS estimate {}".format(component),
        xaxis=dict(title="Time (frames)"),
        yaxis=dict(title="rps"),
    ),
)
fig.show()
# Save the figure in reports directory
fig.write_image("reports/speed_estimate_{}.png".format(component))
