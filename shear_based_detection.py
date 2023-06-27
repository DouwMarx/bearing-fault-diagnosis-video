import cv2
import numpy as np
import plotly.graph_objects as go
import scipy
from scipy.ndimage import convolve1d

# Load the rpm_variable_100_frames_polar.npy data
# frames_polar_array = np.load("rpm_variable_1000_frames_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_1_channels_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")
frames_polar_array = np.load("rpm_variable_23723_frames_0.2_channels_1_space_polar.npy")


print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Extract random sample of 1000 frames
n_samples = 5000
frames_polar_array = frames_polar_array[np.random.choice(frames_polar_array.shape[0], n_samples, replace=False), :, :]

# Robust standardization
frames_polar_array_norm = (frames_polar_array - np.median(frames_polar_array, axis=0)) / (np.percentile(frames_polar_array, 75, axis=0) - np.percentile(frames_polar_array, 25, axis=0))
# Remove NaNs
frames_polar_array_norm[np.isnan(frames_polar_array_norm)] = 0
frames_polar_array_norm[np.isinf(frames_polar_array_norm)] = 0

# Do a moving average over axis 2
window_size = 30

half_filter  = np.ones(window_size) / (2*window_size)
filter = np.concatenate((-half_filter, half_filter))

# Convolve over axis 2
gradients = convolve1d(frames_polar_array_norm, filter, axis=2, mode='constant', cval=0.0)
# # Stack all frames on top of each other
gradient_score = gradients.reshape(-1, gradients.shape[2])

# Get the IQR of the gradient score
variance = np.percentile(gradient_score, 75, axis=0) - np.percentile(gradient_score, 25, axis=0)
# Get the two most prominent positive peaks
peak_indices = scipy.signal.find_peaks(variance, prominence=0.1, distance=10)
print("Peak indices: ", peak_indices)


# Write the peak indexes as npy file as fraction of the total number of channels
np.save("peak_indices.npy", peak_indices[0] / variance.shape[0])

# Plot the variance using plotly under a photo the first frame
fig = go.Figure()

# Plot the first frame as heat map
fig.add_trace(go.Heatmap(z=frames_polar_array[0], colorscale='Viridis'))

fig.add_trace(go.Scatter(x=np.arange(variance.shape[0]), y=variance / variance.max() * frames_polar_array_norm[0].shape[0], mode='lines', name='Variance'))

# Plot the peaks as vertical lines
for peak in peak_indices[0]:
    fig.add_trace(go.Scatter(x=[peak, peak], y=[0, frames_polar_array_norm[0].shape[0]],
                             mode='lines',
                             name='Peak',
                                line=dict(width=4)
                             ))

fig.update_layout(title="Variance of the gradient score", xaxis_title="channel", yaxis_title="Variance")


fig.show()



