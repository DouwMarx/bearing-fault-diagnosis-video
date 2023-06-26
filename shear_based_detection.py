import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import convolve1d

# Load the rpm_variable_100_frames_polar.npy data
# frames_polar_array = np.load("rpm_variable_1000_frames_polar.npy")
frames_polar_array = np.load("rpm_variable_10000_frames_1_channels_polar.npy")

print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Compute the horizontal gradient of the frames
# Standardize the data
frames_polar_array_norm = (frames_polar_array - frames_polar_array.mean(axis=0)) / frames_polar_array.std(axis=0)

# Robust standardization
# frames_polar_array = (frames_polar_array - np.median(frames_polar_array, axis=0)) / (np.percentile(frames_polar_array, 75, axis=0) - np.percentile(frames_polar_array, 25, axis=0))

# Min max scaling
# frames_polar_array = (frames_polar_array - frames_polar_array.min(axis=0)) / (frames_polar_array.max(axis=0) - frames_polar_array.min(axis=0))


# Remove NaNs
frames_polar_array_norm[np.isnan(frames_polar_array_norm)] = 0
frames_polar_array_norm[np.isinf(frames_polar_array_norm)] = 0


# gradients = np.diff(frames_polar_array_norm, axis=2)
# Do a moving average over axis 2
# window_size = 3
# gradients = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='same'), axis=2, arr=frames_polar_array_norm)
# filter = np.array([-0.25,-0.25,0.25,0.25])
filter = np.array([-0.1,-0.1,-0.1, -0.1,- 0.1, 0.1, 0.1, 0.1, 0.1,0.1])

gradients = np.apply_along_axis(lambda x: np.convolve(x, filter, mode='same'), axis=2, arr=frames_polar_array_norm)

# For each frame, sum the gradients over the rows
# gradient_score = np.sum(gradients, axis=1)

# # Stack all frames on top of each other
gradient_score = gradients.reshape(-1, gradients.shape[2])

# Compute the variance for each channel over all samples
# variance = np.var(gradient_score, axis=0)
# Compute IQR for each channel over all samples
variance = np.percentile(gradient_score, 75, axis=0) - np.percentile(gradient_score, 25, axis=0)

# Plot the variance using plotly under a photo the first frame
fig = go.Figure()

# Plot the first frame as heat map
fig.add_trace(go.Heatmap(z=frames_polar_array[0], colorscale='Viridis'))

fig.add_trace(go.Scatter(x=np.arange(variance.shape[0]), y=variance / variance.max() * frames_polar_array_norm[0].shape[0], mode='lines', name='Variance'))
fig.update_layout(title="Variance of the gradient score", xaxis_title="channel", yaxis_title="Variance")
fig.show()


