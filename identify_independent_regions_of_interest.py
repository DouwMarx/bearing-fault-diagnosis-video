import cv2
import numpy as np
import plotly.graph_objects as go
import scipy
from scipy.ndimage import convolve1d

# Load the rpm_variable_100_frames_polar.npy data
frames_polar_array = np.load("rpm_variable_23723_frames_0.1_channels_1_space_polar.npy")


print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Extract random sample of 1000 frames
n_samples = 5000# 50#00
frames_polar_array = frames_polar_array[np.random.choice(frames_polar_array.shape[0], n_samples, replace=False), :, :]

# Robust standardization
frames_polar_array_norm = (frames_polar_array - np.median(frames_polar_array, axis=0)) / (np.percentile(frames_polar_array, 75, axis=0) - np.percentile(frames_polar_array, 25, axis=0))
# Remove NaNs
frames_polar_array_norm[np.isnan(frames_polar_array_norm)] = 0
frames_polar_array_norm[np.isinf(frames_polar_array_norm)] = 0

# Do convolution over the radial direction to detect "shear"
window_size = 15
half_filter  = np.ones(window_size) / (2*window_size)
filter = np.concatenate((-half_filter, half_filter))

# Convolve over axis 2
convolved = convolve1d(frames_polar_array_norm, filter, axis=2, mode='constant', cval=0.0)
# Stack all frames on top of each other so the typcial "shear" variance can be computed for a given channel (radius)
convolved = convolved.reshape(-1, convolved.shape[2])

# Get the IQR of the gradient score
IQR = np.percentile(convolved, 75, axis=0) - np.percentile(convolved, 25, axis=0)
# Get the two most prominent positive peaks
peak_indices = scipy.signal.find_peaks(IQR, prominence=0.1, distance=window_size)
print("Peak indices: ", peak_indices)

# Write the peak indexes as npy file as fraction of the total number of channels
np.save("peak_indices.npy", peak_indices[0] / IQR.shape[0])

# Plot the variance using plotly under a photo the first frame
fig = go.Figure()

# Plot the first frame as heat map
# dont show colorbar
fig.add_trace(go.Heatmap(z=frames_polar_array[0], colorscale='gray', showscale=False))


fig.add_trace(go.Scatter(x=np.arange(IQR.shape[0]), y=IQR / IQR.max() * frames_polar_array_norm[0].shape[0], mode='lines', name='Variance'))

# Plot the peaks as vertical lines
for i,peak in enumerate(peak_indices[0]):
    fig.add_trace(go.Scatter(x=[peak, peak], y=[0, frames_polar_array_norm[0].shape[0]],
                             mode='lines',
                             name='Component boundary {0}'.format(i),
                                line=dict(width=4, color='red')
                             ))
fig.update_layout(title="Variance of the gradient score", xaxis_title="channel", yaxis_title="Variance")
fig.show()

# Write the figure in reports directory as png
fig.write_image("reports/identify_independent_regions_of_interest.png")



