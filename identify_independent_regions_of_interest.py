import cv2
import numpy as np
import plotly.graph_objects as go
import scipy
from scipy.ndimage import convolve1d

# Load the polar transformed and reduced frames
frames_polar_array = np.load("rpm_variable_23723_frames_0.15_channels_1_space_polar.npy")
print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Extract random sample of signals
n_samples = 5000#00
frames_polar_array = frames_polar_array[np.random.choice(frames_polar_array.shape[0], n_samples, replace=False), :, :]

# # Do a robust standardization
# frames_polar_array_norm = (frames_polar_array - np.median(frames_polar_array, axis=0)) / (np.percentile(frames_polar_array, 75, axis=0) - np.percentile(frames_polar_array, 25, axis=0))
# # Remove NaNs

frames_polar_array_norm = (frames_polar_array -np.mean(frames_polar_array, axis=0)) / np.std(frames_polar_array, axis=0)

frames_polar_array_norm[np.isnan(frames_polar_array_norm)] = 0
frames_polar_array_norm[np.isinf(frames_polar_array_norm)] = 0


# Do convolution over the radial direction to detect "shear"
window_size = 20
half_filter  = np.ones(window_size) / (2*window_size)
filter = np.concatenate((-half_filter, half_filter))

# Convolve over axis 2
convolved = convolve1d(frames_polar_array_norm, filter, axis=2, mode='constant', cval=0.0)
# Stack all frames on top of each other so the typcial "shear" variance can be computed for a given channel (radius)
convolved = convolved.reshape(-1, convolved.shape[2])

# Get the IQR of the gradient score
sdev = np.std(convolved, axis=0)
# Get the two most prominent positive peaks
peak_indices = scipy.signal.find_peaks(sdev, prominence=0.1, distance=window_size)
print("Peak indices: ", peak_indices)

# Write the peak indexes as npy file as fraction of the total number of channels
peak_fractions = peak_indices[0] / sdev.shape[0]
np.save("peak_indices.npy", peak_fractions)
print("Peak fractions: ", peak_fractions)

# Plot the variance using plotly under a photo the first frame
fig = go.Figure()

# Plot the first frame as heat map
# dont show colorbar
fig.add_trace(go.Heatmap(z=frames_polar_array[0], colorscale='gray', showscale=False))


fig.add_trace(go.Scatter(x=np.arange(sdev.shape[0]), y=sdev / sdev.max() * frames_polar_array_norm[0].shape[0], mode='lines', name='Filter variance for random angles and time steps'))

# Plot the peaks as vertical lines
for i,peak in enumerate(peak_indices[0]):
    fig.add_trace(go.Scatter(x=[peak, peak], y=[0, frames_polar_array_norm[0].shape[0]],
                             mode='lines',
                             name='Component boundary {0}'.format(i+1),
                                line=dict(width=4, color='red')
                             ))
fig.update_layout(title="Identify independent regions of interest", xaxis_title="Radial axis (Channel)", yaxis_title="Angle (0-360 degrees)")
fig.show()

# Write the figure in reports directory as png
fig.write_image("reports/identify_independent_regions_of_interest.png")



