import cv2
import numpy as np
from sklearn.decomposition import FastICA, PCA, NMF, SparseCoder, SparsePCA, DictionaryLearning

# Load the rpm_variable_100_frames_polar.npy data
# frames_polar_array = np.load("rpm_variable_100_frames_polar.npy")
frames_polar_array = np.load("../rpm_variable_1000_frames_polar.npy")

print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Apply ICA to the first frame over the rows
frame = frames_polar_array[0]
print("Dimensions of the first frame: ", frame.shape)

# Stack each frame on top of each other so that columns are like features
data = np.vstack(frames_polar_array)



print("Dimensions of the data: ", data.shape)


# ica = FastICA(n_components=3, random_state=0 )
# Apply a very sparse ICA to the data
# decompose = FastICA(n_components=3, whiten="unit-variance", fun="exp", random_state=0)
# decompose = SparsePCA(n_components=3,alpha=2)#, alpha=0.0001, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None)
# decompose = PCA(n_components=3)
# Non negative matrix factorization
# decompose = NMF(n_components=4, init='random', random_state=0, l1_ratio=1, max_iter=1000)#, alpha_W=0.01)
decompose = DictionaryLearning(n_components=3, alpha=1)



# Rescale the data to be between 0 and 1
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
# Normalize the data
# data = (data - data.mean(axis=0)) / data.std(axis=0)
# Rescale by dividing by IQR

# Make NaNs 0
data[np.isnan(data)] = 0
data[np.isinf(data)] = 0
decompose.fit(data)
components = decompose.components_

# Plot the components using plotly
import plotly.graph_objects as go

fig = go.Figure()

for i in range(components.shape[0]):
    fig.add_trace(go.Scatter(x=np.arange(components.shape[1]), y=components[i], mode='lines', name='ICA component ' + str(i)))
fig.update_layout(title="ICA Components", xaxis_title="channel", yaxis_title="Amplitude")
fig.show()




