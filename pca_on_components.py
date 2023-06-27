import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, FastICA

# Load the data
# frames_polar_array = np.load("rpm_variable_10000_frames_1_channels_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")

target_file = "rpm_variable_23721_frames_0.2_channels_1_space_polar.npy"

frames_polar_array = np.load(target_file)

print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Load the two fractions of the channels that demarcate the different components
fractions = np.load("peak_indices.npy")

n_channels = frames_polar_array.shape[2]
segment_columns = {"inner":[0, int(fractions[0] * n_channels)],
                     "cage":[ int(fractions[0]*n_channels), int(fractions[1]*n_channels)],
                     "outer":[ int(fractions[1]*n_channels), n_channels]
                   }
print(segment_columns)

# Do PCA on each segment separately, retaining 1 component only and print the explained variance
# pca = IncrementalPCA(n_components=1, batch_size=10000)
pca = PCA(n_components=1)
# pca = FastICA(n_components=1, whiten="unit-variance")
transformed_data = {}
for segment in segment_columns:
    slice = segment_columns[segment]
    print("Segment: ", segment)
    print("Slice: ", slice)
    data = frames_polar_array [:, :,slice[0]:slice[1]]
    # Stack all frames on top of each other

    data = data.reshape(-1, data.shape[2])
    # Standardize the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Replace NaNs with 0s
    data = np.nan_to_num(data)

    # Take 10% of the data randomly
    indices = np.random.choice(data.shape[0], int(data.shape[0]*0.01), replace=False)
    data_train = data[indices, :]

    pca.fit(data_train)
    # print("Explained variance: ", pca.explained_variance_ratio_)
    # Transform the data
    transformed = pca.transform(data)
    transformed = transformed.reshape(frames_polar_array.shape[0], frames_polar_array.shape[1])

    # # Do the Fourier transform over the rows (time)
    # data = np.fft.fft(data, axis=1)


    transformed_data[segment] =  transformed


# Plot a heatmap of the transformed data for the cage segment for each frame using plotly
import plotly.graph_objects as go

for segment in transformed_data:
    print("Segment: ", segment)
    print("Shape: ", transformed_data[segment].shape)
    fig = go.Figure(data=go.Heatmap(
                      z=transformed_data[segment],
                        colorscale='Viridis'),

                    layout=go.Layout(
                        title="Segment: {}".format(segment),
                        xaxis=dict(
                            title="Space (channels)"
                        ),
                        yaxis=dict(
                            title="Time (frames)"
    )
                    ))
    fig.show()



# Write the transformed data to a file

# Save the transformed data to a file
np.save("pca_{}.npy".format(target_file), transformed_data)