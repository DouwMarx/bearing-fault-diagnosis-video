import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from tqdm import tqdm

# Load the data
target_file = "rpm_variable_23723_frames_0.15_channels_1_space_polar.npy"
frames_polar_array = np.load(target_file)
print("Dimensions of the polar transformed frames: ", frames_polar_array.shape)

# Load the two fractions of the channels that demarcate the different components
fractions = np.load("peak_indices.npy")
# Make sure fractions are in ascending order
fractions = np.sort(fractions)



n_channels = frames_polar_array.shape[2]
segment_columns = {"inner":[0, np.ceil(fractions[0] * n_channels).astype(int)],
                     "cage":[ int(fractions[0]*n_channels), np.ceil(fractions[1]*n_channels).astype(int)],
                     "outer":[ int(fractions[1]*n_channels), n_channels +1]
                   }
print("Chosen independent regions of interest: ", segment_columns)

# Do PCA on each segment separately, retaining 1 component only and print the explained variance
train_size = 5000000 # Train using only a random subset of radial locations for different frames
# model = PCA(n_components=1)
model = FastICA(n_components=1, whiten="unit-variance")
transformed_data = {}
for segment_name,segment_bounds in segment_columns.items():
    print("Segment: ", segment_name, "Bounds: ", segment_bounds)
    data = frames_polar_array [:, :, segment_bounds[0]:segment_bounds[1]]
    # Stack all frames on top of each other

    data = data.reshape(-1, data.shape[2])
    # Standardize the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Replace NaNs with 0s
    data = np.nan_to_num(data)

    # Take some of the data randomly
    indices = np.random.choice(data.shape[0], train_size, replace=False)
    data_train = data[indices, :]

    model.fit(data_train)
    # print("Explained variance: ", pca.explained_variance_ratio_)

    # Transform all data data
    transformed = model.transform(data)
    transformed = transformed.reshape(frames_polar_array.shape[0], frames_polar_array.shape[1])
    transformed_data[segment_name] = transformed


# Show a heatmap of the signal that representing a single component as it evolves over time
import plotly.graph_objects as go

for segment in list(transformed_data.keys()):
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

    # Save figure as png in reports folder
    fig.write_image("reports/extracted_signal_with_time_{}.png".format(segment))

# Save the reduced data
np.save("reduced_roi_{}.npy".format(target_file), transformed_data, allow_pickle=True)