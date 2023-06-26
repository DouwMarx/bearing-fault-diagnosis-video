import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, FastICA

# Load the data
# frames_polar_array = np.load("rpm_variable_10000_frames_1_channels_polar.npy")
# frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")
frames_polar_array = np.load("rpm_variable_10000_frames_0.25_channels_0.5_space_polar.npy")
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
pca = IncrementalPCA(n_components=1, batch_size=1000)
# pca = FastICA(n_components=1, max_iter=1000)
transformed_data = {}
for segment in segment_columns:
    slice = segment_columns[segment]
    print("Segment: ", segment)
    print("Slice: ", slice)
    data = frames_polar_array [:, :,slice[0]:slice[1]]
    # Stack all frames on top of each other

    # Replace invalid values with the value of the previous frame
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         for k in range(data.shape[2]):
    #             if np.isnan(data[i,j,k]):
    #                 data[i,j,k] = data[i-1,j,k]
    # print("Number of NaNs: ", np.sum(np.isnan(data)))
    # print("Number of inf: ", np.sum(np.isinf(data)))


    data = data.reshape(-1, data.shape[2])

    # Standardize the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Remove any NaNs

    # # Do PCA
    # pca = PCA(n_components=1)
    # Do incremental PCA
    # pca = PCA(n_components=1, svd_solver='full')
    pca.fit(data)
    # print("Explained variance: ", pca.explained_variance_ratio_)

    # Transform the data
    transformed = pca.transform(data)
    transformed = transformed.reshape(frames_polar_array.shape[0], frames_polar_array.shape[1])
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
                            title="Time (frames)"
                        ),
                        yaxis=dict(
                            title="Angle (degrees)"
    )
                    ))
    fig.show()



# Write the transformed data to a file
np.save("10000_frames_pca.npy", transformed_data)

