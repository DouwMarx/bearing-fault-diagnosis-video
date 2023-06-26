import numpy as np
import plotly.graph_objects as go
from music import music_algorithm
from estimators import LocalLinearSpeedEstimator

# Load the .npy dict with the data


signals = np.load("10000_frames_pca.npy" , allow_pickle=True).item()

wave_data = signals["cage"]
# wave_data = signals["outer"]
# wave_data = signals["inner"]
# Remove the mean
wave_data = wave_data - np.mean(wave_data, axis=0)

window_size =400

fs = 2000
spectra = []
rps = []
for i in range(0, wave_data.shape[0], window_size):
    # window_wave_data = wave_data[i:i+window_size]
    # thetas, music_spectrum = music_algorithm(window_wave_data, 1)
    # spectra.append(music_spectrum)
    # theta_max = thetas[np.argmax(music_spectrum)]
    # rps.append(theta_max *fs / (2*np.pi))

    estimator = LocalLinearSpeedEstimator(wave_data[i:i+window_size, :])
    rps.append(estimator.get_rps_estimate())
    spectra.append(estimator.music_spectrum)

# Plot a heatmap of how the spectrum changes over time

spectra = np.array(spectra)
print("Spectra shape: ", spectra.shape)
fig = go.Figure(data=go.Heatmap(
                        z=spectra,
                        colorscale='Viridis'),
                    layout=go.Layout(
                        title="Spectra over time",
                        xaxis=dict(
                            title="Time (frames)"
                        ),
                        yaxis=dict(
                            title="Angle (degrees)"
                        )
                    ))
fig.show()

# Plot the optimal theta over time
fig = go.Figure(data=go.Scatter(
                        y=rps,
                        mode='lines+markers',
                    ),
                    layout=go.Layout(
                        title="Optimal theta over time",
                        xaxis=dict(
                            title="Time (frames)"
                        ),
                        yaxis=dict(
                            title="Angular velocity (degrees/frame)"
                        )
                    ))
fig.show()






