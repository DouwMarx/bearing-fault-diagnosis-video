import scipy
import numpy as np
import plotly.graph_objects as go

# Load toby .txt speed estimate
toby_inner_speed_estimate = np.loadtxt("toby_speed_estimates/shaft.txt")
toby_cage_speed_estimate = np.loadtxt("toby_speed_estimates/train.txt")

douw_inner_speed_estimate = np.load("inner_speed_profile.npy")
douw_cage_speed_estimate = np.load("cage_speed_profile.npy")

# Plot the signals using plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
                        y=toby_inner_speed_estimate,
                        mode='lines',
                        name="Toby inner speed estimate"
                    ))
fig.add_trace(go.Scatter(
                        y=toby_cage_speed_estimate,
                        mode='lines',
                        name="Toby cage speed estimate"
                    ))
fig.add_trace(go.Scatter(
                        y=douw_inner_speed_estimate,
                        mode='lines',
                        name="Douw inner speed estimate"
                    ))
fig.add_trace(go.Scatter(
                        y=douw_cage_speed_estimate,
                        mode='lines',
                        name="Douw cage speed estimate"
                    ))
fig.update_layout(
    title="Speed estimates",
    xaxis_title="Sample",
    yaxis_title="rps",
)
fig.show()

