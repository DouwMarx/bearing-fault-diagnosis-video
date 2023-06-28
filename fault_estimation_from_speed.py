# Load the signal
import numpy as np
import scipy


# name = "inner"

for name in ["inner", "cage"]:
    print("Segment: ", name)

    fs = 2000
    signal = np.load(name + "_speed_profile.npy")
    time = np.arange(signal.shape[0]) / fs

    # Integrate the signal to get the angle
    angle = np.cumsum(signal) / fs # In revolutions

    # Get a mapping between angle (revs) and the signal
    signal_as_function_of_angle = scipy.interpolate.interp1d(angle, signal,kind="cubic")

    constant_angle = np.linspace(angle[0], angle[-1], signal.shape[0]) # Also revolutions
    angle_sample_rate = 1 / (angle[1] - angle[0]) # Samples/revolution
    resampled_signal = signal_as_function_of_angle(constant_angle) # Resample the signal to have constant angle

    # # Limit signal between 50 and 68 revs
    # resampled_signal = resampled_signal[(constant_angle > 52) & (constant_angle < 67)]
    # constant_angle = constant_angle[(constant_angle > 52) & (constant_angle < 67)]

    # Detrend by removing moving median
    # window_size = 1001
    window_size = 501
    resampled_signal = resampled_signal - scipy.signal.medfilt(resampled_signal, window_size)

    # Replace outliers outside 2*IQR with median
    q1 = np.quantile(resampled_signal, 0.25)
    q3 = np.quantile(resampled_signal, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2*iqr
    upper_bound = q3 + 2*iqr
    resampled_signal[resampled_signal < lower_bound] = np.median(resampled_signal)
    resampled_signal[resampled_signal > upper_bound] = np.median(resampled_signal)

    # Plot the resampled signal using plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter(
                            y=resampled_signal,
                            x=constant_angle,
                            mode='lines',
                        ),
                        layout=go.Layout(
                            title="Resampled signal",
                            xaxis=dict(
                                title="Number of revolutions"
                            ),
                            yaxis=dict(
                                title="De-trended, order tracked angular velocity (rps)"
                            )
                        ))
    fig.show()
    fig.write_image("reports/resampled_signal_{}.png".format(name))

    # Show the spectrum of the resampled signal

    resampled_fft = np.fft.rfft(resampled_signal-np.mean(resampled_signal))
    fft_freqs = np.fft.rfftfreq(resampled_signal.shape[0], d=1/angle_sample_rate)

    fig = go.Figure(data=go.Scatter(
                            y=np.abs(resampled_fft),
                            x=np.arange(resampled_fft.shape[0]),
                            mode='lines',
                        ),
                        layout=go.Layout(
                            title="Resampled signal spectrum",
                            xaxis=dict(
                                title="Angular frequency (events/revolution)",
                              range=[0, 10]
                            ),
                            yaxis=dict(
                                title="Magnitude"
                            )
                        ))
    fig.show()
    fig.write_image("reports/resampled_signal_spectrum_{}.png".format(name))








