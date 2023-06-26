import numpy as np
from numpy.linalg import eig

def music_algorithm(data, num_sources):

    # data is a matrix of size (num_sensors, num_samples) (M, N)
    # Check that N > M
    # if data.shape[1] <= data.shape[0]:
    #     raise ValueError("The number of samples must be greater than the number of sensors.")

    # Compute the correlation matrix
    correlation_matrix = np.matmul( data, data.conj().T) / data.shape[1]

    # Eigenvalue decomposition of the covariance matrix
    # eigenvalues, eigenvectors = eig(correlation_matrix)

    # SV decomposition of the covariance matrix
    U, S, V = np.linalg.svd(correlation_matrix)
    eigenvalues = S
    eigenvectors = U


    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select noise subspace
    noise_subspace = sorted_eigenvectors[:, num_sources:]

    # MUSIC algorithm
    music_spectrum = []
    # thetas =range(-70, 70)
    # thetas = np.linspace(-180, 180, 1000)
    thetas = np.linspace(-70, 70, 1000)
    # thetas = np.linspace(0, 180, 200)
    for theta in thetas:
        # Construct steering vector
        steering_vector = np.exp(-1j * 2 * np.pi * theta / 180 * np.arange(data.shape[0]))

        # Compute the MUSIC spectrum
        pseudo_amplitude = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)

        music_spectrum.append(pseudo_amplitude)

    # Compute the phase velocity in spatial samples / temporal sample for different thetas
    # tan(theta) = time/space
    phase_velocities = np.tan(thetas / 180 * np.pi)


    return thetas, phase_velocities, music_spectrum


def estimate_phase_velocity(wave_data, dx, dt):
    """
    Estimate the phase velocity of a non-dispersive wave using the MUSIC algorithm.

    Args:
    wave_data (numpy.ndarray): Array of wave data over time. Rows represent time and columns represent space.
    dx (float): Spatial resolution.
    dt (float): Temporal resolution.

    Returns:
    float: Estimated phase velocity.
    """
    num_time_points, num_space_points = wave_data.shape

    # Compute the spatial Fourier transform of the wave data
    wave_data_fft = np.fft.fft(wave_data, axis=1)

    # Compute the correlation matrix
    correlation_matrix = np.matmul(wave_data_fft.conj().T, wave_data_fft) / num_time_points

    # Perform eigenvalue decomposition of the correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Find the smallest eigenvalue corresponding to noise
    noise_eigenvalue_index = np.argmin(sorted_eigenvalues)

    # Select the eigenvectors corresponding to the signal
    signal_eigenvectors = sorted_eigenvectors[:, :noise_eigenvalue_index]

    # Compute the wavenumbers
    k_values = np.fft.fftfreq(num_space_points, d=dx) * 2 * np.pi

    # Compute the normalized spatial frequencies
    spatial_frequencies = k_values / (2 * np.pi)

    # Compute the MUSIC spectrum
    music_spectrum = np.sum(np.abs(np.matmul(signal_eigenvectors.conj().T, np.exp(-1j * k_values * spatial_frequencies))),
                            axis=0)

    # Find the peak of the MUSIC spectrum
    phase_velocity_index = np.argmax(music_spectrum)
    estimated_phase_velocity = spatial_frequencies[phase_velocity_index] / dt

    return k_values,music_spectrum,estimated_phase_velocity