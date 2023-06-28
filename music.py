import numpy as np
from numpy.linalg import eig

def music_algorithm(data, num_sources,n_thetas=500,max_expected_rps=40,fs=2000):
    # Compute the correlation matrix
    correlation_matrix = np.matmul(data, data.conj().T) / data.shape[1]

    # Eigenvalue decomposition of the covariance matrix
    # eigenvalues, eigenvectors = eig(correlation_matrix)

    # SVD decomposition of the covariance matrix
    U, S, V = np.linalg.svd(correlation_matrix)
    eigenvalues = S
    eigenvectors = U


    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select noise subspace
    noise_subspace = sorted_eigenvectors[:, num_sources:]

    # MUSIC algorithm
    music_spectrum = []
    max_radians =  max_expected_rps * 2 * np.pi / fs
    thetas = np.linspace(0, max_radians, n_thetas) # In radians
    for theta in thetas:
        # Construct steering vector
        steering_vector = np.exp(-1j * 2 * np.pi * theta * np.arange(data.shape[0]))
        # Compute the MUSIC spectrum
        pseudo_amplitude = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)
        # pseudo_amplitude = 1 / (steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector )
        music_spectrum.append(pseudo_amplitude)

    rotational_speeds = thetas * fs / (2 * np.pi)

    return rotational_speeds, music_spectrum