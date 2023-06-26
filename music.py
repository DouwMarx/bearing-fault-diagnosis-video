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
    # thetas = np.linspace(0, 0.8*np.pi/4, 1000) # In radians
    # thetas = np.linspace(0, 0.7*np.pi/4, 500) # In radians
    thetas = np.linspace(0, 0.25*np.pi/4, 300) # In radians
    # thetas = np.linspace(-0.8*np.pi/4, 0.8*np.pi/4, 1000) # In radians
    for theta in thetas:
        # Construct steering vector
        # steering_vector = np.exp(-1j * 2 * np.pi * theta / 180 * np.arange(data.shape[0]))
        steering_vector = np.exp(-1j * 2 * np.pi * theta * np.arange(data.shape[0]))

        # Compute the MUSIC spectrum
        pseudo_amplitude = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)

        music_spectrum.append(pseudo_amplitude)

    return thetas, music_spectrum
