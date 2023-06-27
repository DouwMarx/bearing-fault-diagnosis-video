import numpy as np
from numpy.linalg import eig

def music_algorithm(data, num_sources,n_thetas=1000,max_expected_rps=40):

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
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select noise subspace
    noise_subspace = sorted_eigenvectors[:, num_sources:]

    # MUSIC algorithm
    music_spectrum = []

    max_radians = max_expected_rps * 2 * np.pi / data.shape[0]


    thetas = np.linspace(0, 0.25*np.pi/4, n_thetas) # In radians
    for theta in thetas:
        # Construct steering vector
        steering_vector = np.exp(-1j * 2 * np.pi * theta * np.arange(data.shape[0]))
        # Compute the MUSIC spectrum
        pseudo_amplitude = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)
        # pseudo_amplitude = 1 / (steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector )
        music_spectrum.append(pseudo_amplitude)

    return thetas, music_spectrum


def music_b(wave_data):

    # (measurement, time)
    d = 1

    r = np.asmatrix(wave_data).T

    num_expected_signals = 1  # Try changing this!
    Nr = wave_data.shape[0]  # Number of sensors
    # part that doesn't change with theta_i
    R = r@ r.conjugate().transpose()  # Calc covariance matrix, it's Nr x Nr
    w, v = np.linalg.eig(R)  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
    v = v[:, eig_val_order]  # sort eigenvectors using this order
    # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
    V = np.asmatrix(np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64))
    for i in range(Nr - num_expected_signals):
        V[:, i] = v[:, i]

    theta_scan = np.linspace(-1 * np.pi, np.pi, 1000)  # -180 to +180 degrees
    results = []
    for theta_i in theta_scan:
        a = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)))  # array factor
        a = a.T
        metric = 1 / (a.H @ V @ V.H @ a)  # The main MUSIC equation
        metric = np.abs(metric[0, 0])  # take magnitude
        metric = 10 * np.log10(metric)  # convert to dB
        results.append(metric)

    results /= np.max(results)  # normalize

    return theta_scan, results




def music_c(sensor_array, num_sources):

    sensor_array = sensor_array.transpose()

    num_elements = len(sensor_array)
    num_snapshots = sensor_array.shape[1]

    # Compute the sample covariance matrix
    cov_matrix = np.matmul(sensor_array, sensor_array.conj().T) / num_snapshots

    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # Sort eigenvalues in descending order and corresponding eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]

    # Estimate the noise subspace
    noise_subspace = sorted_eigvecs[:, num_sources:]

    # Compute the pseudospectrum
    pseudospectrum = np.zeros(num_elements)
    for theta_idx in range(num_elements):
        steering_vector = np.exp(-1j * 2 * np.pi * theta_idx * np.arange(num_elements))
        pseudospectrum[theta_idx] = 1 / (steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)

    return range(num_elements), pseudospectrum