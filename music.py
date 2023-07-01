import numpy as np
from numpy.linalg import eig

def music_algorithm(data, num_sources,n_thetas=2000,max_expected_rps=40,fs=2000):
    # Compute the correlation matrix

    # data = data.transpose()

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
    # max_radians =  np.tanh(max_expected_rps *data.shape[0] /fs)
    thetas = np.linspace(0, max_radians, n_thetas) # In radians
    # thetas = np.linspace(0, 1/800, n_thetas) # In radians
    # thetas = np.linspace(0,1000, n_thetas) # In radians
    for theta in thetas:
        # Construct steering vector
        steering_vector = np.exp(-1j * 2 * np.pi * theta * np.arange(data.shape[0]))
        # Compute the MUSIC spectrum
        pseudo_amplitude = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)
        # pseudo_amplitude = 1 / (steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector )
        music_spectrum.append(pseudo_amplitude)

    revs_per_second = thetas * fs / (2 * np.pi)

    # revs_per_second = thetas # * fs / data.shape[0]
    # revs_per_second = 1/thetas # / fs * (2 * np.pi)

    # n_space_sample_per_time_sample = np.tan(thetas)
    # revs_per_time_sample = n_space_sample_per_time_sample #/data.shape[0]
    # revs_per_second = revs_per_time_sample * fs

    # Velocity = Distance / (Time * sin(Angle))
    # n_space_sample_per_time_sample = (1/data.shape[0])/((1/fs)*np.sin(thetas))
    # # n_space_sample_per_time_sample = np.tan(thetas)
    # revs_per_time_sample = n_space_sample_per_time_sample# /data.shape[0]
    # revs_per_second = revs_per_time_sample * fs
    # revs_per_second= (1/data.shape[0])/((1/fs)*np.sin(thetas))

    # theta is the required time shift for one spatial sample shift
    # revs_per_second = 1/(np.pi/2-thetas)*fs/data.shape[0]
    # revs_per_second = 1/np.tan(thetas)

    return revs_per_second, music_spectrum


def estimate_wave_velocity(wave_data, time_step, space_step):
    # Step 1: Calculate the autocorrelation matrix
    autocorr_matrix = np.dot(wave_data.T, wave_data)

    # Step 2: Find the dominant frequency component
    eigenvalues, eigenvectors = np.linalg.eig(autocorr_matrix)
    dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Step 3: Calculate the phase shift of the dominant frequency component
    phase_shift = np.angle(np.fft.fftshift(np.fft.fft(dominant_eigenvector)))

    # Step 4: Estimate the velocity
    velocity = (phase_shift * space_step) / (2 * np.pi * time_step)

    return velocity

