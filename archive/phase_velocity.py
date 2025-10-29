import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate the mixed signal as a superposition of three independent signals
t = np.linspace(0, 1, 1000)  # Time axis
source1 = np.sin(2 * np.pi * 10 * t)  # Independent source 1
source2 = np.cos(2 * np.pi * 5 * t)  # Independent source 2
source3 = np.random.randn(len(t))  # Independent source 3
mixed_signal = 2 * source1 + 0.5 * source2 + 1.5 * source3  # Mixed signal

# Perform ICA to recover the independent sources
ica = FastICA(n_components=3)
recovered_sources = ica.fit_transform(mixed_signal.reshape(1, -1))

# Plot the mixed signal and the recovered sources
plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.plot(t, mixed_signal)
plt.title("Mixed Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 2)
plt.plot(t, recovered_sources[:, 0])
plt.title("Recovered Source 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(t, recovered_sources[:, 1])
plt.title("Recovered Source 2")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 4)
plt.plot(t, recovered_sources[:, 2])
plt.title("Recovered Source 3")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
