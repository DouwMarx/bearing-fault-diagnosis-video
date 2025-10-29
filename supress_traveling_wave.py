import numpy as np
import matplotlib.pyplot as plt

# Debug the 2D FFT filtering to understand what's happening

# Parameters (simplified for debugging)
T = 2.0
L = 4.0
Nt = 320
Nx = 320
dt = T / Nt
dx = L / Nx
t = np.linspace(0, T, Nt, endpoint=False)
x = np.linspace(0, L, Nx, endpoint=False)

# Create a single wave with known velocity
v_wave_1 = 2.0  # Target velocity
k_1 = 2 * np.pi / (L / 2)  # spatial frequency (4 wavelengths in the domain)
omega_1 = k_1 * v_wave_1  # corresponding angular frequency
f1 = np.sin(omega_1 * t[:, np.newaxis] - k_1 * x[np.newaxis, :])
for harmonic in range(2, 5):  # Add 2nd, 3rd, and 4th harmonics
    f1 += (1 / harmonic) * np.sin(
        harmonic * omega_1 * t[:, np.newaxis] - harmonic * k_1 * x[np.newaxis, :]
    )

# Create another wave that will not be filtered out - spatially localized envelope
envelope_freq = omega_1  # Same frequency as the wave to be suppressed (f1)
duty_cycle = 0.02  # 20% on, 80% off
spatial_extent = 0.1  # 10% of spatial domain
spatial_center = L * 0.3  # Center the region at 30% of domain

# Create temporal envelope with small duty cycle
phase = np.mod(
    envelope_freq * t[:, np.newaxis] / (2 * np.pi), 1.0
)  # Normalize to [0,1]
temporal_envelope = np.where(phase < duty_cycle, 1.0, 0.0)

# Create spatial mask (only active in 10% of spatial domain)
spatial_mask = np.where(
    np.abs(x[np.newaxis, :] - spatial_center) <= spatial_extent * L / 2, 1.0, 0.0
)

# Combine temporal and spatial localization
f2 = temporal_envelope * spatial_mask

# Create the wave: sin(ωt - kx)
f = f1 + f2

print(f"Created wave: v = {v_wave_1:.2f}, ω = {omega_1:.3f}, k = {k_1:.3f}")

# Check which Fourier modes should be dominant
F = np.fft.fft2(f)
freq_t = np.fft.fftfreq(Nt, dt) * 2 * np.pi
k_x = np.fft.fftfreq(Nx, dx) * 2 * np.pi

# Find the dominant modes
F_mag = np.abs(F)
threshold = 0.1 * np.max(F_mag)  # Only look at significant modes
dominant_modes = np.where(F_mag > threshold)

print(f"\nDominant modes (|F| > {threshold:.3f}):")
for i, (m, n) in enumerate(zip(dominant_modes[0], dominant_modes[1])):
    omega_fourier = freq_t[m]
    k_fourier = k_x[n]
    if abs(k_fourier) > 1e-10:
        v_fourier = omega_fourier / k_fourier
        print(
            f"Mode {i}: m={m}, n={n}, ω={omega_fourier:.3f}, k={k_fourier:.3f}, v={v_fourier:.3f}, |F|={F_mag[m, n]:.3f}"
        )
    else:
        print(
            f"Mode {i}: m={m}, n={n}, ω={omega_fourier:.3f}, k={k_fourier:.3f}, k≈0 (DC), |F|={F_mag[m, n]:.3f}"
        )

# Now test filtering
v_target = v_wave_1
tolerance = 0.1

F_filtered = F.copy()
filtered_modes = []

for m in range(Nt):
    for n in range(Nx):
        omega_f = freq_t[m]
        k_f = k_x[n]

        if abs(k_f) > 1e-10:
            v_phase = omega_f / k_f
            # Filter based on absolute velocity (both +v and -v)
            if abs(abs(v_phase) - abs(v_target)) < tolerance:
                F_filtered[m, n] = 0
                filtered_modes.append((m, n, omega_f, k_f, v_phase, F_mag[m, n]))

print(f"\nFiltered modes:")
for i, (m, n, omega_f, k_f, v_phase, mag) in enumerate(filtered_modes):
    print(
        f"Filtered {i}: m={m}, n={n}, ω={omega_f:.3f}, k={k_f:.3f}, v={v_phase:.3f}, |F|={mag:.3f}"
    )

# Reconstruct and check
f_filtered = np.real(np.fft.ifft2(F_filtered))

# Check if the wave was actually removed
print(f"\nComparison:")
print(f"Original max amplitude: {np.max(np.abs(f)):.6f}")
print(f"Filtered max amplitude: {np.max(np.abs(f_filtered)):.6f}")
print(f"Difference max: {np.max(np.abs(f - f_filtered)):.6f}")

# Plot for visual inspection
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original
im1 = axes[0].imshow(f, aspect="auto", origin="lower")
axes[0].set_title("Original Wave")
axes[0].set_xlabel("Space index")
axes[0].set_ylabel("Time index")
plt.colorbar(im1, ax=axes[0])

# Filtered
im2 = axes[1].imshow(f_filtered, aspect="auto", origin="lower")
axes[1].set_title("Filtered Wave")
axes[1].set_xlabel("Space index")
axes[1].set_ylabel("Time index")
plt.colorbar(im2, ax=axes[1])

# Difference
diff = f - f_filtered
im3 = axes[2].imshow(diff, aspect="auto", origin="lower")
axes[2].set_title("Difference (Removed)")
axes[2].set_xlabel("Space index")
axes[2].set_ylabel("Time index")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

# Check the phase relationship
print(f"\nPhase check:")
print(f"Expected: ω/k = {omega_1 / k_1:.3f}")
print(f"Found dominant modes with velocity near {v_target}")
