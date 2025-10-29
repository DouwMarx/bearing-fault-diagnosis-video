import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 4.0  # Time period (e.g., seconds)
L = 10.0  # Space period (e.g., meters)
Nt = 128  # Time samples
Nx = 128  # Space samples
dt = T / Nt
dx = L / Nx
t = np.linspace(0, T, Nt, endpoint=False)  # Time grid
x = np.linspace(0, L, Nx, endpoint=False)  # Space grid

# Generate sample periodic signal: superposition of propagating waves with different velocities
# Using the form sin(ωt - kx) where v = ω/k

v1 = 0.5  # Slow wave velocity (units: L/T, e.g., m/s)
v2 = 2.0  # Fast wave velocity (target to filter)
v3 = 1.5  # Medium wave velocity

# Create waves with specific velocities
# For wave with velocity v: choose ω, then k = ω/v
omega1 = 2 * np.pi  # Base frequency for wave 1
k1 = omega1 / v1  # k = ω/v

omega2 = 3 * np.pi  # Base frequency for wave 2
k2 = omega2 / v2  # k = ω/v

omega3 = 4 * np.pi  # Base frequency for wave 3
k3 = omega3 / v3  # k = ω/v

# Create composite signal: sum of three propagating waves
f = (
    0.4 * np.sin(omega1 * t[:, np.newaxis] - k1 * x[np.newaxis, :])
    + 0.8
    * np.sin(
        omega2 * t[:, np.newaxis] - k2 * x[np.newaxis, :]
    )  # Make target wave stronger
    + 0.3 * np.sin(omega3 * t[:, np.newaxis] - k3 * x[np.newaxis, :])
)

print(f"Created waves:")
print(f"Wave 1: v = {v1:.1f}, ω = {omega1:.3f}, k = {k1:.3f}")
print(f"Wave 2: v = {v2:.1f}, ω = {omega2:.3f}, k = {k2:.3f} <- TARGET")
print(f"Wave 3: v = {v3:.1f}, ω = {omega3:.3f}, k = {k3:.3f}")

# 1. 2D Fourier Decomposition
F = np.fft.fft2(f)  # 2D FFT: coefficients c_{m,n} (complex)
freq_t = np.fft.fftfreq(Nt, dt) * 2 * np.pi  # Angular temporal frequencies ω_m (rad/T)
k_x = np.fft.fftfreq(Nx, dx) * 2 * np.pi  # Spatial wavenumbers k_n (rad/L)

# For visualization, shift to center zero freq (optional, for plotting)
F_shift = np.fft.fftshift(F)
freq_t_shift = np.fft.fftshift(freq_t)
k_x_shift = np.fft.fftshift(k_x)

# Plot magnitude of coefficients (log scale for clarity)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(
    np.abs(F_shift),
    extent=[k_x_shift.min(), k_x_shift.max(), freq_t_shift.min(), freq_t_shift.max()],
    aspect="auto",
    origin="lower",
)
plt.xlabel("Wavenumber k_x (rad/m)")
plt.ylabel("Freq ω (rad/s)")
plt.title("FFT Coefficients |c_{m,n}|")
plt.colorbar(label="Magnitude")

# 2. Reconstruction (to verify decomposition)
f_recon = np.real(np.fft.ifft2(F))  # Inverse 2D FFT
plt.subplot(132)
plt.imshow(
    f, extent=[x.min(), x.max(), t.min(), t.max()], aspect="auto", origin="lower"
)
plt.xlabel("Space x (m)")
plt.ylabel("Time t (s)")
plt.title("Original Signal")
plt.colorbar(label="Amplitude")

plt.subplot(133)
plt.imshow(
    f_recon, extent=[x.min(), x.max(), t.min(), t.max()], aspect="auto", origin="lower"
)
plt.xlabel("Space x (m)")
plt.ylabel("Time t (s)")
plt.title("Reconstructed Signal")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()

# Error: Should be near zero (numerical precision)
print(f"Reconstruction RMSE: {np.sqrt(np.mean((f - f_recon) ** 2)):.2e}")

# 3. Filter: Null out modes propagating at target velocity v_target
v_target = v2  # Velocity to remove (e.g., the v2 wave above)
tolerance = 0.1  # Relative tolerance for v ≈ v_target (adjust based on grid resolution)

F_filtered = F.copy()  # Start with original coefficients

# Alternative approach: Use shifted coordinates for consistent filtering and visualization
F_shifted_for_filter = np.fft.fftshift(F)
freq_t_centered = np.fft.fftshift(freq_t)
k_x_centered = np.fft.fftshift(k_x)

# Filter in the shifted (centered) coordinate system
filtered_count = 0
for m in range(Nt):
    for n in range(Nx):
        omega = freq_t_centered[m]  # Use centered frequency values
        k = k_x_centered[n]  # Use centered wavenumber values

        # Check if this mode corresponds to the target velocity
        if abs(k) > 1e-10:  # Avoid division by zero
            v_phase = omega / k
            # Filter modes with velocity magnitude matching target
            if abs(abs(v_phase) - abs(v_target)) < tolerance:
                F_shifted_for_filter[m, n] = 0  # Null this mode
                filtered_count += 1
                if filtered_count <= 10:  # Limit printed output
                    print(
                        f"Filtered mode: m={m}, n={n}, ω={omega:.3f}, k={k:.3f}, v={v_phase:.3f}, |F|={abs(F_shifted_for_filter[m, n]):.3f}"
                    )

# Convert back to standard layout for reconstruction
F_filtered = np.fft.ifftshift(F_shifted_for_filter)

print(f"Total filtered modes: {filtered_count}")

# Reconstruct filtered signal
f_filtered = np.real(np.fft.ifft2(F_filtered))

# Analysis of filtering effectiveness
energy_original = np.mean(f**2)
energy_filtered = np.mean(f_filtered**2)
energy_removed = np.mean((f - f_filtered) ** 2)

print(f"\nFiltering effectiveness:")
print(f"Original energy: {energy_original:.6f}")
print(f"Filtered energy: {energy_filtered:.6f}")
print(f"Removed energy: {energy_removed:.6f}")
print(f"Energy retention: {energy_filtered / energy_original * 100:.1f}%")
print(f"Energy removed: {energy_removed / energy_original * 100:.1f}%")

# Show the filtered amplitudes using consistent shifted layout
F_filtered_shift = np.fft.fftshift(F_filtered)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(
    np.abs(F_shift),
    extent=[k_x_shift.min(), k_x_shift.max(), freq_t_shift.min(), freq_t_shift.max()],
    aspect="auto",
    origin="lower",
)
plt.xlabel("Wavenumber k_x (rad/m)")
plt.ylabel("Freq ω (rad/s)")
plt.title("Original FFT Coefficients |c_{m,n}|")
plt.colorbar(label="Magnitude")

plt.subplot(132)
plt.imshow(
    np.abs(F_filtered_shift),
    extent=[k_x_shift.min(), k_x_shift.max(), freq_t_shift.min(), freq_t_shift.max()],
    aspect="auto",
    origin="lower",
)
plt.xlabel("Wavenumber k_x (rad/m)")
plt.ylabel("Freq ω (rad/s)")
plt.title("Filtered FFT Coefficients |c_{m,n}|")
plt.colorbar(label="Magnitude")

plt.subplot(133)
# Show the difference (what was removed)
F_diff_shift = F_shift - F_filtered_shift
plt.imshow(
    np.abs(F_diff_shift),
    extent=[k_x_shift.min(), k_x_shift.max(), freq_t_shift.min(), freq_t_shift.max()],
    aspect="auto",
    origin="lower",
)
plt.xlabel("Wavenumber k_x (rad/m)")
plt.ylabel("Freq ω (rad/s)")
plt.title("Removed Components |c_{m,n}|")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()


# Plot filtered result
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(
    f, extent=[x.min(), x.max(), t.min(), t.max()], aspect="auto", origin="lower"
)
plt.xlabel("Space x (m)")
plt.ylabel("Time t (s)")
plt.title("Original Signal")
plt.colorbar(label="Amplitude")

plt.subplot(122)
plt.imshow(
    f_filtered,
    extent=[x.min(), x.max(), t.min(), t.max()],
    aspect="auto",
    origin="lower",
)
plt.xlabel("Space x (m)")
plt.ylabel("Time t (s)")
plt.title(f"Filtered (Removed v ≈ {v_target})")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()

# Optional: Plot a snapshot at fixed time to see spatial effect
t_snap = T / 4  # Snapshot at t = T/4
idx_t = int(t_snap / dt) % Nt
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(x, f[idx_t, :], label="Original")
plt.xlabel("Space x (m)")
plt.ylabel("Amplitude")
plt.title(f"Snapshot at t={t_snap:.1f}")
plt.legend()

plt.subplot(122)
plt.plot(x, f_filtered[idx_t, :], label=f"Filtered (no v={v_target})")
plt.xlabel("Space x (m)")
plt.ylabel("Amplitude")
plt.title(f"Snapshot at t={t_snap:.1f}")
plt.legend()
plt.tight_layout()
plt.show()
