import numpy as np
import matplotlib.pyplot as plt

# Parameters for the synthetic example
n = 50  # Number of spatial points (pixels along the slit)
m = 100  # Number of time steps (frames)
P = 20   # Period in frames (rotation completes every 20 frames)
r = 3    # Number of temporal modes (fundamental + 2 harmonics)
noise_level = 0.05  # Relative noise amplitude (5% of signal strength)

# Sampling interval (in frames, assuming dt=1 for simplicity)
dt = 1.0

# Prescribe temporal frequencies: fundamental omega_1 = 2*pi / P, and harmonics j*omega_1
omega_1 = 2 * np.pi / P
omegas = np.array([j * omega_1 for j in range(1, r + 1)])  # [omega_1, 2*omega_1, 3*omega_1]

# Prescribed eigenvalues (assuming neutral stability: no growth/decay, sigma_j=0)
lambdas = np.exp(1j * omegas * dt)  # lambda_j = exp(i * omega_j * dt)

# Build the Vandermonde matrix Gamma (r x m): row j is [lambda_j^{0}, lambda_j^{1}, ..., lambda_j^{m-1}]
k = np.arange(m).reshape(1, -1)  # Time indices (1 x m)
Gamma = lambdas.reshape(-1, 1) ** k  # Broadcasting: (r x 1) ** (1 x m) -> (r x m)

# Generate synthetic spatial modes phi_j (true ones, for demonstration)
# These are arbitrary along-slit patterns (s = 0 to n-1)
s = np.arange(n)
# Mode 1 (fundamental): Sinusoidal "zebra tape" pattern (spatial frequency 2pi/10 pixels)
phi_1 = np.sin(2 * np.pi * s / 10) + 1j * np.cos(2 * np.pi * s / 10)
# Mode 2 (harmonic): Step-like "helical gear tooth" edge (abrupt change around s=20)
phi_2 = np.tanh((s - 20) / 5) + 1j * np.exp(- (s - 25)**2 / 50)  # Real: step; Imag: Gaussian bump
# Mode 3 (higher harmonic): Higher-frequency oscillation (e.g., finer teeth)
phi_3 = np.sin(4 * np.pi * s / n) + 1j * np.cos(6 * np.pi * s / n)

# True Phi matrix (n x r): columns are the spatial modes
Phi_true = np.column_stack([phi_1, phi_2, phi_3])

# Amplitudes b_j (complex, for phase and strength; here simple scaling)
b = np.array([1.0, 0.8 * 1j, 0.6])  # Absorb into tilde{phi} later, but use here for generation

# Generate clean data: X = Phi @ diag(b) @ Gamma
# (For simplicity, absorb b into Phi_true temporarily)
Phi_full = Phi_true * b[None, :]  # Scale each column by b_j
X_clean = Phi_full @ Gamma

# Add noise (complex Gaussian, relative to signal norm)
noise = noise_level * np.linalg.norm(X_clean) / np.sqrt(n * m) * (np.random.randn(n, m) + 1j * np.random.randn(n, m))
X = X_clean + noise

# Now solve the least-squares problem: tilde{Phi} = X @ Gamma^\dagger
# Gamma^\dagger is the pseudoinverse (r x m) -> (m x r)
Gamma_pinv = np.linalg.pinv(Gamma)
Phi_est = X @ Gamma_pinv  # (n x m) @ (m x r) = (n x r)

# Compute reconstruction
X_recon = Phi_est @ Gamma  # (n x r) @ (r x m) = (n x m)

# Error metrics
recon_error = np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')
print(f"Reconstruction error (relative Frobenius norm): {recon_error:.4f} ({recon_error*100:.2f}%)")

# For comparison: true Phi_full (with b absorbed)
print(f"True Phi norm: {np.linalg.norm(Phi_true, 'fro'):.2f}")
print(f"Estimated Phi norm: {np.linalg.norm(Phi_est, 'fro'):.2f}")

# Visualization (original plots)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Spatial Modes and Reconstruction (Slit Camera: Rotating Shaft Example)')

# Plot estimated spatial modes (real and imag parts)
s_plot = s  # Spatial coordinate
for j in range(r):
    ax = axes[0, j]
    ax.plot(s_plot, Phi_est[:, j].real, label='Real(φ_j)', linewidth=2)
    ax.plot(s_plot, Phi_est[:, j].imag, '--', label='Imag(φ_j)', linewidth=2)
    ax.set_title(f'Estimated Spatial Mode {j+1} (ω_{j+1} = {omegas[j]:.2f} rad/frame)')
    ax.set_xlabel('Slit Position (pixels)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

# Plot a sample snapshot reconstruction (e.g., at k=m//2)
k_sample = m // 2
ax_recon = axes[1, :]
ax_recon[0].plot(s_plot, X[:, k_sample].real, 'o-', label='True Data (Real)', alpha=0.7)
ax_recon[0].plot(s_plot, X_recon[:, k_sample].real, 's-', label='Reconstructed (Real)', linewidth=2)
ax_recon[0].set_title(f'Snapshot at t={k_sample} (Real Part)')
ax_recon[0].legend()
ax_recon[0].grid(True)

# Imag part for same snapshot
ax_recon[1].plot(s_plot, X[:, k_sample].imag, 'o-', label='True Data (Imag)', alpha=0.7)
ax_recon[1].plot(s_plot, X_recon[:, k_sample].imag, 's-', label='Reconstructed (Imag)', linewidth=2)
ax_recon[1].set_title(f'Snapshot at t={k_sample} (Imag Part)')
ax_recon[1].legend()
ax_recon[1].grid(True)

# Temporal evolution at a fixed spatial point (e.g., s=25)
s_fixed = 25
ax_recon[2].plot(np.arange(m), np.abs(X[s_fixed, :]), 'o-', label='True |x(s_fixed, t)|', alpha=0.7)
ax_recon[2].plot(np.arange(m), np.abs(X_recon[s_fixed, :]), '-', label='Reconstructed |x(s_fixed, t)|', linewidth=2)
ax_recon[2].set_title(f'Temporal Evolution at Fixed s={s_fixed} (Magnitude)')
ax_recon[2].set_xlabel('Time Step (frame)')
ax_recon[2].legend()
ax_recon[2].grid(True)

plt.tight_layout()
plt.show()

# Optional: Compare with true modes
fig_true, axes_true = plt.subplots(1, r, figsize=(15, 4))
for j in range(r):
    axes_true[j].plot(s_plot, (Phi_true * b[None, j])[:, j].real, label='True Real(φ_j * b_j)', linewidth=2)
    axes_true[j].plot(s_plot, Phi_est[:, j].real, '--', label='Est. Real(φ̃_j)', linewidth=2)
    axes_true[j].plot(s_plot, (Phi_true * b[None, j])[:, j].imag, label='True Imag(φ_j * b_j)')
    axes_true[j].plot(s_plot, Phi_est[:, j].imag, '--', label='Est. Imag(φ̃_j)')
    axes_true[j].set_title(f'True vs Est. Mode {j+1}')
    axes_true[j].legend()
    axes_true[j].grid(True)
plt.tight_layout()
plt.show()

# NEW: Space-time diagram for 2π evolution of the fundamental mode
# Slice to exactly one period (P frames) for clean visualization
# Use the first P frames of the data/reconstruction
m_period = P
k_period = np.arange(m_period)
X_period = X[:, :m_period]
X_recon_period = X_recon[:, :m_period]

# Focus on fundamental mode only (j=0): contribution from first mode
Gamma_fund_period = Gamma[0:1, :m_period]  # 1 x P
X_fund_period = Phi_est[:, 0:1] @ Gamma_fund_period  # n x 1 @ 1 x P = n x P

# Map time to angle: theta = 2π * k / P (y-axis from 0 to 2π)
theta = 2 * np.pi * k_period / P  # 0 to 2π (excluding 2π for plotting)

# Create meshgrids for pcolormesh (need to transpose the data to match expected format)
S, Theta = np.meshgrid(s, theta, indexing='ij')

# Plot space-time: x = spatial (s), y = angle (theta), color = real part of field (or magnitude)
fig_st, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig_st.suptitle(f'Space-Time Diagram: 2π Evolution of Fundamental Mode (ω_1 = {omega_1:.3f} rad/frame)')

# Left: True noisy data over one period (real part)
im1 = ax1.pcolormesh(S, Theta, np.real(X_period), cmap='RdBu_r', shading='gouraud', vmin=-1.5, vmax=1.5)
ax1.set_xlabel('Slit Position (pixels)')
ax1.set_ylabel('Angle (radians, 0 to 2π)')
ax1.set_title('Noisy Data (Real Part, One Period)')
ax1.set_ylim(0, 2 * np.pi)
plt.colorbar(im1, ax=ax1, label='Intensity')

# Right: Fundamental mode contribution (real part) over one period
im2 = ax2.pcolormesh(S, Theta, np.real(X_fund_period), cmap='RdBu_r', shading='gouraud', vmin=-1.5, vmax=1.5)
ax2.set_xlabel('Slit Position (pixels)')
ax2.set_ylabel('Angle (radians, 0 to 2π)')
ax2.set_title('Fundamental Mode Reconstruction (Real Part, One Period)')
ax2.set_ylim(0, 2 * np.pi)
plt.colorbar(im2, ax=ax2, label='Intensity')

# Optional: Add lines to show periodicity (e.g., every π)
for ax in [ax1, ax2]:
    ax.axhline(y=np.pi, color='k', linestyle='--', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.show()

# Additional: Full reconstruction space-time for comparison (if desired)
fig_full_st, ax_full = plt.subplots(1, 1, figsize=(8, 5))
im_full = ax_full.pcolormesh(S, Theta, np.real(X_recon_period), cmap='RdBu_r', shading='gouraud', vmin=-1.5, vmax=1.5)
ax_full.set_xlabel('Slit Position (pixels)')
ax_full.set_ylabel('Angle (radians, 0 to 2π)')
ax_full.set_title('Full Reconstruction (All Modes, Real Part, One Period)')
ax_full.set_ylim(0, 2 * np.pi)
plt.colorbar(im_full, ax=ax_full, label='Intensity')
plt.show()
