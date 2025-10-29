# NEW SECTION: Prescribing Spatial Basis for Helical Gear (Pitch Angle alpha=30 deg)
# Add this after the original Phi_est computation and before the space-time plots

# Helical gear parameters (for synthetic generation and basis)
alpha = np.deg2rad(30)  # Pitch angle (helix tilt)
P_tooth = 10  # Assumed axial tooth spacing (pixels per tooth projection)
kappa = 2 * np.pi * np.sin(alpha) / P_tooth  # Spatial wavenumber from pitch (rad/pixel)
psi = np.pi / 4  # Fixed phase for helical wrap
s0 = n / 2  # Center of tooth edge (pixel)
w = 3  # Width of edge transition (pixels)
s = np.arange(n)  # Slit positions

# Prescribed spatial basis G (n x q): q=3 templates for helical structures
# Col 1: DC (uniform)
G = np.ones((n, 1), dtype=complex)

# Col 2: Helical traveling wave (sinusoid with pitch-derived kappa)
G_hel_wave = np.exp(1j * (kappa * s + psi))  # Complex exponential for full phase
G = np.column_stack(
    [G, np.real(G_hel_wave), np.imag(G_hel_wave)]
)  # Take real/imag for real basis (or keep complex)

# Col 3: Localized step for tooth edge (tilted by pitch)
step = np.tanh((s - s0) / w)  # Basic step
G_step = step * np.cos(kappa * s + psi)  # Modulate with helical phase (tilted envelope)
G = np.column_stack([G, G_step])

q = G.shape[1]  # q=4 here (DC + wave real/imag + step)

# For helical synthetic truth: Regenerate X with helical spatial modes (for demo)
# Mode 1: Helical wave dominant
phi_hel_1 = 1.2 * np.exp(1j * kappa * s)  # Full complex helical
# Mode 2: Stepped helical edge
phi_hel_2 = 0.9 * G_step + 1j * 0.7 * np.exp(-((s - s0) ** 2) / (2 * w**2))
# Mode 3: Higher harmonic (2*kappa, finer helix)
phi_hel_3 = 0.5 * np.exp(1j * 2 * kappa * s)
Phi_hel_true = np.column_stack([phi_hel_1, phi_hel_2, phi_hel_3])
b_hel = np.array([1.0, 0.8j, 0.6])  # Amplitudes
X_hel_clean = (Phi_hel_true * b_hel[None, :]) @ Gamma
noise_hel = (
    noise_level
    * np.linalg.norm(X_hel_clean)
    / np.sqrt(n * m)
    * (np.random.randn(n, m) + 1j * np.random.randn(n, m))
)
X_hel = X_hel_clean + noise_hel  # Use this as the "data" for helical fit

# Solve for Theta: Theta = G^\dagger X_hel Gamma^\dagger  (but efficient: project X onto G first)
# First, project temporal: Y = X_hel @ Gamma_pinv  (n x r: spatial-time projection)
Y = X_hel @ Gamma_pinv  # Equivalent to earlier Phi_est, but now for helical data

# Then, for each temporal mode j, solve phi_j ≈ G * theta_j (n x q @ q x 1 = n x 1)
# Collect into Theta (q x r)
Theta = np.linalg.pinv(G) @ Y  # (q x n) wait no: pinv(G) is q x n, Y is n x r → q x r

# Reconstruct spatial modes: Phi_hel_est = G @ Theta
Phi_hel_est = G @ Theta  # (n x q) @ (q x r) = (n x r)

# Full reconstruction
X_hel_recon = Phi_hel_est @ Gamma

# Error for helical fit
hel_error = np.linalg.norm(X_hel - X_hel_recon, "fro") / np.linalg.norm(X_hel, "fro")
print(
    f"Helical reconstruction error (relative Frobenius norm): {hel_error:.4f} ({hel_error * 100:.2f}%)"
)

# Visualization: Helical spatial modes and space-time
fig_hel, axes_hel = plt.subplots(2, 3, figsize=(15, 10))
fig_hel.suptitle(
    f"Prescribed Spatial Basis: Helical Gear (α={alpha:.0f} rad, κ={kappa:.2f} rad/pix)"
)

# Plot estimated spatial modes (from prescribed basis)
for j in range(r):
    ax = axes_hel[0, j]
    ax.plot(s, np.real(Phi_hel_est[:, j]), label="Real(φ_j)", linewidth=2)
    ax.plot(s, np.imag(Phi_hel_est[:, j]), "--", label="Imag(φ_j)", linewidth=2)
    ax.set_title(f"Helical Mode {j + 1} (ω_{j + 1} = {omegas[j]:.2f})")
    ax.set_xlabel("Slit Position (pixels)")
    ax.legend()
    ax.grid(True)

# Sample snapshot (real part)
k_sample = m // 4  # Different sample for variety
axes_hel[1, 0].plot(s, np.real(X_hel[:, k_sample]), "o-", label="Noisy Data", alpha=0.7)
axes_hel[1, 0].plot(
    s,
    np.real(X_hel_recon[:, k_sample]),
    "s-",
    label="Reconstructed (Prescribed Spatial)",
    linewidth=2,
)
axes_hel[1, 0].set_title("Snapshot Reconstruction (Real)")
axes_hel[1, 0].legend()
axes_hel[1, 0].grid(True)

# Space-time for helical fundamental (one period, as before)
m_period = P
X_hel_period = X_hel[:, :m_period]
X_hel_recon_period = X_hel_recon[:, :m_period]
Gamma_fund_period = Gamma[0:1, :m_period]
X_hel_fund_period = Phi_hel_est[:, 0:1] @ Gamma_fund_period
theta_period = 2 * np.pi * np.arange(m_period) / P

im_hel_st = axes_hel[1, 1].pcolormesh(
    s,
    theta_period,
    np.real(X_hel_fund_period),
    cmap="RdBu_r",
    shading="gouraud",
    vmin=-1.5,
    vmax=1.5,
)
axes_hel[1, 1].set_xlabel("Slit Position")
axes_hel[1, 1].set_ylabel("Angle (rad)")
axes_hel[1, 1].set_title("Helical Fund. Mode Space-Time")
axes_hel[1, 1].set_ylim(0, 2 * np.pi)
plt.colorbar(im_hel_st, ax=axes_hel[1, 1], label="Real Part")
axes_hel[1, 1].grid(True)

# Coefficients Theta (shows how much each basis contributes per mode)
im_theta = axes_hel[1, 2].imshow(
    np.abs(Theta), aspect="auto", cmap="viridis", extent=[0, r, 0, q]
)
axes_hel[1, 2].set_title("Spatial Basis Coefficients |Θ| (q x r)")
axes_hel[1, 2].set_xlabel("Temporal Mode")
axes_hel[1, 2].set_ylabel("Spatial Basis Index")
plt.colorbar(im_theta, ax=axes_hel[1, 2], label="Magnitude")

plt.tight_layout()
plt.show()

# Optional: Plot the prescribed basis functions G
fig_g, axes_g = plt.subplots(1, q, figsize=(15, 3))
for i in range(q):
    axes_g[i].plot(s, np.real(G[:, i]), linewidth=2, label=f"Basis {i + 1}")
    if np.iscomplexobj(G[:, i]):
        axes_g[i].plot(s, np.imag(G[:, i]), "--", label=f"Imag")
    axes_g[i].set_title(f"Spatial Basis {i + 1}")
    axes_g[i].legend()
    axes_g[i].grid(True)
plt.tight_layout()
plt.show()
