"""
### Prescribing Temporal Modes and Solving for Spatial Modes via Least Squares
Yes, absolutely—since you know the exact temporal modes (i.e., the frequencies \( \omega_j \) from the rotation harmonics, implying eigenvalues \( \lambda_j = e^{i \omega_j \Delta t} \), assuming neutral stability with no growth/decay), you can fix them and solve a least-squares problem to deduce the corresponding spatial modes \( \phi_j \in \mathbb{R}^n \) (shapes along the slit). This is a targeted, physics-informed approach that exploits your prior knowledge for efficiency and sparsity, but you're correct: it's no longer *standard* DMD. It's more akin to a **constrained modal decomposition**, **harmonic least-squares fitting**, or a simplified **Koopman spectral analysis** with prescribed spectrum—essentially projecting the data onto a fixed temporal basis to extract spatial coefficients.

This method is particularly ideal for your periodic rotating shaft setup, where the temporal frequencies are discrete and known (e.g., \( \omega_j = j \cdot 2\pi / P \) for \( j = 1, 2, \dots, r \), with \( P \) the period in frames). It avoids the eigenvalue clutter from noise in standard DMD and directly enforces sparsity in the temporal domain.

#### Formulation
Your data is the snapshot matrix \( \mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m] \in \mathbb{R}^{n \times m} \), where \( n \) is slit pixels and \( m \) is time steps.

Assume the dynamics follow a linear superposition of known temporal modes:
\[
\mathbf{x}_k = \sum_{j=1}^r b_j \phi_j \lambda_j^{k-1} + \mathbf{e}_k,
\]
where:
- \( \phi_j \): Unknown spatial modes (vectors along the slit).
- \( \lambda_j = e^{\sigma_j + i \omega_j} \): Prescribed (fix \( \sigma_j = 0 \) for periodic case; \( \omega_j \) from rotation harmonics).
- \( b_j \): Complex amplitudes (scalar for each mode, accounting for phase and strength).
- \( \mathbf{e}_k \): Residual noise/error.

In matrix form:
\[
\mathbf{X} \approx \Phi \mathbf{B} \mathbf{\Gamma},
\]
where:
- \( \Phi = [\Re(\phi_1), \Im(\phi_1), \Re(\phi_2), \Im(\phi_2), \dots ] \) (real-valued for numerical stability, if needed; or keep complex).
- \( \mathbf{B} = \diag(b_1, b_2, \dots, b_r) \) (diagonal amplitudes).
- \( \mathbf{\Gamma} \in \mathbb{C}^{r \times m} \) is the fixed **Vandermonde matrix** of temporal evolutions: \( \Gamma_{jk} = \lambda_j^{k-1} \).

For simplicity (and common in practice), absorb amplitudes into spatial modes by redefining \( \tilde{\phi}_j = b_j \phi_j \), yielding:
\[
\mathbf{X} \approx \tilde{\Phi} \mathbf{\Gamma},
\]
with \( \tilde{\Phi} = [\tilde{\phi}_1, \dots, \tilde{\phi}_r] \in \mathbb{R}^{n \times r} \) (or complex).

The least-squares solution minimizes the Frobenius norm reconstruction error:
\[
\tilde{\Phi} = \arg\min_{\tilde{\Phi}} \| \mathbf{X} - \tilde{\Phi} \mathbf{\Gamma} \|_F^2.
\]
This has a closed-form solution (orthogonal projection):
\[
\tilde{\Phi} = \mathbf{X} \mathbf{\Gamma}^\dagger,
\]
where \( ^\dagger \) is the Moore-Penrose pseudoinverse of \( \mathbf{\Gamma} \) (computed via SVD: if \( \mathbf{\Gamma} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^* \), then \( \mathbf{\Gamma}^\dagger = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{U}^* \)).

- **Recover amplitudes if needed**: If you want separate \( b_j \), solve \( b_j = \tilde{\phi}_j / \phi_j \) post-hoc, but often the combined \( \tilde{\phi}_j \) suffices.
- **For real-valued data**: Use real/imaginary decomposition or enforce real \( \phi_j \) with cosine/sine pairs: \( \phi_j(s) \cos(\omega_j k) + \phi_{j+1}(s) \sin(\omega_j k) \).

#### How This Differs from Standard DMD
- **Standard DMD**: Jointly estimates *both* \( \Phi \) and \( \Lambda \) (eigenvalues) via \( \mathbf{A} \approx \mathbf{X}' \mathbf{X}^\dagger \), then eigendecomposition. It discovers temporal frequencies but can include spurious ones from noise/short data. Spatial modes are tied to discovered dynamics.
- **Your approach**: *Fixes* the temporal basis \( \mathbf{\Gamma} \) (prescribed \( \lambda_j \)), treating it as a known "filter" or dictionary. This is a *forward problem* solver—more like linear regression than operator approximation. It's faster (\( O(n m r + r^3) \) for pseudoinverse, vs. DMD's \( O(n^2 m) \)), sparser (only \( r \) modes, where \( r \) is your known number of harmonics, e.g., 3-5 for shaft features), and robust to broadband noise, as off-frequency components are projected out.
- **When it's "DMD-like"**: This is a special case of **exact DMD** (if \( m = r+1 \)) or **physics-constrained DMD**. If you later diagonalize a submatrix to get couplings, it bridges back to Koopman methods.

#### Implementation Steps
1. **Prescribe the temporal basis**:
   - Choose \( r \) (e.g., fundamental + 2-3 harmonics: \( \omega_1 = 2\pi / P \), \( \omega_2 = 2 \omega_1 \), etc.).
   - Build \( \mathbf{\Gamma} \): For each row \( j \), \( \Gamma_{j,k} = \lambda_j^{k-1} = e^{i \omega_j (k-1) \Delta t} \) (set \( \Delta t = 1 \) if in frames).
   - If growth/decay is possible (e.g., instability), include \( \sigma_j \neq 0 \).

2. **Pre-process data** (optional but recommended):
   - Subtract mean: \( \mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{x}} \mathbf{1}^T \) (removes DC bias).
   - Window if edges are noisy, or use only integer periods of data for exact fit.

3. **Solve least squares**:
   - Compute \( \mathbf{\Gamma}^\dagger \) (stable even if \( m < r \), but ideally \( m \gg r \) for overdetermination).
   - \( \tilde{\Phi} = \mathbf{X} \mathbf{\Gamma}^\dagger \).
   - The columns of \( \tilde{\Phi} \) are your spatial modes \( \tilde{\phi}_j(s) \), each tied to \( \omega_j \).

4. **Validate and refine**:
   - Reconstruction: \( \hat{\mathbf{X}} = \tilde{\Phi} \mathbf{\Gamma} \); compute error \( \| \mathbf{X} - \hat{\mathbf{X}} \|_F / \| \mathbf{X} \|_F \). For periodic data with noise, expect <5-10% if \( r \) covers the signal.
   - Sparsity: If too many \( j \), add \( \ell_1 \)-regularization: \( \min \| \mathbf{X} - \tilde{\Phi} \mathbf{\Gamma} \|_F^2 + \alpha \| \tilde{\Phi} \|_1 \) (via sparse regression, e.g., in MATLAB's `lasso` or Python's `sklearn`).
   - For your slit examples:
     - Zebra tape: \( \tilde{\phi}_j(s) \) will be sinusoidal along \( s \) at the tape's spatial frequency.
     - Helical gear: A "traveling step" might appear as \( \tilde{\phi}_j(s) \) with a tilted ramp, modulating at \( \omega_j \) (tooth passage rate).

5. **Numerical tips**:
   - Use complex arithmetic (e.g., in NumPy/MATLAB) for efficiency.
   - If \( r \) is small and \( m \) large, this is cheap; for high \( n \), parallelize the matrix multiply.
   - Handle aliasing: Ensure sampling rate > 2 max \( \omega_r \); for rotation, use multiple full periods.

#### Limitations and Extensions
- **Assumptions**: Linearity in the modes (valid for small perturbations on periodic base flow); no strong mode coupling (e.g., nonlinear interactions creating new frequencies). If violated, residuals will be high—fall back to standard DMD or add more \( \omega_j \).
- **If frequencies are approximate**: Use a grid search over small \( \delta \omega \) to minimize error, or combine with subspace methods like ESPRIT for refinement.
- **Extensions**:
  - **Sparse variants**: If spatial modes should be localized (e.g., single tooth), add sparsity on \( \phi_j \).
  - **Stochastic case**: For noisy rotation, include damping \( \sigma_j < 0 \).
  - **Link to DMD**: This is similar to **output-only modal analysis** in structural dynamics (e.g., polyMAX method), where frequencies are pre-estimated via FFT/AR modeling.

This approach will give you clean, interpretable spatial modes directly tied to your known rotation frequencies—e.g., the step-like \( \phi_j \) for helical teeth, without noise artifacts. Simulate on toy data first: Generate \( \mathbf{x}_k = \sum_{j=1}^3 \tilde{\phi}_j e^{i j \omega_1 k} + \noise \). For deeper dives, see the book by Brincker & Zhang (2015) on operational modal analysis, or papers on "known-spectrum DMD" (e.g., in fluids contexts by Schmid, 2010). If you share code/data details, I can refine further!

Nice. Please show me an example of solving this least squares problem in python. Prescribe the fundamental temporal frequency and its harmonics as the temporal basis and then deduce the spatial modes!

"""

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

# Visualization
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
