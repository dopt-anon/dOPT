#!/usr/bin/env python3
"""
Switched Linear System (SLS) data generation:
    x_{t+1} = A_{σ_i} x_t
    
Guarantees: ρ(A_i) < 1 for all modes i (stable system)
Two modes with mode switching at t = T/2
"""
import numpy as np
import torch
from typing import Tuple, Dict, List

from .lyapunov_block_sdp import (
    build_common_lyapunov_block_sdp,
    solve_block_sdp_cvxpy_reference,
)

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    cp = None
    HAS_CVXPY = False

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization skipped")


def generate_stable_random_matrix(n: int, spectral_radius: float = 0.9, seed: int = None) -> np.ndarray:
    """
    Generate a stable random matrix A such that ρ(A) < spectral_radius < 1.
    
    Method:
    1. Generate random matrix
    2. Normalize by spectral radius to ensure stability
    
    Args:
        n: Matrix dimension
        spectral_radius: Target spectral radius (must be < 1 for stability)
        seed: Random seed for reproducibility
        
    Returns:
        A: n × n stable random matrix with ρ(A) ≈ spectral_radius
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random matrix from Gaussian
    A_raw = np.random.randn(n, n)
    
    # Compute spectral radius
    eigs = np.linalg.eigvals(A_raw)
    rho_current = np.max(np.abs(eigs))
    
    # Normalize to target spectral radius
    A = (spectral_radius / rho_current) * A_raw
    
    # Verify ρ(A) < 1
    eigs_final = np.linalg.eigvals(A)
    rho_final = np.max(np.abs(eigs_final))
    assert rho_final < 1.0, f"ρ(A) = {rho_final} not < 1"
    
    return A


def generate_common_lyapunov_modes(
    n_state: int,
    n_modes: int = 2,
    contraction: float = 0.85,
    lyapunov_condition_number: float = 5.0,
    seed: int = 42,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Generate modes with a known common quadratic Lyapunov function.

    We first construct a positive definite matrix ``P0`` and contractions
    ``B_i`` satisfying ``||B_i||_2 = contraction < 1``.  Defining

        A_i = P0^{-1/2} B_i P0^{1/2}

    gives

        P0 - A_i.T @ P0 @ A_i
        = P0^{1/2} (I - B_i.T @ B_i) P0^{1/2} >> 0.

    Thus all modes share ``V(x) = x.T @ P0 @ x`` as a Lyapunov function,
    which certifies stability under arbitrary switching. ``P0`` is normalized
    to have unit trace, matching the stability-margin SDP below.
    """
    if n_state < 1:
        raise ValueError("n_state must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if not 0.0 < contraction < 1.0:
        raise ValueError("contraction must lie strictly between 0 and 1")
    if lyapunov_condition_number < 1.0:
        raise ValueError("lyapunov_condition_number must be at least 1")

    rng = np.random.default_rng(seed)

    # A nontrivial metric makes the example stronger than Euclidean contraction.
    q_raw = rng.standard_normal((n_state, n_state))
    q, _ = np.linalg.qr(q_raw)
    p_eigenvalues = np.geomspace(1.0, lyapunov_condition_number, n_state)
    p0 = q @ np.diag(p_eigenvalues) @ q.T
    p0 /= np.trace(p0)

    eigvals, eigvecs = np.linalg.eigh(p0)
    p_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    p_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    modes = {}
    for mode_idx in range(n_modes):
        b_raw = rng.standard_normal((n_state, n_state))
        op_norm = np.linalg.svd(b_raw, compute_uv=False)[0]
        b_i = (contraction / op_norm) * b_raw
        a_i = p_inv_sqrt @ b_i @ p_sqrt
        modes[mode_idx] = a_i

        residual = p0 - a_i.T @ p0 @ a_i
        min_residual_eig = np.linalg.eigvalsh(residual).min()
        if min_residual_eig <= 0.0:
            raise RuntimeError(
                f"Failed to construct a common Lyapunov certificate for mode {mode_idx}"
            )

    return modes, p0


def solve_common_lyapunov_margin(
    modes: Dict[int, np.ndarray],
    p_epsilon: float = 1e-6,
    solver: str = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Solve the common-Lyapunov max-margin SDP for fixed system modes.

    The problem is

        maximize    t
        subject to  P >= p_epsilon * I, trace(P) = 1,
                    P - A_i.T @ P @ A_i >= t * I,  for every mode i.

    A positive optimum certifies exponential stability under arbitrary
    switching.  This function is deliberately independent of ``dSDP``: it is
    the forward reference problem used before building the differentiable
    block-SDP representation.
    """
    if not HAS_CVXPY:
        raise ImportError("cvxpy is required to solve the stability-margin SDP")
    if not modes:
        raise ValueError("modes must contain at least one system matrix")

    mode_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in modes.items()}
    first_shape = next(iter(mode_arrays.values())).shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("each mode must be a square matrix")
    if any(a_i.shape != first_shape for a_i in mode_arrays.values()):
        raise ValueError("all mode matrices must have the same shape")

    n_state = first_shape[0]
    identity = np.eye(n_state)
    p_var = cp.Variable((n_state, n_state), symmetric=True)
    t_var = cp.Variable()
    constraints = [p_var >> p_epsilon * identity, cp.trace(p_var) == 1.0]
    constraints.extend(
        p_var - a_i.T @ p_var @ a_i >> t_var * identity
        for a_i in mode_arrays.values()
    )
    problem = cp.Problem(cp.Maximize(t_var), constraints)

    if solver is None:
        solver = cp.SCS
    solve_kwargs = {"solver": solver, "verbose": verbose}
    if solver == cp.SCS or str(solver).upper() == "SCS":
        solve_kwargs.update({"eps": 1e-7, "max_iters": 100_000})
    problem.solve(**solve_kwargs)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"stability-margin SDP failed with status {problem.status}")

    p_star = np.asarray(p_var.value)
    t_star = float(t_var.value)
    residual_min_eigenvalues = {
        key: float(np.linalg.eigvalsh(p_star - a_i.T @ p_star @ a_i - t_star * identity).min())
        for key, a_i in mode_arrays.items()
    }
    return {
        "P": p_star,
        "t": t_star,
        "status": problem.status,
        "P_min_eigenvalue": float(np.linalg.eigvalsh(p_star).min()),
        "residual_min_eigenvalues": residual_min_eigenvalues,
    }


def create_common_lyapunov_switching_data(
    n_state: int = 5,
    T: int = 50,
    N: int = 100,
    n_modes: int = 2,
    contraction: float = 0.85,
    lyapunov_condition_number: float = 5.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Create switched-system trajectories with a known common certificate."""
    modes, p0 = generate_common_lyapunov_modes(
        n_state=n_state,
        n_modes=n_modes,
        contraction=contraction,
        lyapunov_condition_number=lyapunov_condition_number,
        seed=seed,
    )

    # Use balanced contiguous segments so every mode is represented while the
    # trajectories remain easy to inspect visually.
    mode_sequence = [min(n_modes - 1, time_idx * n_modes // T) for time_idx in range(T)]
    data = generate_sls_data(
        n_state=n_state,
        T=T,
        N=N,
        modes=modes,
        mode_sequence=mode_sequence,
        noise_std=noise_std,
        x0_std=1.0,
        seed=seed,
    )
    data["common_lyapunov_P"] = torch.tensor(p0, dtype=torch.float32)
    data["params"].update(
        {
            "n_modes": n_modes,
            "contraction": contraction,
            "lyapunov_condition_number": lyapunov_condition_number,
        }
    )
    return data


def generate_sls_data(
    n_state: int,
    T: int,
    N: int,
    modes: Dict[str, np.ndarray],
    mode_sequence: List[int],
    noise_std: float = 0.0,
    x0_std: float = 1.0,
    seed: int = None
) -> Dict[str, torch.Tensor]:
    """
    Generate trajectories from switched linear system.
    
    Args:
        n_state: State dimension
        T: Trajectory length
        N: Number of trajectories
        modes: Dict {mode_id: A_i}, where A_i is n_state × n_state matrix
        mode_sequence: List of length T with mode indices for each timestep
        noise_std: Standard deviation of Gaussian process noise (optional)
        x0_std: Standard deviation of initial condition distribution
        seed: Random seed
        
    Returns:
        data: Dict with keys:
            'X': Trajectories, shape (N, T, n_state)
            'mode_seq': Mode sequence, shape (T,)
            'modes': Dict of A matrices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    assert len(mode_sequence) == T, f"mode_sequence length {len(mode_sequence)} != T {T}"
    
    # Initialize storage
    X = np.zeros((N, T, n_state))
    
    # Generate N trajectories
    for traj_idx in range(N):
        # Sample initial condition
        x = np.random.randn(n_state) * x0_std
        X[traj_idx, 0, :] = x
        
        # Roll out trajectory
        for t in range(T - 1):
            mode_id = mode_sequence[t]
            A = modes[mode_id]
            
            # Transition with optional noise
            x = A @ x
            if noise_std > 0:
                x += np.random.randn(n_state) * noise_std
            
            X[traj_idx, t + 1, :] = x
    
    # Convert to torch tensors
    data = {
        'X': torch.tensor(X, dtype=torch.float32),
        'mode_seq': torch.tensor(mode_sequence, dtype=torch.long),
        'modes': {k: torch.tensor(v, dtype=torch.float32) for k, v in modes.items()},
        'params': {
            'n_state': n_state,
            'T': T,
            'N': N,
            'noise_std': noise_std,
            'x0_std': x0_std,
        }
    }
    
    return data


def create_two_mode_switching_data(
    n_state: int = 5,
    T: int = 50,
    N: int = 100,
    n_mode1: int = None,
    spectral_radius: float = 0.9,
    noise_std: float = 0.0,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Convenience function: create SLS data with exactly 2 modes switching at T/2.
    
    Mode sequence:
        t ∈ [0, T/2): mode 1 (A_1)
        t ∈ [T/2, T): mode 2 (A_2)
    
    Args:
        n_state: State dimension
        T: Total trajectory length
        N: Number of trajectories
        n_mode1: Length of mode 1 phase (default: T//2)
        spectral_radius: Target spectral radius for stable matrices
        noise_std: Process noise level
        seed: Random seed for reproducibility
        
    Returns:
        data: Dict with 'X', 'mode_seq', 'modes', 'params'
    """
    if n_mode1 is None:
        n_mode1 = T // 2
    
    # Generate stable matrices for mode 1 and mode 2
    if seed is not None:
        np.random.seed(seed)
    
    A1 = generate_stable_random_matrix(n_state, spectral_radius=spectral_radius, seed=seed)
    A2 = generate_stable_random_matrix(n_state, spectral_radius=spectral_radius, seed=seed + 1)
    
    modes = {
        0: A1,
        1: A2,
    }
    
    # Create mode sequence: [0, 0, ..., 0, 1, 1, ..., 1]
    mode_sequence = [0] * n_mode1 + [1] * (T - n_mode1)
    
    print(f"Generated 2-mode SLS data:")
    print(f"  State dim: {n_state}, Trajectory length: {T}, Num trajectories: {N}")
    print(f"  Mode 1 length: {n_mode1}, Mode 2 length: {T - n_mode1}")
    print(f"  ρ(A_1) = {np.max(np.abs(np.linalg.eigvals(A1))):.4f}")
    print(f"  ρ(A_2) = {np.max(np.abs(np.linalg.eigvals(A2))):.4f}")
    print(f"  Process noise std: {noise_std}")
    
    # Generate data
    data = generate_sls_data(
        n_state=n_state,
        T=T,
        N=N,
        modes=modes,
        mode_sequence=mode_sequence,
        noise_std=noise_std,
        x0_std=1.0,
        seed=seed
    )
    
    return data


def visualize_trajectories(
    data: Dict[str, torch.Tensor],
    n_plot: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    savepath: str = None
) -> None:
    """
    Visualize a subset of trajectories with mode switching highlighted.
    
    Args:
        data: Output from generate_sls_data
        n_plot: Number of trajectories to plot
        figsize: Figure size
        savepath: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return
    
    X = data['X'].numpy()
    mode_seq = data['mode_seq'].numpy()
    T = data['params']['T']
    switch_times = np.where(np.diff(mode_seq) != 0)[0] + 1
    
    n_plot = min(n_plot, X.shape[0])
    n_state = X.shape[2]
    
    fig, axes = plt.subplots(n_state, 1, figsize=figsize)
    if n_state == 1:
        axes = [axes]
    
    for state_idx in range(n_state):
        for traj_idx in range(n_plot):
            x_traj = X[traj_idx, :, state_idx]
            axes[state_idx].plot(x_traj, alpha=0.6, label=f"Traj {traj_idx}")
        
        # Highlight mode switch
        for switch_idx, switch_time in enumerate(switch_times):
            axes[state_idx].axvline(
                x=switch_time,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Mode switch' if switch_idx == 0 else None,
            )
        axes[state_idx].set_ylabel(f"$x_{{{state_idx}}}(t)$")
        axes[state_idx].set_xlabel("Time $t$")
        axes[state_idx].grid(True, alpha=0.3)
        if state_idx == 0:
            axes[state_idx].legend()
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {savepath}")
    plt.show()


def verify_stability(data: Dict[str, torch.Tensor]) -> None:
    """
    Verify that trajectories decay towards zero (system is stable).
    """
    X = data['X'].numpy()
    
    # Compute norm of each state over time
    norms = np.linalg.norm(X, axis=2)  # Shape: (N, T)
    
    # Average across trajectories
    norm_mean = norms.mean(axis=0)
    norm_std = norms.std(axis=0)
    
    print("\nStability verification:")
    print(f"  Initial state norm (mean): {norm_mean[0]:.4f}")
    print(f"  Final state norm (mean): {norm_mean[-1]:.4f}")
    print(f"  Decay ratio: {norm_mean[-1] / (norm_mean[0] + 1e-10):.4f}")
    
    # Plot norm decay (if matplotlib available)
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(norm_mean))
    ax.plot(t, norm_mean, 'b-', linewidth=2, label='Mean ||x(t)||')
    ax.fill_between(t, norm_mean - norm_std, norm_mean + norm_std, alpha=0.3)
    mode_seq = data['mode_seq'].numpy()
    switch_times = np.where(np.diff(mode_seq) != 0)[0] + 1
    for switch_idx, switch_time in enumerate(switch_times):
        ax.axvline(
            x=switch_time,
            color='red',
            linestyle='--',
            label='Mode switch' if switch_idx == 0 else None,
        )
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('State norm ||x(t)||')
    ax.set_title('Decay of state trajectories (verifying stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Step 1: generate data with a known common Lyapunov certificate.
    print("=" * 60)
    print("Common-Lyapunov Switched Linear System")
    print("=" * 60)

    data = create_common_lyapunov_switching_data(
        n_state=5,
        T=10,
        N=100,
        n_modes=2,
        contraction=0.95,
        lyapunov_condition_number=5.0,
        noise_std=0.01,
        seed=42
    )

    print(f"\nData shapes:")
    print(f"  X (trajectories): {data['X'].shape}")
    print(f"  mode_seq: {data['mode_seq'].shape}")

    p0 = data["common_lyapunov_P"].numpy().astype(np.float64)
    known_margins = {
        mode_idx: np.linalg.eigvalsh(
            p0 - a_i.numpy().astype(np.float64).T @ p0 @ a_i.numpy().astype(np.float64)
        ).min()
        for mode_idx, a_i in data["modes"].items()
    }
    print("\nKnown common Lyapunov certificate:")
    print(f"  trace(P0): {np.trace(p0):.8f}")
    print(f"  min eigenvalue(P0): {np.linalg.eigvalsh(p0).min():.8f}")
    print(f"  per-mode margins: {known_margins}")

    # Step 2: independently solve the max-margin reference SDP.
    reference_solution = solve_common_lyapunov_margin(
        {mode_idx: a_i.numpy() for mode_idx, a_i in data["modes"].items()}
    )
    print("\nCVXPY common-Lyapunov margin SDP:")
    print(f"  status: {reference_solution['status']}")
    print(f"  optimal t: {reference_solution['t']:.8f}")
    print(f"  min eigenvalue(P*): {reference_solution['P_min_eigenvalue']:.8f}")
    print(f"  constraint residual eigenvalues: {reference_solution['residual_min_eigenvalues']}")

    lifted_modes = {
        mode_idx: a_i.to(dtype=torch.float64)
        for mode_idx, a_i in data["modes"].items()
    }
    objective, a_matrix, b_vector, layout = build_common_lyapunov_block_sdp(
        lifted_modes,
        t_bound=1.0,
    )
    lifted_solution = solve_block_sdp_cvxpy_reference(
        objective,
        a_matrix,
        b_vector,
        layout,
    )
    print("\nLifted single-PSD standard form:")
    print(f"  matrix size: {layout.matrix_size} x {layout.matrix_size}")
    print(f"  equality constraints: {b_vector.numel()}")
    print(f"  optimal t: {lifted_solution['t'].item():.8f}")
    print(f"  equality residual: {lifted_solution['equality_residual']:.3e}")
    print(
        "  direct/lifted t difference: "
        f"{abs(reference_solution['t'] - lifted_solution['t'].item()):.3e}"
    )

    # Visualize
    print("\nVisualizing first 5 trajectories...")
    visualize_trajectories(data, n_plot=5)

    # Verify stability
    verify_stability(data)

    print("\n" + "=" * 60)
    print("Data generation and forward SDP verification complete!")
    print("=" * 60)
