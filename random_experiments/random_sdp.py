#!/usr/bin/env python3
"""
Clean and clear dSDP test: create an SDP, check eigenvalue uniqueness, and compare methods.
"""
import time 
import argparse
import csv
import os
import sys
from pathlib import Path

# Allow both `python -m random_experiments.random_sdp` and direct execution.
if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cvxpy as cp
from src.dSDP import dSDPLayer, precompute_M_transform, precompute_T_inv
from src.utils.SDP_utils import (
    dsdp_solve,
    diffcp_solve,
    cvxpylayers_solve,
    check_unique_eigenvalues,
    solve_sdp_cvxpy,
)


def ffolayer_solve(C, A_matrix, b, P=None, layer=None, *, eps=1e-8,
                   solver="mosek"):
    """Solve the SDP with FFOLayer, exposing every SDP datum as a parameter."""
    if P is not None:
        raise NotImplementedError(
            "The FFOLayer random-SDP adapter currently supports the linear SDP only."
        )
    from third_party.FFOLayer.src.ffolayer import FFOLayer

    if layer is None:
        n = C.shape[0]
        m = b.shape[0]
        Z = cp.Variable((n, n), symmetric=True, name="Z")
        C_param = cp.Parameter((n, n), name="C")
        A_param = cp.Parameter((m, n * n), name="A")
        b_param = cp.Parameter(m, name="b")
        vec_Z = cp.vec(Z, order="F")
        problem = cp.Problem(
            cp.Minimize(cp.trace(C_param @ Z)),
            [Z >> 0, A_param @ vec_Z == b_param],
        )
        if not problem.is_dpp():
            raise ValueError("The FFOLayer SDP formulation must satisfy DPP.")
        layer = FFOLayer(
            problem,
            parameters=[C_param, A_param, b_param],
            variables=[Z],
            eps=eps,
            backward_eps=eps,
        )

    C_ffo = C.detach().clone().cpu().requires_grad_(True)
    A_ffo = A_matrix.detach().clone().cpu().requires_grad_(True)
    b_ffo = b.detach().clone().cpu().requires_grad_(True)
    solver_map = {"mosek": cp.MOSEK, "scs": cp.SCS}
    solver_name = solver.lower()
    if solver_name not in solver_map:
        raise ValueError(f"Unsupported FFOLayer solver: {solver}")
    solver_args = {
        "solver": solver_map[solver_name],
        "warm_start": False,
        "verbose": False,
    }
    if solver_name == "scs":
        solver_args["eps"] = eps
    else:
        # FFOLayer deep-copies the CVXPY problem. CVXPY 1.9 can otherwise
        # re-canonicalize that clone with stale parameter ids (KeyError).
        solver_args["ignore_dpp"] = False
        # Use CVXPY's solver-agnostic tolerance entry point, exactly as dSDP's
        # CVXPY/MOSEK forward solve does.
        solver_args["eps"] = float(eps)
    # FFOLayer 0.1.2's conic backward incorrectly retains a leading singleton
    # dimension for an unbatched solve. Use an explicit batch of one while
    # keeping the returned leaf tensors unbatched for gradient comparison.
    start_time = time.perf_counter()

    Z_ffo, = layer(
        C_ffo.unsqueeze(0), A_ffo.unsqueeze(0), b_ffo.unsqueeze(0),
        solver_args=solver_args,
    )
    forward_time = time.perf_counter() - start_time

    if solver_name != "scs":
        # FFOLayer 0.1.2 unconditionally injects the SCS-style max_iters
        # option into its backward arguments. CVXPY/MOSEK rejects that option.
        layer._solver_args_bwd.pop("max_iters", None)
    # FFOLayer 0.1.2 keeps a leading singleton batch dimension even for
    # unbatched inputs; normalize it to the interface used by this benchmark.
    if Z_ffo.ndim == C_ffo.ndim + 1 and Z_ffo.shape[0] == 1:
        Z_ffo = Z_ffo[0]
    # The native forward solve already produced these duals. Read them before
    # backward solves FFOLayer's perturbed problem and overwrites CVXPY state.
    bundle = layer.bundles[0]
    if len(bundle["psd_cones"]) != 1 or len(bundle["eq_constraints"]) != 1:
        raise RuntimeError("random SDP expects one PSD cone and one equality block")
    S_ffo = torch.as_tensor(
        np.asarray(bundle["psd_cones"][0].dual_value), dtype=C_ffo.dtype
    )
    nu_ffo = torch.as_tensor(
        np.asarray(bundle["eq_constraints"][0].dual_value).reshape(-1),
        dtype=b_ffo.dtype,
    )
    return Z_ffo, S_ffo, nu_ffo, C_ffo, A_ffo, b_ffo, forward_time, layer


def parameter_gradient_difference(candidate, reference):
    """Maximum relative gradient error over C, A, and b."""
    errors = {}
    for name, cand, ref in zip(("C", "A", "b"), candidate, reference):
        if cand.grad is None or ref.grad is None:
            errors[name] = None
            continue
        ref_grad = ref.grad.detach().cpu()
        cand_grad = cand.grad.detach().cpu()
        errors[name] = (
            torch.linalg.norm(cand_grad - ref_grad)
            / max(torch.linalg.norm(ref_grad).item(), 1e-12)
        ).item()
    finite_errors = [value for value in errors.values() if value is not None]
    return errors, max(finite_errors) if finite_errors else None
     
def create_sdp_instance_matrix(n: int = 5, m: int = 3, random_seed: int = 42, use_quadratic: bool = True):
    """
    Create SDP instance with matrix constraints: generate C, A matrix, b vector, and optional P matrix, solve it, and check eigenvalues.
    SDP format: min <C,Z> + 1/2*vec(Z)^T*P*vec(Z) s.t. A*vec(Z) = b, Z >= 0
    """
    torch.manual_seed(random_seed)

    # print(f"Creating {n}×{n} SDP with {m} constraints (A*vec(Z) = b)...")
    if use_quadratic:
        print("   Including quadratic term 1/2*vec(Z)^T*P*vec(Z)")
    
    # 1. Generate objective matrix C (positive definite)
    A_rand = torch.randn(n, n, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A_rand)
    eigenvals = torch.logspace(0, 1, n, dtype=torch.float64)  # 1 to 10
    C = Q @ torch.diag(eigenvals) @ Q.T
    
    # 2. Generate quadratic matrix P = Q^T*Q + 1e-5*I
    P = None
    if use_quadratic:
        Q_rand = torch.randn(n*n, n*n, dtype=torch.float64)
        P = Q_rand.T @ Q_rand + 1e-5 * torch.eye(n*n, dtype=torch.float64)
        print(f"P matrix condition number: {torch.linalg.cond(P):.2f}")
    
    # 3. Generate random constraint matrix A (m x n^2)
    A_matrix = torch.randn(m, n*n, dtype=torch.float64)
    
    # 4. Generate feasible point Z0 with well-separated eigenvalues
    eigenvals_z0 = torch.logspace(0, 1.5, n, dtype=torch.float64)  # 1 to ~31.6
    eigenvals_z0 = torch.sort(eigenvals_z0, descending=True)[0]
    Q_z0 = torch.randn(n, n, dtype=torch.float64)
    Q_z0, _ = torch.linalg.qr(Q_z0)
    Z0 = Q_z0 @ torch.diag(eigenvals_z0) @ Q_z0.T
    
    # 5. Compute b = A * vec(Z0) to ensure feasibility
    vec_Z0 = Z0.flatten()
    b = A_matrix @ vec_Z0
    
    # print(f"Generated SDP with matrix constraints:")
    # print(f"C condition number: {torch.linalg.cond(C):.2f}")
    # print(f"A_matrix shape: {A_matrix.shape}")
    # if P is not None:
        # print(f"P matrix shape: {P.shape}")
    # print(f"Z0 eigenvalues: {eigenvals_z0}")
    # print(f"b values range: [{b.min():.3f}, {b.max():.3f}]")
    
    return C, A_matrix, b, P

def cvxpy_solve(C, A_matrix, b, P=None):
    """Solve using CVXPY with matrix constraints and verify"""
    print("\nCVXPY solving...")
    
    Z_star, S_star, nu_star, eigenvals, eigenvecs, status = solve_sdp_cvxpy(C.detach().numpy(), A_matrix.detach().numpy(), b.detach().numpy(), P.detach().numpy() if P is not None else None)
    
    if Z_star is not None:
        print(f"Status: {status}")
        # Check eigenvalues
        unique, eigenvals_sorted = check_unique_eigenvalues(Z_star)
        print(f"Eigenvalues: {eigenvals}")
        return Z_star, unique
    else:
        print(f"CVXPY failed: {status}")
        return None, False




def solution_accuracy(Z, C, A_matrix, b, P=None, S=None, nu=None, method_name="Unknown"):
    """
    Measure solution accuracy for SDP: min <C,Z> + 1/2*vec(Z)^T*P*vec(Z) s.t. A*vec(Z) = b, Z >= 0
    
    Args:
        Z: Primal solution (n x n matrix)
        C: Objective matrix (n x n)
        A_matrix: Constraint matrix (m x n^2)
        b: Constraint vector (m x 1)
        P: Quadratic term matrix (n^2 x n^2), optional
        S: Dual matrix for PSD constraint (n x n), optional
        nu: Dual vector for equality constraint (m x 1), optional
        method_name: Name of the method for printing
    
    Returns:
        dict: Dictionary containing all accuracy metrics
    """
    
    # Convert to numpy if needed
    if torch.is_tensor(Z):
        Z_np = Z.detach().cpu().numpy()
        C_np = C.detach().cpu().numpy()
        A_np = A_matrix.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        P_np = P.detach().cpu().numpy() if P is not None else None
        S_np = S.detach().cpu().numpy() if S is not None else None
        nu_np = nu.detach().cpu().numpy() if nu is not None else None
    else:
        Z_np, C_np, A_np, b_np = Z, C, A_matrix, b
        P_np, S_np, nu_np = P, S, nu
    
    n = Z_np.shape[0]
    m = b_np.shape[0]
    
    # print(f"\nSolution Accuracy Analysis - {method_name}")
    # print("=" * 50)
    
    # 1. Primal Feasibility
    # Check Z >= 0 (positive semidefinite)
    eigenvals_Z = np.linalg.eigvalsh(Z_np)
    min_eigval = float(np.min(eigenvals_Z))
    psd_violation = max(0.0, -min_eigval)
    psd_feasible = min_eigval >= -1e-6
    
    # Check A*vec(Z) = b
    vec_Z = Z_np.flatten(order='F')  # Fortran order to match CVXPY
    constraint_residual = A_np @ vec_Z - b_np
    primal_residual_norm = np.linalg.norm(constraint_residual)
    primal_residual_max = np.max(np.abs(constraint_residual))
    constraint_feasible = primal_residual_norm < 1e-6
    
    # print("Primal Feasibility:")
    # print(f"  PSD constraint:  (min eigenvalue: {min_eigval:.2e})")
    # print(f"    max|A*vec(Z) - b|: {primal_residual_max:.2e}")
    
    # 2. Dual Feasibility (if dual variables provided)
    dual_residual = None
    dual_feasible = None
    min_eigval_S = None
    dual_psd_violation = None
    if S_np is not None and nu_np is not None:
        M = precompute_M_transform(n).numpy()
        A_T_nu = A_np.T @ nu_np  # Shape: n^2 x 1
        
        
        if P_np is not None:
            # For quadratic case: A^T*nu + S = C + P*vec(Z)
            P_vec_Z = P_np @ vec_Z  # Shape: n^2 x 1
            dual_residual = M @ A_T_nu - M @ S_np.reshape(-1,order="F") + M @ C_np.reshape(-1,order="F") + M @ P_vec_Z
        else:
            # For linear case: A^T*nu + S = C
            dual_residual = M @ A_T_nu - M @ S_np.reshape(-1,order="F") + M @ C_np.reshape(-1,order="F")
        dual_residual = float(np.max(np.abs(dual_residual)))
        dual_feasible = dual_residual < 1e-6
        
        # Check S >= 0
        eigenvals_S = np.linalg.eigvalsh(S_np)
        min_eigval_S = float(np.min(eigenvals_S))
        dual_psd_violation = max(0.0, -min_eigval_S)
        S_psd_feasible = min_eigval_S >= -1e-8
        
        # print("Dual Feasibility:")
        # print(f"    ||C + A^T*nu - S||_F: {dual_residual:.2e}")
        # print(f"  S >= 0: (min eigenvalue: {min_eigval_S:.2e})")
    
    # 3. Complementary Slackness and Duality Gap
    # Primal objective
    primal_obj = np.trace(C_np @ Z_np)
    if P_np is not None:
        primal_obj += 0.5 * vec_Z.T @ P_np @ vec_Z
    
    # Dual objective (if dual variables available)
    dual_obj = None
    duality_gap = None
    complementarity_violation = None
    
    if S_np is not None and nu_np is not None:
        dual_obj = -b_np.T @ nu_np
        duality_gap = abs(primal_obj - dual_obj)
        
        # Complementary slackness: <Z, S> = 0
        complementarity_product = np.trace(Z_np @ S_np)
        complementarity_violation = abs(complementarity_product)
        complementarity_satisfied = complementarity_violation < 1e-6
        
        # print("Optimality Conditions:")
        # print(f" <Z, S>: {complementarity_violation:.2e}")
    else:
        print(f"  (Dual variables not provided)")
    
    # 4. Overall Assessment
    overall_feasible = psd_feasible and constraint_feasible
    if dual_residual is not None:
        overall_optimal = overall_feasible and dual_feasible and (abs(duality_gap) < 1e-6) and (complementarity_violation < 1e-6)
    else:
        overall_optimal = overall_feasible
    
    # Return all metrics
    metrics = {
        'method_name': method_name,
        'primal_objective': primal_obj,
        'dual_objective': dual_obj,
        'duality_gap': duality_gap,
        'primal_residual_norm': primal_residual_norm,
        'primal_residual_max': primal_residual_max,
        'dual_residual': dual_residual,
        'dual_psd_violation': dual_psd_violation,
        'complementarity_violation': complementarity_violation,
        'min_eigenvalue_Z': min_eigval,
        'min_eigenvalue_S': min_eigval_S,
        'psd_feasible': psd_feasible,
        'constraint_feasible': constraint_feasible,
        'dual_feasible': dual_feasible,
        'overall_feasible': overall_feasible,
        'overall_optimal': overall_optimal
    }
    
    return metrics


def analytic_gradient(Z_star, nu_star, n, m):
    """
    Calculate analytic gradients using envelope theorem:
    - dL/dC = Z*
    - dL/db = -nu*
    - dL/dA[i] = nu*[i] * Z*.flatten()
    
    Args:
        Z_star: Optimal primal solution (n x n matrix)
        nu_star: Optimal dual solution for equality constraints (m x 1 vector)
        n: Size of matrix Z
        m: Number of equality constraints
    
    Returns:
        dict: Dictionary containing analytic gradients
    """
    # Convert to numpy if needed
    if torch.is_tensor(Z_star):
        Z_np = Z_star.detach().cpu().numpy()
        nu_np = nu_star.detach().cpu().numpy()
    else:
        Z_np = Z_star
        nu_np = nu_star
    
    # Analytic gradients according to envelope theorem
    grad_C_analytic = Z_np  # dL/dC = Z*
    grad_b_analytic = -nu_np  # dL/db = -nu*
    
    # dL/dA[i] = nu*[i] * Z*.flatten()
    vec_Z = Z_np.flatten(order='F')  # Fortran order to match CVXPY
    grad_A_analytic = np.outer(nu_np, vec_Z)  # Shape: (m, n^2)
    
    return {
        'grad_C': grad_C_analytic,
        'grad_b': grad_b_analytic, 
        'grad_A': grad_A_analytic
    }


def compare_solutions_and_gradients(Z_dsdp, Z_cvxpy, 
                                   C_dsdp, A_dsdp, b_dsdp, P_dsdp,
                                   C_cvxpy, A_cvxpy, b_cvxpy, P_cvxpy):
    """Compare solutions and gradients for matrix constraint format"""
    # print("\nComparing solutions and gradients...")  # Muted
    # print(f"dSDP solution device: {Z_dsdp.device}, CVXPYLayers solution device: {Z_cvxpy.device}")  # Muted
    
    # Ensure both solutions are on the same device for comparison
    if Z_dsdp.device != Z_cvxpy.device:
        Z_cvxpy = Z_cvxpy.to(Z_dsdp.device)
        # print(f"Moved CVXPYLayers solution to {Z_dsdp.device}")  # Muted
    
    # Compare solutions
    sol_diff = torch.norm(Z_dsdp - Z_cvxpy) / torch.norm(Z_cvxpy)
    sol_diff = sol_diff.item()
    # print(f"Solution difference: {sol_diff:.2e}")  # Muted
    
    # Z_dsdp.sum().backward()
    # Z_cvxpy.sum().backward()
    
    # Compare gradients (muted output)
    grad_C_abs = torch.norm(C_dsdp.grad - C_cvxpy.grad).item()
    grad_C_ref = torch.norm(C_cvxpy.grad).item()
    grad_C_diff = grad_C_abs / max(grad_C_ref, 1e-12)
    print(
        f"C gradient difference: {grad_C_diff:.2e} relative "
        f"({grad_C_abs:.2e} absolute; reference norm {grad_C_ref:.2e})"
    )
    # Compare A gradients
    if A_dsdp.grad is not None and A_cvxpy.grad is not None:
        grad_A_diff = torch.norm(A_dsdp.grad - A_cvxpy.grad) / torch.norm(A_cvxpy.grad)
        grad_A_diff = grad_A_diff.item()
        print(f"A gradient difference: {grad_A_diff:.2e}")
        
        # print(f"dSDP A gradient:\n{A_dsdp.grad}")
        # print(f"CVXPYLayers A gradient:\n{A_cvxpy.grad}")
    else:
        grad_A_diff = 0.0
        print(f"A gradients: dOPT={A_dsdp.grad is not None}, CVXPYLayers={A_cvxpy.grad is not None}")
    
    # Compare b gradients
    if b_dsdp.grad is not None and b_cvxpy.grad is not None:
        grad_b_diff = torch.norm(b_dsdp.grad - b_cvxpy.grad) / torch.norm(b_cvxpy.grad)
        grad_b_diff = grad_b_diff.item()
        print(f"b gradient difference: {grad_b_diff:.2e}")
        
        # print(f"dSDP b gradient:\n{b_dsdp.grad}")
        # print(f"CVXPYLayers b gradient:\n{b_cvxpy.grad}")
    else:
        grad_b_diff = 0.0
        print(f"b gradients: dOPT={b_dsdp.grad is not None}, CVXPYLayers={b_cvxpy.grad is not None}")
    
    # Compare P gradients
    grad_P_diff = 0.0
    if P_dsdp is not None and P_cvxpy is not None:
        if P_dsdp.grad is not None and P_cvxpy.grad is not None:
            grad_P_diff = torch.norm(P_dsdp.grad - P_cvxpy.grad) 
            grad_P_diff = grad_P_diff.item()
            print(f"P gradient difference: {grad_P_diff:.2e}")
        else:
            print(f"P gradients: dOPT={P_dsdp.grad is not None}, CVXPYLayers={P_cvxpy.grad is not None}")
    
    return sol_diff, max(grad_C_diff, grad_A_diff, grad_b_diff, grad_P_diff)


def compare_gradients_with_analytic(C_dsdp, A_dsdp, b_dsdp, C_cvxpy, A_cvxpy, b_cvxpy, 
                                   Z_star, nu_star, n, m):
    """
    Compare computed gradients with analytic gradients and compute relative errors
    
    Args:
        C_dsdp, A_dsdp, b_dsdp: dSDP parameter tensors with gradients
        C_cvxpy, A_cvxpy, b_cvxpy: CVXPYLayers parameter tensors with gradients  
        Z_star: Optimal solution
        nu_star: Optimal dual variables
        n: Matrix size
        m: Number of constraints
    """
    print("\nGradient Comparison with Analytic Solution")
    print("=" * 60)
    
    # Calculate analytic gradients
    analytic_grads = analytic_gradient(Z_star, nu_star, n, m)
    
    # Convert analytic gradients to torch tensors for comparison
    grad_C_analytic = torch.tensor(analytic_grads['grad_C'], dtype=torch.float64)
    grad_b_analytic = torch.tensor(analytic_grads['grad_b'], dtype=torch.float64) 
    grad_A_analytic = torch.tensor(analytic_grads['grad_A'], dtype=torch.float64)
    
    print(f"{'Method':<12} {'C Rel Error':<12} {'A Rel Error':<12} {'b Rel Error':<12}")
    print("-" * 60)
    
    # dSDP gradient comparison
    if C_dsdp.grad is not None:
        # C gradient relative error
        C_dsdp_error = torch.norm(C_dsdp.grad - grad_C_analytic) / torch.norm(grad_C_analytic)
        
        # A gradient relative error  
        A_dsdp_error = 0.0
        if A_dsdp.grad is not None:
            A_dsdp_error = torch.norm(A_dsdp.grad - grad_A_analytic) / torch.norm(grad_A_analytic)
        
        # b gradient relative error
        b_dsdp_error = 0.0
        if b_dsdp.grad is not None:
            b_dsdp_error = torch.norm(b_dsdp.grad - grad_b_analytic) / torch.norm(grad_b_analytic)
        
        print(f"{'dOPT':<12} {C_dsdp_error.item():<12.2e} {A_dsdp_error if isinstance(A_dsdp_error, float) else A_dsdp_error.item():<12.2e} {b_dsdp_error if isinstance(b_dsdp_error, float) else b_dsdp_error.item():<12.2e}")
    
    # CVXPYLayers gradient comparison
    if C_cvxpy.grad is not None:
        # C gradient relative error
        C_cvxpy_error = torch.norm(C_cvxpy.grad - grad_C_analytic) / torch.norm(grad_C_analytic)
        
        # A gradient relative error
        A_cvxpy_error = 0.0
        if A_cvxpy.grad is not None:
            A_cvxpy_error = torch.norm(A_cvxpy.grad - grad_A_analytic) / torch.norm(grad_A_analytic)
        
        # b gradient relative error
        b_cvxpy_error = 0.0
        if b_cvxpy.grad is not None:
            b_cvxpy_error = torch.norm(b_cvxpy.grad - grad_b_analytic) / torch.norm(grad_b_analytic)
        
        print(f"{'CVXPYLayers':<12} {C_cvxpy_error.item():<12.2e} {A_cvxpy_error if isinstance(A_cvxpy_error, float) else A_cvxpy_error.item():<12.2e} {b_cvxpy_error if isinstance(b_cvxpy_error, float) else b_cvxpy_error.item():<12.2e}")


def dim2num_var(n):
    return n*(n+1) / 2


def write_summary_csv(path, rows):
    """Upsert summaries, keeping one row per experiment configuration/method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    key_fields = ("n", "m", "n_test", "base_seed", "eps", "method")
    existing = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    replacement_keys = {
        tuple(str(row[field]) for field in key_fields) for row in rows
    }
    existing = [
        row for row in existing
        if tuple(row.get(field, "") for field in key_fields) not in replacement_keys
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)


def main(
    n=10,
    n_test=3,
    base_seed=888,
    eps=1e-8,
    methods=("dOPT", "cvxpylayers"),
    ffo_solver="mosek",
    output_dir=Path("random_experiments/results/random_sdp"),
    no_warmup=False,
):
    # GPU/CPU device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64  # Keep float64 for numerical stability
    print(f" Using device: {device}")

    m = int(dim2num_var(n) - 1) // 2

    use_quadratic = False  # Set to False to test linear objective only


    def _safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else None
    
    
    aggregate_stats = {
        'dOPT': {
            'forward': [],
            'backward': [],
            'total': [],
            'primal_residual_max': [],
            'dual_residual': [],
            'dual_psd_violation': [],
            'duality_gap_abs': [],
            'complementarity_violation': [],
            'psd_violation': [],
        },
        'CVXPYLayers': {
            'forward': [],
            'backward': [],
            'total': [],
            'primal_residual_max': [],
            'dual_residual': [],
            'dual_psd_violation': [],
            'duality_gap_abs': [],
            'complementarity_violation': [],
            'psd_violation': [],
        },
        'FFOLayer': {
            'forward': [],
            'backward': [],
            'total': [],
            'primal_residual_max': [],
            'dual_residual': [],
            'dual_psd_violation': [],
            'duality_gap_abs': [],
            'complementarity_violation': [],
            'psd_violation': [],
            'gradient_diff': [],
            'C_gradient_diff': [],
            'A_gradient_diff': [],
            'b_gradient_diff': [],
        },
        'comparison': {
            'sol_diff': [],
            'grad_diff': [],
        }
    }
    
    
    methods = set(methods)
    run_dsdp = "dOPT" in methods
    run_cvxpylayers = "cvxpylayers" in methods
    run_ffolayer = "ffolayer" in methods
    cvxpy_layer = None
    ffo_layer = None

    output_dir = Path(output_dir)
    all_results_dir = output_dir / "all"
    all_results_dir.mkdir(parents=True, exist_ok=True)
    raw_headers = [
        "seed", "method", "dim", "m", "eps", "forward_solver",
        "forward_s", "backward_s", "total_s", "primal_psd_violation",
        "primal_residual_max", "dual_psd_violation", "dual_residual_max",
        "duality_gap_abs", "complementarity_abs",
        "solution_rel_diff_vs_cvxpylayers",
        "gradient_rel_diff_vs_cvxpylayers", "C_gradient_rel_diff",
        "A_gradient_rel_diff", "b_gradient_rel_diff",
    ]
    raw_path = all_results_dir / f"dim_{n}.csv"
    selected_labels = set(methods)
    persisted_rows = []
    if raw_path.exists():
        with raw_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == raw_headers:
                persisted_rows = [
                    row for row in reader if row["method"] not in selected_labels
                ]

    def write_raw_results():
        with raw_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_headers)
            writer.writeheader()
            writer.writerows(persisted_rows)

    # Clear stale rows only for methods selected in this invocation, matching
    # the random-SOCP benchmark's resumable per-dimension layout.
    write_raw_results()
    run_rows = []
    
    # Build the dSDP problem once, before the timed random-instance loop.  Every
    # instance below has the same (n, m) dimensions and objective type.
    dsdp_layer = None
    if run_dsdp:
        dsdp_transforms = (
            precompute_M_transform(n, device, dtype),
            precompute_T_inv(n, device, dtype),
        )
        dsdp_settings = {
            "solver": cp.MOSEK,
            "solver_args": {"eps": eps},
        }
        dsdp_layer = dSDPLayer(
            n,
            m,
            settings=dsdp_settings,
            transforms=dsdp_transforms,
            has_quadratic_term=use_quadratic,
        )

    # Match the random-SOCP benchmark: initialize every selected layer and run
    # one complete forward/backward pass before collecting timings. In
    # particular, FFOLayer lazily builds its copied forward/perturbed problems
    # on the first call; that setup is intentionally outside the benchmark.
    if not no_warmup:
        warm_C, warm_A, warm_b, warm_P = create_sdp_instance_matrix(
            n, m, random_seed=base_seed - 1, use_quadratic=use_quadratic
        )
        warm_C = warm_C.to(device=device, dtype=dtype)
        warm_A = warm_A.to(device=device, dtype=dtype)
        warm_b = warm_b.to(device=device, dtype=dtype)
        if warm_P is not None:
            warm_P = warm_P.to(device=device, dtype=dtype)
        if run_dsdp:
            warm_Z, *_ = dsdp_solve(
                warm_C, warm_A, warm_b, warm_P, dsdp_layer=dsdp_layer
            )
            warm_Z.sum().backward()
        if run_cvxpylayers:
            warm_result = cvxpylayers_solve(
                warm_C, warm_A, warm_b, warm_P, cvxpy_layer, eps=eps
            )
            warm_Z_cvxpy, *_, cvxpy_layer = warm_result
            warm_Z_cvxpy.sum().backward()
        if run_ffolayer:
            warm_Z_ffo, *_, ffo_layer = ffolayer_solve(
                warm_C, warm_A, warm_b, warm_P, ffo_layer,
                eps=eps, solver=ffo_solver,
            )
            warm_Z_ffo.sum().backward()
    
    for i_test in range(n_test):
        print("\n" + "#" * 80)
        print(f"Test Instance {i_test + 1}/{n_test}")
            
        C, A, b, P = create_sdp_instance_matrix(
            n, m, random_seed=base_seed + i_test, use_quadratic=use_quadratic
        )
        # Move tensors to selected device
        C = C.to(device=device, dtype=dtype)
        A = A.to(device=device, dtype=dtype)
        b = b.to(device=device, dtype=dtype)
        if P is not None:
            P = P.to(device=device, dtype=dtype)
        
        
        # 2. Use CVXPY to verify this SDP has unique eigenvalues
        # Z_cvxpy_ref, unique_ref = cvxpy_solve(C, A, b, P)
        
        # if not unique_ref:
        #     raise ValueError("SDP does not have unique eigenvalues; adjustment required")
        
        Z_dsdp = S_dsdp = nu_dsdp = None
        C_dsdp = A_dsdp = b_dsdp = P_dsdp = None
        dsdp_acc = None
        dsdp_fwd_time = dsdp_bwd_time = None
        if run_dsdp:
            # 3. Solve using dSDP
            Z_dsdp, S_dsdp, nu_dsdp, C_dsdp, A_dsdp, b_dsdp, P_dsdp, dsdp_fwd_time = dsdp_solve(
                C,
                A,
                b,
                P,
                dsdp_layer=dsdp_layer,
            )
            dsdp_acc = solution_accuracy(
                Z_dsdp, C, A, b, P, S=S_dsdp, nu=nu_dsdp,
                method_name="dOPT Solution",
            )
            if Z_dsdp is None:
                print("dOPT solving failed")

            start_time = time.perf_counter()
            Z_dsdp.sum().backward()
            dsdp_bwd_time = time.perf_counter() - start_time
        
        # grad = np.random.randn(int(dim2num_var(n)))
        M = precompute_M_transform(n).numpy()
        grad = M @ np.ones(n**2)
        # 4. Solve using diffcp
        # (Z_diffcp, 
        #  diffcp_fwd_time, diffcp_bwd_time, 
        # dA_diffcp, db_diffcp, dc_diffcp) = diffcp_solve(C, A, b, P, mode="",grad=grad)
        
        # print(f"Diffcp dq: {dc_diffcp[:5]}")
    
        # 6. Solve using CVXPYLayers
        Z_cvxpy = S_cvxpy = nu_cvxpy = None
        C_cvxpy = A_cvxpy = b_cvxpy = P_cvxpy = None
        cvxpylayers_acc = None
        dcvxpylayer_fwd_time = dcvxpylayer_bwd_time = None
        if run_cvxpylayers:
            print("CvxpyLayer solving")
            (Z_cvxpy, S_cvxpy, nu_cvxpy,
             C_cvxpy, A_cvxpy, b_cvxpy, P_cvxpy,
             dcvxpylayer_fwd_time, cvxpy_layer) = cvxpylayers_solve(
                C, A, b, P, cvxpy_layer, eps=eps
            )
            cvxpylayers_acc = solution_accuracy(
                Z_cvxpy, C_cvxpy, A_cvxpy, b_cvxpy, P_cvxpy,
                S=S_cvxpy, nu=nu_cvxpy,
                method_name="CVXPYLayers Solution",
            )

            start_time = time.perf_counter()
            Z_cvxpy.sum().backward()
            dcvxpylayer_bwd_time = time.perf_counter() - start_time

        # 7. Solve using FFOLayer. All linear-SDP data C, A, and b are
        # differentiable inputs; CVXPYLayer is used as the gradient reference
        # when both methods are selected.
        Z_ffo = None
        S_ffo = nu_ffo = None
        C_ffo = A_ffo = b_ffo = None
        ffo_acc = None
        ffo_fwd_time = ffo_bwd_time = None
        ffo_grad_diff = None
        if run_ffolayer:
            print("FFOLayer solving")
            (Z_ffo, S_ffo, nu_ffo, C_ffo, A_ffo, b_ffo,
             ffo_fwd_time, ffo_layer) = ffolayer_solve(
                C, A, b, P, ffo_layer,
                eps=eps,
                solver=ffo_solver,
            )
            ffo_acc = solution_accuracy(
                Z_ffo, C_ffo, A_ffo, b_ffo, S=S_ffo, nu=nu_ffo,
                method_name="FFOLayer Solution"
            )
            start_time = time.perf_counter()
            Z_ffo.sum().backward()
            ffo_bwd_time = time.perf_counter() - start_time
            grad_presence = {
                "C": C_ffo.grad is not None,
                "A": A_ffo.grad is not None,
                "b": b_ffo.grad is not None,
            }
            print(f"FFOLayer gradients present: {grad_presence}")
            if run_cvxpylayers and C_cvxpy is not None:
                ffo_errors, ffo_grad_diff = parameter_gradient_difference(
                    (C_ffo, A_ffo, b_ffo),
                    (C_cvxpy, A_cvxpy, b_cvxpy),
                )
                formatted = {
                    name: ("N/A" if value is None else f"{value:.2e}")
                    for name, value in ffo_errors.items()
                }
                print(f"FFOLayer relative gradient errors vs CVXPYLayer: {formatted}")
            else:
                ffo_errors = {}
        
        # if Z_cvxpy is None:
        #     print("CVXPYLayers solving failed")
        
        # 6. Compare results between dSDP and CVXPYLayers.
        sol_diff = grad_diff = None
        if run_dsdp and run_cvxpylayers and Z_dsdp is not None and Z_cvxpy is not None:
            sol_diff, grad_diff = compare_solutions_and_gradients(
                Z_dsdp, Z_cvxpy,
                C_dsdp, A_dsdp, b_dsdp, P_dsdp,
                C_cvxpy, A_cvxpy, b_cvxpy, P_cvxpy
            )
        
        
        
        
        
        
        # ============================================================================
        # Performance Summary
        # ============================================================================
        def print_summary():
            print("\n" + "=" * 80)
            print("PERFORMANCE COMPARISON SUMMARY")
            print("=" * 80)
            
            # Collect methods and their timing data
            methods = {}
            
            # dSDP
            if Z_dsdp is not None:
                methods['dOPT'] = {
                    'forward': dsdp_fwd_time,
                    'backward': dsdp_bwd_time,
                    'total': dsdp_fwd_time + dsdp_bwd_time,
                    'solved': True
                }
            else:
                methods['dOPT'] = {'solved': False}
            
            # CVXPYLayers  
            if Z_cvxpy is not None:
                methods['CVXPYLayers'] = {
                    'forward': dcvxpylayer_fwd_time,
                    'backward': dcvxpylayer_bwd_time,
                    'total': dcvxpylayer_fwd_time + dcvxpylayer_bwd_time,
                    'solved': True
                }
            else:
                methods['CVXPYLayers'] = {'solved': False}
            
            # diffcp (if enabled)
            # if 'diffcp_fwd_time' in locals() and diffcp_fwd_time is not None:
            #     methods['diffcp'] = {
            #         'forward': diffcp_fwd_time,
            #         'backward': diffcp_bwd_time, 
            #         'total': diffcp_fwd_time + diffcp_bwd_time,
            #         'solved': True
            #     }
            
            # Print timing table
            solved_methods = {k: v for k, v in methods.items() if v.get('solved', False)}
            
            if solved_methods:
                print(f"{'Method':<12} {'Forward':<10} {'Backward':<10} {'Total':<10}")
                print("-" * 50)
                
                for method, data in methods.items():
                    if data.get('solved', False):
                        print(f"{method:<12} {data['forward']:<10.4f} {data['backward']:<10.4f} {data['total']:<10.4f}")
                    else:
                        print(f"{method:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            
                # Calculate speedups (using dSDP as baseline)
                if 'dOPT' in solved_methods:
                    baseline = solved_methods['dOPT']
                    print("\nSpeedup Analysis (vs dOPT):")
                    print(f"{'Method':<12} {'Forward':<10} {'Backward':<10} {'Total':<10}")
                    print("-" * 50)
                    
                    for method, data in solved_methods.items():
                        if method != 'dOPT':
                            fwd_speedup = data['forward'] / baseline['forward']
                            bwd_speedup = data['backward'] / baseline['backward'] 
                            total_speedup = data['total'] / baseline['total']
                            print(f"{method:<12} {fwd_speedup:<10.2f}x {bwd_speedup:<10.2f}x {total_speedup:<10.2f}x")
            
                # Solution accuracy comparison
                print("\nSolution Accuracy:")
                if Z_dsdp is not None and Z_cvxpy is not None:
                    print(f"  dOPT vs CVXPYLayers solution difference: {sol_diff:.2e}")
                    print(f"  dOPT vs CVXPYLayers gradient difference: {grad_diff:.2e}")
                    
                    success = sol_diff < 1e-3 and grad_diff < 1e-3
                    if success:
                        print("  Methods are consistent")
                    else:
                        print("  Methods have significant differences")
                
                # Accuracy metrics comparison table
                print("\nAccuracy Metrics:")
                print(f"{'Method':<12} {'PSD Viol':<10} {'Dual PSD':<10} {'Primal Res':<12} {'Dual Res':<10} {'Complement':<12}")
                print("-" * 78)
                
                # dSDP accuracy
                if  dsdp_acc is not None:
                    psd_viol = max(0, -dsdp_acc['min_eigenvalue_Z']) if dsdp_acc['min_eigenvalue_Z'] < 0 else 0
                    dual_psd_viol = "N/A"
                    if  S_dsdp is not None:
                        S_eigenvals = torch.linalg.eigvals(S_dsdp.detach()).real
                        dual_psd_viol = f"{max(0, -torch.min(S_eigenvals).item()):.2e}"
                    
                    primal_res = dsdp_acc['primal_residual_max']
                    dual_res = dsdp_acc['dual_residual'] if dsdp_acc['dual_residual'] is not None else "N/A"
                    complement = dsdp_acc['complementarity_violation'] if dsdp_acc['complementarity_violation'] is not None else "N/A"
                    
                    print(f"{'dOPT':<12} {psd_viol:<10.2e} {dual_psd_viol:<10} {primal_res:<12.2e} {dual_res if isinstance(dual_res, str) else f'{dual_res:.2e}':<10} {complement if isinstance(complement, str) else f'{complement:.2e}':<12}")
                
                # CVXPYLayers accuracy  
                if  cvxpylayers_acc is not None:
                    psd_viol = max(0, -cvxpylayers_acc['min_eigenvalue_Z']) if cvxpylayers_acc['min_eigenvalue_Z'] < 0 else 0
                    dual_psd_viol = "N/A"
                    if  S_cvxpy is not None:
                        S_eigenvals = torch.linalg.eigvals(S_cvxpy.detach()).real
                        dual_psd_viol = f"{max(0, -torch.min(S_eigenvals).item()):.2e}"
                    
                    primal_res = cvxpylayers_acc['primal_residual_max']
                    dual_res = cvxpylayers_acc['dual_residual'] if cvxpylayers_acc['dual_residual'] is not None else "N/A"
                    complement = cvxpylayers_acc['complementarity_violation'] if cvxpylayers_acc['complementarity_violation'] is not None else "N/A"
                    
                    print(f"{'CVXPYLayers':<12} {psd_viol:<10.2e} {dual_psd_viol:<10} {primal_res:<12.2e} {dual_res if isinstance(dual_res, str) else f'{dual_res:.2e}':<10} {complement if isinstance(complement, str) else f'{complement:.2e}':<12}")
            
            
            
            else:
                print("No methods solved successfully")
        
        
        # print_summary()
    
        # Aggregate per-test metrics for final summary
        if Z_dsdp is not None:
            aggregate_stats['dOPT']['forward'].append(dsdp_fwd_time)
            aggregate_stats['dOPT']['backward'].append(dsdp_bwd_time)
            aggregate_stats['dOPT']['total'].append(dsdp_fwd_time + dsdp_bwd_time)
            if dsdp_acc is not None:
                aggregate_stats['dOPT']['primal_residual_max'].append(dsdp_acc['primal_residual_max'])
                if dsdp_acc['dual_residual'] is not None:
                    aggregate_stats['dOPT']['dual_residual'].append(dsdp_acc['dual_residual'])
                if dsdp_acc['dual_psd_violation'] is not None:
                    aggregate_stats['dOPT']['dual_psd_violation'].append(dsdp_acc['dual_psd_violation'])
                if dsdp_acc['duality_gap'] is not None:
                    aggregate_stats['dOPT']['duality_gap_abs'].append(dsdp_acc['duality_gap'])
                if dsdp_acc['complementarity_violation'] is not None:
                    aggregate_stats['dOPT']['complementarity_violation'].append(dsdp_acc['complementarity_violation'])
                aggregate_stats['dOPT']['psd_violation'].append(max(0, -dsdp_acc['min_eigenvalue_Z']))
    
        if Z_cvxpy is not None:
            aggregate_stats['CVXPYLayers']['forward'].append(dcvxpylayer_fwd_time)
            aggregate_stats['CVXPYLayers']['backward'].append(dcvxpylayer_bwd_time)
            aggregate_stats['CVXPYLayers']['total'].append(dcvxpylayer_fwd_time + dcvxpylayer_bwd_time)
            if cvxpylayers_acc is not None:
                aggregate_stats['CVXPYLayers']['primal_residual_max'].append(cvxpylayers_acc['primal_residual_max'])
                if cvxpylayers_acc['dual_residual'] is not None:
                    aggregate_stats['CVXPYLayers']['dual_residual'].append(cvxpylayers_acc['dual_residual'])
                if cvxpylayers_acc['dual_psd_violation'] is not None:
                    aggregate_stats['CVXPYLayers']['dual_psd_violation'].append(cvxpylayers_acc['dual_psd_violation'])
                if cvxpylayers_acc['duality_gap'] is not None:
                    aggregate_stats['CVXPYLayers']['duality_gap_abs'].append(cvxpylayers_acc['duality_gap'])
                if cvxpylayers_acc['complementarity_violation'] is not None:
                    aggregate_stats['CVXPYLayers']['complementarity_violation'].append(cvxpylayers_acc['complementarity_violation'])
                aggregate_stats['CVXPYLayers']['psd_violation'].append(max(0, -cvxpylayers_acc['min_eigenvalue_Z']))

        if Z_ffo is not None:
            aggregate_stats['FFOLayer']['forward'].append(ffo_fwd_time)
            aggregate_stats['FFOLayer']['backward'].append(ffo_bwd_time)
            aggregate_stats['FFOLayer']['total'].append(ffo_fwd_time + ffo_bwd_time)
            if ffo_grad_diff is not None:
                aggregate_stats['FFOLayer']['gradient_diff'].append(ffo_grad_diff)
                for name in ("C", "A", "b"):
                    value = ffo_errors.get(name)
                    if value is not None:
                        aggregate_stats['FFOLayer'][f'{name}_gradient_diff'].append(value)
            if ffo_acc is not None:
                aggregate_stats['FFOLayer']['primal_residual_max'].append(ffo_acc['primal_residual_max'])
                if ffo_acc['dual_residual'] is not None:
                    aggregate_stats['FFOLayer']['dual_residual'].append(ffo_acc['dual_residual'])
                if ffo_acc['dual_psd_violation'] is not None:
                    aggregate_stats['FFOLayer']['dual_psd_violation'].append(ffo_acc['dual_psd_violation'])
                if ffo_acc['duality_gap'] is not None:
                    aggregate_stats['FFOLayer']['duality_gap_abs'].append(ffo_acc['duality_gap'])
                if ffo_acc['complementarity_violation'] is not None:
                    aggregate_stats['FFOLayer']['complementarity_violation'].append(ffo_acc['complementarity_violation'])
                aggregate_stats['FFOLayer']['psd_violation'].append(max(0, -ffo_acc['min_eigenvalue_Z']))

        if sol_diff is not None and grad_diff is not None:
            aggregate_stats['comparison']['sol_diff'].append(sol_diff)
            aggregate_stats['comparison']['grad_diff'].append(grad_diff)

        solver_names = {
            "dOPT": "MOSEK",
            "cvxpylayers": "diffcp/SCS",
            "ffolayer": ffo_solver.upper(),
        }

        def make_raw_row(method_name, forward, backward, accuracy, **diffs):
            return {
                "seed": base_seed + i_test,
                "method": method_name,
                "dim": n,
                "m": m,
                "eps": eps,
                "forward_solver": solver_names[method_name],
                "forward_s": forward,
                "backward_s": backward,
                "total_s": forward + backward,
                "primal_psd_violation": (
                    max(0, -accuracy["min_eigenvalue_Z"]) if accuracy else ""
                ),
                "primal_residual_max": accuracy["primal_residual_max"] if accuracy else "",
                "dual_psd_violation": accuracy["dual_psd_violation"] if accuracy and accuracy["dual_psd_violation"] is not None else "",
                "dual_residual_max": accuracy["dual_residual"] if accuracy and accuracy["dual_residual"] is not None else "",
                "duality_gap_abs": accuracy["duality_gap"] if accuracy and accuracy["duality_gap"] is not None else "",
                "complementarity_abs": accuracy["complementarity_violation"] if accuracy and accuracy["complementarity_violation"] is not None else "",
                "solution_rel_diff_vs_cvxpylayers": diffs.get("solution", ""),
                "gradient_rel_diff_vs_cvxpylayers": diffs.get("gradient", ""),
                "C_gradient_rel_diff": diffs.get("C", ""),
                "A_gradient_rel_diff": diffs.get("A", ""),
                "b_gradient_rel_diff": diffs.get("b", ""),
            }

        seed_rows = []
        if Z_dsdp is not None:
            seed_rows.append(make_raw_row(
                "dOPT", dsdp_fwd_time, dsdp_bwd_time, dsdp_acc,
                solution=sol_diff if sol_diff is not None else "",
                gradient=grad_diff if grad_diff is not None else "",
            ))
        if Z_cvxpy is not None:
            seed_rows.append(make_raw_row(
                "cvxpylayers", dcvxpylayer_fwd_time,
                dcvxpylayer_bwd_time, cvxpylayers_acc,
            ))
        if Z_ffo is not None:
            ffo_solution_diff = ""
            if Z_cvxpy is not None:
                ffo_solution_diff = float(
                    torch.linalg.vector_norm(Z_ffo.detach().cpu() - Z_cvxpy.detach().cpu())
                    / max(torch.linalg.vector_norm(Z_cvxpy.detach().cpu()).item(), 1e-12)
                )
            seed_rows.append(make_raw_row(
                "ffolayer", ffo_fwd_time, ffo_bwd_time, ffo_acc,
                solution=ffo_solution_diff,
                gradient=ffo_grad_diff if ffo_grad_diff is not None else "",
                C=ffo_errors.get("C", ""), A=ffo_errors.get("A", ""),
                b=ffo_errors.get("b", ""),
            ))
        run_rows.extend(seed_rows)
        persisted_rows.extend(seed_rows)
        write_raw_results()
    
        # Compare gradients with analytic solution
        # compare_gradients_with_analytic(
        #     C_dsdp, A_dsdp, b_dsdp, 
        #     C_cvxpy, A_cvxpy, b_cvxpy,
        #     Z_dsdp, nu_dsdp, n, m  # Use dSDP solution as reference
        # )
    
    
    print("\n" + "#" * 80)
    print("FINAL AVERAGE SUMMARY ACROSS ALL TESTS")
    print("#" * 80)
    
    selected_methods = []
    if run_dsdp:
        selected_methods.append('dOPT')
    if run_cvxpylayers:
        selected_methods.append('CVXPYLayers')
    if run_ffolayer:
        selected_methods.append('FFOLayer')

    for method_name in selected_methods:
        forward_mean = _safe_mean(aggregate_stats[method_name]['forward'])
        backward_mean = _safe_mean(aggregate_stats[method_name]['backward'])
        total_mean = _safe_mean(aggregate_stats[method_name]['total'])
    
        primal_res_mean = _safe_mean(aggregate_stats[method_name]['primal_residual_max'])
        dual_res_mean = _safe_mean(aggregate_stats[method_name]['dual_residual'])
        dual_psd_viol_mean = _safe_mean(aggregate_stats[method_name]['dual_psd_violation'])
        duality_gap_mean = _safe_mean(aggregate_stats[method_name]['duality_gap_abs'])
        comp_mean = _safe_mean(aggregate_stats[method_name]['complementarity_violation'])
        psd_viol_mean = _safe_mean(aggregate_stats[method_name]['psd_violation'])
    
        solved_count = len(aggregate_stats[method_name]['total'])
        print(f"\n{method_name} (solved {solved_count}/{n_test}):")
    
        if total_mean is not None:
            print(f"  Avg runtime - Forward: {forward_mean:.4f}s, Backward: {backward_mean:.4f}s, Total: {total_mean:.4f}s")
        else:
            print("  Avg runtime - N/A")
    
        print(
            "  Avg accuracy - "
            f"PSD violation: {psd_viol_mean:.2e} "
            if psd_viol_mean is not None else
            "  Avg accuracy - PSD violation: N/A"
        )
        print(
            f"                 Primal residual: {primal_res_mean:.2e}, "
            f"Dual residual: {dual_res_mean:.2e}, "
            f"Dual PSD violation: {dual_psd_viol_mean:.2e}"
            if (primal_res_mean is not None and dual_res_mean is not None and dual_psd_viol_mean is not None)
            else "                 Primal/Dual feasibility: partial or N/A"
        )
        print(
            f"                 |Duality gap|: {duality_gap_mean:.2e}, "
            f"Complementarity: {comp_mean:.2e}"
            if (duality_gap_mean is not None and comp_mean is not None)
            else "                 Duality gap/Complementarity: partial or N/A"
        )
    
    sol_diff_mean = _safe_mean(aggregate_stats['comparison']['sol_diff'])
    grad_diff_mean = _safe_mean(aggregate_stats['comparison']['grad_diff'])
    
    if run_dsdp and run_cvxpylayers:
        print("\nCross-method consistency (dOPT vs CVXPYLayers):")
        if sol_diff_mean is not None and grad_diff_mean is not None:
            print(f"  Avg solution difference: {sol_diff_mean:.2e}")
            print(f"  Avg gradient difference: {grad_diff_mean:.2e}")
        else:
            print("  N/A (insufficient successful paired runs)")

    summary_headers = [
        "dim", "method", "m", "eps", "forward_solver", "seed_start",
        "n_prob", "n_solved", "forward_s", "backward_s", "total_s",
        "primal_psd_violation", "primal_residual_max",
        "dual_psd_violation", "dual_residual_max", "duality_gap_abs",
        "complementarity_abs", "solution_rel_diff_vs_cvxpylayers",
        "gradient_rel_diff_vs_cvxpylayers", "C_gradient_rel_diff",
        "A_gradient_rel_diff", "b_gradient_rel_diff",
    ]
    mean_fields = summary_headers[8:]
    summaries = []
    for method_name in ("dOPT", "cvxpylayers", "ffolayer"):
        method_rows = [row for row in run_rows if row["method"] == method_name]
        if not method_rows:
            continue
        summary = {key: method_rows[0].get(key, "") for key in summary_headers}
        summary.update({
            "seed_start": base_seed, "n_prob": n_test,
            "n_solved": len(method_rows),
        })
        for field in mean_fields:
            values = [float(row[field]) for row in method_rows if row[field] != ""]
            summary[field] = float(np.mean(values)) if values else ""
        summaries.append(summary)

    summary_path = output_dir / "random_sdp_results.csv"
    existing = []
    if summary_path.exists():
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == summary_headers:
                existing = list(reader)
    existing = [
        row for row in existing
        if not (int(row["dim"]) == n and row["method"] in selected_labels)
    ]
    existing.extend(summaries)
    existing.sort(key=lambda row: (int(row["dim"]), row["method"]))
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(existing)
    print(f"\nPer-instance CSV: {raw_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random SDP benchmarks.")
    parser.add_argument("--dim", type=int, default=10, help="PSD matrix size")
    parser.add_argument("--n-prob", type=int, default=10, help="number of random instances")
    parser.add_argument("--seed-start", type=int, default=888, help="first random seed")
    parser.add_argument("--eps", type=float, default=1e-8, help="forward solver tolerance")
    parser.add_argument(
        "--methods", nargs="+", required=True,
        choices=("dOPT", "cvxpylayers", "ffolayer"),
        help="one or more benchmark methods to run",
    )
    parser.add_argument(
        "--ffo-solver",
        choices=("mosek", "scs"),
        default="mosek",
        help="solver used by FFOLayer for both forward and perturbed backward solves",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "random_sdp",
        help="directory containing random_sdp_results.csv and all/dim_N.csv",
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.dim < 2:
        parser.error("--dim must be at least 2")
    if args.n_prob < 1:
        parser.error("--n-prob must be positive")
    if args.eps <= 0:
        parser.error("--eps must be positive")
    main(
        n=args.dim,
        n_test=args.n_prob,
        base_seed=args.seed_start,
        eps=args.eps,
        methods=args.methods,
        ffo_solver=args.ffo_solver,
        output_dir=args.output_dir,
        no_warmup=args.no_warmup,
    )
