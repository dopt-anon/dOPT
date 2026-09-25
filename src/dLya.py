"""Differentiable common-Lyapunov layer in its native cone formulation.

The mathematical problem is

    maximize    t
    over        P in S^n, t in R
    subject to  P - p_floor I >= 0, trace(P) = 1,
                G_i(P,t) = P - A_i.T P A_i - t I >= 0.

No explicit slack variables or slack-definition equalities are part of the
layer interface or its backward KKT system.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch

from .lyapunov_problem import LyapunovProblemData, lyapunov_operator_values


def _vech(matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = torch.triu_indices(
        matrix.shape[-1], matrix.shape[-1], device=matrix.device
    )
    return matrix[rows, cols]


def _symmetric_cotangent_vech(matrix: torch.Tensor) -> torch.Tensor:
    """g such that g @ vech(dP) equals <matrix, dP> for symmetric dP."""
    rows, cols = torch.triu_indices(
        matrix.shape[-1], matrix.shape[-1], device=matrix.device
    )
    result = matrix[rows, cols].clone()
    off_diagonal = rows != cols
    result[off_diagonal] += matrix[cols[off_diagonal], rows[off_diagonal]]
    return result


def _symmetric_basis(n: int, *, dtype, device) -> torch.Tensor:
    rows, cols = torch.triu_indices(n, n, device=device)
    basis = torch.zeros((len(rows), n, n), dtype=dtype, device=device)
    indices = torch.arange(len(rows), device=device)
    basis[indices, rows, cols] = 1.0
    basis[indices, cols, rows] = 1.0
    return basis


def _active_U(eigenvectors: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """PSD critical-cone directions in local vech coordinates."""
    n = eigenvectors.shape[0]
    rows, cols = torch.triu_indices(n, n, device=eigenvectors.device)
    vectors = eigenvectors[:, active]
    if vectors.shape[1] == 0:
        return torch.empty(
            (len(rows), 0), dtype=eigenvectors.dtype,
            device=eigenvectors.device,
        )
    left, right = torch.triu_indices(
        vectors.shape[1], vectors.shape[1], device=eigenvectors.device
    )
    u, v = vectors[:, left], vectors[:, right]
    values = u[rows] * v[cols]
    off_diagonal = rows != cols
    values[off_diagonal] += (
        u[cols[off_diagonal]] * v[rows[off_diagonal]]
    )
    return values


def _spectral_hessian(
    primal: torch.Tensor, dual: torch.Tensor, eps_active: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the theorem's PSD Hessian block and active-U block."""
    eigenvalues, eigenvectors = torch.linalg.eigh(primal)
    active = torch.nonzero(eigenvalues < eps_active, as_tuple=True)[0]
    inactive = torch.nonzero(eigenvalues >= eps_active, as_tuple=True)[0]
    U = _active_U(eigenvectors, active)
    q = primal.shape[0] * (primal.shape[0] + 1) // 2
    if len(active) == 0:
        return torch.zeros((q, q), dtype=primal.dtype, device=primal.device), U
    if len(inactive) == 0:
        primal_pinv = torch.zeros_like(primal)
    else:
        vectors = eigenvectors[:, inactive]
        primal_pinv = (
            vectors @ torch.diag(1.0 / eigenvalues[inactive]) @ vectors.T
        )

    n = primal.shape[0]
    rows, cols = torch.triu_indices(n, n, device=primal.device)
    off_diagonal = rows != cols
    ri, rj = rows[:, None], cols[:, None]
    ck, cl = rows[None, :], cols[None, :]
    row_off, col_off = off_diagonal[:, None], off_diagonal[None, :]
    H = (
        primal_pinv[ri, ck] * dual[rj, cl]
        + dual[ri, ck] * primal_pinv[rj, cl]
    )
    H += row_off * (
        primal_pinv[rj, ck] * dual[ri, cl]
        + dual[rj, ck] * primal_pinv[ri, cl]
    )
    H += col_off * (
        primal_pinv[ri, cl] * dual[rj, ck]
        + dual[ri, cl] * primal_pinv[rj, ck]
    )
    H += row_off * col_off * (
        primal_pinv[rj, cl] * dual[ri, ck]
        + dual[rj, cl] * primal_pinv[ri, ck]
    )
    return H, U


def _solve_adjoint(
    modes: torch.Tensor,
    P_star: torch.Tensor,
    cone_primals: list[torch.Tensor],
    cone_duals: list[torch.Tensor],
    grad_P: torch.Tensor,
    grad_t: torch.Tensor,
    eps_active: float,
):
    """Solve the direct KKT and return dP, dt, and mode gradients."""
    n_state = P_star.shape[0]
    q = n_state * (n_state + 1) // 2
    reduced_dim = q + 1
    rows, cols = torch.triu_indices(n_state, n_state, device=P_star.device)
    identity_vech = (rows == cols).to(P_star.dtype)
    basis = _symmetric_basis(
        n_state, dtype=P_star.dtype, device=P_star.device
    )

    cone_jacobians = []
    first = torch.zeros((q, reduced_dim), dtype=P_star.dtype, device=P_star.device)
    first[:, :q] = torch.eye(q, dtype=P_star.dtype, device=P_star.device)
    cone_jacobians.append(first)
    for mode in modes:
        jacobian = torch.zeros_like(first)
        mapped_basis = mode.T @ basis @ mode
        jacobian[:, :q] = torch.eye(q, dtype=P_star.dtype, device=P_star.device)
        jacobian[:, :q] -= mapped_basis[:, rows, cols].T
        jacobian[:, -1] = -identity_vech
        cone_jacobians.append(jacobian)

    reduced_H = torch.zeros(
        (reduced_dim, reduced_dim), dtype=P_star.dtype, device=P_star.device
    )
    reduced_U_parts = []
    local_data = []
    for primal, dual, jacobian in zip(cone_primals, cone_duals, cone_jacobians):
        H, U = _spectral_hessian(primal, dual, eps_active)
        reduced_H += jacobian.T @ H @ jacobian
        reduced_U_parts.append(jacobian.T @ U)
        local_data.append((H, U, jacobian))
    reduced_U = torch.cat(reduced_U_parts, dim=1)
    trace_column = torch.zeros(
        (reduced_dim, 1), dtype=P_star.dtype, device=P_star.device
    )
    trace_column[:q, 0] = identity_vech
    coupling = torch.cat((reduced_U, trace_column), dim=1)
    dual_dim = coupling.shape[1]
    KKT = torch.cat((
        torch.cat((reduced_H, -coupling), dim=1),
        torch.cat((
            -coupling.T,
            torch.zeros((dual_dim, dual_dim), dtype=P_star.dtype, device=P_star.device),
        ), dim=1),
    ), dim=0)
    output_gradient = torch.cat((
        _symmetric_cotangent_vech(grad_P), grad_t.reshape(1)
    ))
    rhs = torch.cat((
        output_gradient,
        torch.zeros(dual_dim, dtype=P_star.dtype, device=P_star.device),
    ))
    try:
        solution = torch.linalg.solve(KKT, rhs)
    except torch.linalg.LinAlgError:
        solution = torch.linalg.lstsq(KKT, rhs).solution

    adjoint_direction = solution[:reduced_dim]
    dP_vech, dt = -adjoint_direction[:q], -adjoint_direction[-1]
    dP = torch.zeros_like(P_star)
    dP[rows, cols] = dP_vech
    dP[cols, rows] = dP_vech

    active_multiplier = solution[reduced_dim:reduced_dim + reduced_U.shape[1]]
    dcone_duals = []
    active_offset = 0
    for H, U, jacobian in local_data:
        width = U.shape[1]
        local_direction = jacobian @ adjoint_direction
        residual = H @ local_direction
        if width:
            residual -= U @ active_multiplier[active_offset:active_offset + width]
        dcone_duals.append(-residual)
        active_offset += width

    with torch.enable_grad():
        differentiable_modes = modes.detach().requires_grad_(True)
        contraction = torch.zeros((), dtype=P_star.dtype, device=P_star.device)
        for mode, dual, ddual in zip(
            differentiable_modes, cone_duals[1:], dcone_duals[1:]
        ):
            contraction += torch.dot(_vech(dual), _vech(mode.T @ dP @ mode))
            contraction -= torch.dot(ddual, _vech(mode.T @ P_star @ mode))
        mode_grad = torch.autograd.grad(contraction, differentiable_modes)[0]
    return mode_grad, dP, dt


class dLyaFunction(torch.autograd.Function):
    """Autograd function for the native common-Lyapunov problem."""

    @staticmethod
    def forward(ctx, modes, problem, settings):
        operator_values = lyapunov_operator_values(modes.detach().cpu().numpy())
        for parameter, value in zip(problem.parameters, operator_values):
            parameter.value = value
        problem.problem.solve(
            solver=settings.get("solver", cp.SCS),
            **settings.get("solver_args", {}),
        )
        if problem.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Lyapunov SDP failed: {problem.problem.status}")
        P, t = (variable.value for variable in problem.variables)
        cone_primals = [
            constraint.args[0].value for constraint in problem.psd_constraints
        ]
        cone_duals = [constraint.dual_value for constraint in problem.psd_constraints]
        device, dtype = modes.device, modes.dtype
        P = torch.as_tensor(P, dtype=dtype, device=device)
        t = torch.as_tensor(t, dtype=dtype, device=device)
        primal_tensors = [torch.as_tensor(x, dtype=dtype, device=device) for x in cone_primals]
        dual_tensors = [torch.as_tensor(x, dtype=dtype, device=device) for x in cone_duals]
        ctx.save_for_backward(modes, P, *primal_tensors, *dual_tensors)
        ctx.n_cones = len(primal_tensors)
        ctx.settings = settings
        return P, t

    @staticmethod
    def backward(ctx, grad_P, grad_t):
        saved = ctx.saved_tensors
        modes, P_star = saved[:2]
        start = 2
        cone_primals = list(saved[start:start + ctx.n_cones])
        start += ctx.n_cones
        cone_duals = list(saved[start:start + ctx.n_cones])
        mode_grad, _, _ = _solve_adjoint(
            modes, P_star, cone_primals, cone_duals,
            grad_P, grad_t, ctx.settings.get("eps_active", 1e-5),
        )
        return mode_grad, None, None


class dLyaLayer(torch.nn.Module):
    """Native differentiable layer for a common Lyapunov certificate."""

    def __init__(
        self,
        n_state: int,
        n_modes: int,
        *,
        problem: LyapunovProblemData,
        settings: dict | None = None,
    ):
        super().__init__()
        if n_state < 1 or n_modes < 1:
            raise ValueError("n_state and n_modes must be positive")
        self.n_state, self.n_modes = n_state, n_modes
        self.settings = {
            "solver": cp.SCS,
            "solver_args": {"eps": 1e-7},
            "eps_active": 1e-5,
        }
        if settings:
            self.settings.update(settings)
        if len(problem.parameters) != n_modes:
            raise ValueError("problem parameter count does not match n_modes")
        self.problem = problem

    def forward(self, modes: torch.Tensor | dict[int, torch.Tensor]):
        if isinstance(modes, dict):
            modes = torch.stack([modes[key] for key in sorted(modes)])
        if modes.shape != (self.n_modes, self.n_state, self.n_state):
            raise ValueError(
                "modes must have shape "
                f"{(self.n_modes, self.n_state, self.n_state)}"
            )
        P, t = dLyaFunction.apply(modes, self.problem, self.settings)
        return {"P": P, "t": t}
