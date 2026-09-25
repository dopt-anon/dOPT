"""Lift a common-Lyapunov margin problem into dSDP standard form.

The differentiable layer in ``dSDP.py`` accepts

    minimize    <C, Z>
    subject to  A_matrix @ vec(Z) = b,
                Z >= 0.

This module converts

    maximize    t
    subject to  P >= 0, trace(P) = 1,
                P - A_i.T @ P @ A_i >= t I

to that form while preserving PyTorch's gradient path from every mode matrix
``A_i`` to the equality coefficients.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping

import cvxpy as cp
import numpy as np
import torch


@dataclass(frozen=True)
class LyapunovBlockLayout:
    """Locations of semantic blocks in the lifted PSD matrix."""

    n_state: int
    mode_keys: List[int]
    p_slice: slice
    slack_slices: Dict[int, slice]
    t_slice: slice
    matrix_size: int
    t_bound: float
    p_floor: float


def _entry_basis(
    matrix_size: int,
    row: int,
    col: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return symmetric H satisfying <H, X> = X[row, col]."""
    basis = torch.zeros((matrix_size, matrix_size), dtype=dtype, device=device)
    if row == col:
        basis[row, col] = 1.0
    else:
        basis[row, col] = 0.5
        basis[col, row] = 0.5
    return basis


def _make_layout(
    n_state: int,
    mode_keys: List[int],
    t_bound: float,
    p_floor: float,
) -> LyapunovBlockLayout:
    cursor = 0
    p_slice = slice(cursor, cursor + n_state)
    cursor += n_state

    slack_slices = {}
    for mode_key in mode_keys:
        slack_slices[mode_key] = slice(cursor, cursor + n_state)
        cursor += n_state

    t_slice = slice(cursor, cursor + 2)
    cursor += 2
    return LyapunovBlockLayout(
        n_state=n_state,
        mode_keys=mode_keys,
        p_slice=p_slice,
        slack_slices=slack_slices,
        t_slice=t_slice,
        matrix_size=cursor,
        t_bound=t_bound,
        p_floor=p_floor,
    )


def build_common_lyapunov_block_sdp(
    modes: Mapping[int, torch.Tensor],
    *,
    t_bound: float = 10.0,
    p_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LyapunovBlockLayout]:
    """Build ``(C, A_matrix, b, layout)`` for ``dSDPLayer``.

    The lifted variable is

        Z = blockdiag(P, S_1, ..., S_m, T),
        T = [[t_bound, t], [t, t_bound]].

    The PSD constraint on ``T`` represents a free scalar ``t`` inside
    ``[-t_bound, t_bound]`` without the non-uniqueness caused by splitting it
    into positive and negative parts. Choose ``t_bound`` large enough that the
    optimum is not clipped, but not excessively large because scaling matters
    for first-order SDP solvers.
    """
    if not modes:
        raise ValueError("modes must be nonempty")
    if t_bound <= 0:
        raise ValueError("t_bound must be positive")

    mode_keys = list(modes)
    first = modes[mode_keys[0]]
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("each mode must be square")
    n_state = first.shape[0]
    if any(a_i.shape != first.shape for a_i in modes.values()):
        raise ValueError("all modes must have the same shape")
    if any(
        a_i.dtype != first.dtype or a_i.device != first.device
        for a_i in modes.values()
    ):
        raise ValueError("all modes must have the same dtype and device")

    dtype, device = first.dtype, first.device
    if not 0 <= p_floor < 1.0 / n_state:
        raise ValueError("p_floor must satisfy 0 <= p_floor < 1 / n_state")
    layout = _make_layout(n_state, mode_keys, t_bound, p_floor)
    size = layout.matrix_size
    constraint_matrices = []
    rhs_values = []

    # The PSD block represents Q = P - p_floor I. Thus Q >= 0 enforces
    # P >= p_floor I, and trace(P) = 1 becomes trace(Q) = 1 - n*p_floor.
    trace_constraint = torch.zeros((size, size), dtype=dtype, device=device)
    for state_idx in range(n_state):
        global_idx = layout.p_slice.start + state_idx
        trace_constraint[global_idx, global_idx] = 1.0
    constraint_matrices.append(trace_constraint)
    rhs_values.append(torch.as_tensor(
        1.0 - n_state * p_floor, dtype=dtype, device=device
    ))

    # Fix diag(T), leaving T[0, 1] as the margin t.
    for local_idx in range(2):
        global_idx = layout.t_slice.start + local_idx
        constraint_matrices.append(
            _entry_basis(size, global_idx, global_idx, dtype=dtype, device=device)
        )
        rhs_values.append(torch.as_tensor(t_bound, dtype=dtype, device=device))

    # S_i - P + A_i^T P A_i + t I = 0, entry by entry.
    for mode_key, a_i in modes.items():
        slack_slice = layout.slack_slices[mode_key]
        for row in range(n_state):
            for col in range(row, n_state):
                constraint = torch.zeros(
                    (size, size), dtype=dtype, device=device
                )

                p_entry = _entry_basis(
                    n_state, row, col, dtype=dtype, device=device
                )
                a_col_outer = torch.outer(a_i[:, row], a_i[:, col])
                ata_coefficient = 0.5 * (a_col_outer + a_col_outer.T)
                constraint[layout.p_slice, layout.p_slice] = (
                    -p_entry + ata_coefficient
                )

                slack_row = slack_slice.start + row
                slack_col = slack_slice.start + col
                constraint += _entry_basis(
                    size, slack_row, slack_col, dtype=dtype, device=device
                )

                if row == col:
                    t_row = layout.t_slice.start
                    t_col = layout.t_slice.start + 1
                    constraint += _entry_basis(
                        size, t_row, t_col, dtype=dtype, device=device
                    )

                constraint_matrices.append(constraint)
                # Substituting P = Q + p_floor I gives
                # S-Q+A'QA+tI = p_floor * (I-A'A).
                floor_rhs = -p_floor * torch.dot(a_i[:, row], a_i[:, col])
                if row == col:
                    floor_rhs = floor_rhs + p_floor
                rhs_values.append(floor_rhs)

    # <C,Z> = -t, hence minimizing the standard objective maximizes t.
    t_row = layout.t_slice.start
    t_col = layout.t_slice.start + 1
    objective = -_entry_basis(
        size, t_row, t_col, dtype=dtype, device=device
    )

    # dSDP uses column-major vec(Z). Every coefficient matrix above is
    # symmetric, so its row-major and column-major flattened vectors coincide.
    a_matrix = torch.stack(
        [matrix.flatten() for matrix in constraint_matrices]
    )
    b_vector = torch.stack(rhs_values)
    return objective, a_matrix, b_vector, layout


def extract_common_lyapunov_solution(
    z_star: torch.Tensor,
    layout: LyapunovBlockLayout,
) -> Dict[str, object]:
    """Extract ``P``, ``t`` and the mode slacks from a lifted solution."""
    q_block = z_star[layout.p_slice, layout.p_slice]
    p_block = q_block + layout.p_floor * torch.eye(
        layout.n_state, dtype=z_star.dtype, device=z_star.device
    )
    return {
        "P": p_block,
        "Q": q_block,
        "t": z_star[
            layout.t_slice.start,
            layout.t_slice.start + 1,
        ],
        "slacks": {
            key: z_star[block_slice, block_slice]
            for key, block_slice in layout.slack_slices.items()
        },
    }


def solve_block_sdp_cvxpy_reference(
    objective: torch.Tensor,
    a_matrix: torch.Tensor,
    b_vector: torch.Tensor,
    layout: LyapunovBlockLayout,
) -> Dict[str, object]:
    """Solve the product-cone problem directly with CVXPY for validation."""
    size = layout.matrix_size
    block_slices = [
        layout.p_slice,
        *(layout.slack_slices[key] for key in layout.mode_keys),
        layout.t_slice,
    ]
    block_columns = np.asarray([
        row + col * size
        for block in block_slices
        for col in range(block.start, block.stop)
        for row in range(block.start, block.stop)
    ], dtype=np.int64)
    block_variables = [
        cp.Variable(
            (block.stop - block.start, block.stop - block.start),
            symmetric=True,
        )
        for block in block_slices
    ]
    block_vector = cp.hstack([
        cp.vec(variable, order="F") for variable in block_variables
    ])
    objective_coefficients = (
        objective.detach().cpu().numpy().reshape(-1, order="F")[block_columns]
    )
    block_a_matrix = a_matrix.detach().cpu().numpy()[:, block_columns]
    constraints = [
        *(variable >> 0 for variable in block_variables),
        block_a_matrix @ block_vector == b_vector.detach().cpu().numpy(),
    ]
    problem = cp.Problem(
        cp.Minimize(objective_coefficients @ block_vector),
        constraints,
    )
    problem.solve(
        solver=cp.SCS,
        eps=1e-7,
        max_iters=100_000,
        verbose=False,
    )
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"lifted SDP failed with status {problem.status}")

    z_np = np.zeros((size, size))
    for block, variable in zip(block_slices, block_variables):
        z_np[block, block] = np.asarray(variable.value)
    z_star = torch.as_tensor(
        z_np,
        dtype=objective.dtype,
        device=objective.device,
    )
    extracted = extract_common_lyapunov_solution(z_star, layout)
    extracted.update(
        {
            "Z": z_star,
            "status": problem.status,
            "equality_residual": float(
                np.linalg.norm(
                    a_matrix.detach().cpu().numpy()
                    @ z_np.reshape(-1, order="F")
                    - b_vector.detach().cpu().numpy()
                )
            ),
            "Z_min_eigenvalue": float(np.linalg.eigvalsh(z_np).min()),
        }
    )
    return extracted
