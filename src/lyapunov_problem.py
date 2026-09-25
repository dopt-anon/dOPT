"""Shared CVXPY formulation for the common-Lyapunov SDP."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class LyapunovProblemData:
    """Objects needed by both CVXPYLayer and the native dLya layer."""

    problem: cp.Problem
    parameters: tuple[cp.Parameter, ...]
    variables: tuple[cp.Variable, cp.Variable]
    psd_constraints: tuple[cp.Constraint, ...]
    trace_constraint: cp.Constraint


def build_common_lyapunov_problem(
    n_state: int,
    n_modes: int,
    p_floor: float = 0.0,
) -> LyapunovProblemData:
    """Build the single non-lifted ``(P, t)`` formulation used by all backends."""
    if n_state < 1 or n_modes < 1:
        raise ValueError("n_state and n_modes must be positive")
    if not 0 <= p_floor < 1.0 / n_state:
        raise ValueError("p_floor must satisfy 0 <= p_floor < 1/n_state")

    rows, cols = np.triu_indices(n_state)
    q = len(rows)
    selection = np.zeros((q, n_state * n_state))
    duplication = np.zeros((n_state * n_state, q))
    for index, (row, col) in enumerate(zip(rows, cols)):
        selection[index, row + col * n_state] = 1.0
        duplication[row + col * n_state, index] = 1.0
        duplication[col + row * n_state, index] = 1.0

    P = cp.Variable((n_state, n_state), symmetric=True, name="P")
    t = cp.Variable(name="t")
    p_vech = selection @ cp.vec(P, order="F")
    identity_vech = np.asarray(rows == cols, dtype=float)
    parameters = tuple(
        cp.Parameter((q, q), name=f"K_{index}") for index in range(n_modes)
    )

    psd_constraints = [P - p_floor * np.eye(n_state) >> 0]
    for operator in parameters:
        g_vech = operator @ p_vech - t * identity_vech
        g_matrix = cp.reshape(
            duplication @ g_vech, (n_state, n_state), order="F"
        )
        psd_constraints.append(g_matrix >> 0)
    # Written with the shared vech representation because cvxtorch (used by
    # FFOLayer) does not implement CVXPY's Trace atom.
    trace_constraint = identity_vech @ p_vech == 1.0
    problem = cp.Problem(
        cp.Minimize(-t), [*psd_constraints, trace_constraint]
    )
    if not problem.is_dpp():
        raise RuntimeError("common Lyapunov formulation must be DPP")
    return LyapunovProblemData(
        problem=problem,
        parameters=parameters,
        variables=(P, t),
        psd_constraints=tuple(psd_constraints),
        trace_constraint=trace_constraint,
    )


def lyapunov_operator_values(modes: np.ndarray) -> list[np.ndarray]:
    """Return K(A_i) such that K(A_i) vech(P)=vech(P-A_i.T P A_i)."""
    n_state = modes.shape[-1]
    rows, cols = np.triu_indices(n_state)
    q = len(rows)
    basis = np.zeros((q, n_state, n_state), dtype=modes.dtype)
    basis[np.arange(q), rows, cols] = 1.0
    basis[np.arange(q), cols, rows] = 1.0
    identity = np.eye(q, dtype=modes.dtype)
    return [
        identity - (mode.T @ basis @ mode)[:, rows, cols].T
        for mode in modes
    ]
