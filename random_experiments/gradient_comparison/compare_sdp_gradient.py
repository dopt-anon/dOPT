#!/usr/bin/env python3
"""Extend the verified SDP gradient comparison with FFOLayer."""

import argparse
import csv
import sys
from contextlib import contextmanager
from pathlib import Path

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.dSDP as dsdp_module
from random_experiments.random_sdp import (
    create_sdp_instance_matrix,
    dim2num_var,
    ffolayer_solve,
)
from src.dSDP import dSDPLayer, precompute_M_transform, precompute_T_inv


VARIABLES = ("C", "A", "b")


def flatten(groups):
    values = []

    def append(group):
        if isinstance(group, (tuple, list)):
            for item in group:
                append(item)
        elif group is not None:
            values.append(group.detach().reshape(-1))

    append(groups)
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float64)


def build_cvxpylayer(n, m):
    Z = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n))
    A = cp.Parameter((m, n * n))
    b = cp.Parameter(m)
    constraints = [Z >> 0, A @ cp.vec(Z, order="F") == b]
    problem = cp.Problem(cp.Minimize(cp.trace(C @ Z)), constraints)
    return CvxpyLayer(problem, parameters=[C, A, b], variables=[Z])


def companion_primal_dual(C, A, b, eps):
    """Compute the shared SCS primal-dual point used by all comparisons."""
    n = C.shape[0]
    Z = cp.Variable((n, n), symmetric=True)
    constraints = [
        Z >> 0,
        A.detach().cpu().numpy() @ cp.vec(Z, order="F")
        == b.detach().cpu().numpy(),
    ]
    problem = cp.Problem(
        cp.Minimize(cp.trace(C.detach().cpu().numpy() @ Z)), constraints
    )
    problem.solve(solver=cp.SCS, eps=eps, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"companion CVXPY solve failed: {problem.status}")
    return (
        torch.tensor(Z.value, dtype=torch.float64),
        torch.tensor(constraints[0].dual_value, dtype=torch.float64),
        torch.tensor(constraints[1].dual_value, dtype=torch.float64),
    )


@contextmanager
def shared_dsdp_forward(Z, S, nu):
    """Make dOPT consume the same forward solution as the other methods."""
    original = dsdp_module.solve_sdp_cvxpy

    def solve(*unused_args, **unused_kwargs):
        return (
            Z.detach().cpu().numpy(),
            S.detach().cpu().numpy(),
            nu.detach(),
            "optimal",
        )

    dsdp_module.solve_sdp_cvxpy = solve
    try:
        yield
    finally:
        dsdp_module.solve_sdp_cvxpy = original


def dsdp_value_gradients(layer, C, A, b, Z, S, nu):
    leaves = [value.detach().clone().requires_grad_(True) for value in (C, A, b)]
    with shared_dsdp_forward(Z, S, nu):
        Z_dopt, _, _ = layer(*leaves)
    torch.trace(leaves[0] @ Z_dopt).backward()
    return Z_dopt.detach(), tuple(value.grad for value in leaves)


def envelope_gradients(Z, nu):
    return Z, torch.outer(nu, Z.flatten()), -nu


def error_metrics(candidate, reference):
    candidate = flatten(candidate).cpu()
    reference = flatten(reference).cpu()
    absolute = torch.linalg.vector_norm(candidate - reference).item()
    reference_norm = torch.linalg.vector_norm(reference).item()
    relative = absolute / max(reference_norm, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        candidate, reference, dim=0, eps=1e-12
    ).item()
    return absolute, relative, reference_norm, cosine


def comparison_rows(seed, dim, candidate_name, candidate, reference_name,
                    reference, solution_diff, companion_diff):
    rows = []
    for variable, candidate_group, reference_group in zip(
        VARIABLES, candidate, reference
    ):
        absolute, relative, reference_norm, cosine = error_metrics(
            candidate_group, reference_group
        )
        rows.append({
            "seed": seed,
            "dim": dim,
            "candidate": candidate_name,
            "reference": reference_name,
            "variable": variable,
            "absolute_error": absolute,
            "relative_error": relative,
            "reference_norm": reference_norm,
            "cosine_similarity": cosine,
            "solution_abs_diff": solution_diff,
            "companion_solution_abs_diff": companion_diff,
        })
    absolute, relative, reference_norm, cosine = error_metrics(candidate, reference)
    rows.append({
        "seed": seed,
        "dim": dim,
        "candidate": candidate_name,
        "reference": reference_name,
        "variable": "all",
        "absolute_error": absolute,
        "relative_error": relative,
        "reference_norm": reference_norm,
        "cosine_similarity": cosine,
        "solution_abs_diff": solution_diff,
        "companion_solution_abs_diff": companion_diff,
    })
    return rows


def cvxpylayer_gradients(layer, C, A, b, eps, objective):
    """Keep the verified CVXPYLayers call unchanged."""
    leaves = [value.detach().clone().requires_grad_(True) for value in (C, A, b)]
    Z, = layer(*leaves, solver_args={"eps": eps})
    loss = Z.sum() if objective == "solution_sum" else torch.trace(leaves[0] @ Z)
    loss.backward()
    return Z.detach(), tuple(value.grad for value in leaves)


def dsdp_solution_gradients(layer, C, A, b, Z, S, nu):
    leaves = [value.detach().clone().requires_grad_(True) for value in (C, A, b)]
    with shared_dsdp_forward(Z, S, nu):
        Z_dsdp, _, _ = layer(*leaves)
    Z_dsdp.sum().backward()
    return Z_dsdp.detach(), tuple(value.grad for value in leaves)


def ffolayer_gradients(layer, C, A, b, eps, objective):
    Z, _, _, C_leaf, A_leaf, b_leaf, _, layer = ffolayer_solve(
        C, A, b, None, layer, eps=eps, solver="scs"
    )
    loss = Z.sum() if objective == "solution_sum" else torch.trace(C_leaf @ Z)
    loss.backward()
    return Z.detach(), (C_leaf.grad, A_leaf.grad, b_leaf.grad), layer


def solution_relative_error(candidate, reference):
    absolute = torch.linalg.vector_norm(candidate.cpu() - reference.cpu()).item()
    reference_norm = torch.linalg.vector_norm(reference.cpu()).item()
    return absolute / max(reference_norm, 1e-12)


def append_rows(rows, seed, dim, objective, candidate_name, candidate,
                reference_name, reference, solution_error, companion_error):
    additions = comparison_rows(
        seed, dim, candidate_name, candidate, reference_name, reference,
        solution_error, companion_error,
    )
    for row in additions:
        row["objective"] = objective
    rows.extend(additions)


def write_table(rows, output):
    groups = {}
    for row in rows:
        if row["variable"] != "all":
            continue
        key = (
            row["objective"], row["candidate"], row["reference"], row["dim"]
        )
        groups.setdefault(key, []).append(row)

    comparisons = (
        ("Solution gradient: dOPT vs. CVXPYLayers", "solution_sum", "dOPT", "cvxpylayers"),
        ("Value gradient: dOPT vs. GT", "value", "dOPT", "envelope"),
        ("Value gradient: CVXPYLayers vs. GT", "value", "cvxpylayers", "envelope"),
    )
    dims = sorted({row["dim"] for row in rows})
    table_rows = []
    for label, objective, candidate, reference in comparisons:
        table_row = {"gradient_comparison": label}
        for dim in dims:
            group = groups[(objective, candidate, reference, dim)]
            value = sum(row["relative_error"] for row in group) / len(group)
            table_row[f"n={dim}"] = f"{value:.2e}"
        table_rows.append(table_row)

    table_path = output.with_name(f"{output.stem}_table.csv")
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"wrote {table_path.resolve()}")


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.dims:
        m = int(dim2num_var(n) - 1) // 2
        cvxpylayer = build_cvxpylayer(n, m)
        ffo_layer = None
        dsdp_layer = dSDPLayer(
            n, m,
            transforms=(
                precompute_M_transform(n, device="cpu", dtype=torch.float64),
                precompute_T_inv(n, device="cpu", dtype=torch.float64),
            ),
            has_quadratic_term=False,
        )

        for seed in range(args.seed_start, args.seed_start + args.n_prob):
            C, A, b, P = create_sdp_instance_matrix(
                n, m, random_seed=seed, use_quadratic=False
            )
            if P is not None:
                raise RuntimeError("comparison requires the linear random SDP")

            Z_shared, S_shared, nu_shared = companion_primal_dual(
                C, A, b, args.eps
            )
            for objective in ("solution_sum", "value"):
                Z_cvx, grad_cvx = cvxpylayer_gradients(
                    cvxpylayer, C, A, b, args.eps, objective
                )
                Z_ffo, grad_ffo, ffo_layer = ffolayer_gradients(
                    ffo_layer, C, A, b, args.eps, objective
                )
                if objective == "solution_sum":
                    Z_dsdp, grad_dsdp = dsdp_solution_gradients(
                        dsdp_layer, C, A, b, Z_shared, S_shared, nu_shared
                    )
                else:
                    Z_dsdp, grad_dsdp = dsdp_value_gradients(
                        dsdp_layer, C, A, b, Z_shared, S_shared, nu_shared
                    )

                solutions = {
                    "dOPT": Z_dsdp,
                    "ffolayer": Z_ffo,
                    "cvxpylayers": Z_cvx,
                }
                gradients = {
                    "dOPT": grad_dsdp,
                    "ffolayer": grad_ffo,
                    "cvxpylayers": grad_cvx,
                }
                companion_error = solution_relative_error(Z_cvx, Z_shared)
                for candidate, reference in (
                    ("dOPT", "ffolayer"),
                    ("dOPT", "cvxpylayers"),
                    ("ffolayer", "cvxpylayers"),
                ):
                    append_rows(
                        rows, seed, n, objective,
                        candidate, gradients[candidate],
                        reference, gradients[reference],
                        solution_relative_error(
                            solutions[candidate], solutions[reference]
                        ),
                        companion_error,
                    )

                if objective == "value":
                    envelope = envelope_gradients(Z_shared, nu_shared)
                    for method in ("dOPT", "ffolayer", "cvxpylayers"):
                        append_rows(
                            rows, seed, n, objective,
                            method, gradients[method], "envelope", envelope,
                            solution_relative_error(solutions[method], Z_shared),
                            companion_error,
                        )

            instance_rows = [
                row for row in rows
                if row["seed"] == seed and row["dim"] == n
                and row["variable"] == "all"
            ]
            print(
                f"dim={n} seed={seed} "
                + " ".join(
                    f"{row['objective']}:{row['candidate']}_vs_"
                    f"{row['reference']}={row['relative_error']:.3e}"
                    for row in instance_rows
                ),
                flush=True,
            )

    fieldnames = [
        "seed", "dim", "objective", "candidate", "reference", "variable",
        "absolute_error", "relative_error", "reference_norm",
        "cosine_similarity", "solution_abs_diff",
        "companion_solution_abs_diff",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output.resolve()}")
    write_table(rows, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[20])
    parser.add_argument("--n-prob", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=121)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
