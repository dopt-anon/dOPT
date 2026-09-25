#!/usr/bin/env python3
"""Compare SOCP optimal-value gradients against the envelope theorem."""

import argparse
import csv
import sys
from pathlib import Path

import diffcp
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dSOCP import dSOCPLayer
from src.utils import socp_utils


NAMES = ("q", "A", "b", "c", "d", "F", "g", "G", "h")


def _flatten_gradients(*groups):
    flat = []

    def append(group):
        if group is None:
            return
        if isinstance(group, (list, tuple)):
            for item in group:
                append(item)
            return
        flat.append(group.detach().reshape(-1))

    for group in groups:
        append(group)
    return torch.cat(flat) if flat else torch.empty(0, dtype=torch.float64)


def clone_parameters(raw):
    q, A, b, c, d, F, g, G, h = raw

    def leaf(value):
        return None if value is None else value.detach().clone().requires_grad_(True)

    return (
        leaf(q), [leaf(value) for value in A], [leaf(value) for value in b],
        [leaf(value) for value in c], [leaf(value) for value in d],
        leaf(F), leaf(g), leaf(G), leaf(h),
    )


def cone_data(q, A, b, c, d, F, g, G, h):
    blocks, rhs = [], []
    n_eq = 0 if F is None else F.shape[0]
    n_ineq = 0 if G is None else G.shape[0]
    if F is not None:
        blocks.append(sp.csc_matrix(F.detach().cpu().numpy()))
        rhs.append(g.detach().cpu().numpy())
    if G is not None:
        blocks.append(sp.csc_matrix(G.detach().cpu().numpy()))
        rhs.append(h.detach().cpu().numpy())

    soc_ranges = []
    row = n_eq + n_ineq
    for Ai, bi, ci, di in zip(A, b, c, d):
        matrix = np.vstack((ci.detach().cpu().numpy()[None, :],
                            Ai.detach().cpu().numpy()))
        vector = np.hstack((di.detach().cpu().numpy(),
                            bi.detach().cpu().numpy()))
        blocks.append(sp.csc_matrix(-matrix))
        rhs.append(vector)
        soc_ranges.append((row, row + matrix.shape[0]))
        row += matrix.shape[0]

    matrix = sp.vstack(blocks, format="csc")
    vector = np.concatenate(rhs)
    objective = q.detach().cpu().numpy()
    cones = {
        "z": n_eq,
        "l": n_ineq,
        "q": [Ai.shape[0] + 1 for Ai in A],
    }
    return matrix, vector, objective, cones, soc_ranges, n_eq, n_ineq


def split_adjoint(dmatrix, dvector, dobjective, A, n_eq, n_ineq):
    dense = dmatrix.toarray() if sp.issparse(dmatrix) else np.asarray(dmatrix)
    offset = 0
    grad_F = torch.tensor(dense[:n_eq], dtype=torch.float64) if n_eq else None
    grad_g = torch.tensor(dvector[:n_eq], dtype=torch.float64) if n_eq else None
    offset += n_eq
    grad_G = torch.tensor(dense[offset:offset + n_ineq], dtype=torch.float64) if n_ineq else None
    grad_h = torch.tensor(dvector[offset:offset + n_ineq], dtype=torch.float64) if n_ineq else None
    offset += n_ineq

    grad_A, grad_b, grad_c, grad_d = [], [], [], []
    for Ai in A:
        size = Ai.shape[0] + 1
        block_matrix = dense[offset:offset + size]
        block_vector = dvector[offset:offset + size]
        grad_c.append(torch.tensor(-block_matrix[0], dtype=torch.float64))
        grad_A.append(torch.tensor(-block_matrix[1:], dtype=torch.float64))
        grad_d.append(torch.tensor(block_vector[0], dtype=torch.float64))
        grad_b.append(torch.tensor(block_vector[1:], dtype=torch.float64))
        offset += size

    return (
        torch.tensor(dobjective, dtype=torch.float64),
        grad_A, grad_b, grad_c, grad_d,
        grad_F, grad_g, grad_G, grad_h,
    )


def solve_once_with_diffcp(raw, eps, max_iters):
    q, A, b, c, d, F, g, G, h = raw
    matrix, vector, objective, cones, soc_ranges, n_eq, n_ineq = cone_data(*raw)
    x, y, slack, derivative, adjoint = diffcp.solve_and_derivative(
        matrix, vector, objective, cones, eps=eps, max_iters=max_iters,
    )
    # For V(theta) = q(theta)^T x*(theta), the indirect upstream
    # derivative with respect to x* is q. The direct q derivative is x*.
    dmatrix, dvector, dobjective = adjoint(
        objective, np.zeros_like(y), np.zeros_like(slack)
    )
    gradients = list(split_adjoint(
        dmatrix, dvector, dobjective, A, n_eq, n_ineq
    ))
    gradients[0] = gradients[0] + torch.tensor(x, dtype=torch.float64)

    soc_duals = [np.asarray(y[start:stop]) for start, stop in soc_ranges]
    solution = {
        "x": np.asarray(x),
        "y": np.asarray(y),
        "slack": np.asarray(slack),
        "mu": np.asarray(y[:n_eq]) if n_eq else None,
        "lambda": np.asarray(y[n_eq:n_eq + n_ineq]) if n_ineq else None,
        "soc_duals": soc_duals,
    }
    return solution, tuple(gradients), adjoint


class SharedForwardSolver:
    """Return one precomputed diffcp primal/dual point to dSOCP."""

    def __init__(self, solution):
        self.solution = solution

    def solve(self, q, A, b, c, d, F=None, g=None, G=None, h=None,
              eps=None, return_soc_duals=False, **unused):
        sol = self.solution
        nu = np.asarray([dual[0] for dual in sol["soc_duals"]])
        result = (sol["x"].copy(), sol["mu"], nu, sol["lambda"])
        if return_soc_duals:
            return (*result, [dual.copy() for dual in sol["soc_duals"]])
        return result


def dsocp_value_gradients(raw, solution, eps):
    q, A, b, c, d, F, g, G, h = clone_parameters(raw)
    n_as = [Ai.shape[0] for Ai in A]
    layer = dSOCPLayer(
        socp_solver="cvxpy",
        mode="dense",
        eps=eps,
        forward_solver=SharedForwardSolver(solution),
    )
    x, _, _, _ = layer(
        q, n_as, torch.vstack(A), torch.hstack(b),
        torch.vstack(c), torch.hstack(d), F, g, G, h,
    )
    value = q @ x.squeeze()
    value.backward()
    gradients = (
        q.grad, [item.grad for item in A], [item.grad for item in b],
        [item.grad for item in c], [item.grad for item in d],
        F.grad if F is not None else None,
        g.grad if g is not None else None,
        G.grad if G is not None else None,
        h.grad if h is not None else None,
    )
    return x.detach().cpu(), gradients


def dsocp_solution_gradients(raw, solution, eps):
    q, A, b, c, d, F, g, G, h = clone_parameters(raw)
    n_as = [Ai.shape[0] for Ai in A]
    layer = dSOCPLayer(
        socp_solver="cvxpy",
        mode="dense",
        eps=eps,
        forward_solver=SharedForwardSolver(solution),
    )
    x, _, _, _ = layer(
        q, n_as, torch.vstack(A), torch.hstack(b),
        torch.vstack(c), torch.hstack(d), F, g, G, h,
    )
    x.sum().backward()
    gradients = (
        q.grad, [item.grad for item in A], [item.grad for item in b],
        [item.grad for item in c], [item.grad for item in d],
        F.grad if F is not None else None,
        g.grad if g is not None else None,
        G.grad if G is not None else None,
        h.grad if h is not None else None,
    )
    return x.detach().cpu(), gradients


def diffcp_solution_gradients(raw, solution, adjoint):
    _, A, _, _, _, F, _, G, _ = raw
    n_eq = 0 if F is None else F.shape[0]
    n_ineq = 0 if G is None else G.shape[0]
    dmatrix, dvector, dobjective = adjoint(
        np.ones_like(solution["x"]),
        np.zeros_like(solution["y"]),
        np.zeros_like(solution["slack"]),
    )
    return split_adjoint(
        dmatrix, dvector, dobjective, A, n_eq, n_ineq
    )


def envelope_gradients(raw, solution):
    q, A, b, c, d, F, g, G, h = raw
    x = torch.tensor(solution["x"], dtype=torch.float64)
    soc_duals = [torch.tensor(value, dtype=torch.float64)
                 for value in solution["soc_duals"]]

    # diffcp uses A_cone x + s = b. For an SOC block,
    # A_cone = -[c^T; A] and b_cone = [d; b].
    grad_A = [-dual[1:, None] @ x[None, :] for dual in soc_duals]
    grad_b = [-dual[1:] for dual in soc_duals]
    grad_c = [-dual[0] * x for dual in soc_duals]
    grad_d = [-dual[0] for dual in soc_duals]

    mu = None if solution["mu"] is None else torch.tensor(
        solution["mu"], dtype=torch.float64
    )
    lam = None if solution["lambda"] is None else torch.tensor(
        solution["lambda"], dtype=torch.float64
    )
    grad_F = None if mu is None else mu[:, None] @ x[None, :]
    grad_g = None if mu is None else -mu
    grad_G = None if lam is None else lam[:, None] @ x[None, :]
    grad_h = None if lam is None else -lam
    return (x, grad_A, grad_b, grad_c, grad_d,
            grad_F, grad_g, grad_G, grad_h)


def metrics(candidate, reference):
    candidate_vector = _flatten_gradients(candidate).cpu()
    reference_vector = _flatten_gradients(reference).cpu()
    absolute = torch.linalg.vector_norm(candidate_vector - reference_vector).item()
    reference_norm = torch.linalg.vector_norm(reference_vector).item()
    relative = absolute / max(reference_norm, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        candidate_vector, reference_vector, dim=0, eps=1e-12
    ).item()
    return absolute, relative, reference_norm, cosine


def compare_groups(seed, dim, objective, candidate_name, candidate,
                   reference_name, reference, solution_diff):
    rows = []
    for variable, cand_group, ref_group in zip(NAMES, candidate, reference):
        absolute, relative, reference_norm, cosine = metrics(cand_group, ref_group)
        rows.append({
            "seed": seed, "dim": dim, "objective": objective,
            "candidate": candidate_name, "reference": reference_name,
            "variable": variable, "absolute_error": absolute,
            "relative_error": relative, "reference_norm": reference_norm,
            "cosine_similarity": cosine, "solution_abs_diff": solution_diff,
        })
    absolute, relative, reference_norm, cosine = metrics(candidate, reference)
    rows.append({
        "seed": seed, "dim": dim, "objective": objective,
        "candidate": candidate_name, "reference": reference_name,
        "variable": "all", "absolute_error": absolute,
        "relative_error": relative, "reference_norm": reference_norm,
        "cosine_similarity": cosine, "solution_abs_diff": solution_diff,
    })
    return rows


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
        ("Solution gradient: dOPT vs. diffcp", "solution_sum", "dOPT", "diffcp"),
        ("Value gradient: dOPT vs. GT", "value", "dOPT", "envelope"),
        ("Value gradient: diffcp vs. GT", "value", "diffcp", "envelope"),
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
    print(f"wrote {table_path}")


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dim in args.dims:
        n_soc = dim // 2
        n_as = [dim // 3] * n_soc
        for seed in range(args.seed_start, args.seed_start + args.n_prob):
            raw = socp_utils.generate_random_socp(
                dim, n_soc, n_as, 10, 2, seed=seed
            )
            solution, diffcp_value_grad, adjoint = solve_once_with_diffcp(
                raw, args.eps, args.max_iters
            )
            x_dsocp, dsocp_value_grad = dsocp_value_gradients(
                raw, solution, args.eps
            )
            _, dsocp_solution_grad = dsocp_solution_gradients(
                raw, solution, args.eps
            )
            diffcp_solution_grad = diffcp_solution_gradients(
                raw, solution, adjoint
            )
            envelope_grad = envelope_gradients(raw, solution)
            x_reference = torch.tensor(solution["x"], dtype=torch.float64)
            solution_diff = torch.linalg.vector_norm(
                x_dsocp.squeeze() - x_reference
            ).item()
            rows.extend(compare_groups(
                seed, dim, "solution_sum", "dOPT", dsocp_solution_grad,
                "diffcp", diffcp_solution_grad, solution_diff,
            ))
            rows.extend(compare_groups(
                seed, dim, "value", "dOPT", dsocp_value_grad,
                "envelope", envelope_grad, solution_diff,
            ))
            rows.extend(compare_groups(
                seed, dim, "value", "diffcp", diffcp_value_grad,
                "envelope", envelope_grad, 0.0,
            ))
            overall = next(
                row for row in reversed(rows)
                if row["objective"] == "solution_sum" and row["variable"] == "all"
            )
            print(
                f"dim={dim} seed={seed} solution_diff={solution_diff:.3e} "
                f"dOPT_vs_diffcp_solution_grad_rel={overall['relative_error']:.3e}",
                flush=True,
            )

    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    write_table(rows, output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SOCP optimal-value gradients with envelope gradients."
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[20])
    parser.add_argument("--n-prob", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=121)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("results") / "socp_value_gradients.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
