#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark differentiable solvers on randomly generated dense SOCPs."""


import time
import os
import csv
import argparse
import sys
import gc as garbage_collector
import numpy as np
import torch
import cvxpy as cp
import scipy.sparse as sp
import diffcp
from cvxpylayers.torch import CvxpyLayer

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import socp_utils as utils
from src.dSOCP import dSOCPLayer
from random_experiments.socp_forward_solver import cvxpy_solve as cvxpy_forward_solver


def ffolayer_solve(q, A, b, c, d, F=None, g=None, G=None, h=None,
                   layer=None, *, eps=1e-8, ignore_dpp=True):
    """Solve and differentiate the SOCP with FFOLayer and MOSEK."""
    from third_party.FFOLayer.src.ffolayer import FFOLayer

    dim, n_soc = q.shape[0], len(A)
    if layer is None:
        x_cp = cp.Variable(dim, name="x")
        q_cp = cp.Parameter(dim, name="q")
        A_cp = [cp.Parameter(Ai.shape, name=f"A_{i}") for i, Ai in enumerate(A)]
        b_cp = [cp.Parameter(bi.shape, name=f"b_{i}") for i, bi in enumerate(b)]
        c_cp = [cp.Parameter(ci.shape, name=f"c_{i}") for i, ci in enumerate(c)]
        d_cp = [cp.Parameter(di.shape, name=f"d_{i}") for i, di in enumerate(d)]
        constraints = [
            cp.SOC(c_cp[i] @ x_cp + d_cp[i], A_cp[i] @ x_cp + b_cp[i])
            for i in range(n_soc)
        ]
        parameters = [q_cp]
        if F is not None:
            F_cp = cp.Parameter(F.shape, name="F")
            g_cp = cp.Parameter(g.shape, name="g")
            constraints.append(F_cp @ x_cp == g_cp)
            parameters.extend([F_cp, g_cp])
        if G is not None:
            G_cp = cp.Parameter(G.shape, name="G")
            h_cp = cp.Parameter(h.shape, name="h")
            constraints.append(G_cp @ x_cp <= h_cp)
            parameters.extend([G_cp, h_cp])
        parameters.extend(A_cp + b_cp + c_cp + d_cp)
        problem = cp.Problem(cp.Minimize(q_cp @ x_cp), constraints)
        if not problem.is_dpp():
            raise ValueError("The FFOLayer SOCP formulation must satisfy DPP.")
        layer = FFOLayer(
            problem, parameters=parameters, variables=[x_cp], eps=eps,
            backward_eps=eps,
        )

    parameter_tensors = [q]
    if F is not None:
        parameter_tensors.extend([F, g])
    if G is not None:
        parameter_tensors.extend([G, h])
    parameter_tensors.extend(A + b + c + d)
    mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": float(eps),
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": float(eps),
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": float(eps),
    }
    solver_args = {
        "solver": cp.MOSEK,
        "ignore_dpp": ignore_dpp,
        "mosek_params": mosek_params,
    }
    start = time.perf_counter()
    x_ffo, = layer(
        *(tensor.unsqueeze(0) for tensor in parameter_tensors),
        solver_args=solver_args,
    )
    forward = time.perf_counter() - start
    # FFOLayer 0.1.2 injects the SCS-only max_iters option into backward.
    layer._solver_args_bwd.pop("max_iters", None)
    x_ffo = x_ffo[0] if x_ffo.ndim == 2 and x_ffo.shape[0] == 1 else x_ffo

    # Read the dual variables from FFOLayer's original forward solve before
    # its perturbed backward solve can overwrite any CVXPY constraint state.
    bundle = layer.bundles[0]
    soc_duals = []
    for constraint in bundle["soc_constraints"]:
        tau, vec = constraint.dual_value
        tau_value = float(np.asarray(tau).reshape(-1)[0])
        vec_value = np.asarray(vec, dtype=float).reshape(-1)
        soc_duals.append(torch.tensor(np.r_[tau_value, vec_value], dtype=torch.float64))
    mu = None
    if F is not None:
        mu = torch.tensor(
            np.asarray(bundle["eq_constraints"][0].dual_value).reshape(-1),
            dtype=torch.float64,
        )
    lam = None
    if G is not None:
        lam = torch.tensor(
            np.asarray(bundle["scalar_ineq_constraints"][0].dual_value).reshape(-1),
            dtype=torch.float64,
        )
    nu = torch.tensor([dual[0] for dual in soc_duals], dtype=torch.float64)
    return x_ffo, mu, nu, lam, soc_duals, forward, layer


def diffcp_solve(q, A, b, c, d, F=None, g=None, G=None, h=None,
                 eps=1e-8, max_iters=100000):
    """Solve and differentiate the SOCP directly in diffcp cone form."""
    dim, n_soc = q.shape[0], len(A)
    blocks, rhs = [], []
    n_eq = 0 if F is None else F.shape[0]
    n_ineq = 0 if G is None else G.shape[0]
    if F is not None:
        blocks.append(sp.csc_matrix(F.detach().cpu().numpy()))
        rhs.append(g.detach().cpu().numpy())
    if G is not None:
        blocks.append(sp.csc_matrix(G.detach().cpu().numpy()))
        rhs.append(h.detach().cpu().numpy())

    soc_ranges, row = [], n_eq + n_ineq
    for Ai, bi, ci, di in zip(A, b, c, d):
        Mi = np.vstack((ci.detach().cpu().numpy()[None, :],
                        Ai.detach().cpu().numpy()))
        ri = np.hstack((di.detach().cpu().numpy(), bi.detach().cpu().numpy()))
        blocks.append(sp.csc_matrix(-Mi)); rhs.append(ri)
        soc_ranges.append((row, row + Mi.shape[0])); row += Mi.shape[0]

    A_cone = sp.vstack(blocks, format="csc")
    b_cone = np.concatenate(rhs)
    q_np = q.detach().cpu().numpy()
    cones = {"z": n_eq, "l": n_ineq,
             "q": [Ai.shape[0] + 1 for Ai in A]}
    start = time.perf_counter()
    x_np, y_np, s_np, _, adjoint = diffcp.solve_and_derivative(
        A_cone, b_cone, q_np, cones, eps=eps, max_iters=max_iters,
    )
    forward = time.perf_counter() - start
    start = time.perf_counter()
    dA, db, dq = adjoint(np.ones_like(x_np), np.zeros_like(y_np),
                         np.zeros_like(s_np))
    backward = time.perf_counter() - start
    dA = dA.toarray() if sp.issparse(dA) else np.asarray(dA)

    offset = 0
    grad_F = torch.tensor(dA[:n_eq], dtype=torch.float64) if n_eq else None
    grad_g = torch.tensor(db[:n_eq], dtype=torch.float64) if n_eq else None
    offset += n_eq
    grad_G = torch.tensor(dA[offset:offset+n_ineq], dtype=torch.float64) if n_ineq else None
    grad_h = torch.tensor(db[offset:offset+n_ineq], dtype=torch.float64) if n_ineq else None
    offset += n_ineq
    grad_A, grad_b, grad_c, grad_d = [], [], [], []
    for Ai in A:
        size = Ai.shape[0] + 1
        block_A, block_b = dA[offset:offset+size], db[offset:offset+size]
        grad_c.append(torch.tensor(-block_A[0], dtype=torch.float64))
        grad_A.append(torch.tensor(-block_A[1:], dtype=torch.float64))
        grad_d.append(torch.tensor(block_b[0], dtype=torch.float64))
        grad_b.append(torch.tensor(block_b[1:], dtype=torch.float64))
        offset += size

    soc_duals = [torch.tensor(y_np[s:e], dtype=torch.float64)
                 for s, e in soc_ranges]
    return (
        torch.tensor(x_np, dtype=torch.float64),
        torch.tensor(y_np[:n_eq], dtype=torch.float64) if n_eq else None,
        torch.tensor([dual[0] for dual in soc_duals], dtype=torch.float64),
        torch.tensor(y_np[n_eq:n_eq+n_ineq], dtype=torch.float64) if n_ineq else None,
        soc_duals,
        torch.tensor(dq, dtype=torch.float64), grad_A, grad_b, grad_c, grad_d,
        grad_F, grad_g, grad_G, grad_h, forward, backward,
    )

def cvxpy_solve(q, A, b, c, d, F, g, G=None, h=None, cvxpy_cache=None,
                compute_reference_duals=True, phase_log=False, **solver_args):
    dim = q.shape[0]
    nSOC = len(A)
    n_As = [Ai.shape[0] for Ai in A]
    if cvxpy_cache is None:
        if phase_log:
            print("[cvxpylayers] constructing/canonicalizing layer", flush=True)
        q_cp = cp.Parameter(dim)
        x = cp.Variable(dim)

        A_cp = [cp.Parameter((n_As[i], dim)) for i in range(nSOC)]
        b_cp = [cp.Parameter(n_As[i]) for i in range(nSOC)]
        c_cp = [cp.Parameter(dim) for i in range(nSOC)]
        d_cp = [cp.Parameter() for i in range(nSOC)]

        soc_constraints = [
            cp.SOC(c_cp[i].T @ x + d_cp[i], A_cp[i] @ x + b_cp[i]) for i in range(nSOC)
        ]

        constraints = soc_constraints

        # Optional equality constraint F x = g
        if F is not None:
            nEq = F.shape[0]
            F_cp = cp.Parameter((nEq, dim))
            g_cp = cp.Parameter(nEq)
            constraints += [F_cp @ x == g_cp]

        # Optional inequality constraint G x <= h
        if G is not None and h is not None:
            G_cp = cp.Parameter((G.shape[0], G.shape[1]))
            h_cp = cp.Parameter(G.shape[0])
            constraints += [G_cp @ x <= h_cp]

        problem = cp.Problem(cp.Minimize(q_cp.T @ x), constraints)

        param_list = [q_cp]
        if F is not None:
            param_list += [F_cp, g_cp]
        if G is not None and h is not None:
            param_list += [G_cp, h_cp]
        param_list += A_cp + b_cp + c_cp + d_cp
        cvxpylayer = CvxpyLayer(problem, parameters=param_list, variables=[x])
        cvxpy_cache = {
            "layer": cvxpylayer,
            "problem": problem,
            "constraints": constraints,
            "x": x,
            "q": q_cp,
            "F": F_cp if F is not None else None,
            "g": g_cp if F is not None else None,
            "G": G_cp if G is not None and h is not None else None,
            "h": h_cp if G is not None and h is not None else None,
            "A": A_cp,
            "b": b_cp,
            "c": c_cp,
            "d": d_cp,
        }
    else:
        cvxpylayer = cvxpy_cache["layer"]
        problem = cvxpy_cache["problem"]
        constraints = cvxpy_cache["constraints"]
        x = cvxpy_cache["x"]
        q_cp = cvxpy_cache["q"]
        F_cp, g_cp = cvxpy_cache["F"], cvxpy_cache["g"]
        G_cp, h_cp = cvxpy_cache["G"], cvxpy_cache["h"]
        A_cp, b_cp = cvxpy_cache["A"], cvxpy_cache["b"]
        c_cp, d_cp = cvxpy_cache["c"], cvxpy_cache["d"]
    arg_list = [q]
    if F is not None:
        arg_list += [F, g]
    if G is not None and h is not None:
        arg_list += [G, h]
    arg_list += A + b + c + d

    start_time = time.perf_counter()
    if phase_log:
        print("[cvxpylayers] forward", flush=True)
    solution, = cvxpylayer(*arg_list,solver_args=solver_args)
    forward_time = time.perf_counter() - start_time

    # Backward pass
    start_time = time.perf_counter()
    if phase_log:
        print("[cvxpylayers] backward", flush=True)
    solution.sum().backward()
    backward_time = time.perf_counter() - start_time

    if not compute_reference_duals:
        grad_q = q.grad
        grad_F = F.grad if F is not None else None
        grad_g = g.grad if g is not None else None
        grad_G = G.grad if G is not None else None
        grad_h = h.grad if h is not None else None
        grad_A = [Ai.grad for Ai in A]
        grad_b = [bi.grad for bi in b]
        grad_c = [ci.grad for ci in c]
        grad_d = [di.grad for di in d]
        return (
            solution, None, None, None, None, None,
            grad_q, grad_A, grad_b, grad_c, grad_d,
            grad_F, grad_g, grad_G, grad_h,
            forward_time, backward_time, cvxpy_cache,
        )

    if phase_log:
        print("[cvxpylayers] reference SCS dual solve", flush=True)
    q_cp.value = q.detach().cpu().numpy()
    if F is not None:
        F_cp.value = F.detach().cpu().numpy()
        g_cp.value = g.detach().cpu().numpy()
    if G is not None and h is not None:
        G_cp.value = G.detach().cpu().numpy()
        h_cp.value = h.detach().cpu().numpy()
    for i in range(nSOC):
        A_cp[i].value = A[i].detach().cpu().numpy()
        b_cp[i].value = b[i].detach().cpu().numpy()
        c_cp[i].value = c[i].detach().cpu().numpy()
        d_cp[i].value = d[i].detach().cpu().numpy()

    problem.solve(solver=cp.SCS, **solver_args)

    # Extract the primal solution and the complete SOC duals from the same
    # CVXPY/SCS solve. Each SOC dual is (scalar_part, vector_part).
    cvxpy_primal = torch.tensor(x.value, dtype=q.dtype)
    nu = np.concatenate([constraints[i].dual_value[0] for i in range(nSOC)])
    soc_dual_vectors = [
        torch.tensor(np.asarray(constraints[i].dual_value[1]).reshape(-1), dtype=q.dtype)
        for i in range(nSOC)
    ]

    mu = constraints[nSOC].dual_value if F is not None else None
    lam = constraints[nSOC + 1].dual_value if (G is not None and h is not None and F is not None) else (
        constraints[nSOC].dual_value if (G is not None and h is not None and F is None) else None
    )

    # Gradients
    grad_q = q.grad
    grad_F = F.grad if F is not None else None
    grad_g = g.grad if g is not None else None
    grad_G = G.grad if G is not None else None
    grad_h = h.grad if h is not None else None
    grad_A = [A[i].grad for i in range(nSOC)]
    grad_b = [b[i].grad for i in range(nSOC)]
    grad_c = [c[i].grad for i in range(nSOC)]
    grad_d = [d[i].grad for i in range(nSOC)]

    return solution, cvxpy_primal, mu, nu, soc_dual_vectors, lam, grad_q, grad_A, grad_b, grad_c, grad_d, grad_F, grad_g, grad_G, grad_h, forward_time, backward_time, cvxpy_cache


def lin_solve(eps_active,x_star,nu_star,q, A, b, c, d, F, g):
    '''compute the gradient in the general case'''
    act_ind = []
    rs = []

    for i in range(nSOC):
        r = c[i].T @ x_star + d[i] - torch.norm(A[i] @ x_star + b[i])
        # print(r.item())
        if r < eps_active:
            act_ind.append(i)
            rs.append(r)

    n_act = len(act_ind)
    print('#act: ',n_act)
    # Some stuff can be precomputed! Can be improved in efficiency
    KKT_A_Q = 0
    KKT_A_C = torch.zeros((dim,n_act),dtype=torch.float64)
    KKT_b_act = torch.zeros(n_act,dtype=torch.float64)
    KKT_b_q = -q
    for i in range(n_act):
        j = act_ind[i]
        Axb = A[j]@x_star + b[j]
        norm_x = torch.norm(Axb)
        Axb_unit = Axb / norm_x

        Q_j = nu_star[j]/norm_x * A[j].T  @  ( torch.eye(n_As[j]) - torch.outer( Axb_unit , Axb_unit)   ) @ A[j]
        KKT_A_Q += Q_j

        C_j = A[j].T @  Axb_unit - c[j]
        KKT_A_C[:,i] = C_j
        KKT_b_act_tmp =  rs[i] + C_j @ x_star
        KKT_b_act[i] =  KKT_b_act_tmp


        KKT_b_q += Q_j @ x_star

    if F is not None:
        KKT_A_C = torch.hstack(( KKT_A_C, F.T ))
    KKT_A = torch.vstack((
            torch.hstack((
                KKT_A_Q,
                KKT_A_C
                )),
        torch.hstack((
            KKT_A_C.T, torch.zeros((n_act + nEq,n_act + nEq))
            ))
        ))
    KKT_b = torch.hstack(( KKT_b_q,  KKT_b_act))


    if F is not None:
        KKT_b = torch.hstack((KKT_b, g))
    try:
        sol = torch.linalg.solve(KKT_A,KKT_b)
    except Exception as e:
        print(e)
        sol = torch.linalg.lstsq(KKT_A,KKT_b).solution
    x = sol[:dim]
    nu_act = sol[dim:dim+n_act]
    nu = torch.zeros(nSOC,dtype=torch.float64)
    nu[act_ind] = nu_act
    mu = sol[dim+n_act:] if F is not None else None



    x.sum().backward()
    grad_q = q.grad
    if F is not None:
        grad_F = F.grad
        grad_g = g.grad
    else:
        grad_F,grad_g = None,None

    grad_A = []
    grad_b = []
    grad_c = []
    grad_d = []
    for i in range(nSOC):
        if i in act_ind:
            grad_A.append(A[i].grad)
            grad_b.append(b[i].grad)
            grad_c.append(c[i].grad)
            grad_d.append(d[i].grad)
        else:
            grad_A.append(torch.zeros((n_As[i], n_As[i]), dtype=torch.float64))
            grad_b.append(torch.zeros(n_As[i], dtype=torch.float64))
            grad_c.append(torch.zeros(dim, dtype=torch.float64))
            grad_d.append(torch.zeros(1, dtype=torch.float64))


    return x,mu,nu,grad_q,grad_A,grad_b,grad_c,grad_d,grad_F,grad_g

def init_para(q,A,b,c,d,F,g,G,h):
    '''Eliminate grad of parameters'''

    q_ls = q.clone().detach()
    q_ls.requires_grad_(True)
    if F is not None:
        F_ls = F.clone().detach()
        F_ls.requires_grad_(True)
        g_ls = g.clone().detach()
        g_ls.requires_grad_(True)
    else:
        F_ls, g_ls = None, None

    if G is not None:
        G_ls = G.clone().detach()
        G_ls.requires_grad_(True)
        h_ls = h.clone().detach()
        h_ls.requires_grad_(True)
    else:
        G_ls, h_ls = None, None

    A_ls = []
    b_ls = []
    c_ls = []
    d_ls = []
    for i in range(nSOC):
        Ai = A[i].clone().detach()
        Ai.requires_grad_(True)
        A_ls.append(Ai)

        bi = b[i].clone().detach()
        bi.requires_grad_(True)
        b_ls.append(bi)

        ci = c[i].clone().detach()
        ci.requires_grad_(True)
        c_ls.append(ci)

        di = d[i].clone().detach()
        di.requires_grad_(True)
        d_ls.append(di)

    return q_ls,A_ls,b_ls,c_ls,d_ls,F_ls,g_ls,G_ls,h_ls

def primal_obj(q, x):
    x = x.squeeze()
    return (q @ x).item()


def dual_obj_socp(x, nu, soc_dual_vectors, A, b, c, d, F=None, g=None,
                  mu=None, G=None, h=None, lam=None):
    """
    Compute SOC dual objective using available dual variables:
      d* = -sum_i (tau_i * d_i + s_i^T b_i) - mu^T g - lam^T h

    The scalar and vector parts must be the complete SOC dual returned by the
    solver; no vector-dual reconstruction is performed here.
    """
    x = x.squeeze()
    nu = nu.squeeze()
    dual_obj = torch.tensor(0.0, dtype=x.dtype)
    for i in range(len(A)):
        tau_i = nu[i]
        s_i = torch.as_tensor(soc_dual_vectors[i], dtype=x.dtype)
        dual_obj = dual_obj - tau_i * d[i] - torch.dot(s_i, b[i])

    if F is not None and g is not None and mu is not None:
        dual_obj = dual_obj - torch.dot(mu.squeeze(), g.squeeze())
    if G is not None and h is not None and lam is not None:
        dual_obj = dual_obj - torch.dot(lam.squeeze(), h.squeeze())

    return dual_obj.item()


def duality_gap_socp(q, x, nu, soc_dual_vectors, A, b, c, d, F=None, g=None,
                     mu=None, G=None, h=None, lam=None):
    p_obj = primal_obj(q, x)
    d_obj = dual_obj_socp(
        x, nu, soc_dual_vectors, A, b, c, d,
        F=F, g=g, mu=mu, G=G, h=h, lam=lam,
    )
    gap = abs(p_obj - d_obj)
    return p_obj, d_obj, gap


def _flatten_gradients(*grad_groups):
    """Flatten tensors/lists of tensors into one dense gradient vector."""
    flat = []
    for group in grad_groups:
        if group is None:
            continue
        tensors = group if isinstance(group, (list, tuple)) else [group]
        for tensor in tensors:
            if tensor is None:
                continue
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            flat.append(tensor.detach().reshape(-1))
    return torch.cat(flat) if flat else torch.empty(0, dtype=torch.float64)


def print_gradient_breakdown(candidate_groups, reference_groups):
    names = ("q", "A", "b", "c", "d", "F", "g", "G", "h")
    candidate_all = _flatten_gradients(*candidate_groups).cpu()
    reference_all = _flatten_gradients(*reference_groups).cpu()
    total_error_sq = torch.linalg.vector_norm(candidate_all - reference_all).item() ** 2
    for name, candidate, reference in zip(names, candidate_groups, reference_groups):
        candidate_vec = _flatten_gradients(candidate).cpu()
        reference_vec = _flatten_gradients(reference).cpu()
        absolute = torch.linalg.vector_norm(candidate_vec - reference_vec).item()
        denominator = max(torch.linalg.vector_norm(reference_vec).item(), 1e-12)
        relative = absolute / denominator
        contribution = absolute ** 2 / max(total_error_sq, 1e-24)
        print(
            f"gradient_group={name} abs={absolute:.12e} "
            f"rel={relative:.12e} error_sq_share={contribution:.12e}",
            flush=True,
        )


def gradient_diff(candidate, reference, eps=1e-12):
    """Absolute and reference-relative L2 difference between gradient groups."""
    candidate_vec = _flatten_gradients(*candidate)
    reference_vec = _flatten_gradients(*reference)
    abs_diff = torch.linalg.vector_norm(candidate_vec - reference_vec).item()
    ref_norm = torch.linalg.vector_norm(reference_vec).item()
    return abs_diff, abs_diff / max(ref_norm, eps)


def _flatten_parameter_grads(*parameter_groups):
    """Flatten parameter grads, representing a missing grad by matching zeros."""
    flat = []
    for group in parameter_groups:
        if group is None:
            continue
        parameters = group if isinstance(group, (list, tuple)) else [group]
        for parameter in parameters:
            if parameter is None:
                continue
            grad = parameter.grad
            if grad is None:
                grad = torch.zeros_like(parameter)
            if grad.is_sparse:
                grad = grad.to_dense()
            flat.append(grad.detach().reshape(-1))
    return torch.cat(flat) if flat else torch.empty(0, dtype=torch.float64)


def run_benchmark(args):
    """Benchmark dOPT, FFOLayer, and direct diffcp on shared problems."""
    global nSOC, n_As
    methods = set(args.methods)
    run_dopt = "dOPT" in methods
    run_diffcp = "diffcp" in methods
    run_ffo = "ffolayer" in methods
    dim = args.dim
    n_soc = dim // 2
    n_as = [dim // 3] * n_soc
    nSOC, n_As = n_soc, n_as
    n_ineq, n_eq = 10, 2
    all_results_dir = os.path.join(args.output_dir, "all")
    os.makedirs(all_results_dir, exist_ok=True)

    headers = [
        "seed", "method", "dim", "n_soc", "n_ineq", "n_eq", "mode", "eps",
        "forward_s", "backward_s", "total_s", "duality_gap",
        "solution_abs_diff", "solution_rel_diff", "gradient_abs_diff",
        "gradient_rel_diff", "n_active_soc", "n_active_ineq", "n_active_total",
        "n_active_apex_soc",
    ]
    temp_path = os.path.join(all_results_dir, f"dim_{dim}.csv")
    method_labels = {
        "dOPT": "dOPT", "diffcp": "diffcp", "ffolayer": "ffolayer",
    }
    selected_labels = {method_labels[method] for method in methods}
    temp_rows = []
    if os.path.exists(temp_path):
        with open(temp_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == headers:
                temp_rows = [
                    row for row in reader if row["method"] not in selected_labels
                ]

    def write_temp_results():
        """Persist every completed method for this dimension immediately."""
        with open(temp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(temp_rows)

    # Replace only the selected methods; keep completed results for the others.
    write_temp_results()
    layer = None
    if run_dopt:
        cvxpy_forward_solver_name = {
            "mosek": cp.MOSEK,
            "scs": cp.SCS,
        }[args.forward_solver]
        # Build the fixed CVXPY problem once, outside all measured calls.
        forward_solver = cvxpy_forward_solver(
            dim, n_as, n_eq=n_eq, n_ineq=n_ineq, eps=args.eps,
            solver=cvxpy_forward_solver_name,
            solver_args={"ignore_dpp": args.ignore_dpp},
        )
        layer = dSOCPLayer(
            socp_solver="cvxpy", mode="dense", eps=args.eps,
            forward_solver=forward_solver,
        )
    ffo_layer = None

    # Warm up only the selected methods; this pass never enters timing/output.
    if not args.no_warmup:
        warm_raw = utils.generate_random_socp(
            dim, n_soc, n_as, n_ineq, n_eq, seed=args.seed_start - 1
        )
        if run_dopt:
            warm = init_para(*warm_raw)
            q, A, b, c, d, F, g, G, h = warm
            x, *_ = layer(q, n_as, torch.vstack(A), torch.hstack(b),
                          torch.vstack(c), torch.hstack(d), F, g, G, h)
            x.sum().backward()
        if run_ffo:
            warm_ffo = init_para(*warm_raw)
            x_ffo, *_, ffo_layer = ffolayer_solve(
                *warm_ffo, layer=ffo_layer, eps=args.eps,
                ignore_dpp=args.ignore_dpp,
            )
        if run_diffcp:
            diffcp_solve(*init_para(*warm_raw), eps=args.eps)

    # Release the warm-up graph before allocating the first timed problem.
    if not args.no_warmup:
        if run_dopt:
            del warm, q, A, b, c, d, F, g, G, h, x
        if run_ffo:
            del warm_ffo, x_ffo
        del warm_raw
    garbage_collector.collect()

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_prob):
        raw = utils.generate_random_socp(
            dim, n_soc, n_as, n_ineq, n_eq, seed=seed
        )
        q0, A0, b0, c0, d0, F0, g0, G0, h0 = raw
        common = {
            "seed": seed, "dim": dim, "n_soc": n_soc,
            "n_ineq": n_ineq, "n_eq": n_eq, "mode": "dense", "eps": args.eps,
        }
        seed_rows = []
        dsocp_row = None
        x = grad = None
        if run_dopt:
            q, A, b, c, d, F, g, G, h = init_para(*raw)
            try:
                print(f"[seed {seed}] dOPT forward starting", flush=True)
                start = time.perf_counter()
                x, mu, nu, lam = layer(
                    q, n_as, torch.vstack(A), torch.hstack(b),
                    torch.vstack(c), torch.hstack(d), F, g, G, h,
                )
                fwd = time.perf_counter() - start
                print(f"[seed {seed}] dOPT forward complete ({fwd:.3f}s)", flush=True)
                print(f"[seed {seed}] dOPT backward starting", flush=True)
                start = time.perf_counter(); x.sum().backward()
                bwd = time.perf_counter() - start
                print(f"[seed {seed}] dOPT backward complete ({bwd:.3f}s)", flush=True)
                soc_duals = layer.vec_duals
                if soc_duals is None:
                    raise RuntimeError("dOPT forward did not expose complete SOC duals")
                gap = duality_gap_socp(
                    q0, x.detach(), nu.detach(), [v[1:] for v in soc_duals],
                    A0, b0, c0, d0, F0, g0,
                    mu.detach() if mu is not None else None, G0, h0,
                    lam.detach() if lam is not None else None,
                )[2]
                grad_groups = (
                    q.grad, [v.grad for v in A], [v.grad for v in b],
                    [v.grad for v in c], [v.grad for v in d],
                    F.grad, g.grad, G.grad, h.grad,
                )
                grad = _flatten_gradients(*grad_groups).cpu()
                slacks = torch.stack([
                    c0[j] @ x.detach() + d0[j]
                    - torch.norm(A0[j] @ x.detach() + b0[j])
                    for j in range(n_soc)
                ])
                active = slacks < layer.eps_active
                n_act_soc = int(active.sum())
                n_apex = sum(
                    1 for j in range(n_soc)
                    if active[j] and torch.norm(A0[j] @ x.detach() + b0[j])
                    < layer.eps_active
                )
                n_act_ineq = int(
                    ((G0 @ x.detach() - h0).abs() < layer.eps_active).sum()
                )
                dsocp_row = {
                    **common, "method": "dOPT", "forward_s": fwd,
                    "backward_s": bwd, "total_s": fwd + bwd,
                    "duality_gap": gap, "solution_abs_diff": "",
                    "solution_rel_diff": "", "gradient_abs_diff": "",
                    "gradient_rel_diff": "", "n_active_soc": n_act_soc,
                    "n_active_ineq": n_act_ineq,
                    "n_active_total": n_act_soc + n_act_ineq,
                    "n_active_apex_soc": n_apex,
                }
                seed_rows.append(dsocp_row)
                temp_rows.append(dsocp_row)
                write_temp_results()
            except Exception as exc:
                print(f"dOPT failed for seed={seed}: {exc}", flush=True)

        ffo_row = None
        ffo_solution = ffo_gradient = None
        if run_ffo:
            try:
                q_ffo, A_ffo, b_ffo, c_ffo, d_ffo, F_ffo, g_ffo, G_ffo, h_ffo = init_para(*raw)
                print(f"[seed {seed}] FFOLayer forward starting", flush=True)
                (x_ffo, mu_ffo, nu_ffo, lam_ffo, soc_duals_ffo,
                 fwd_ffo, ffo_layer) = ffolayer_solve(
                    q_ffo, A_ffo, b_ffo, c_ffo, d_ffo, F_ffo, g_ffo,
                    G_ffo, h_ffo, layer=ffo_layer, eps=args.eps,
                    ignore_dpp=args.ignore_dpp,
                )
                print(f"[seed {seed}] FFOLayer forward complete ({fwd_ffo:.3f}s)", flush=True)
                gap_ffo = duality_gap_socp(
                    q0, x_ffo.detach(), nu_ffo,
                    [dual[1:] for dual in soc_duals_ffo], A0, b0, c0, d0,
                    F0, g0, mu_ffo, G0, h0, lam_ffo,
                )[2]
                print(f"[seed {seed}] FFOLayer backward starting", flush=True)
                start = time.perf_counter(); x_ffo.sum().backward()
                bwd_ffo = time.perf_counter() - start
                print(f"[seed {seed}] FFOLayer backward complete ({bwd_ffo:.3f}s)", flush=True)
                missing_ffo_grads = [
                    name for name, parameter in (
                        [("q", q_ffo), ("F", F_ffo), ("g", g_ffo),
                         ("G", G_ffo), ("h", h_ffo)]
                        + [(f"A[{i}]", value) for i, value in enumerate(A_ffo)]
                        + [(f"b[{i}]", value) for i, value in enumerate(b_ffo)]
                        + [(f"c[{i}]", value) for i, value in enumerate(c_ffo)]
                        + [(f"d[{i}]", value) for i, value in enumerate(d_ffo)]
                    )
                    if parameter is not None and parameter.grad is None
                ]
                if missing_ffo_grads:
                    print(
                        f"[seed {seed}] FFOLayer missing {len(missing_ffo_grads)} "
                        "parameter gradients; recording them as zeros",
                        flush=True,
                    )
                ffo_solution = x_ffo.detach().squeeze().cpu()
                ffo_gradient = _flatten_parameter_grads(
                    q_ffo, A_ffo, b_ffo, c_ffo, d_ffo,
                    F_ffo, g_ffo, G_ffo, h_ffo,
                ).cpu()
                ffo_row = {
                    **common, "method": "ffolayer", "forward_s": fwd_ffo,
                    "backward_s": bwd_ffo, "total_s": fwd_ffo + bwd_ffo,
                    "duality_gap": gap_ffo, "solution_abs_diff": "",
                    "solution_rel_diff": "", "gradient_abs_diff": "",
                    "gradient_rel_diff": "", "n_active_soc": "",
                    "n_active_ineq": "", "n_active_total": "",
                    "n_active_apex_soc": "",
                }
                seed_rows.append(ffo_row)
                temp_rows.append(ffo_row)
                write_temp_results()
                if dsocp_row is not None and not run_diffcp:
                    ffo_sol_abs = torch.linalg.vector_norm(
                        ffo_solution - x.detach().squeeze().cpu()
                    ).item()
                    ffo_sol_rel = ffo_sol_abs / max(
                        torch.linalg.vector_norm(
                            x.detach().squeeze().cpu()
                        ).item(),
                        1e-12,
                    )
                    ffo_row.update({
                        "solution_abs_diff": ffo_sol_abs,
                        "solution_rel_diff": ffo_sol_rel,
                    })
                    write_temp_results()
                    print(
                        f"[seed {seed}] FFOLayer vs dOPT solution diff: "
                        f"abs={ffo_sol_abs:.6e}, rel={ffo_sol_rel:.6e}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"FFOLayer failed for seed={seed}: {exc}", flush=True)

        if not run_diffcp:
            rows.extend(seed_rows)
            continue

        # Run the reference second. Its adjoint uses the same upstream gradient
        # (all ones) as x.sum().backward() above.
        q_ref, A_ref, b_ref, c_ref, d_ref, F_ref, g_ref, G_ref, h_ref = init_para(*raw)
        try:
            (x_ref, mu_ref, nu_ref, lam_ref, soc_duals_ref, gq_ref, gA_ref,
             gb_ref, gc_ref, gd_ref, gF_ref, gg_ref, gG_ref, gh_ref,
             fwd_ref, bwd_ref) = diffcp_solve(
                q_ref, A_ref, b_ref, c_ref, d_ref, F_ref, g_ref, G_ref, h_ref,
                eps=args.eps,
            )
            gap_ref = duality_gap_socp(
                q0, x_ref, nu_ref, [dual[1:] for dual in soc_duals_ref],
                A0, b0, c0, d0, F0, g0, mu_ref, G0, h0, lam_ref,
            )[2]
            ref_gradient = _flatten_gradients(
                gq_ref, gA_ref, gb_ref, gc_ref, gd_ref,
                gF_ref, gg_ref, gG_ref, gh_ref,
            ).cpu()
            if dsocp_row is not None:
                if args.gradient_breakdown:
                    print_gradient_breakdown(
                        grad_groups,
                        (gq_ref, gA_ref, gb_ref, gc_ref, gd_ref,
                         gF_ref, gg_ref, gG_ref, gh_ref),
                    )
                sol_abs = torch.linalg.vector_norm(
                    x.detach().squeeze().cpu() - x_ref.cpu()
                ).item()
                sol_rel = sol_abs / max(
                    torch.linalg.vector_norm(x_ref).item(), 1e-12
                )
                grad_abs = torch.linalg.vector_norm(grad - ref_gradient).item()
                grad_rel = grad_abs / max(
                    torch.linalg.vector_norm(ref_gradient).item(), 1e-12
                )
                dsocp_row.update({
                    "solution_abs_diff": sol_abs,
                    "solution_rel_diff": sol_rel,
                    "gradient_abs_diff": grad_abs,
                    "gradient_rel_diff": grad_rel,
                })
                if args.gradient_breakdown:
                    reference_groups = (
                        gq_ref, gA_ref, gb_ref, gc_ref, gd_ref,
                        gF_ref, gg_ref, gG_ref, gh_ref,
                    )
                    total_sq = max(grad_abs * grad_abs, 1e-24)
                    print(f"[seed {seed}] gradient breakdown", flush=True)
                    for name, candidate_group, reference_group in zip(
                        ("q", "A", "b", "c", "d", "F", "g", "G", "h"),
                        grad_groups, reference_groups,
                    ):
                        candidate_vec = _flatten_gradients(candidate_group).cpu()
                        reference_vec = _flatten_gradients(reference_group).cpu()
                        absolute = torch.linalg.vector_norm(
                            candidate_vec - reference_vec
                        ).item()
                        relative = absolute / max(
                            torch.linalg.vector_norm(reference_vec).item(), 1e-12
                        )
                        contribution = absolute * absolute / total_sq
                        print(
                            f"  {name}: abs={absolute:.6e}, rel={relative:.6e}, "
                            f"total_sq_share={contribution:.6%}", flush=True,
                        )
            if ffo_row is not None:
                ffo_sol_abs = torch.linalg.vector_norm(ffo_solution - x_ref.cpu()).item()
                ffo_sol_rel = ffo_sol_abs / max(
                    torch.linalg.vector_norm(x_ref).item(), 1e-12
                )
                ffo_grad_abs = torch.linalg.vector_norm(
                    ffo_gradient - ref_gradient
                ).item()
                ffo_grad_rel = ffo_grad_abs / max(
                    torch.linalg.vector_norm(ref_gradient).item(), 1e-12
                )
                ffo_row.update({
                    "solution_abs_diff": ffo_sol_abs,
                    "solution_rel_diff": ffo_sol_rel,
                    "gradient_abs_diff": ffo_grad_abs,
                    "gradient_rel_diff": ffo_grad_rel,
                })
            diffcp_row = {
                **common, "method": "diffcp", "forward_s": fwd_ref,
                "backward_s": bwd_ref, "total_s": fwd_ref + bwd_ref,
                "duality_gap": gap_ref, "solution_abs_diff": "",
                "solution_rel_diff": "", "gradient_abs_diff": "",
                "gradient_rel_diff": "", "n_active_soc": "",
                "n_active_ineq": "", "n_active_total": "",
                "n_active_apex_soc": "",
            }
            seed_rows.append(diffcp_row)
            rows.extend(seed_rows)
            temp_rows.append(diffcp_row)
            write_temp_results()
        except Exception as exc:
            print(f"diffcp failed for seed={seed}: {exc}", flush=True)
            rows.extend(seed_rows)
            # Completed dOPT/FFOLayer rows remain safely on disk without diffs.
            continue

    result_headers = ["dim", "method", "n_soc", "n_ineq", "n_eq", "mode", "eps",
                      "seed_start", "n_prob", "n_solved", "forward_s", "backward_s",
                      "total_s", "duality_gap", "solution_abs_diff", "solution_rel_diff",
                      "gradient_abs_diff", "gradient_rel_diff", "n_active_soc",
                      "n_active_ineq", "n_active_total", "n_active_apex_soc"]
    summaries = []
    for method in ("diffcp", "dOPT", "ffolayer"):
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        def method_mean(field):
            vals = [float(r[field]) for r in method_rows if r[field] != ""]
            return float(np.mean(vals)) if vals else ""
        summary = {k: method_rows[0].get(k, "") for k in result_headers}
        summary.update({"n_solved": len(method_rows), "seed_start": args.seed_start,
                        "n_prob": args.n_prob})
        for field in result_headers[10:]: summary[field] = method_mean(field)
        summaries.append(summary)
    result_path = args.result_path or os.path.join(
        args.output_dir, "random_socp_results.csv"
    )
    existing = []
    if os.path.exists(result_path):
        with open(result_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == result_headers: existing = list(reader)
    # Replace only selected methods at this dimension and preserve the rest.
    existing = [r for r in existing if (
        r["method"] in {"diffcp", "dOPT", "ffolayer"}
        and not (
            int(r["dim"]) == dim and r["mode"] == "dense"
            and r["method"] in selected_labels
        )
    )]
    existing.extend(summaries)
    existing.sort(key=lambda r: (int(r["dim"]), r["method"]))
    with open(result_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result_headers); w.writeheader(); w.writerows(existing)
    print(f"Completed dim={dim}: diffcp={sum(r['method'] == 'diffcp' for r in rows)}, "
          f"dOPT={sum(r['method'] == 'dOPT' for r in rows)}, "
          f"ffolayer={sum(r['method'] == 'ffolayer' for r in rows)}")
    print(f"Results: {result_path}")


parser = argparse.ArgumentParser(description="Benchmark random SOCP methods.")
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--n-prob", type=int, default=10)
parser.add_argument("--seed-start", type=int, default=121)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--no-warmup", action="store_true")
parser.add_argument("--gradient-breakdown", action="store_true")
parser.add_argument(
    "--forward-solver", choices=("mosek", "scs"), default="mosek",
    help="CVXPY forward solver used by dOPT (default: mosek)",
)
parser.add_argument(
    "--ignore-dpp", action=argparse.BooleanOptionalAction, default=True,
    help="whether CVXPY should bypass DPP compilation (default: true)",
)
parser.add_argument(
    "--methods", nargs="+", required=True,
    choices=("dOPT", "diffcp", "ffolayer"),
    help="one or more benchmark methods to run",
)
parser.add_argument(
    "--output-dir",
    default=os.path.join(os.path.dirname(__file__), "results", "random_socp"),
)
parser.add_argument(
    "--result-path",
    default=None,
    help="optional aggregate CSV path; per-dimension CSVs still use output-dir",
)
run_benchmark(parser.parse_args())
raise SystemExit(0)
