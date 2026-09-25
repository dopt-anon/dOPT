#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:40:45 2025

: Project contributors"""

import numpy as np
import torch
from torch import nn
from random_experiments.socp_forward_solver import mosek_solve
from random_experiments.socp_forward_solver import gurobi_solve

from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

def coo_to_csr(M):
    # Convert torch coo matrix into scipy csr matrix. 
    # Might need to be improved
    indices = M.indices().numpy()
    values = M.values().numpy()
    shape = M.shape
    M = csr_matrix((values, (indices[0], indices[1])), shape=shape)
    return M

class dSOCPFunction(torch.autograd.Function):
    ''' Solves and differentiate
    min_x      q.T * x
    s.t.       ||A_i * x + b_i || <= c_i.T * x + d_i
               Fx = g
    '''

        
    @staticmethod
    def forward(ctx, A_dim, q, A_stack, b_stack, c_stack, d_stack, F=None, g=None, G=None, h=None, eps_active=1e-5, eps=None, socp_solver="mosek", mode="dense", dual_holder=None, forward_solver=None):
        
        solvers = {"mosek": mosek_solve,
                   "gurobi": gurobi_solve}
        solver = forward_solver.solve if forward_solver is not None else solvers[socp_solver]
        
        # Extract A_i from the row blk matrix A_stack
        A = []
        b = []
        c = []
        d = []
        start = 0
        if mode=="sparse":
            A_stack = A_stack.to_dense()
        for i,dim in enumerate(A_dim):
            A_i = A_stack[start:start+dim]
            #TODO: Try to avoid converting back and forth 
            if mode=="sparse":
                A.append(A_i.to_sparse_coo())
            else:
                A.append(A_i)
            
            b_i = b_stack[start:start+dim]
            b.append(b_i)
            
            c.append(c_stack[i])
            d.append(d_stack[i])
            start += dim
            
        
        q_ = q.detach().numpy()
        b_ = [ bi.detach().numpy() for bi in b]
        c_ = [ ci.detach().numpy() for ci in c]
        d_ = [ di.detach().numpy() for di in d]
        
        G_,h_,F_,g_ = None, None, None, None
        if mode == "sparse":
            A_ = [ coo_to_csr(Ai) for Ai in A]
            if F is not None:
                F_ = coo_to_csr(F)
                g_ = g.detach().numpy()
            if G is not None:
                G_ = coo_to_csr(G)
                h_ = h.detach().numpy()
        else:
            A_ = [ Ai.detach().numpy() for Ai in A]
            if F is not None:
                F_,g_ = F.detach().numpy(),g.detach().numpy()
            if G is not None:
                G_,h_ = G.detach().numpy(),h.detach().numpy()
            
            
        if socp_solver in {"mosek", "cvxpy"}:
            x_star, mu_star, nu_star, lamb_star, soc_duals = solver(
                q_, A_, b_, c_, d_, F_, g_, G_, h_, eps=eps,
                return_soc_duals=True,
            )
        else:
            x_star, mu_star, nu_star, lamb_star = solver(q_, A_, b_, c_, d_, F_, g_, G_, h_)
            soc_duals = None
        
        x_star = torch.tensor(x_star, dtype=torch.float64)
        mu_star = torch.tensor(mu_star, dtype=torch.float64) if mu_star is not None else None
        nu_star = torch.tensor(nu_star, dtype=torch.float64)
        lamb_star = torch.tensor(lamb_star, dtype=torch.float64) if lamb_star is not None else None
        
        ctx.save_for_backward(x_star, nu_star, mu_star, lamb_star, q, F, g, G,h)
        ctx.nSOC = len(A)
        ctx.nIneq = G.shape[0] if G is not None else 0
        ctx.nEq = F.shape[0] if F is not None else 0
        ctx.dim = q.shape[0]
        ctx.eps_active = eps_active
        ctx.A = A
        ctx.b = b
        ctx.c = c
        ctx.d = d
        ctx.F = F
        ctx.g = g
        ctx.mode = mode
        ctx.soc_duals = (
            [torch.tensor(dual, dtype=torch.float64) for dual in soc_duals]
            if soc_duals is not None else None
        )
        if dual_holder is not None:
            dual_holder["soc_duals"] = ctx.soc_duals
        

        return x_star,mu_star,nu_star,lamb_star
    
    @staticmethod
    def backward(ctx, grad_output, grad_mu, grad_nu, grad_lamb):
        x_star, nu_star, mu_star, lamb_star, q, F, g, G, h = ctx.saved_tensors
        A, b, c, d = ctx.A, ctx.b, ctx.c, ctx.d
        dim, nSOC = ctx.dim, ctx.nSOC
        nIneq, nEq = ctx.nIneq, ctx.nEq
        eps_active = ctx.eps_active

        if ctx.mode == "sparse":
            A = [Ai.to_dense() for Ai in A]
            if F is not None:
                F = F.to_dense()
            if G is not None:
                G = G.to_dense()

        regular_soc = []
        apex_soc = []
        norm_y = {}
        for j in range(nSOC):
            y_j = A[j] @ x_star + b[j]
            norm_j = torch.norm(y_j)
            slack_j = c[j] @ x_star + d[j] - norm_j
            if slack_j >= eps_active:
                continue
            if norm_j >= eps_active:
                regular_soc.append(j)
                norm_y[j] = norm_j
                continue

            if ctx.soc_duals is None:
                raise RuntimeError(
                    "An active SOC is at the cone apex, but the forward solver "
                    "did not return a complete cone dual."
                )
            cone_dual = ctx.soc_duals[j]
            dual_margin = cone_dual[0] - torch.norm(cone_dual[1:])
            if dual_margin <= eps_active:
                raise RuntimeError(
                    f"SOC {j} is biactive/degenerate at the cone apex: "
                    f"||Ax+b||={norm_j.item():.3e}, "
                    f"tau-||v||={dual_margin.item():.3e}."
                )
            apex_soc.append(j)

        if G is not None:
            ineq_residual = G @ x_star - h
            ineq_act_ind = (ineq_residual.abs() < eps_active).nonzero(as_tuple=True)[0]
        else:
            ineq_act_ind = torch.empty(0, dtype=torch.long)
        n_act_ineq = len(ineq_act_ind)

        # Only regular (non-apex) SOCs contribute curvature. Strictly
        # complementary apex SOCs contribute equality blocks -M.T/-M.
        KKT_Q = torch.zeros((dim, dim), dtype=q.dtype, device=q.device)
        constraint_blocks = []
        for j in regular_soc:
            y_j = A[j] @ x_star + b[j]
            y_unit = y_j / norm_y[j]
            Ay_unit = A[j].T @ y_unit
            KKT_Q = KKT_Q + nu_star[j] / norm_y[j] * (
                A[j].T @ A[j] - torch.outer(Ay_unit, Ay_unit)
            )
            constraint_blocks.append((Ay_unit - c[j]).reshape(dim, 1))

        for j in apex_soc:
            M_j = torch.vstack((c[j].reshape(1, -1), A[j]))
            constraint_blocks.append(-M_j.T)

        if G is not None and n_act_ineq:
            constraint_blocks.append(G[ineq_act_ind].T)
        if F is not None:
            constraint_blocks.append(F.T)

        KKT_C = (
            torch.hstack(constraint_blocks)
            if constraint_blocks
            else torch.empty((dim, 0), dtype=q.dtype, device=q.device)
        )
        n_constraints = KKT_C.shape[1]
        KKT = torch.vstack((
            torch.hstack((KKT_Q, KKT_C)),
            torch.hstack((
                KKT_C.T,
                torch.zeros(
                    (n_constraints, n_constraints),
                    dtype=q.dtype,
                    device=q.device,
                ),
            )),
        ))
        rhs = torch.hstack((
            grad_output,
            torch.zeros(n_constraints, dtype=q.dtype, device=q.device),
        ))

        if ctx.mode == "sparse":
            KKT_np = csc_matrix(KKT.detach().cpu().numpy())
            rhs_np = rhs.detach().cpu().numpy()
            try:
                dsol_np = splu(KKT_np).solve(rhs_np)
            except Exception:
                dsol_np = np.linalg.lstsq(KKT_np.toarray(), rhs_np, rcond=None)[0]
            dsol = torch.tensor(dsol_np, dtype=q.dtype, device=q.device)
        else:
            try:
                dsol = torch.linalg.solve(KKT, rhs)
            except Exception:
                dsol = torch.linalg.lstsq(KKT, rhs).solution

        dx = -dsol[:dim]
        offset = dim
        dnu_regular = {}
        for j in regular_soc:
            dnu_regular[j] = -dsol[offset]
            offset += 1

        dz_apex = {}
        for j in apex_soc:
            cone_dim = 1 + A[j].shape[0]
            dz_apex[j] = -dsol[offset:offset + cone_dim]
            offset += cone_dim

        dlamb_act = -dsol[offset:offset + n_act_ineq]
        offset += n_act_ineq
        dmu = -dsol[offset:offset + nEq]

        dq = dx
        dF = dg = dG = dh = None
        if F is not None:
            dg = -dmu
            dF = torch.outer(mu_star, dx) + torch.outer(dmu, x_star)
        if G is not None:
            dh = torch.zeros(nIneq, dtype=q.dtype, device=q.device)
            dh[ineq_act_ind] = -dlamb_act
            dG = torch.zeros((nIneq, dim), dtype=q.dtype, device=q.device)
            dG[ineq_act_ind] = (
                torch.outer(dlamb_act, x_star)
                + torch.outer(lamb_star[ineq_act_ind], dx)
            )

        dA, db, dc, dd = [], [], [], []
        for j in range(nSOC):
            if j in regular_soc:
                y_j = A[j] @ x_star + b[j]
                y_unit = y_j / norm_y[j]
                Ay_unit = A[j].T @ y_unit
                P = nu_star[j] / norm_y[j] * (
                    A[j] - torch.outer(y_unit, Ay_unit)
                )
                dnu_j = dnu_regular[j]
                db_j = P @ dx + y_unit * dnu_j
                dA_j = (
                    nu_star[j] * torch.outer(y_unit, dx)
                    + torch.outer(P @ dx, x_star)
                    + dnu_j * torch.outer(y_unit, x_star)
                )
                dc_j = -(nu_star[j] * dx + dnu_j * x_star)
                dd_j = -dnu_j
            elif j in apex_soc:
                z_j = ctx.soc_duals[j].to(dtype=q.dtype, device=q.device)
                dz_j = dz_apex[j]
                tau_j, v_j = z_j[0], z_j[1:]
                dtau_j, dv_j = dz_j[0], dz_j[1:]
                dc_j = -(tau_j * dx + dtau_j * x_star)
                dA_j = -(
                    torch.outer(v_j, dx) + torch.outer(dv_j, x_star)
                )
                dd_j = -dtau_j
                db_j = -dv_j
            else:
                dA_j = torch.zeros_like(A[j])
                db_j = torch.zeros_like(b[j])
                dc_j = torch.zeros_like(c[j])
                dd_j = torch.zeros_like(d[j])

            dA.append(dA_j)
            db.append(db_j)
            dc.append(dc_j)
            dd.append(dd_j)

        return (
            None, dq, torch.vstack(dA), torch.hstack(db), torch.vstack(dc),
            torch.hstack(dd), dF, dg, dG, dh, None, None, None, None, None,
            None,
        )
    def build_settings(solve_type="dense",eps_active=1e-5,eps=None,socp_solver=None):
        settings = {
            "solve_type" : solve_type,
            "socp_solver": socp_solver,
            "eps_active": eps_active,
            "eps": eps,
            # "eps_abs": eps_active,
            # "eps_rel": eps_active,
            # "lin_solver": lin_solver,
            }
        return settings


class dSOCPLayer(nn.Module):
    def __init__(self, eps_active=1e-5, eps=None, socp_solver="mosek",
                 mode="dense", forward_solver=None):
        super().__init__()
        self.eps_active = eps_active
        self.eps = eps
        self.socp_solver = socp_solver
        self.mode = mode
        self.forward_solver = forward_solver
        self.vec_duals = None
    def forward(self, q, A_dim, A, b, c, d, F=None, g=None, G=None, h=None):
        dual_holder = {}
        result = dSOCPFunction.apply(
            A_dim, q, A, b, c, d, F, g, G, h, self.eps_active,
            self.eps, self.socp_solver, self.mode, dual_holder,
            self.forward_solver,
        )
        self.vec_duals = dual_holder.get("soc_duals")
        return result
    
   
