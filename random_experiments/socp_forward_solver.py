#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:05:02 2025

: Project contributors
Forward solvers used to solve general socp

"""
import os
import time
import torch
import numpy as np
from scipy.sparse import csr_matrix

import cvxpy as cp 
from cvxpylayers.torch import CvxpyLayer

import mosek.fusion as mf

def cvxpylayer_solve(q,A,b,c,d,F,g):
    '''
    Using CvxpyLayers to solve a general socp
    
    Just use to retrieve the dual variable for evaluation
    
    '''
    
    dim = len(q)
    nSOC = len(A)
    n_As = [A_i.shape[0] for A_i in A]
    
    x = cp.Variable(dim)
    q_cp = cp.Parameter(dim)
    
    
    A_cp = [cp.Parameter((n_As[i],dim)) for i in range(nSOC)]
    b_cp = [cp.Parameter(n_As[i]) for i in range(nSOC)]
    c_cp = [cp.Parameter(dim) for i in range(nSOC)]
    d_cp = [cp.Parameter() for i in range(nSOC)]
     
    
    q_np = q.detach().numpy()
    if F is not None:
        F_np,g_np = F.detach().numpy(),g.detach().numpy()
    A_np = [ Ai.detach().numpy() for Ai in A]
    b_np = [ bi.detach().numpy() for bi in b]
    c_np = [ ci.detach().numpy() for ci in c]
    d_np = [ di.detach().numpy() for di in d]
    
    if F is not None:
        
        
        nEq = F.shape[0]
        F_cp = cp.Parameter((nEq, dim))
        g_cp = cp.Parameter(nEq)
        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [
            cp.SOC(c_cp[i].T@x + d_cp[i], A_cp[i]@x + b_cp[i]) for i in range(nSOC)
        ]
        constraints = soc_constraints + [F_cp@x == g_cp]
        problem = cp.Problem(cp.Minimize(q_cp.T@x),
                          constraints)
        
        
        # cvxpylayer = CvxpyLayer(problem, parameters=[q_cp,F_cp,g_cp] + A_cp + b_cp + c_cp + d_cp
        #                         , variables=[x])
        cvxpylayer = CvxpyLayer(problem, parameters=[q_cp,F_cp,g_cp] + A_cp + b_cp + c_cp + d_cp
                                , variables=[x])
        
        
        solution, = cvxpylayer(q,F,g,*A,*b,*c,*d)
                
        q_cp.value = q_np
       
        F_cp.value = F_np
        g_cp.value = g_np 
        for i in range(nSOC):
            A_cp[i].value = A_np[i]
            b_cp[i].value = b_np[i]
            c_cp[i].value = c_np[i]
            # Need to change later
            d_cp[i].value = d_np[i]

        problem.solve()
        nu = np.concatenate([(constraints[i].dual_value[0]) for i in range(nSOC)])
        
        mu = constraints[-1].dual_value
        

    else:
        
        
        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [
            cp.SOC(c_cp[i].T@x + d_cp[i], A_cp[i]@x + b_cp[i]) for i in range(nSOC)
        ]
        constraints = soc_constraints 
        problem = cp.Problem(cp.Minimize(q_cp.T@x),
                          constraints)
        
        
        cvxpylayer = CvxpyLayer(problem, parameters=[q_cp] + A_cp + b_cp + c_cp + d_cp
                                , variables=[x])
        
        
        solution, = cvxpylayer(q,*A,*b,*c,*d)
        
        q_cp.value = q_np
       
        
        for i in range(nSOC):
            A_cp[i].value = A_np[i]
            b_cp[i].value = b_np[i]
            c_cp[i].value = c_np[i]
            d_cp[i].value = d_np[i]

        problem.solve(solver=cp.SCS)
        nu = np.concatenate([(constraints[i].dual_value[0]) for i in range(nSOC)])
        
        mu = None
        
        
    return solution,mu,nu

def mosek_solve(q,A,b,c,d,F=None,g=None,G=None,h=None,eps=None,return_soc_duals=False):
    '''
    Using auxiliary variables y, z to solve general SOCP as formulated below:
    
    min_{x,y,z} q.T * x
    
    s.t.        ||y_i|| <= z_i
                
                y_i = A_i * x + b_i
                z_i = c_i.T * x + d_i
                Fx = g
    
    
    Parameters
    ----------
    q : numpy array with size (dim,)
    A : list, len(A) = # cone constraints, 
            A[i]: i) dense, numpy array with size (m_i,dim) 
                 ii) sparse scipy.sparse.csr_matrix
    b : list, len(b) = # cone constraints, b[i]: numpy array with size (m_i,).
    c : list, len(c) = # cone constraints, c[i]: numpy array with size (dim,).
    d : list, len(d) = # cone constraints, d[i]: numpy array with size ().
    
    F : None/ numpy array with size (nEq, dim)/ scipy.sparse.csr_matrix
    g : numpy array with size (nEq, ), could be None
    eps : None/float, optional MOSEK feasibility/gap tolerance.
    return_soc_duals : bool, optional
        Return complete quadratic-cone duals ``[tau_i, v_i]``.

    Returns
    -------
    x : 1D numpy array, primal solution
    nu: 1D numpy array, dual solution of cone constraints
    mu: 1D numpy array, dual solution of Fx=g. 
    '''
    dim = len(q)
    nSOC = len(A)
    diagnostic_start = time.perf_counter()
    M = mf.Model()
    if eps is not None:
        M.setSolverParam("intpntCoTolPfeas", float(eps))
        M.setSolverParam("intpntCoTolDfeas", float(eps))
        M.setSolverParam("intpntCoTolRelGap", float(eps))

    x = M.variable("x", dim, mf.Domain.unbounded())
    q = mf.Matrix.dense(q.reshape(1, -1))
    M.objective(mf.ObjectiveSense.Minimize, mf.Expr.mul(q, x))
     
    for i in range(nSOC):
        if isinstance(A[0], csr_matrix):
            row, col = A[i].nonzero()
            val = A[i].data
            m, n = A[i].shape
            Ai = mf.Matrix.sparse(m, n, row, col, val)
        else:
            Ai = mf.Matrix.dense(A[i])
            
        
        bi = mf.Matrix.dense(b[i].reshape(-1, 1))
        ci = mf.Matrix.dense(c[i].reshape(1, -1))
        di = float(d[i])
        k_i = A[i].shape[0]
    
        y = M.variable(f"y_{i}", k_i, mf.Domain.unbounded())
        z = M.variable(f"z_{i}", 1, mf.Domain.unbounded())
    
    
        M.constraint(f"y_eq_{i}", mf.Expr.sub(y, mf.Expr.add(mf.Expr.mul(Ai, x), bi)), mf.Domain.equalsTo(0))
        M.constraint(f"z_eq_{i}", mf.Expr.sub(z, mf.Expr.add(mf.Expr.mul(ci, x), di)), mf.Domain.equalsTo(0))
    
        M.constraint(f"cone_{i}", mf.Expr.vstack(z, y), mf.Domain.inQCone())

    if F is not None:
        if isinstance(F, csr_matrix):
            row, col = F.nonzero()
            val = F.data
            m, n = F.shape
            F_mtx = mf.Matrix.sparse(m, n, row, col, val)
        else:
            F_mtx = mf.Matrix.dense(F)
        g_vec = mf.Matrix.dense(g.reshape(-1, 1))    
        M.constraint("eq", mf.Expr.sub(mf.Expr.mul(F_mtx, x), g_vec), mf.Domain.equalsTo(0))

    # === Added Gx <= h constraint ===
    if G is not None and h is not None:
        if isinstance(G, csr_matrix):
            row, col = G.nonzero()
            val = G.data
            m, n = G.shape
            G_mtx = mf.Matrix.sparse(m, n, row, col, val)
        else:
            G_mtx = mf.Matrix.dense(G)
        h_vec = mf.Matrix.dense(h.reshape(-1, 1))
        M.constraint("ineq", mf.Expr.sub(mf.Expr.mul(G_mtx, x), h_vec), mf.Domain.lessThan(0))
    
    # Objective
    
    build_time = time.perf_counter() - diagnostic_start
    solve_start = time.perf_counter()
    M.solve()
    solve_wall = time.perf_counter() - solve_start
    try:
        optimizer_time = float(M.getSolverDoubleInfo("optimizerTime"))
    except Exception:
        optimizer_time = float("nan")
    
    extract_start = time.perf_counter()
    x = x.level()
    soc_duals = [
        np.asarray(M.getConstraint(f"cone_{i}").dual(), dtype=float)
        for i in range(nSOC)
    ]
    nu = np.array([dual[0] for dual in soc_duals])
    mu = None
    if F is not None:
        mu = -M.getConstraint("eq").dual()
    # === Dual for Gx <= h ===
    lamb = None
    if G is not None and h is not None:
        lamb = -M.getConstraint("ineq").dual()
    extract_time = time.perf_counter() - extract_start
    if os.environ.get("DSOCP_TIMING_DIAGNOSTIC") == "1":
        interface_time = (
            solve_wall - optimizer_time
            if np.isfinite(optimizer_time) else float("nan")
        )
        print(
            f"[timing][dSOCP] fusion_build={build_time:.6f}s "
            f"solve_wall={solve_wall:.6f}s optimizer={optimizer_time:.6f}s "
            f"solve_interface={interface_time:.6f}s "
            f"extract={extract_time:.6f}s",
            flush=True,
        )
    # Fusion owns native solver resources; release them as soon as all primal
    # and dual values have been copied out of the model.
    M.dispose()
    if return_soc_duals:
        return x,mu,nu,lamb,soc_duals
    return x,mu,nu,lamb


class cvxpy_solve:
    """Reusable CVXPY model for a fixed SOCP structure.

    Construction creates the CVXPY variables, parameters, constraints, and
    problem once. ``solve`` only updates parameter values and solves the same
    problem, so Python-level model setup can live outside benchmark timing.
    """

    def __init__(self, dim, n_as, n_eq=0, n_ineq=0, eps=None,
                 solver=cp.MOSEK, solver_args=None):
        self.dim = int(dim)
        self.n_as = tuple(int(size) for size in n_as)
        self.n_eq = int(n_eq)
        self.n_ineq = int(n_ineq)
        self.eps = eps
        self.solver = solver
        self.solver_args = dict(solver_args or {})

        self.x = cp.Variable(self.dim, name="x")
        self.q = cp.Parameter(self.dim, name="q")
        self.A = [
            cp.Parameter((size, self.dim), name=f"A_{i}")
            for i, size in enumerate(self.n_as)
        ]
        self.b = [
            cp.Parameter(size, name=f"b_{i}")
            for i, size in enumerate(self.n_as)
        ]
        self.c = [
            cp.Parameter(self.dim, name=f"c_{i}")
            for i in range(len(self.n_as))
        ]
        self.d = [
            cp.Parameter(name=f"d_{i}") for i in range(len(self.n_as))
        ]

        self.soc_constraints = [
            cp.SOC(
                self.c[i] @ self.x + self.d[i],
                self.A[i] @ self.x + self.b[i],
            )
            for i in range(len(self.n_as))
        ]
        constraints = list(self.soc_constraints)

        self.F = self.g = self.eq_constraint = None
        if self.n_eq:
            self.F = cp.Parameter((self.n_eq, self.dim), name="F")
            self.g = cp.Parameter(self.n_eq, name="g")
            self.eq_constraint = self.F @ self.x == self.g
            constraints.append(self.eq_constraint)

        self.G = self.h = self.ineq_constraint = None
        if self.n_ineq:
            self.G = cp.Parameter((self.n_ineq, self.dim), name="G")
            self.h = cp.Parameter(self.n_ineq, name="h")
            self.ineq_constraint = self.G @ self.x <= self.h
            constraints.append(self.ineq_constraint)

        self.problem = cp.Problem(cp.Minimize(self.q @ self.x), constraints)
        if not self.problem.is_dpp():
            raise ValueError("CVXPY SOCP model must satisfy DPP")

    @staticmethod
    def _dense(value):
        return value.toarray() if hasattr(value, "toarray") else np.asarray(value)

    def solve(self, q, A, b, c, d, F=None, g=None, G=None, h=None,
              eps=None, return_soc_duals=False, solver=None,
              solver_args=None):
        if len(A) != len(self.n_as):
            raise ValueError("SOC block count does not match cached problem")
        self.q.value = np.asarray(q)
        for i, size in enumerate(self.n_as):
            Ai = self._dense(A[i])
            if Ai.shape != (size, self.dim):
                raise ValueError(
                    f"A[{i}] must have shape {(size, self.dim)}, got {Ai.shape}"
                )
            self.A[i].value = Ai
            self.b[i].value = np.asarray(b[i])
            self.c[i].value = np.asarray(c[i])
            self.d[i].value = float(np.asarray(d[i]))

        if self.n_eq:
            if F is None or g is None:
                raise ValueError("cached problem requires F and g")
            self.F.value = self._dense(F)
            self.g.value = np.asarray(g)
        elif F is not None or g is not None:
            raise ValueError("cached problem was built without equalities")

        if self.n_ineq:
            if G is None or h is None:
                raise ValueError("cached problem requires G and h")
            self.G.value = self._dense(G)
            self.h.value = np.asarray(h)
        elif G is not None or h is not None:
            raise ValueError("cached problem was built without inequalities")

        selected_solver = self.solver if solver is None else solver
        options = dict(self.solver_args)
        options.update(solver_args or {})
        tolerance = self.eps if eps is None else eps
        if str(selected_solver).upper() == "MOSEK" and tolerance is not None:
            mosek_params = dict(options.pop("mosek_params", {}))
            mosek_params.update({
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": float(tolerance),
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": float(tolerance),
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": float(tolerance),
            })
            options["mosek_params"] = mosek_params
        elif str(selected_solver).upper() == "SCS" and tolerance is not None:
            options.setdefault("eps", float(tolerance))
            # diffcp.solve_and_derivative starts each solve from a cold state.
            # Match that behavior so a benchmark warm-up cannot change the
            # primal/dual point used by dSOCP's analytic backward pass.
            options.setdefault("warm_start", False)
            options.setdefault("max_iters", 100000)
        self.problem.solve(solver=selected_solver, **options)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"CVXPY SOCP solve failed with status {self.problem.status}"
            )

        x = np.asarray(self.x.value, dtype=float).reshape(-1)
        soc_duals = []
        for constraint in self.soc_constraints:
            tau, vec = constraint.dual_value
            soc_duals.append(np.r_[
                float(np.asarray(tau).reshape(-1)[0]),
                np.asarray(vec, dtype=float).reshape(-1),
            ])
        nu = np.asarray([dual[0] for dual in soc_duals], dtype=float)
        mu = (
            np.asarray(self.eq_constraint.dual_value, dtype=float).reshape(-1)
            if self.eq_constraint is not None else None
        )
        lamb = (
            np.asarray(self.ineq_constraint.dual_value, dtype=float).reshape(-1)
            if self.ineq_constraint is not None else None
        )
        if return_soc_duals:
            return x, mu, nu, lamb, soc_duals
        return x, mu, nu, lamb

def gurobi_solve(q,A,b,c,d,F=None,g=None,G=None,h=None,eps=None):
    '''
    To note:
        1. Getting dual from Gurobi can add significant time, documented:
            https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#qcpdual
        2. Need to formulate the cone condition as quadratic condition, but it still solves it as SOCP. 
        3. (Technical) When creating variables, the default lower bound is 0 instead of -inf
    '''
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "The optional Gurobi forward solver requires gurobipy."
        ) from exc

    dim = q.shape[0]
    nSOC = len(A)

    
    m = gp.Model()
    x = m.addMVar(shape=dim,lb=-GRB.INFINITY)
    m.setObjective(q @ x, GRB.MINIMIZE)
    
    
    m.Params.QCPDual = 1 
    cones = []
    for i in range(nSOC):
        A_i,b_i,c_i,d_i = A[i],b[i],c[i],d[i]
        
        y_i = m.addMVar(shape=A_i.shape[0],lb=-GRB.INFINITY)
        z_i = m.addVar()
        
        m.addConstr(z_i == c_i @ x + d_i)
        m.addConstr(y_i == A_i @ x + b_i)
        cone_i = m.addConstr(gp.quicksum(y_i[j] * y_i[j] for j in range(y_i.shape[0])) <= z_i * z_i)
        cones.append(cone_i)
    if F is not None:
        eq_constr = m.addConstr(F @ x == g)

    # === Added Gx <= h constraint ===
    if G is not None and h is not None:
        ineq_constr = m.addConstr(G @ x <= h)
    
    m.optimize()
    x = x.X
    nu = -np.array([cone.QCPi for cone in cones])
    
    # Convert nu of the quadratic condition (norm^2 <= ) to the one of norm condition (norm <=) 
    nu = [nu[i] * (2*np.linalg.norm(A[i]@x+b[i])) for i in range(nSOC)]
    mu = None
    if F is not None:
        mu = -np.array([c.Pi for c in eq_constr])
    # === Dual for Gx <= h ===
    lamb = None
    if G is not None and h is not None:
        lamb = -np.array([c.Pi for c in ineq_constr])
    return x,mu,nu,lamb
