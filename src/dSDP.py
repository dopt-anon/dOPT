#!/usr/bin/env python3
"""
Clean and clear dSDP test: Create SDP → Check unique eigenvalue → Compare solutions and gradients of two methods
"""
import time 
import torch
import numpy as np
import cvxpy as cp
from scipy import sparse as scipy_sparse
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from . import lin_solvers


def torch_csc_to_scipy(tensor, output_format="csc"):
    if tensor.layout not in {torch.sparse_csr, torch.sparse_csc}:
        raise TypeError(f"expected torch sparse CSR/CSC, got {tensor.layout}")
    tensor = tensor.detach().cpu()
    if tensor.layout == torch.sparse_csc:
        matrix = scipy_sparse.csc_matrix(
            (tensor.values().numpy(), tensor.row_indices().numpy(), tensor.ccol_indices().numpy()),
            shape=tuple(tensor.shape),
        )
    else:
        matrix = scipy_sparse.csr_matrix(
            (tensor.values().numpy(), tensor.col_indices().numpy(), tensor.crow_indices().numpy()),
            shape=tuple(tensor.shape),
        )
    return matrix if output_format == "csc" else matrix.tocsr()


def _cvxpy_sparse_value(matrix):
    matrix = matrix.tocoo()
    return scipy_sparse.coo_array(
        (matrix.data, (matrix.row, matrix.col)), shape=matrix.shape
    )


def _pattern_indices(template):
    matrix = torch_csc_to_scipy(template).tocoo()
    return matrix.row, matrix.col


def _same_csc_pattern(matrix, template):
    if matrix.layout != template.layout or tuple(matrix.shape) != tuple(template.shape): return False
    if matrix.layout == torch.sparse_csr:
        return torch.equal(matrix.crow_indices(), template.crow_indices()) and torch.equal(matrix.col_indices(), template.col_indices())
    return torch.equal(matrix.ccol_indices(), template.ccol_indices()) and torch.equal(matrix.row_indices(), template.row_indices())


def _vech_pairs(n):
    rows, cols = [], []
    for row in range(n):
        for col in range(row, n):
            rows.append(row); cols.append(col)
    return np.asarray(rows), np.asarray(cols)


def _sparse_spectral_hessian(Z_inv, S, zero_tol=0.0, chunk_size=256):
    """Build M(kron(Z,S)+kron(S,Z))M.T directly as CSC chunks."""
    Z = Z_inv.detach().cpu().numpy(); S = S.detach().cpu().numpy()
    vr, vc = _vech_pairs(Z.shape[0]); nv = len(vr)
    out_rows, out_cols, out_values = [], [], []
    col_off = vc != vr
    for start in range(0, nv, chunk_size):
        stop = min(start + chunk_size, nv)
        rr, rc = vr[start:stop, None], vc[start:stop, None]
        cr, cc = vr[None, :], vc[None, :]
        block = Z[rr, cr] * S[rc, cc] + S[rr, cr] * Z[rc, cc]
        row_off = rc != rr
        block += row_off * (Z[rc, cr] * S[rr, cc] + S[rc, cr] * Z[rr, cc])
        block += col_off * (Z[rr, cc] * S[rc, cr] + S[rr, cc] * Z[rc, cr])
        block += row_off * col_off * (Z[rc, cc] * S[rr, cr] + S[rc, cc] * Z[rr, cr])
        keep = np.abs(block) > zero_tol
        local_r, local_c = np.nonzero(keep)
        out_rows.append(local_r + start); out_cols.append(local_c)
        out_values.append(block[local_r, local_c])
    return scipy_sparse.csc_matrix(
        (np.concatenate(out_values), (np.concatenate(out_rows), np.concatenate(out_cols))),
        shape=(nv, nv),
    )


_VECH_TORCH_INDEX_CACHE = {}


def _torch_vech_indices(n, device):
    """Return cached upper-triangular index pairs used by direct H_vech assembly."""
    key = (n, str(device))
    if key not in _VECH_TORCH_INDEX_CACHE:
        rows, cols = torch.triu_indices(n, n, device=device)
        _VECH_TORCH_INDEX_CACHE[key] = (rows, cols, rows != cols)
    return _VECH_TORCH_INDEX_CACHE[key]


def _direct_vech_spectral_hessian(Z_inv, S):
    """Build M(kron(Z,S)+kron(S,Z))M.T directly in vech space.

    This avoids materializing the much larger n^2-by-n^2 Kronecker matrix.
    """
    n = Z_inv.shape[0]
    vr, vc, offdiag = _torch_vech_indices(n, Z_inv.device)

    row_i, row_j = vr[:, None], vc[:, None]
    col_k, col_l = vr[None, :], vc[None, :]
    row_offdiag = offdiag[:, None]
    col_offdiag = offdiag[None, :]

    hessian_vech = (
        Z_inv[row_i, col_k] * S[row_j, col_l]
        + S[row_i, col_k] * Z_inv[row_j, col_l]
    )
    hessian_vech = hessian_vech + row_offdiag * (
        Z_inv[row_j, col_k] * S[row_i, col_l]
        + S[row_j, col_k] * Z_inv[row_i, col_l]
    )
    hessian_vech = hessian_vech + col_offdiag * (
        Z_inv[row_i, col_l] * S[row_j, col_k]
        + S[row_i, col_l] * Z_inv[row_j, col_k]
    )
    hessian_vech = hessian_vech + row_offdiag * col_offdiag * (
        Z_inv[row_j, col_l] * S[row_i, col_k]
        + S[row_j, col_l] * Z_inv[row_i, col_k]
    )

    return hessian_vech


def _direct_vech_spectral_hessian_as_sparse(Z_inv, S, zero_tol=0.0):
    """Build the direct dense vech Hessian, then convert it once to CSC."""
    hessian_np = _direct_vech_spectral_hessian(
        Z_inv, S
    ).detach().cpu().numpy()
    if zero_tol > 0.0:
        hessian_np[np.abs(hessian_np) <= zero_tol] = 0.0
    return scipy_sparse.csc_matrix(hessian_np)


def _matrix_free_kkt_operator(Z_inv, S, C_block, P=None):
    """Return the KKT LinearOperator without materializing H or the KKT matrix."""
    Z = Z_inv.detach().cpu().numpy()
    dual_slack = S.detach().cpu().numpy()
    n = Z.shape[0]
    vech_rows, vech_cols = _vech_pairs(n)
    offdiag = vech_rows != vech_cols
    p, q = C_block.shape

    def apply_M_transpose(x):
        matrix = np.zeros((n, n), dtype=x.dtype)
        matrix[vech_rows, vech_cols] = x
        matrix[vech_cols[offdiag], vech_rows[offdiag]] = x[offdiag]
        return matrix

    def apply_M(matrix):
        result = matrix[vech_rows, vech_cols].copy()
        result[offdiag] += matrix[vech_cols[offdiag], vech_rows[offdiag]]
        return result

    def hessian_matvec(x):
        matrix = apply_M_transpose(x)
        output = Z @ matrix @ dual_slack + dual_slack @ matrix @ Z
        result = apply_M(output)
        if P is not None:
            result += apply_M((P @ matrix.reshape(-1)).reshape(n, n))
        return result

    def kkt_matvec(vector):
        primal = vector[:p]
        dual = vector[p:]
        return np.concatenate((
            hessian_matvec(primal) - C_block @ dual,
            -(C_block.T @ primal),
        ))

    return scipy_sparse.linalg.LinearOperator(
        (p + q, p + q),
        matvec=kkt_matvec,
        rmatvec=kkt_matvec,
        dtype=np.result_type(Z.dtype, dual_slack.dtype),
    )


def _matrix_free_diagonal_preconditioner(
    Z_inv,
    S,
    C_block,
    P=None,
    relative_shift=1e-6,
    scheme="block-diagonal",
):
    """Build an inexpensive SPD diagonal preconditioner.

    The returned LinearOperator applies the inverse of the SPD block-diagonal
    preconditioner expected by SciPy MINRES.
    """
    Z = Z_inv.detach().cpu().numpy()
    dual_slack = S.detach().cpu().numpy()
    n = Z.shape[0]
    rows, cols = _vech_pairs(n)
    offdiag = rows != cols

    hessian_diagonal = (
        Z[rows, rows] * dual_slack[cols, cols]
        + dual_slack[rows, rows] * Z[cols, cols]
    )
    if np.any(offdiag):
        r, c = rows[offdiag], cols[offdiag]
        hessian_diagonal[offdiag] += 2.0 * (
            Z[c, r] * dual_slack[r, c]
            + dual_slack[c, r] * Z[r, c]
        )
        hessian_diagonal[offdiag] += (
            Z[c, c] * dual_slack[r, r]
            + dual_slack[c, c] * Z[r, r]
        )

    if P is not None:
        first = rows * n + cols
        second = cols * n + rows
        hessian_diagonal += np.asarray(P[first, first]).reshape(-1)
        hessian_diagonal[offdiag] += (
            np.asarray(P[first[offdiag], second[offdiag]]).reshape(-1)
            + np.asarray(P[second[offdiag], first[offdiag]]).reshape(-1)
            + np.asarray(P[second[offdiag], second[offdiag]]).reshape(-1)
        )

    hessian_scale = max(float(np.max(np.abs(hessian_diagonal))), 1.0)
    hessian_floor = relative_shift * hessian_scale
    hessian_diagonal = np.maximum(np.abs(hessian_diagonal), hessian_floor)

    if scheme == "block-diagonal":
        inverse_hessian_diagonal = 1.0 / hessian_diagonal
        dual_diagonal = np.asarray(
            C_block.power(2).T @ inverse_hessian_diagonal
        ).reshape(-1)
    elif scheme == "kkt-equilibration":
        squared_C = C_block.power(2)
        primal_C_norm = np.sqrt(np.asarray(squared_C.sum(axis=1)).reshape(-1))
        dual_diagonal = np.sqrt(np.asarray(squared_C.sum(axis=0)).reshape(-1))
        hessian_diagonal = np.sqrt(hessian_diagonal**2 + primal_C_norm**2)
    else:
        raise ValueError(f"unknown diagonal preconditioner scheme: {scheme}")

    dual_scale = max(float(np.max(np.abs(dual_diagonal))), 1.0)
    dual_floor = relative_shift * dual_scale
    dual_diagonal = np.maximum(np.abs(dual_diagonal), dual_floor)

    p = len(hessian_diagonal)

    def apply_inverse(vector):
        return np.concatenate((
            vector[:p] / hessian_diagonal,
            vector[p:] / dual_diagonal,
        ))

    operator = scipy_sparse.linalg.LinearOperator(
        (p + len(dual_diagonal), p + len(dual_diagonal)),
        matvec=apply_inverse,
        rmatvec=apply_inverse,
        dtype=np.result_type(Z.dtype, dual_slack.dtype),
    )
    diagnostics = {
        "hessian_diagonal_min": float(np.min(hessian_diagonal)),
        "hessian_diagonal_max": float(np.max(hessian_diagonal)),
        "dual_diagonal_min": float(np.min(dual_diagonal)),
        "dual_diagonal_max": float(np.max(dual_diagonal)),
    }
    return operator, diagnostics


def _sparse_U(eigenvectors, active_set, n, zero_tol=0.0):
    vr, vc = _vech_pairs(n); columns = []
    eig = eigenvectors.detach().cpu().numpy()
    active = [int(i) for i in active_set]
    for left_index, left in enumerate(active):
        for right in active[left_index:]:
            u, v = eig[:, left], eig[:, right]
            values = u[vr] * v[vc]
            off = vr != vc
            values[off] += u[vc[off]] * v[vr[off]]
            values[np.abs(values) <= zero_tol] = 0.0
            columns.append(scipy_sparse.csc_matrix(values[:, None]))
    return scipy_sparse.hstack(columns, format="csc")


def _dense_U(eigenvectors, active_set):
    """Vectorized complementarity basis in local vech coordinates."""
    n = eigenvectors.shape[0]
    rows, cols = torch.triu_indices(n, n, device=eigenvectors.device)
    active_vectors = eigenvectors[:, active_set]
    if active_vectors.shape[1] == 0:
        return torch.empty(
            (len(rows), 0), dtype=eigenvectors.dtype, device=eigenvectors.device
        )
    left, right = torch.triu_indices(
        active_vectors.shape[1], active_vectors.shape[1],
        device=eigenvectors.device,
    )
    u = active_vectors[:, left]
    v = active_vectors[:, right]
    values = u[rows, :] * v[cols, :]
    offdiag = rows != cols
    values[offdiag, :] += u[cols[offdiag], :] * v[rows[offdiag], :]
    return values


def _normalize_psd_blocks(psd_blocks, n):
    normalized = []
    covered = []
    for block in psd_blocks:
        start, stop = (block.start, block.stop) if isinstance(block, slice) else block
        if start is None or stop is None or not (0 <= start < stop <= n):
            raise ValueError(f"invalid PSD block: {block}")
        normalized.append((int(start), int(stop)))
        covered.extend(range(start, stop))
    if sorted(covered) != list(range(n)) or len(set(covered)) != n:
        raise ValueError("psd_blocks must be non-overlapping and cover the matrix")
    return normalized


def _solve_block_reduced_kkt(
    Z_star,
    S_star,
    A_matrix,
    grad_Z,
    psd_blocks,
    eps_active,
    M_transform,
    P=None,
    linear_solver="dense",
    minres_rtol=1e-8,
    minres_maxiter=None,
):
    """Solve the backward KKT in concatenated block-vech coordinates."""
    if P is not None:
        raise NotImplementedError("block-reduced backward does not yet support P")
    n = Z_star.shape[0]
    blocks = _normalize_psd_blocks(psd_blocks, n)
    global_rows, global_cols = _vech_pairs(n)
    global_pair_to_index = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(zip(global_rows, global_cols))
    }

    block_indices = []
    block_sizes = []
    hessian_blocks = []
    U_blocks = []
    operator_blocks = []
    active_counts = []
    for start, stop in blocks:
        size = stop - start
        local_rows, local_cols = _vech_pairs(size)
        indices = torch.as_tensor(
            [
                global_pair_to_index[(start + int(row), start + int(col))]
                for row, col in zip(local_rows, local_cols)
            ],
            dtype=torch.long,
            device=Z_star.device,
        )
        block_indices.append(indices)
        block_sizes.append(len(indices))

        Z_block = Z_star[start:stop, start:stop]
        S_block = S_star[start:stop, start:stop]
        eigenvalues, eigenvectors = torch.linalg.eigh(Z_block)
        active = torch.nonzero(eigenvalues < eps_active, as_tuple=True)[0]
        inactive = torch.nonzero(eigenvalues >= eps_active, as_tuple=True)[0]
        active_counts.append(len(active))

        if len(active) > 0:
            if len(inactive) > 0:
                inactive_vectors = eigenvectors[:, inactive]
                Z_inverse = (
                    inactive_vectors
                    @ torch.diag(1.0 / eigenvalues[inactive])
                    @ inactive_vectors.T
                )
            else:
                Z_inverse = torch.zeros_like(Z_block)
            hessian_blocks.append(
                _direct_vech_spectral_hessian(Z_inverse, S_block)
            )
            U_blocks.append(_dense_U(eigenvectors, active))
        else:
            p_block = size * (size + 1) // 2
            hessian_blocks.append(
                torch.zeros((p_block, p_block), dtype=Z_star.dtype, device=Z_star.device)
            )
            U_blocks.append(
                torch.empty((p_block, 0), dtype=Z_star.dtype, device=Z_star.device)
            )
            Z_inverse = torch.zeros_like(Z_block)
        operator_blocks.append((Z_inverse, S_block, size))

    selected_indices = torch.cat(block_indices)
    selected_rows = torch.as_tensor(
        global_rows[selected_indices.detach().cpu().numpy()],
        dtype=torch.long,
        device=Z_star.device,
    )
    selected_cols = torch.as_tensor(
        global_cols[selected_indices.detach().cpu().numpy()],
        dtype=torch.long,
        device=Z_star.device,
    )
    H_reduced = torch.block_diag(*hessian_blocks)
    total_complementarity = sum(block.shape[1] for block in U_blocks)
    U_reduced = torch.zeros(
        (len(selected_indices), total_complementarity),
        dtype=Z_star.dtype,
        device=Z_star.device,
    )
    row_offset = 0
    col_offset = 0
    for p_block, U_block in zip(block_sizes, U_blocks):
        U_reduced[
            row_offset:row_offset + p_block,
            col_offset:col_offset + U_block.shape[1],
        ] = U_block
        row_offset += p_block
        col_offset += U_block.shape[1]

    if A_matrix.layout in {torch.sparse_csr, torch.sparse_csc}:
        A_vech_full = torch.sparse.mm(A_matrix, M_transform.T)
        A_reduced = A_vech_full[:, selected_indices]
    else:
        primary = selected_rows * n + selected_cols
        mirrored = selected_cols * n + selected_rows
        A_reduced = A_matrix[:, primary]
        off_diagonal = selected_rows != selected_cols
        A_reduced = A_reduced + (
            off_diagonal.to(A_matrix.dtype)[None, :]
            * A_matrix[:, mirrored]
        )
    grad_Z_reduced = grad_Z[selected_rows, selected_cols]
    grad_Z_reduced = grad_Z_reduced + (
        (selected_rows != selected_cols).to(grad_Z.dtype)
        * grad_Z[selected_cols, selected_rows]
    )
    C_reduced = torch.cat((U_reduced, A_reduced.T), dim=1)
    q = C_reduced.shape[1]
    KKT = torch.cat((
        torch.cat((H_reduced, -C_reduced), dim=1),
        torch.cat((
            -C_reduced.T,
            torch.zeros((q, q), dtype=Z_star.dtype, device=Z_star.device),
        ), dim=1),
    ), dim=0)
    rhs = torch.cat((
        grad_Z_reduced,
        torch.zeros(q, dtype=Z_star.dtype, device=Z_star.device),
    ))
    minres_stats = None
    if linear_solver == "scipy MINRES matrix-free":
        C_numpy = C_reduced.detach().cpu().numpy()
        block_offsets = np.cumsum([0, *block_sizes])

        def block_hessian_matvec(primal_numpy):
            outputs = []
            for block_index, (Z_inverse, S_block, size) in enumerate(operator_blocks):
                start = block_offsets[block_index]
                stop = block_offsets[block_index + 1]
                local_vector = primal_numpy[start:stop]
                rows, cols = np.triu_indices(size)
                offdiag = rows != cols
                local_matrix = np.zeros((size, size), dtype=primal_numpy.dtype)
                local_matrix[rows, cols] = local_vector
                local_matrix[cols[offdiag], rows[offdiag]] = local_vector[offdiag]
                Z_numpy = Z_inverse.detach().cpu().numpy()
                S_numpy = S_block.detach().cpu().numpy()
                output_matrix = (
                    Z_numpy @ local_matrix @ S_numpy
                    + S_numpy @ local_matrix @ Z_numpy
                )
                output = output_matrix[rows, cols].copy()
                output[offdiag] += output_matrix[cols[offdiag], rows[offdiag]]
                outputs.append(output)
            return np.concatenate(outputs)

        def kkt_matvec(vector):
            primal_numpy = vector[:len(selected_indices)]
            dual_numpy = vector[len(selected_indices):]
            return np.concatenate((
                block_hessian_matvec(primal_numpy) - C_numpy @ dual_numpy,
                -(C_numpy.T @ primal_numpy),
            ))

        operator = scipy_sparse.linalg.LinearOperator(
            (len(rhs), len(rhs)),
            matvec=kkt_matvec,
            rmatvec=kkt_matvec,
            dtype=np.float64,
        )
        solution_numpy, minres_stats = lin_solvers.matrix_free_minres(
            operator,
            rhs.detach().cpu().numpy(),
            rtol=minres_rtol,
            maxiter=minres_maxiter,
        )
        if minres_stats["info"] != 0:
            raise RuntimeError(
                "block matrix-free MINRES did not converge: "
                f"info={minres_stats['info']}, "
                f"iterations={minres_stats['iterations']}, "
                f"relative_residual={minres_stats['relative_residual']:.3e}"
            )
        solution = torch.as_tensor(
            solution_numpy, dtype=Z_star.dtype, device=Z_star.device
        )
    elif linear_solver in lin_solvers.get_sparse_solvers():
        KKT_sparse = scipy_sparse.csc_matrix(KKT.detach().cpu().numpy())
        solution_numpy, _ = lin_solvers.sparse_solve(
            KKT_sparse,
            rhs.detach().cpu().numpy(),
            linear_solver=linear_solver,
        )
        solution = torch.as_tensor(
            np.asarray(solution_numpy).reshape(-1),
            dtype=Z_star.dtype,
            device=Z_star.device,
        )
    else:
        try:
            solution = torch.linalg.solve(KKT, rhs)
        except torch.linalg.LinAlgError:
            # Degenerate SDP optima can make the sensitivity KKT singular.
            # Use the minimum-norm implicit derivative in that case.
            solution = torch.linalg.lstsq(KKT, rhs).solution

    primal = solution[:len(selected_indices)]
    dZ = torch.zeros_like(Z_star)
    offset = 0
    for (start, stop), p_block in zip(blocks, block_sizes):
        size = stop - start
        rows, cols = torch.triu_indices(size, size, device=Z_star.device)
        values = -primal[offset:offset + p_block]
        local = torch.zeros((size, size), dtype=Z_star.dtype, device=Z_star.device)
        local[rows, cols] = values
        local[cols, rows] = values
        dZ[start:stop, start:stop] = local
        offset += p_block

    dual_offset = len(selected_indices) + total_complementarity
    dnu = -solution[dual_offset:dual_offset + A_matrix.shape[0]]
    diagnostics = {
        "block_primal_dim": len(selected_indices),
        "equality_constraints": A_matrix.shape[0],
        "complementarity_constraints": total_complementarity,
        "kkt_dim": len(selected_indices) + q,
        "active_counts": active_counts,
        "minres_stats": minres_stats,
    }
    return dZ, dnu, diagnostics


class CVXPYSDPProblem:
    """A parameterized CVXPY problem reused by one dSDP layer."""

    def __init__(self, n, m, has_quadratic_term=False, mode="dense", sparsity=None,
                 psd_blocks=None):
        self.n = n
        self.m = m
        self.has_quadratic_term = has_quadratic_term
        self.mode = mode

        self.psd_blocks = (
            _normalize_psd_blocks(psd_blocks, n)
            if psd_blocks is not None else None
        )
        if self.psd_blocks is not None:
            if mode != "dense":
                raise NotImplementedError(
                    "block-cone CVXPY forward currently supports dense mode only"
                )
            if has_quadratic_term:
                raise NotImplementedError(
                    "block-cone CVXPY forward currently supports linear objectives only"
                )
            self.block_columns = np.asarray([
                row + col * n
                for start, stop in self.psd_blocks
                for col in range(start, stop)
                for row in range(start, stop)
            ], dtype=np.int64)
            self.block_variables = [
                cp.Variable((stop - start, stop - start), symmetric=True)
                for start, stop in self.psd_blocks
            ]
            block_vector = cp.hstack([
                cp.vec(variable, order="F") for variable in self.block_variables
            ])
            reduced_n = len(self.block_columns)
            self.C_parameter = cp.Parameter(reduced_n)
            self.A_parameter = cp.Parameter((m, reduced_n))
            self.b_parameter = cp.Parameter(m)
            self.P_parameter = None
            self.psd_constraints = [
                variable >> 0 for variable in self.block_variables
            ]
            self.equality_constraint = (
                self.A_parameter @ block_vector == self.b_parameter
            )
            self.constraints = [*self.psd_constraints, self.equality_constraint]
            self.problem = cp.Problem(
                cp.Minimize(self.C_parameter @ block_vector),
                self.constraints,
            )
            return

        self.Z = cp.Variable((n, n), symmetric=True)
        if mode == "sparse":
            if sparsity is None or "C" not in sparsity or "A" not in sparsity:
                raise ValueError("sparse mode requires C and A templates")
            self.C_parameter = cp.Parameter((n, n), sparsity=_pattern_indices(sparsity["C"]))
            self.A_parameter = cp.Parameter((m, n*n), sparsity=_pattern_indices(sparsity["A"]))
        else:
            self.C_parameter = cp.Parameter((n, n), symmetric=True)
            self.A_parameter = cp.Parameter((m, n * n))
        self.b_parameter = cp.Parameter(m)
        self.P_parameter = None

        vec_Z = cp.vec(self.Z, order="F")
        self.constraints = [
            self.Z >> 0,
            self.A_parameter @ vec_Z == self.b_parameter,
        ]
        C_expr = self.C_parameter if mode == "dense" else 0.5*(self.C_parameter+self.C_parameter.T)
        objective_expression = cp.trace(C_expr @ self.Z)
        if has_quadratic_term:
            if mode == "sparse":
                if "P" not in sparsity:
                    raise ValueError("sparse quadratic mode requires a P template")
                self.P_parameter = cp.Parameter((n*n,n*n), sparsity=_pattern_indices(sparsity["P"]))
                P_expr = cp.psd_wrap(0.5*(self.P_parameter+self.P_parameter.T))
            else:
                self.P_parameter = cp.Parameter((n * n, n * n), PSD=True)
                P_expr = self.P_parameter
            objective_expression += 0.5 * cp.quad_form(
                vec_Z,
                P_expr,
            )
        self.problem = cp.Problem(
            cp.Minimize(objective_expression),
            self.constraints,
        )
    def set_values(self, C, A_matrix, b, P=None):
        if self.psd_blocks is not None:
            if P is not None:
                raise ValueError("block-cone cached problem does not accept P")
            C_dense = np.asarray(C)
            A_dense = np.asarray(A_matrix)
            self.C_parameter.value = C_dense.reshape(-1, order="F")[self.block_columns]
            self.A_parameter.value = A_dense[:, self.block_columns]
            self.b_parameter.value = np.asarray(b)
            return
        if self.mode == "sparse":
            self.C_parameter.value_sparse = _cvxpy_sparse_value(C)
            self.A_parameter.value_sparse = _cvxpy_sparse_value(A_matrix)
        else:
            self.C_parameter.value = 0.5 * (C + C.T)
            self.A_parameter.value = A_matrix
        self.b_parameter.value = b
        if self.has_quadratic_term:
            if P is None:
                raise ValueError("cached SDP expects a quadratic objective matrix P")
            if self.mode == "sparse":
                self.P_parameter.value_sparse = _cvxpy_sparse_value(P)
            else:
                self.P_parameter.value = 0.5 * (P + P.T)
        elif P is not None:
            raise ValueError("cached linear SDP does not accept a quadratic term P")


def solve_sdp_cvxpy(
    C,
    A_matrix,
    b,
    P=None,
    solver=None,
    cached_problem=None,
    **solver_args,
):
    """Solve an SDP, using an already-set-up problem when provided."""
    if not scipy_sparse.issparse(C): C = np.asarray(C)
    if not scipy_sparse.issparse(A_matrix): A_matrix = np.asarray(A_matrix)
    b = np.asarray(b)
    if P is not None and not scipy_sparse.issparse(P): P = np.asarray(P)
    n = C.shape[0]
    m = b.shape[0]
    if C.shape != (n, n):
        raise ValueError("C must be square")
    if A_matrix.shape != (m, n * n):
        raise ValueError(
            f"A_matrix must have shape {(m, n * n)}, got {A_matrix.shape}"
        )
    if P is not None and P.shape != (n * n, n * n):
        raise ValueError(
            f"P must have shape {(n * n, n * n)}, got {P.shape}"
        )

    if cached_problem is None:
        cached_problem = CVXPYSDPProblem(
            n, m, has_quadratic_term=P is not None
        )
    elif (
        cached_problem.n != n
        or cached_problem.m != m
        or cached_problem.has_quadratic_term != (P is not None)
    ):
        raise ValueError("cached problem does not match the current SDP")
    cached_problem.set_values(C, A_matrix, b, P)

    if solver is None:
        solver = cp.SCS
    cached_problem.problem.solve(solver=solver, verbose=False, **solver_args)
    if cached_problem.problem.status == cp.OPTIMAL:
        if cached_problem.psd_blocks is None:
            Z_star = cached_problem.Z.value
            S_star = cached_problem.constraints[0].dual_value
            nu_star = torch.as_tensor(cached_problem.constraints[1].dual_value)
        else:
            Z_star = np.zeros((n, n))
            S_star = np.zeros((n, n))
            for (start, stop), variable, constraint in zip(
                cached_problem.psd_blocks,
                cached_problem.block_variables,
                cached_problem.psd_constraints,
            ):
                Z_star[start:stop, start:stop] = variable.value
                S_star[start:stop, start:stop] = constraint.dual_value
            nu_star = torch.as_tensor(
                cached_problem.equality_constraint.dual_value
            )
        return Z_star, S_star, nu_star, cached_problem.problem.status
    return None, None, None, cached_problem.problem.status


def extract_dual_eigenvalues_from_S(Z_star, S_star, eps_active=1e-6):
    """
    Extract dual eigenvalues (μ*) from slack matrix S*
    """
    # Step 1: Eigendecomposition of Z*
    # TODO: only need eigen-decomp for 0 eigenvalues 
    eigenvals_Z, eigenvecs_Z = torch.linalg.eigh(Z_star)
    
    # Step 2: Find active set J = {j : λⱼ = 0} (zero eigenvalues)
    active_set = eigenvals_Z < eps_active
    J = active_set.nonzero(as_tuple=True)[0]
    U_J = eigenvecs_Z[:,J]
    S_eigblk_with_rot = U_J.T @ S_star @ U_J
    _,R = torch.linalg.eigh(S_eigblk_with_rot)
    
    S_J_eigvec = U_J @ R 
    mu_star = torch.diag(S_J_eigvec.T @ S_star @  S_J_eigvec)
    # # Step 3: For zero eigenvalue eigenvectors, compute μⱼ* = uⱼᵀ S* uⱼ
    # mu_star = torch.zeros_like(eigenvals_Z)
    
    # for j in range(len(eigenvals_Z)):
    #     if j in J:  
    #         u_j = eigenvecs_Z[:, j]
    #         mu_j_star = u_j @ S_star @ u_j

    #         mu_star[j] = mu_j_star
    
    return mu_star, J, eigenvals_Z, eigenvecs_Z

def build_settings(solve_type="dense",eps_active=1e-5,solver=None,lin_solver="scipy SPLU"):
    settings = {
        "solve_type" : solve_type,
        "solver": solver,
        "lin_solver": lin_solver,
        "eps_active": eps_active,
        # "eps_abs": eps_active,
        # "eps_rel": eps_active,
        # "lin_solver": lin_solver,
        }
    return settings

def precompute_M_transform(n, device=None, dtype=None):
    """Precompute M_transform matrix for vech = M @ vec transformation
    
    Args:
        n: Matrix dimension
        device: torch device 
        dtype: torch dtype
        
    Returns:
        M_transform: [n_vech × n_vec] transformation matrix
    """
    n_vec = n * n
    n_vech = n * (n + 1) // 2
    
    M_transform = torch.zeros(n_vech, n_vec, device=device, dtype=dtype)
    
    vech_idx = 0
    for i in range(n):
        for j in range(i, n):  # Upper triangular
            vec_idx = i * n + j  # (i,j) position in vec format
            M_transform[vech_idx, vec_idx] = 1.0
            
            if i != j:  # Add symmetric part
                vec_idx_sym = j * n + i  # (j,i) position in vec format
                M_transform[vech_idx, vec_idx_sym] = 1.0
            
            vech_idx += 1
    
    return M_transform

def precompute_T_inv(n, device=None, dtype=None):
    """Precompute T_inv matrix for vec = T_inv @ vech transformation (vech_to_matrix)
    
    Args:
        n: Matrix dimension
        device: torch device 
        dtype: torch dtype
        
    Returns:
        T_inv: [n_vec × n_vech] inverse transformation matrix
    """
    n_vec = n * n
    n_vech = n * (n + 1) // 2
    
    T_inv = torch.zeros(n_vec, n_vech, device=device, dtype=dtype)
    
    vech_idx = 0
    for i in range(n):
        for j in range(i, n):  # Upper triangular
            vec_idx = i * n + j  # (i,j) position in vec format
            T_inv[vec_idx, vech_idx] = 1.0
            
            if i != j:  # Add symmetric part
                vec_idx_sym = j * n + i  # (j,i) position in vec format
                T_inv[vec_idx_sym, vech_idx] = 1.0
            
            vech_idx += 1
    
    return T_inv

def efficient_vec_to_vech_transform(H, n):
    """
    Efficiently transform H from vec space (n²×n²) to vech space (n_vech×n_vech)
    using vectorized operations with masks.
    
    Equivalent to: M @ H @ M.T where M is the vech transformation matrix,
    but avoids the expensive matrix multiplications.
    
    Args:
        H: [n² × n²] matrix in vectorized space
        n: Matrix dimension
        
    Returns:
        H_vech: [n_vech × n_vech] matrix in half-vectorized space
    """
    n_vech = n * (n + 1) // 2
    device, dtype = H.device, H.dtype
    
    # Create index mappings for vech space (i,j) where i <= j
    vech_i_indices = []
    vech_j_indices = []
    for i in range(n):
        for j in range(i, n):
            vech_i_indices.append(i)
            vech_j_indices.append(j)
    
    vech_i = torch.tensor(vech_i_indices, device=device, dtype=torch.long)
    vech_j = torch.tensor(vech_j_indices, device=device, dtype=torch.long)
    
    # Create diagonal masks (where i == j)
    diag_mask_row = (vech_i == vech_j)  # [n_vech]
    diag_mask_col = diag_mask_row  # Same for columns
    
    # Compute vec indices for each vech position
    # For vech element at (i,j), vec positions are i*n+j and j*n+i
    vec_idx_1 = vech_i * n + vech_j  # Primary index: i*n+j
    vec_idx_2 = vech_j * n + vech_i  # Symmetric index: j*n+i
    
    # Extract 4 blocks from H corresponding to all symmetric combinations
    # H_vech[r,c] = H[vec_r1, vec_c1] + H[vec_r2, vec_c1] + H[vec_r1, vec_c2] + H[vec_r2, vec_c2]
    # where vec_r1, vec_r2 are the two vec indices for vech row r
    # and vec_c1, vec_c2 are the two vec indices for vech column c
    
    # Use advanced indexing to extract all elements at once
    # Shape: [n_vech, n_vech] for each block
    block_11 = H[vec_idx_1[:, None], vec_idx_1[None, :]]  # H[i*n+j, k*n+l]
    block_12 = H[vec_idx_1[:, None], vec_idx_2[None, :]]  # H[i*n+j, l*n+k]
    block_21 = H[vec_idx_2[:, None], vec_idx_1[None, :]]  # H[j*n+i, k*n+l]
    block_22 = H[vec_idx_2[:, None], vec_idx_2[None, :]]  # H[j*n+i, l*n+k]
    
    # Create masks for when to include each block
    # block_21 should be added only when row is not diagonal (i != j)
    # block_12 should be added only when col is not diagonal (k != l)
    # block_22 should be added only when both are not diagonal
    
    mask_row_offdiag = (~diag_mask_row).float()[:, None]  # [n_vech, 1]
    mask_col_offdiag = (~diag_mask_col).float()[None, :]  # [1, n_vech]
    
    # Sum all contributions with appropriate masks
    H_vech = (block_11 + 
              block_21 * mask_row_offdiag + 
              block_12 * mask_col_offdiag + 
              block_22 * mask_row_offdiag * mask_col_offdiag)
    
    return H_vech

# dSDP core class with optimized M_transform precomputation
class dSDPFunction(torch.autograd.Function):
    """
    Differentiable SDP solver with optimized vech transformation precomputation.
    
    Key optimization: Both M_transform and T_inv matrices can be precomputed outside 
    the model and passed as input parameters, eliminating all computation during 
    forward/backward passes. This provides maximum speedup by replacing expensive 
    match_vech and vech_to_matrix operations with fast matrix multiplications.
    
    Usage:
        # Precompute transformation matrices once
        M_transform = precompute_M_transform(n, device, dtype)
        T_inv = precompute_T_inv(n, device, dtype)
        transforms = (M_transform, T_inv)
        
        # Use in multiple forward/backward passes
        layer = dSDPLayer(n, m, settings=settings, transforms=transforms)
        result = layer(C, A_matrix, b, P)
    """
    # Class-level cache for transformation matrices
    _M_transform_cache = {}
    _T_inv_cache = {}
    
    @staticmethod
    def get_M_transform(n, device=None, dtype=None):
        """Get or create cached vech transformation matrix M"""
        cache_key = (n, str(device), str(dtype))
        if cache_key not in dSDPFunction._M_transform_cache:
            M_transform = precompute_M_transform(n, device, dtype)
            dSDPFunction._M_transform_cache[cache_key] = M_transform
        
        return dSDPFunction._M_transform_cache[cache_key]
    
    @staticmethod
    def get_T_inv_transform(n, device=None, dtype=None):
        """Get or create cached vech transformation matrix M"""
        cache_key = (n, str(device), str(dtype))
        if cache_key not in dSDPFunction._T_inv_cache:
            T_inv = precompute_T_inv(n, device, dtype)
            dSDPFunction._T_inv_cache[cache_key] = T_inv

        return dSDPFunction._T_inv_cache[cache_key]


    
    @staticmethod
    def forward(
        ctx,
        C,
        A_matrix,
        b,
        P=None,
        settings=None,
        transforms=None,
        cached_problem=None,
    ):
        # A_matrix is m x n^2 matrix representing constraints A*vec(Z) = b
        # P is optional n^2 x n^2 matrix for quadratic term
        # transforms is optional tuple (M_transform, T_inv) of precomputed matrices
        n = C.shape[0]
        m = b.shape[0]
        device, dtype = C.device, C.dtype
        
        # Use provided transforms or compute if not provided
        if transforms is None:
            M_transform = dSDPFunction.get_M_transform(n, device, dtype)
            T_inv = dSDPFunction.get_T_inv_transform(n, device, dtype)
        else:
            M_transform, T_inv = transforms
            # Ensure transforms are on the correct device
            if M_transform.device != device:
                M_transform = M_transform.to(device=device, dtype=dtype)
            if T_inv.device != device:
                T_inv = T_inv.to(device=device, dtype=dtype)
        
        if settings["solve_type"] == "sparse":
            C_np = torch_csc_to_scipy(C)
            A_np = torch_csc_to_scipy(A_matrix)
            P_np = torch_csc_to_scipy(P) if P is not None else None
        else:
            C_np = C.detach().cpu().numpy()
            A_np = A_matrix.detach().cpu().numpy()
            P_np = P.detach().cpu().numpy() if P is not None else None
        b_np = b.detach().cpu().numpy()
        
        # Solve using CVXPY, get dual information
        solver_args = settings.get('solver_args', {})
        Z_star, S_star, nu_star, status = solve_sdp_cvxpy(
            C_np,
            A_np,
            b_np,
            P_np,
            solver=settings.get('solver', None),
            cached_problem=cached_problem,
            **solver_args,
        )
        
        if Z_star is None:
            raise RuntimeError(f"SDP solver failed with status: {status}")
        
        # Convert back to torch with correct device and dtype
        Z_star = torch.tensor(Z_star, device=device, dtype=dtype)
        S_star = torch.tensor(S_star, device=device, dtype=dtype) 
        nu_star = nu_star.to(device=device, dtype=dtype)
        
        if not S_star.is_contiguous():
            S_star = S_star.contiguous()
            
        if not Z_star.is_contiguous():
            Z_star = Z_star.contiguous()
    
        # Spectral information is computed lazily in backward. In block mode
        # this avoids a full lifted eigendecomposition and lets each PSD cone
        # be decomposed independently.
        ctx.save_for_backward(
            Z_star, S_star, nu_star, C, A_matrix, P, M_transform, T_inv
        )
           
        ctx.b = b
        ctx.settings = settings
        ctx.n = C.shape[0]
        ctx.m = b.shape[0]
        
        return Z_star, S_star, nu_star
    
    # @profile
    @staticmethod
    def backward(ctx, grad_Z, grad_S, grad_nu):
        backward_start = time.perf_counter()
        profile_backward = ctx.settings.get("profile_backward", False)
        # Handle both cases: with and without precomputed T_inv
        Z_star, S_star, nu_star, C, A_matrix, P, M_transform, T_inv = ctx.saved_tensors
        
            
        b = ctx.b
        eps_active = ctx.settings['eps_active']
        n = ctx.n
        m = ctx.m
        
        verbose = ctx.settings.get("verbose", False)
        if verbose:
            print("Starting gradient computation...")

       
        
        # Use precomputed transformation matrices from context
        device, dtype = Z_star.device, Z_star.dtype
        
        # Vectorization helper functions
        def vec(X):
            return X.flatten()
        
        def fast_match_vech(M_matrix):
            """Fast vech transformation using precomputed matrix M_transform"""
            if M_matrix.dim() == 2 and M_matrix.shape[1] == n * n:
                # Multiple constraints case: m × n²
                return M_matrix @ M_transform.T  # Result: m × n_vech
            elif M_matrix.dim() == 1:
                # Single vector case
                return M_transform @ M_matrix  # Result: n_vech
            else:
                # Single matrix flattened case
                return M_transform @ M_matrix.flatten()
        

        def vech(matrix):
            """Extract vech (half-vectorization) from symmetric matrix
            Returns upper triangular elements including diagonal as a vector
            """
            n = matrix.shape[-1]
            # Create mask for upper triangular elements (including diagonal)
            vech_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=0)
            
            if matrix.dim() == 2:
                # Single matrix
                return matrix[vech_mask]
            elif matrix.dim() == 3:
                # Batch of matrices
                return matrix[:, vech_mask]
            else:
                raise ValueError(f"Unsupported matrix dimension: {matrix.dim()}")
            
        

        # Use precomputed T_inv if available, otherwise compute it
        if T_inv is None:
            T_inv = precompute_T_inv(n, device, dtype)
        
        def vech_to_mat(vech_vec):
            """Convert vech vector back to n×n symmetric matrix using precomputed matrix multiplication"""
            # Fast matrix multiplication: vec = T_inv @ vech
            matrix_vec = T_inv @ vech_vec
            # Reshape to n×n matrix
            return matrix_vec.view(n, n)   
             
        n_vec = n * n
        n_vech = n * (n + 1) // 2

        psd_blocks = ctx.settings.get("psd_blocks")
        if psd_blocks is not None:
            block_start = time.perf_counter()
            block_linear_solver = (
                "dense"
                if ctx.settings["solve_type"] == "dense"
                else ctx.settings["lin_solver"]
            )
            dZ, dnu, block_diagnostics = _solve_block_reduced_kkt(
                Z_star,
                S_star,
                A_matrix,
                grad_Z,
                psd_blocks,
                eps_active,
                M_transform,
                P=P,
                linear_solver=block_linear_solver,
                minres_rtol=ctx.settings.get("minres_rtol", 1e-8),
                minres_maxiter=ctx.settings.get("minres_maxiter", None),
            )
            vec_Z_star = vec(Z_star)
            vec_dZ = vec(dZ)
            dC = dZ
            dA_matrix = (
                torch.outer(nu_star, vec_dZ)
                - torch.outer(dnu, vec_Z_star)
            )
            db = dnu
            dP = None
            if profile_backward:
                print(
                    "dSDP block-reduced backward: "
                    f"blocks={psd_blocks}, "
                    f"active={block_diagnostics['active_counts']}, "
                    f"primal={block_diagnostics['block_primal_dim']}, "
                    f"equalities={block_diagnostics['equality_constraints']}, "
                    f"complementarity={block_diagnostics['complementarity_constraints']}, "
                    f"KKT={block_diagnostics['kkt_dim']}x{block_diagnostics['kkt_dim']}, "
                    f"solver={block_linear_solver}, "
                    f"minres={block_diagnostics['minres_stats']}, "
                    f"total={1e3*(time.perf_counter()-block_start):.3f}ms, "
                    f"||dZ||={torch.linalg.norm(dZ).item():.3e}"
                )
            return dC, dA_matrix, db, dP, None, None, None

        # The standard single-cone path needs the full spectral decomposition.

        mu_star, active_set_J, eigenvals_Z, eigenvecs_Z = (
            extract_dual_eigenvalues_from_S(
                Z_star, S_star, eps_active
            )
        )
        n_active = len(active_set_J)
        if verbose:
            print(f"n_active: {n_active}")
            print(f"Active set J = {active_set_J}")
            if n_active == 0:
                print("Warning: No active constraints!")
        
        # Build Hessian block of KKT matrix
        # Start with quadratic term P if present
        sparse_mode = ctx.settings["solve_type"] == "sparse"
        matrix_free = (
            sparse_mode
            and ctx.settings["lin_solver"] == "scipy MINRES matrix-free"
        )
        if sparse_mode:
            M_csr = scipy_sparse.csr_matrix(M_transform.detach().cpu().numpy())
            H_sparse = None if matrix_free else scipy_sparse.csc_matrix((n_vech, n_vech))
            if P is not None and not matrix_free:
                P_csc = torch_csc_to_scipy(P, "csc")
                H_sparse = (M_csr @ P_csc @ M_csr.T).tocsc()
        elif P is not None:
            # Transform P from n²×n² to n_vech×n_vech space
            if P.layout in {torch.sparse_csr, torch.sparse_csc}:
                H_block = M_transform @ torch.sparse.mm(P, M_transform.T)
            else:
                H_block = M_transform @ P @ M_transform.T
        else:
            H_block = torch.zeros(n_vech, n_vech, dtype=dtype, device=device)
        hessian_init_done = time.perf_counter()
        
        # Constraint matrices
        if n_active >= 1: 
            inactive_set_Jc = [k for k in range(n) if k not in active_set_J]

            n_J = len(active_set_J)
            n_Jc = len(inactive_set_Jc)
            
            U_J = eigenvecs_Z[:, active_set_J]
            U_Jc = eigenvecs_Z[:, inactive_set_Jc]
            
            # Build constraint matrix U for pairs j, j' in J (upper triangular)
            # Constraints: <dZ, u_j u_j'^T> = 0 for all j <= j' in J
            n_constraints = n_active * (n_active + 1) // 2  # |J|(|J|+1)/2 constraints
            if sparse_mode:
                U_sparse = _sparse_U(
                    eigenvecs_Z, active_set_J, n,
                    zero_tol=ctx.settings.get("sparse_zero_tol", 0.0),
                )
            else:
                U = torch.zeros(n_vec, n_constraints, dtype=dtype, device=device)
                constraint_idx = 0
                for i, j in enumerate(active_set_J):
                    for k, j_prime in enumerate(active_set_J):
                        if k >= i:
                            u_j = eigenvecs_Z[:, j]
                            u_j_prime = eigenvecs_Z[:, j_prime]
                            U[:, constraint_idx] = torch.outer(u_j, u_j_prime).flatten()
                            constraint_idx += 1
                U = M_transform @ U
            u_build_done = time.perf_counter()
            
            # n_constraints = n_active  
            # U = torch.zeros(n_vec, n_constraints, dtype=dtype, device=device)
            # mu_S, S_eig = torch.linalg.eigh(S_star)
            # S_active_set_J = [k for k,mu in enumerate(mu_S) if mu >= eps_active]
            
            # for i, j in enumerate(S_active_set_J):  # j ∈ J
            #     u_j = S_eig[:, j]
                
            #     # Constraint: <dZ, u_j u_j'^T> = 0
            #     u_outer_vec = torch.outer(u_j, u_j).flatten()
            #     U[:, i] = u_outer_vec
                
            # U = M_transform @ U
            
            
            
            # U = U.sum(dim=1).unsqueeze(-1)
            # n_constraints = 1 
            
            # Hessian_contrib = (U_J_U_Jc + U_Jc_U_J @ Pi_kron) @ M_inv @ U_J_U_Jc.T
            # H_block += M_transform @ Hessian_contrib @ M_transform.T 
            lambda_Jc = eigenvals_Z[inactive_set_Jc]
            Lambda_Jc_inv = torch.diag(1/lambda_Jc)
            Z_inv = U_Jc @ Lambda_Jc_inv @ U_Jc.T
            # Hessian_contrib = torch.kron(Z_inv, S_star) + torch.kron(S_star, Z_inv)
            # H_block += M_transform @ Hessian_contrib @ M_transform.T 
            # 
            if sparse_mode:
                if not matrix_free:
                    # Assemble the final dense H_vech block directly, without
                    # the full n^2-by-n^2 Kronecker intermediate.
                    H_sparse = H_sparse + _direct_vech_spectral_hessian_as_sparse(
                        Z_inv,
                        S_star,
                        zero_tol=ctx.settings.get("sparse_zero_tol", 0.0),
                    )
            else:
                # Dense BLAS-backed Kronecker assembly is faster than direct
                # advanced indexing at the current CPU problem sizes.
                Hessian_contrib = torch.kron(Z_inv, S_star) + torch.kron(S_star, Z_inv)
                H_block += efficient_vec_to_vech_transform(Hessian_contrib, n)
            spectral_hessian_done = time.perf_counter()
            

        # if there is no active constraints
        else:
            # Transform A_matrix constraints
            A_vech = fast_match_vech(A_matrix)  # m × n_vech
            C_blk = A_vech.T  # n_vech × m
            if matrix_free:
                Z_inv = torch.zeros_like(Z_star)
            u_build_done = hessian_init_done
            spectral_hessian_done = hessian_init_done
        
            
            
        if ctx.settings["solve_type"] == "sparse":
            A_csr = torch_csc_to_scipy(A_matrix, "csr")
            A_vech = A_csr @ M_csr.T
            if n_active >= 1:
                n_complementarity_constraints = n_constraints
                C_blk_sparse = scipy_sparse.hstack(
                    [U_sparse, A_vech.T],
                    format="csc",
                )
            else:
                n_complementarity_constraints = 0
                C_blk_sparse = A_vech.T.tocsc()
            q = C_blk_sparse.shape[1]
            if matrix_free:
                P_csc = torch_csc_to_scipy(P, "csc") if P is not None else None
                KKT_operator = _matrix_free_kkt_operator(
                    Z_inv, S_star, C_blk_sparse, P=P_csc,
                )
                minres_preconditioner = None
                preconditioner_stats = None
                preconditioner_name = ctx.settings.get(
                    "minres_preconditioner", "none"
                )
                if preconditioner_name in {"block-diagonal", "kkt-equilibration"}:
                    minres_preconditioner, preconditioner_stats = (
                        _matrix_free_diagonal_preconditioner(
                            Z_inv,
                            S_star,
                            C_blk_sparse,
                            P=P_csc,
                            relative_shift=ctx.settings.get(
                                "minres_preconditioner_shift", 1e-6
                            ),
                            scheme=preconditioner_name,
                        )
                    )
                elif preconditioner_name != "none":
                    raise ValueError(
                        "minres_preconditioner must be 'none', "
                        "'block-diagonal', or 'kkt-equilibration'"
                    )
            else:
                KKT_matrix = scipy_sparse.bmat(
                    [[H_sparse, -C_blk_sparse],
                     [-C_blk_sparse.T, scipy_sparse.csc_matrix((q, q))]],
                    format="csc",
                )
        else:
            A_vech = fast_match_vech(A_matrix)
            if n_active >= 1:
                n_complementarity_constraints = n_constraints
                C_blk = torch.cat([U, A_vech.T], dim=1)
            else:
                n_complementarity_constraints = 0
                C_blk = A_vech.T
            q = C_blk.shape[1]
            KKT_matrix = torch.cat([
                torch.cat([H_block, -C_blk], dim=1),
                torch.cat([-C_blk.T, torch.zeros(q, q, dtype=dtype, device=device)], dim=1),
            ], dim=0)
        kkt_assembly_done = time.perf_counter()


        
        # Right-hand side using fast transformation
        grad_Z_vech = fast_match_vech(grad_Z.flatten())
        rhs = torch.cat([grad_Z_vech, torch.zeros(q, dtype=dtype, device=device)])
    
        start_time = time.time()
        # solution = torch.linalg.lstsq(KKT_matrix, rhs).solution
        # cond = torch.linalg.cond(KKT_matrix)
        # print(f"KKT matrix condition number: {cond.item():.2e}")
        # solution = torch.linalg.lstsq(KKT_matrix, rhs).solution

        minres_stats = None
        if matrix_free:
            solution_np, minres_stats = lin_solvers.matrix_free_minres(
                KKT_operator,
                rhs.detach().cpu().numpy(),
                rtol=ctx.settings.get("minres_rtol", 1e-8),
                maxiter=ctx.settings.get("minres_maxiter", None),
                preconditioner=minres_preconditioner,
            )
            if minres_stats["info"] != 0:
                raise RuntimeError(
                    "matrix-free MINRES did not converge: "
                    f"info={minres_stats['info']}, "
                    f"iterations={minres_stats['iterations']}, "
                    f"relative_residual={minres_stats['relative_residual']:.3e}"
                )
            solution = torch.as_tensor(solution_np, device=device, dtype=dtype)
        elif ctx.settings["solve_type"] == "sparse":
            solution_np, _ = lin_solvers.sparse_solve(
                KKT_matrix,
                rhs.detach().cpu().numpy(),
                linear_solver=ctx.settings["lin_solver"],
            )
            solution = torch.as_tensor(np.asarray(solution_np).squeeze(), device=device, dtype=dtype)
        else:
            solution = torch.linalg.solve(KKT_matrix, rhs)
        kkt_solve_done = time.perf_counter()
        if verbose:
            print("KKT solve time: {:.4f} seconds".format(time.time() - start_time))
        # Extract solution components
        dZ = -vech_to_mat(solution[:n_vech])
        if n_active > 0:
            dmu = -solution[n_vech:n_vech+n_complementarity_constraints]
            dnu = -solution[n_vech+n_complementarity_constraints:n_vech+n_complementarity_constraints+m]
        else:
            dmu = torch.tensor([])
            dnu = -solution[n_vech:n_vech+m]

        vec_Z_star = vec(Z_star)
        vec_dZ = vec(dZ)

        # Gradient representation is independent of the KKT solve mode.
        # Sparse mode accelerates the numerical solve, then returns the same
        # complete dense gradients as dense mode.
        dC = dZ
        dA_matrix = (
            torch.outer(nu_star, vec_dZ)
            - torch.outer(dnu, vec_Z_star)
        )
        
        # b gradients
        db = dnu 
        
        # P gradients (for quadratic term)
        if P is not None:
            dP = 0.5 * torch.outer(vec_Z_star, vec_dZ)
            dP = dP + dP.T
        else:
            dP = None

        if ctx.settings.get("profile_rank", False):
            C_diagnostic = (
                C_blk_sparse.toarray()
                if sparse_mode
                else C_blk.detach().cpu().numpy()
            )
            if n_active >= 1:
                U_diagnostic = (
                    U_sparse.toarray()
                    if sparse_mode
                    else U.detach().cpu().numpy()
                )
                A_diagnostic = (
                    A_vech.T.toarray()
                    if sparse_mode
                    else A_vech.T.detach().cpu().numpy()
                )
            else:
                U_diagnostic = np.empty((n_vech, 0))
                A_diagnostic = C_diagnostic

            def numerical_rank_and_basis(matrix):
                left, singular_values, _ = np.linalg.svd(
                    matrix, full_matrices=False
                )
                if singular_values.size == 0:
                    return 0, left[:, :0]
                tolerance = (
                    max(matrix.shape)
                    * np.finfo(singular_values.dtype).eps
                    * singular_values[0]
                )
                rank = int(np.sum(singular_values > tolerance))
                return rank, left[:, :rank]

            rank_A, basis_A = numerical_rank_and_basis(A_diagnostic)
            rank_U, _ = numerical_rank_and_basis(U_diagnostic)
            U_perpendicular = U_diagnostic - basis_A @ (
                basis_A.T @ U_diagnostic
            )
            rank_U_perpendicular, _ = numerical_rank_and_basis(
                U_perpendicular
            )
            singular_values_C = np.linalg.svd(
                C_diagnostic, compute_uv=False
            )
            rank_tolerance = (
                max(C_diagnostic.shape)
                * np.finfo(singular_values_C.dtype).eps
                * singular_values_C[0]
            )
            rank_C = int(np.sum(singular_values_C > rank_tolerance))
            sigma_min_C = float(singular_values_C[-1])
            condition_C = (
                float(singular_values_C[0] / sigma_min_C)
                if sigma_min_C > 0.0 else float("inf")
            )
            primal_solution = solution[:n_vech]
            primal_solution_np = primal_solution.detach().cpu().numpy()
            constraint_residual = np.linalg.norm(
                C_diagnostic.T @ primal_solution_np
            )
            if matrix_free:
                kkt_rank_description = "KKT rank not materialized"
            else:
                KKT_diagnostic = (
                    KKT_matrix.toarray()
                    if scipy_sparse.issparse(KKT_matrix)
                    else KKT_matrix.detach().cpu().numpy()
                )
                singular_values_KKT = np.linalg.svd(
                    KKT_diagnostic, compute_uv=False
                )
                kkt_rank_tolerance = (
                    max(KKT_diagnostic.shape)
                    * np.finfo(singular_values_KKT.dtype).eps
                    * singular_values_KKT[0]
                )
                rank_KKT = int(
                    np.sum(singular_values_KKT > kkt_rank_tolerance)
                )
                sigma_min_KKT = float(singular_values_KKT[-1])
                condition_KKT = (
                    float(singular_values_KKT[0] / sigma_min_KKT)
                    if sigma_min_KKT > 0.0 else float("inf")
                )
                kkt_rank_description = (
                    f"rank(KKT)={rank_KKT}/{KKT_diagnostic.shape[0]}, "
                    f"sigma_min(KKT)={sigma_min_KKT:.3e}, "
                    f"cond(KKT)={condition_KKT:.3e}"
                )
            if matrix_free:
                kkt_description = f"operator={KKT_operator.shape[0]}x{KKT_operator.shape[1]}"
            elif scipy_sparse.issparse(KKT_matrix):
                kkt_nnz = KKT_matrix.nnz
                kkt_description = (
                    f"KKT={KKT_matrix.shape[0]}x{KKT_matrix.shape[1]}, nnz={kkt_nnz}"
                )
            else:
                kkt_nnz = int(torch.count_nonzero(KKT_matrix).item())
                kkt_description = (
                    f"KKT={KKT_matrix.shape[0]}x{KKT_matrix.shape[1]}, nnz={kkt_nnz}"
                )
            minres_description = ""
            if minres_stats is not None:
                minres_description = (
                    f", iterations={minres_stats['iterations']}, "
                    f"relres={minres_stats['relative_residual']:.3e}"
                )
                if preconditioner_stats is not None:
                    minres_description += (
                        f", precond={preconditioner_name}, "
                        f"diagH=[{preconditioner_stats['hessian_diagonal_min']:.2e},"
                        f"{preconditioner_stats['hessian_diagonal_max']:.2e}], "
                        f"diagD=[{preconditioner_stats['dual_diagonal_min']:.2e},"
                        f"{preconditioner_stats['dual_diagonal_max']:.2e}]"
                    )
            print(
                "dSDP backward profile: "
                f"n={n}, active={n_active}, constraints={n_complementarity_constraints}, "
                f"{kkt_description}{minres_description}; "
                f"init={1e3*(hessian_init_done-backward_start):.3f}ms, "
                f"U={1e3*(u_build_done-hessian_init_done):.3f}ms, "
                f"H={1e3*(spectral_hessian_done-u_build_done):.3f}ms, "
                f"assembly={1e3*(kkt_assembly_done-spectral_hessian_done):.3f}ms, "
                f"solve={1e3*(kkt_solve_done-kkt_assembly_done):.3f}ms, "
                f"grad={1e3*(time.perf_counter()-kkt_solve_done):.3f}ms"
                f"; ||x||={np.linalg.norm(primal_solution_np):.3e}, "
                f"||dZ||={torch.linalg.norm(dZ).item():.3e}, "
                f"rank(C)={rank_C}/{C_diagnostic.shape[1]}, "
                f"rank(A)={rank_A}/{A_diagnostic.shape[1]}, "
                f"rank(U)={rank_U}/{U_diagnostic.shape[1]}, "
                f"rank(U_perp)={rank_U_perpendicular}/{U_diagnostic.shape[1]}, "
                f"sigma_min(C)={sigma_min_C:.3e}, cond(C)={condition_C:.3e}, "
                f"||C^T x||={constraint_residual:.3e}, {kkt_rank_description}"
            )
        
        return dC, dA_matrix, db, dP, None, None, None
    

class dSDPLayer(torch.nn.Module):
    def __init__(self, n, m, mode="dense", sparsity=None, settings=None, transforms=None,
                 has_quadratic_term=False, psd_blocks=None):
        """Create a differentiable semidefinite-programming layer.

        Args:
            n: Dimension of the symmetric primal matrix ``Z``.
            m: Number of equality constraints in ``A @ vec(Z) = b``.
            mode: ``"dense"`` or ``"sparse"``. Sparse mode requires fixed
                nonzero patterns and currently accepts Torch CSR inputs.
            sparsity: Sparse templates for the fixed nonzero patterns. Use a
                dictionary containing ``"C"`` and ``"A"``, and also ``"P"``
                when a quadratic objective is present. Ignored in dense mode.
            settings: Optional solver and differentiation settings overriding
                :func:`build_settings`, such as ``solver``, ``solver_args``,
                ``eps_active`` and ``lin_solver``.
            transforms: Optional precomputed ``(M_transform, T_inv)`` used for
                vec/vech conversion. Supplying them avoids rebuilding these
                matrices for multiple layers of the same size.
            has_quadratic_term: Whether the objective includes
                ``0.5 * vec(Z).T @ P @ vec(Z)``.
            psd_blocks: Optional sequence of ``(start, stop)`` pairs or slices
                describing independent diagonal PSD blocks. The blocks must be
                non-overlapping and cover all indices ``0, ..., n - 1``.
        Notes:
            The block-cone CVXPY forward currently supports dense problems with
            a linear objective. For sparse or quadratic problems, omit
            ``psd_blocks`` to use the standard single-cone formulation. When
            ``psd_blocks`` is provided, ``A`` and ``b`` must already contain
            only the meaningful equality constraints; do not include rows that
            merely force off-block entries of a full lifted matrix to zero.
        """
        super(dSDPLayer, self).__init__()
        
        if mode not in {"dense", "sparse"}:
            raise ValueError("mode must be 'dense' or 'sparse'")
        settings_default = build_settings() # call with defaults
        if settings is not None:
            for key, value in settings.items():
                settings_default[key] = value
        settings = settings_default
        settings["solve_type"] = mode
        requested_psd_blocks = (
            psd_blocks if psd_blocks is not None else settings.get("psd_blocks")
        )
        settings["psd_blocks"] = (
            _normalize_psd_blocks(requested_psd_blocks, n)
            if requested_psd_blocks is not None else None
        )
        if mode == "sparse" and settings["lin_solver"] not in lin_solvers.get_sparse_solvers():
            raise ValueError(f"unknown sparse linear solver: {settings['lin_solver']}")
        self.settings = settings
        self.mode = mode
        self.sparsity = sparsity
        # TODO: maybe another name 
        self.transforms = transforms  # Store precomputed transformation matrices (M_transform, T_inv)
        self.problem = CVXPYSDPProblem(
            n,
            m,
            has_quadratic_term,
            mode,
            sparsity,
            psd_blocks=settings["psd_blocks"],
        )

    def forward(self, C, A_matrix, b, P=None):
        if self.mode == "sparse":
            matrices = {"C": C, "A": A_matrix}
            if P is not None: matrices["P"] = P
            for name, matrix in matrices.items():
                if matrix.layout != torch.sparse_csr:
                    raise TypeError(f"sparse mode requires {name} in torch CSR format; forward converts it to CSC")
                if not _same_csc_pattern(matrix, self.sparsity[name]):
                    raise ValueError(f"sparsity pattern changed for {name}")

        if P is not None:
            return dSDPFunction.apply(
                C,
                A_matrix,
                b,
                P,
                self.settings,
                self.transforms,
                self.problem,
            )
        else:
            return dSDPFunction.apply(
                C,
                A_matrix,
                b,
                None,
                self.settings,
                self.transforms,
                self.problem,
            )
