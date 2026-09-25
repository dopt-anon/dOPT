import time 
import torch
import numpy as np
import cvxpy as cp
from ..dSDP import dSDPLayer, precompute_M_transform, precompute_T_inv
from cvxpylayers.torch import CvxpyLayer
from scipy.sparse import coo_matrix
from scipy import sparse

# --------------------------- CVXPY ----------------------------------
def solve_sdp_cvxpy(C, A_matrix, b, P=None,solver=None,**solver_args):
    """Solve SDP using CVXPY with matrix constraints A*vec(Z) = b and optional quadratic term, return complete dual information"""
    n = C.shape[0]
    Z = cp.Variable((n, n), symmetric=True)
    
    constraints = [Z >> 0]  # Only PSD constraint
    
    # Add matrix constraint A * vec(Z) = b
    vec_Z = cp.vec(Z, order="F")
    constraints.append(A_matrix @ vec_Z == b)
    
    # Objective: <C,Z> + 1/2 * vec(Z)^T * P * vec(Z)
    linear_term = cp.trace(C @ Z)
    if P is not None:
        quadratic_term = 0.5 * cp.quad_form(vec_Z, P)
        objective = cp.Minimize(linear_term + quadratic_term)
    else:
        objective = cp.Minimize(linear_term)
    prob = cp.Problem(objective, constraints)
    
    # Handle solve_method parameter from CVXPYLayers
    if 'solve_method' in solver_args:
        solve_method = solver_args.pop('solve_method')
        if solve_method == "Clarabel":
            solver = cp.CLARABEL
        elif solve_method == "SCS":
            solver = cp.SCS
        elif solve_method == "MOSEK":
            solver = cp.MOSEK
        # Add more solver mappings as needed
    
    if solver is None:
        solver = cp.SCS
    prob.solve(solver=solver, verbose=False, **solver_args)
    if prob.status == cp.OPTIMAL:
        Z_star = Z.value
        S_star = constraints[0].dual_value  # PSD constraint dual matrix
        nu_star = constraints[1].dual_value  # Matrix constraint dual vector
        
        S_star = torch.from_numpy(S_star)
        nu_star = torch.from_numpy(nu_star)
        eigenvalues, eigenvectors = np.linalg.eigh(Z_star)
        return Z_star, S_star, nu_star, eigenvalues, eigenvectors, prob.status
    else:
        return None, None, None, None, prob.status

# --------------------------- dSDP ----------------------------------
def dsdp_solve(
    C,
    A_matrix,
    b,
    P=None,
    settings=None,
    dsdp_layer=None,
    mode="dense",
):
    """Solve using dSDP with matrix constraints and verify"""
    # print(f"\n dSDP solving on device: {C.device}...")
    
    try:
        n = C.shape[0]
        device, dtype = C.device, C.dtype
        if dsdp_layer is None:
            M = precompute_M_transform(n, device, dtype)
            T_inv = precompute_T_inv(n, device, dtype)
            transforms = (M, T_inv)
            sparsity = None
            if mode == "sparse":
                sparsity = {"C": C, "A": A_matrix}
                if P is not None:
                    sparsity["P"] = P
            dsdp_layer = dSDPLayer(
                n,
                b.shape[0],
                mode=mode,
                sparsity=sparsity,
                settings=settings,
                transforms=transforms,
                has_quadratic_term=P is not None,
            )
        
        C_test = C.detach().clone().requires_grad_(True)
        A_matrix_test = A_matrix.detach().clone().requires_grad_(True)
        b_test = b.detach().clone().requires_grad_(True)
        P_test = P.detach().clone().requires_grad_(True) if P is not None else None
        start_time = time.perf_counter()
        Z_dsdp, S_dsdp, nu_dsdp = dsdp_layer(C_test, A_matrix_test, b_test, P_test)
        forward_time = time.perf_counter() - start_time

        return Z_dsdp, S_dsdp, nu_dsdp, C_test, A_matrix_test, b_test, P_test, forward_time

    except Exception as e:
        print(f"❌ dSDP failed: {e}")
        return None, None, None, None, None, None, None, None


def check_unique_eigenvalues(Z, threshold=1e-6):
    """Check if solution has unique eigenvalues"""
    # print("\n🔍 Checking eigenvalues...")
    
    eigenvals = np.linalg.eigvals(Z.real)
    eigenvals_sorted = np.sort(eigenvals)[::-1]  # Descending order
    
    # print(f"Eigenvalues: {eigenvals_sorted}")
    
    # Compute gaps between adjacent eigenvalues
    gaps = np.abs(np.diff(eigenvals_sorted))
    min_gap = np.min(gaps)
    
    unique = min_gap > threshold
    # print(f"Minimum gap: {min_gap:.2e}")
    print(f"Unique eigenvalues: {'✅' if unique else '❌'}")
    
    return unique, eigenvals_sorted


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


# --------------------------- Diffcp ----------------------------------

def diffcp_solve(C, A_matrix, b, P=None,mode="",eps=1e-6, grad=None):
    """Solve using diffcp with matrix constraints and verify"""
    
    try:
        import diffcp
    
        print("\n diffcp solving...")
        
        n = C.shape[0]
        m = b.shape[0]
        C_np = C.detach().numpy()
        A_np = A_matrix.detach().numpy()
        b_np = b.detach().numpy()
        
        # Convert problem to SCS format for diffcp
        # Create CVXPY problem to extract SCS data
        Z = cp.Variable((n, n), symmetric=True)
        constraints = [Z >> 0]
        vec_Z = cp.vec(Z,order="F")  
        constraints.append(A_np @ vec_Z == b_np)
        
        # Objective: <C,Z> + 1/2 * vec(Z)^T * P * vec(Z)
        linear_term = cp.trace(C_np @ Z)
        if P is not None:
            P_sqrt = torch.linalg.cholesky(P, upper=True)
            P_sqrt_np = P_sqrt.detach().numpy()
            quadratic_term = 0.5 * cp.sum_squares(P_sqrt_np @ vec_Z)
            objective = cp.Minimize(linear_term + quadratic_term)
        else:
            objective = cp.Minimize(linear_term)
        
        prob = cp.Problem(objective, constraints)
        A_scs, b_scs, c_scs, cone_dims = scs_data_from_cvxpy_problem(prob)
        
        # Solve with diffcp
        start_time = time.time()
        if mode=="lpgd":
            x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
            A_scs, b_scs, c_scs, cone_dims, eps=eps, mode="lpgd")
        else:
            x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
                A_scs, b_scs, c_scs, cone_dims)
        forward_time = time.time() - start_time
        
        # Extract solution Z from x
        # Note: diffcp returns numpy arrays, need to convert to torch tensors
        T_inv = precompute_T_inv(n, device=C.device, dtype=C.dtype)
        Z_vec = T_inv @ torch.from_numpy(x)
        Z_diffcp = Z_vec.reshape((n, n))  # Reshape to matrix form
        
        # Create gradient-enabled variables for consistency with other test functions
        C_diffcp = C.clone().requires_grad_(True)
        A_diffcp = A_matrix.clone().requires_grad_(True) 
        b_diffcp = b.clone().requires_grad_(True)
        P_diffcp = P.clone().requires_grad_(True) if P is not None else None
        
        # Create a differentiable version of Z_diffcp by computing it from the inputs
        # This is a workaround since diffcp doesn't provide PyTorch gradients directly
        Z_diffcp = Z_diffcp.detach().requires_grad_(True)
        
        # Derivative computation using proper M transform for sum(Z) loss
        start_time = time.time()
        
        # Get the M transform matrix for proper gradient computation
        M_transform = precompute_M_transform(n, device='cpu', dtype=torch.float64)
        M_np = M_transform.detach().cpu().numpy()
        
        # The seed gradient for sum(Z) loss should be M @ ones(n^2)
        # This accounts for the symmetric structure in the optimization
        # ones_n2 = np.ones(n * n, dtype=np.float64)

        # dx_seed = M_np @ ones_n2

        dx_seed = grad

        if mode=="lpgd":
            lpgd_args = dict(tau=0.1, rho=0.1)
            dA, db, dc = adjoint_derivative(dx_seed, np.zeros(y.size), np.zeros(s.size), **lpgd_args)
        else:
            lsqr_args = dict(atol=1e-5, btol=1e-5)
            dA, db, dc = adjoint_derivative(dx_seed, np.zeros(y.size), np.zeros(s.size), **lsqr_args)

        backward_time = time.time() - start_time
        
        return Z_diffcp, forward_time, backward_time, dA, db, dc
        
    except Exception as e:
        print(f"❌ diffcp failed: {e}")
        return None, None, None, None, None, None

# --------------------------- CVXPYLayers ----------------------------------

def cvxpylayers_solve(C, A_matrix, b, P=None,cvxpy_layer=None, **solver_args):
    """Solve using CVXPYLayers with matrix constraints and verify"""
    
    try:
        if not cvxpy_layer:
            # Create CVXPYLayer
            n = C.shape[0]
            m = b.shape[0]
            Z = cp.Variable((n, n), symmetric=True)
            C_param = cp.Parameter((n, n))
            A_param = cp.Parameter((m, n*n))
            b_param = cp.Parameter(m)
            
            constraints = [Z >> 0]
            vec_Z = cp.vec(Z,order="F")
            constraints.append(A_param @ vec_Z == b_param)
            
            # Objective: <C,Z> + 1/2 * vec(Z)^T * P * vec(Z)
            linear_term = cp.trace(C_param @ Z)
            if P is not None:
                P_sqrt_param = cp.Parameter((n*n, n*n))  
                quadratic_term = 0.5 * cp.sum_squares( P_sqrt_param @ vec_Z )
                objective = cp.Minimize(linear_term + quadratic_term)
                parameters = [C_param, A_param, b_param, P_sqrt_param]
            else:
                objective = cp.Minimize(linear_term)
                parameters = [C_param, A_param, b_param]
            
            problem = cp.Problem(objective, constraints)
            
            # Create differentiable layer
            cvxpy_layer = CvxpyLayer(problem, 
                                    parameters=parameters, 
                                    variables=[Z])
            
        # Prepare inputs
        C_cvxpy = C.clone().requires_grad_(True)
        A_cvxpy = A_matrix.clone().requires_grad_(True)
        b_cvxpy = b.clone().requires_grad_(True)
        if P is not None:
            P_cvxpy = P.clone().requires_grad_(True)
            P_sqrt = torch.linalg.cholesky(P_cvxpy,upper=True)
            inputs = [C_cvxpy, A_cvxpy, b_cvxpy, P_sqrt]
        else:
            P_cvxpy = None
            inputs = [C_cvxpy, A_cvxpy, b_cvxpy]
        
        # Solve
        start_time = time.perf_counter()
        Z_cvxpy, = cvxpy_layer(*inputs, solver_args=solver_args)
        forward_time = time.perf_counter() - start_time
    
        # print(f"Solution Z =\n{Z_cvxpy.detach().numpy()}")

        # Solve using CVXPY to get dual variables

        Z_, S_cvxpy, nu_cvxpy, _, _, status = solve_sdp_cvxpy(C.detach().numpy(), A_matrix.detach().numpy(), b.detach().numpy(), P.detach().numpy() if P is not None else None, **solver_args)
        
        
        return Z_cvxpy, S_cvxpy, nu_cvxpy, C_cvxpy, A_cvxpy, b_cvxpy, P_cvxpy, forward_time, cvxpy_layer
        
    except Exception as e:
        print(f"❌ CVXPYLayers failed: {e}")
        return None, None, None, None, None, None, None, None
