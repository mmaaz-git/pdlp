import torch
import time

def solve(
    c: torch.Tensor,
    G: torch.Tensor, h: torch.Tensor,
    A: torch.Tensor, b: torch.Tensor,
    l: torch.Tensor, u: torch.Tensor,
    iteration_limit: int | float = 10000,
    time_sec_limit: float = float('inf'),
    primal_weight_update_smoothing: float = 0.5,
    ruiz_iterations: int = 10,
    pock_chambolle_alpha: float = 1.0,
    eps_tol: float = 1e-6,
    eps_primal_infeasible: float = 1e-8,
    eps_dual_infeasible: float = 1e-8,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str, dict]:
    """
    Solve a Linear Program using the PDLP algorithm.

    Problem formulation:
        minimize    c^T x
        subject to  G x >= h  (inequality constraints)
                    A x  = b  (equality constraints)
                    l <= x <= u  (variable bounds)

    Args:
        G: Inequality constraint matrix (m1, n)
        A: Equality constraint matrix (m2, n)
        c: Objective coefficient vector (n,)
        h: Inequality constraint right-hand side (m1,)
        b: Equality constraint right-hand side (m2,)
        l: Variable lower bounds (n,) - use -inf for unbounded
        u: Variable upper bounds (n,) - use +inf for unbounded
        iteration_limit: Maximum number of PDHG iterations (default 10000, use float('inf') for unlimited)
        time_sec_limit: Maximum solve time in seconds (default float('inf') for unlimited)
        primal_weight_update_smoothing: Smoothing factor for primal weight updates (0-1)
        ruiz_iterations: Number of Ruiz equilibration iterations
        pock_chambolle_alpha: Pock-Chambolle rescaling parameter (0 = disable)
        eps_tol: Convergence tolerance for optimality (default 1e-6)
        eps_primal_infeasible: Tolerance for primal infeasibility detection (default 1e-8)
        eps_dual_infeasible: Tolerance for dual infeasibility detection (default 1e-8)
        verbose: Print detailed solver information

    Returns:
        x_sol: Primal solution (n,)
        y_sol: Dual solution (m1 + m2,) where y[:m1] are inequality duals, y[m1:] are equality duals
        status: "optimal", "primal_infeasible", "dual_infeasible", "iteration_limit", or "time_limit"
        info: Dict with solve statistics.
            - Always contains: solve_time_sec, iterations
            - For optimal/iteration_limit/time_limit also contains:
                primal_obj, dual_obj, duality_gap, relative_gap,
                primal_residual, dual_residual, kkt_error_sq
            - For primal_infeasible/dual_infeasible also contains:
                ray, certificate_quality
    """
    # -----------------------------
    # Shape checks / setup
    # -----------------------------
    assert G.ndim == 2 and A.ndim == 2
    assert c.ndim == h.ndim == b.ndim == l.ndim == u.ndim == 1
    assert G.shape[0] == h.shape[0]
    assert A.shape[0] == b.shape[0]
    assert G.shape[1] == A.shape[1] == c.shape[0] == l.shape[0] == u.shape[0]
    assert G.is_sparse == A.is_sparse, "G and A must both be sparse or both be dense"
    assert not (iteration_limit == float('inf') and time_sec_limit == float('inf')), "At least one of iteration_limit or time_sec_limit must be finite"

    device = c.device
    dtype = c.dtype
    is_sparse = G.is_sparse
    eps_zero = 1e-12
    termination_check_frequency = 100 # how frequently to check for termination
    max_inner_iters = 1000 # max iterations between restarts
    max_backtrack = 50 # max backtracking steps in adaptive step size

    start_time = time.time()

    m1, n = G.shape
    m2 = A.shape[0]
    m = m1 + m2

    if verbose:
        print(f"\nPDLP Solver")
        print(f"  Problem: {m1} inequalities, {m2} equalities, {n} variables")

    # save originals (for termination checks on original problem)
    G_orig, h_orig, A_orig, b_orig = G.clone(), h.clone(), A.clone(), b.clone()
    c_orig, l_orig, u_orig = c.clone(), l.clone(), u.clone()

    # stack constraints: K = [G; A], q = [h; b]
    K = torch.cat([G, A], dim=0)
    q = torch.cat([h, b], dim=0)
    K_orig = torch.cat([G_orig, A_orig], dim=0)
    q_orig = torch.cat([h_orig, b_orig], dim=0)

    def slice_rows(M, start, end):
        if not is_sparse: return M[start:end, :]
        M = M.coalesce()
        indices, values = M.indices(), M.values()
        mask = (indices[0] >= start) & (indices[0] < end)
        new_indices = indices[:, mask].clone()
        new_indices[0] -= start
        return torch.sparse_coo_tensor(new_indices, values[mask], (end-start, M.shape[1]), dtype=M.dtype, device=M.device)

    if is_sparse:
        col_max = lambda M: torch.zeros(M.shape[1], dtype=M.dtype, device=M.device).scatter_reduce_(0, (M_c := M.abs().coalesce()).indices()[1], M_c.values(), reduce='amax', include_self=True)
        row_max = lambda M: torch.zeros(M.shape[0], dtype=M.dtype, device=M.device).scatter_reduce_(0, (M_c := M.abs().coalesce()).indices()[0], M_c.values(), reduce='amax', include_self=True)
    else:
        col_max = lambda M: M.max(dim=0)[0]
        row_max = lambda M: M.max(dim=1)[0]

    # -----------------------------
    # Trivial cases: no variables or no constraints
    # -----------------------------
    if n == 0:
        # no variables: check h <= 0 and b = 0 to be feasible
        if torch.all(h <= eps_zero) and torch.all(b.abs() <= eps_zero):
            info = {"primal_obj": 0.0, "dual_obj": 0.0, "solve_time_sec": time.time()-start_time, "iterations": 0}
            if verbose: print(f"\n  Status: converged (trivial, no variables) in {info['solve_time_sec']:.6f}s\n  Primal objective: 0.000000e+00")
            return torch.zeros(0, device=device, dtype=dtype), torch.zeros(m, device=device, dtype=dtype), "optimal", info
        # Construct Farkas certificate: y with violations
        y_ray = torch.cat([torch.where(h > eps_zero, 1.0, 0.0), torch.where(b.abs() > eps_zero, 1.0, 0.0)])
        y_ray = y_ray / torch.linalg.norm(y_ray, ord=float('inf')).clamp_min(eps_zero)
        info = {"ray": y_ray, "dual_ray_obj": (q_orig @ y_ray).item(), "dual_residual": 0.0, "certificate_quality": 0.0, "solve_time_sec": time.time()-start_time, "iterations": 0}
        if verbose:
            print(f"\n  Status: PRIMAL INFEASIBLE after 0 iterations in {info['solve_time_sec']:.3f}s")
            print(f"    Farkas certificate (dual ray): y = {info['ray']}")
            print(f"      K^T y ≈ 0:  ||K^T y|| = {info['dual_residual']:.3e}")
            print(f"      q^T y > 0:  q^T y = {info['dual_ray_obj']:.3e}")
            print(f"      Relative certificate quality: {info['certificate_quality']:.3e}")
        return torch.zeros(0, device=device, dtype=dtype), torch.zeros(m, device=device, dtype=dtype), "primal_infeasible", info

    if m == 0:
        # no constraints: optimal if all variables bounded in direction of objective
        x_sol = torch.where(c < -eps_zero, u, l)  # minimize c^T x: go to u if c<0, else l
        unbounded_mask = ((c < -eps_zero) & torch.isinf(u)) | ((c > eps_zero) & torch.isinf(l))
        if unbounded_mask.any():
            idx = torch.where(unbounded_mask)[0][0]
            x_ray = torch.zeros(n, device=device, dtype=dtype)
            x_ray[idx] = 1.0 if c[idx] < 0 else -1.0
            info = {"ray": x_ray, "primal_ray_obj": (c_orig @ x_ray).item(), "max_primal_residual": 0.0, "certificate_quality": 0.0, "solve_time_sec": time.time() - start_time, "iterations": 0}
            if verbose:
                print(f"\n  Status: DUAL INFEASIBLE (primal unbounded) after 0 iterations in {info['solve_time_sec']:.3f}s")
                print(f"    Unboundedness certificate (primal ray): x = {info['ray']}")
                print(f"      K x ≈ 0:  max_residual = {info['max_primal_residual']:.3e}")
                print(f"      c^T x < 0:  c^T x = {info['primal_ray_obj']:.3e}")
                print(f"      Relative certificate quality: {info['certificate_quality']:.3e}")
            return x_sol, torch.zeros(0, device=device, dtype=dtype), "dual_infeasible", info
        obj = (c_orig @ x_sol).item()
        info = {"primal_obj": obj, "dual_obj": obj, "solve_time_sec": time.time()-start_time, "iterations": 0}
        if verbose: print(f"\n  Status: converged (trivial, no constraints) in {info['solve_time_sec']:.6f}s\n  Primal objective: {obj:.6e}")
        return x_sol, torch.zeros(0, device=device, dtype=dtype), "optimal", info

    # -----------------------------
    # Rescaling / Preconditioning
    # -----------------------------
    constraint_rescaling = torch.ones(m, device=device, dtype=dtype)
    variable_rescaling = torch.ones(n, device=device, dtype=dtype)

    # Ruiz rescaling (L-infinity equilibration)
    for _ in range(ruiz_iterations):
        # column rescaling: sqrt(max(|K[:,j]|, |c[j]|))
        col_max_vals = torch.maximum(col_max(K.abs()), c.abs())
        col_rescale = torch.sqrt(col_max_vals)
        col_rescale = torch.where(col_max_vals > eps_zero, col_rescale.clamp_min(eps_zero), torch.ones_like(col_rescale)) # scale 1 if all 0's
        # row rescaling: sqrt(max(|K[i,:]|))
        row_max_vals = row_max(K.abs())
        row_rescale = torch.sqrt(row_max_vals)
        row_rescale = torch.where(row_max_vals > eps_zero, row_rescale.clamp_min(eps_zero), torch.ones_like(row_rescale))

        # apply rescaling
        c, l, u = c / col_rescale, l * col_rescale, u * col_rescale
        q = q / row_rescale
        K = K * (1.0 / row_rescale).unsqueeze(1) * (1.0 / col_rescale).unsqueeze(0)

        constraint_rescaling *= row_rescale
        variable_rescaling *= col_rescale

    # Pock-Chambolle rescaling (operator norm <= 1)
    if pock_chambolle_alpha > 0:
        # column rescaling: sqrt(sum_i |K[i,j]|^(2-alpha))
        col_sum = (K.abs() ** (2 - pock_chambolle_alpha)).sum(dim=0)
        col_sum = col_sum.to_dense() if is_sparse else col_sum
        col_rescale = torch.sqrt(col_sum)
        col_rescale = torch.where(col_sum > eps_zero, col_rescale.clamp_min(eps_zero), torch.ones_like(col_rescale)) # scale 1 if all 0s
        # row rescaling: sqrt(sum_j |K[i,j]|^alpha)
        row_sum = (K.abs() ** pock_chambolle_alpha).sum(dim=1)
        row_sum = row_sum.to_dense() if is_sparse else row_sum
        row_rescale = torch.sqrt(row_sum)
        row_rescale = torch.where(row_sum > eps_zero, row_rescale.clamp_min(eps_zero), torch.ones_like(row_rescale))

        # apply rescaling
        c, l, u = c / col_rescale, l * col_rescale, u * col_rescale
        q = q / row_rescale
        K = K * (1.0 / row_rescale).unsqueeze(1) * (1.0 / col_rescale).unsqueeze(0)

        constraint_rescaling *= row_rescale
        variable_rescaling *= col_rescale

    if verbose:
        print(f"  Rescaling: Ruiz iters={ruiz_iterations}, Pock-Chambolle alpha={pock_chambolle_alpha}")
        if not is_sparse: # operator norm not supported for sparse
            print(f"  ||K||_2 (operator norm) after rescaling: {torch.linalg.matrix_norm(K, ord=2).item():.3e}")

    # split scaled K, q back into G, h, A, b (for algorithm)
    G, h = slice_rows(K, 0, m1), q[:m1]
    A, b = slice_rows(K, m1, m), q[m1:]

    # -----------------------------
    # Subprocedures
    # -----------------------------
    # project x onto box constraints [l, u]
    def proj_X(x: torch.Tensor) -> torch.Tensor: return torch.clamp(x, l, u)

    # project y onto dual feasible set (y[:m1] >= 0, y[m1:] free).
    def proj_Y(y: torch.Tensor) -> torch.Tensor: return torch.cat([torch.clamp(y[:m1], min=0.0), y[m1:]])

    @torch.no_grad()
    def primal_weight_update(
        x_new: torch.Tensor, y_new: torch.Tensor,
        x_old: torch.Tensor, y_old: torch.Tensor,
        w_old: torch.Tensor,
    ) -> torch.Tensor:
        """Updates primal weight"""
        dx = torch.linalg.norm(x_new - x_old)
        dy = torch.linalg.norm(y_new - y_old)

        if (dx > eps_zero) and (dy > eps_zero):
            ratio = (dy / dx).clamp_min(eps_zero)
            # exponential moving average: w_new = ratio^alpha * w_old^(1-alpha)
            w_new = (ratio ** primal_weight_update_smoothing) * (w_old ** (1.0 - primal_weight_update_smoothing))
            return w_new.clamp_min(eps_zero)
        return w_old

    @torch.no_grad()
    def compute_lambda_for_box(x: torch.Tensor, g: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Computes lambda, the normal-cone component for box constraints at x."""
        lam = torch.zeros_like(x)
        at_l = torch.isfinite(lower) & (x <= lower + eps_tol)
        at_u = torch.isfinite(upper) & (x >= upper - eps_tol)
        # handle numerically-tight boxes where both flags fire
        both = at_l & at_u
        if both.any():
            at_l = torch.where(both, (x - lower).abs() <= (upper - x).abs(), at_l)
            at_u = torch.where(both, (upper - x).abs() < (x - lower).abs(), at_u)
        lam[at_l] = torch.clamp(g[at_l], min=0.0)
        lam[at_u] = torch.clamp(g[at_u], max=0.0)
        return lam

    def sum_finite_products(values: torch.Tensor, multipliers: torch.Tensor) -> torch.Tensor:
        """Sum values[i] * multipliers[i] where values[i] is finite (not inf)."""
        finite = torch.isfinite(values)
        return (values[finite] * multipliers[finite]).sum() if finite.any() else torch.tensor(0.0, device=values.device, dtype=values.dtype)

    @torch.no_grad()
    def compute_dual_objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes dual objective q^T y + l^T λ+ - u^T λ- for the original (unscaled) problem."""
        g = c_orig - (K_orig.T @ y)
        lam = compute_lambda_for_box(x, g, l_orig, u_orig)
        lam_pos = torch.clamp(lam, min=0.0)
        lam_neg = torch.clamp(-lam, min=0.0)

        l_term = sum_finite_products(l_orig, lam_pos)
        u_term = sum_finite_products(u_orig, lam_neg)

        return q_orig @ y + l_term - u_term

    @torch.no_grad()
    def kkt_error_sq(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Equation (5) in the paper."""
        w = torch.as_tensor(w, device=x.device, dtype=x.dtype).clamp_min(eps_zero)

        # primal residuals
        r_eq = b - (A @ x)
        r_ineq = torch.clamp(h - (G @ x), min=0.0)
        term1 = (w**2) * (r_eq @ r_eq + r_ineq @ r_ineq)

        # stationarity residual + box multipliers
        g = c - (K.T @ y)
        lam = compute_lambda_for_box(x, g, l, u)
        rs = g - lam
        term2 = (1.0 / (w**2)) * (rs @ rs)

        # scalar gap-ish term
        lam_pos = torch.clamp(lam, min=0.0)
        lam_minus = torch.clamp(-lam, min=0.0)

        l_term = sum_finite_products(l, lam_pos)
        u_term = sum_finite_products(u, lam_minus)

        scalar = (q @ y) + l_term - u_term - (c @ x)
        term3 = scalar * scalar

        return term1 + term2 + term3

    @torch.no_grad()
    def termination_criteria(x_scaled: torch.Tensor, y_scaled: torch.Tensor) -> tuple[str, dict]:
        """
        Check termination on original problem. Returns (status, info)
        where status is '' if continuing, 'optimal', 'primal_infeasible', or 'dual_infeasible' if done.
        """
        x_unscaled, y_unscaled = x_scaled / variable_rescaling, y_scaled / constraint_rescaling

        # Check for primal infeasibility (Farkas certificate via dual ray)
        dual_norm_inf = torch.linalg.norm(y_unscaled, ord=float('inf'))
        if dual_norm_inf > eps_zero:
            y_ray = y_unscaled / dual_norm_inf
            dual_ray_obj = (q_orig @ y_ray).item()
            if dual_ray_obj > 0:
                dual_residual = torch.linalg.norm(K_orig.T @ y_ray, ord=float('inf')).item()
                relative_infeas = dual_residual / dual_ray_obj
                if relative_infeas < eps_primal_infeasible:
                    return "primal_infeasible", {
                        "ray": y_ray,
                        "certificate_quality": relative_infeas,
                        "dual_ray_obj": dual_ray_obj,
                        "dual_residual": dual_residual,
                    }

        # Check for dual infeasibility (primal unbounded via primal ray)
        primal_norm_inf = torch.linalg.norm(x_unscaled, ord=float('inf'))
        if primal_norm_inf > eps_zero:
            x_ray = x_unscaled / primal_norm_inf
            primal_ray_obj = (c_orig @ x_ray).item()
            if primal_ray_obj < 0:
                Kx = K_orig @ x_ray
                residual_vec = torch.cat([torch.clamp(-Kx[:m1], min=0.0), Kx[m1:]])
                max_primal_residual = torch.linalg.norm(residual_vec, ord=float('inf')).item()
                relative_infeas = max_primal_residual / (-primal_ray_obj)
                if relative_infeas < eps_dual_infeasible:
                    return "dual_infeasible", {
                        "ray": x_ray,
                        "certificate_quality": relative_infeas,
                        "primal_ray_obj": primal_ray_obj,
                        "max_primal_residual": max_primal_residual,
                    }

        # Check for optimality
        dual_obj = compute_dual_objective(x_unscaled, y_unscaled)
        primal_obj = c_orig @ x_unscaled

        # condition (1): relative duality gap: |primal_obj - dual_obj| / (1 + |primal_obj| + |dual_obj|)
        gap_num = torch.abs(dual_obj - primal_obj)
        gap_den = 1.0 + torch.abs(dual_obj) + torch.abs(primal_obj)
        gap_ok = gap_num <= eps_tol * (1.0 + torch.abs(dual_obj) + torch.abs(primal_obj))

        # condition (2): primal feasibility
        r_eq = b_orig - (A_orig @ x_unscaled)
        r_ineq = torch.clamp(h_orig - (G_orig @ x_unscaled), min=0.0)
        feas = torch.sqrt((r_eq @ r_eq) + (r_ineq @ r_ineq))
        feas_ok = feas <= eps_tol * (1.0 + torch.linalg.norm(q_orig))

        # condition (3): stationarity (dual feasibility)
        g_orig = c_orig - (K_orig.T @ y_unscaled)
        lam = compute_lambda_for_box(x_unscaled, g_orig, l_orig, u_orig)
        stat = torch.linalg.norm(g_orig - lam)
        stat_ok = stat <= eps_tol * (1.0 + torch.linalg.norm(c_orig))

        optimal = bool(gap_ok and feas_ok and stat_ok)
        # return objectives so they don't need to be recomputed for printing
        return "optimal" if optimal else "", {
            "primal_obj": primal_obj.item(),
            "dual_obj": dual_obj.item(),
        }

    @torch.no_grad()
    def adaptive_step_pdhg(
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        eta_hat: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One adaptive PDHG step"""
        eta = torch.as_tensor(eta_hat, device=x.device, dtype=x.dtype).clamp_min(eps_zero)

        kp1 = float(k + 1)
        fac1 = 1.0 if k == 0 else 1.0 - (kp1 ** -0.3)
        fac2 = 1.0 + (kp1 ** -0.6)

        for _ in range(max_backtrack):
            x_p = proj_X(x - (eta / w) * (c - K.T @ y))
            y_p = proj_Y(y + (eta * w) * (q - K @ (2.0 * x_p - x)))

            dx = x_p - x
            dy = y_p - y

            num = w * (dx @ dx) + (dy @ dy) / w
            denom = 2 * torch.abs(dy @ K @ dx)
            bar_eta = torch.tensor(float("inf"), device=dx.device, dtype=dx.dtype) if denom <= eps_zero else num / denom

            eta_p = torch.minimum(fac1 * bar_eta, fac2 * eta).clamp_min(eps_zero)

            if eta <= bar_eta:
                return x_p, y_p, eta, eta_p
            eta = eta_p

        return x_p, y_p, eta, eta

    # -----------------------------
    # Main Algorithm
    # -----------------------------

    # Initializations

    # initialize to zeros
    x0 = proj_X(torch.zeros(n, device=device, dtype=dtype))
    y0 = torch.zeros(m, device=device, dtype=dtype)

    # step size 1/||K||_inf
    row_sums = K.abs().sum(dim=1)
    if is_sparse: row_sums = row_sums.to_dense() # clamp doesn't work on sparse
    eta_hat = (1.0 / row_sums.max().clamp_min(eps_zero)).to(device=device, dtype=dtype)

    # initial weight
    w = (torch.linalg.norm(c) / torch.linalg.norm(q).clamp_min(eps_zero)).clamp_min(eps_zero)

    x, y = x0.clone(), y0.clone() # current iterate
    x_prev, y_prev = x.clone(), y.clone() # past iterate

    # initialize candidate at t=0
    x_c, y_c = x.clone(), y.clone()

    # restart parameters
    beta_sufficient = 0.2 # used for sufficient progress condition
    beta_necessary = 0.8 # used for necessary progress condition
    beta_artificial = 0.36 # used for artificial restart condition

    n_iterations = 0 # total iteration counter
    status = ""
    info = {}
    x_unscaled_last, y_unscaled_last = x / variable_rescaling, y / constraint_rescaling

    while True:
        # compute KKT of last restart point with current primal weight
        kkt_last_restart = kkt_error_sq(x, y, w)

        # Save unscaled values for final return
        x_unscaled, y_unscaled = x / variable_rescaling, y / constraint_rescaling
        x_unscaled_last, y_unscaled_last = x_unscaled.clone(), y_unscaled.clone()

        # reset averaging at start of each outer loop
        eta_sum = 0.0
        x_bar, y_bar = x.clone(), y.clone()
        kkt_c_prev = kkt_last_restart # initialize for first iteration

        for t in range(max_inner_iters):
            x, y, eta_used, eta_hat = adaptive_step_pdhg(x, y, w, eta_hat, n_iterations)

            # weighted average of iterates, weighted by step-size
            eta_sum += float(eta_used)
            alpha = float(eta_used) / eta_sum
            x_bar = x_bar + alpha * (x - x_bar)
            y_bar = y_bar + alpha * (y - y_bar)

            # choose restart candidate: choose one with lower KKT
            kkt_current = kkt_error_sq(x, y, w)
            kkt_averaged = kkt_error_sq(x_bar, y_bar, w)
            x_c_new, y_c_new = (x, y) if (kkt_current < kkt_averaged) else (x_bar, y_bar)
            kkt_c_new = kkt_current if (kkt_current < kkt_averaged) else kkt_averaged

            n_iterations += 1

            # check iteration and time limits
            if n_iterations >= iteration_limit:
                status = "iteration_limit"
                x_unscaled_last, y_unscaled_last = x / variable_rescaling,y / constraint_rescaling
                break
            if time.time() - start_time >= time_sec_limit:
                status = "time_limit"
                x_unscaled_last, y_unscaled_last = x / variable_rescaling, y / constraint_rescaling
                break

            # check termination and restart: first 10 iters, then every frequency
            if n_iterations <= 10 or n_iterations % termination_check_frequency == 0:
                status, info = termination_criteria(x, y)
                if verbose:
                    print(f"  Iter {n_iterations:5d}: primal_obj = {info['primal_obj']:+.6e}, dual_obj = {info['dual_obj']:+.6e}, gap = {abs(info['primal_obj'] - info['dual_obj']):.3e}, KKT = {torch.sqrt(kkt_current).item():.3e}")
                if status:
                    # ignore infeasibility detections before iteration 10 (early false positives)
                    if n_iterations < 10 and status in ["primal_infeasible", "dual_infeasible"]:
                        status = "" # reset status, keep iterating
                    else:
                        # save the iterate where we terminated
                        x_unscaled_last, y_unscaled_last = x / variable_rescaling, y / constraint_rescaling
                        break # optimal or detected infeas/unbound after warm-up

                # check restart criteria (only if didn't terminate above)
                cond_i  = (kkt_c_new <= (beta_sufficient**2) * kkt_last_restart) # sufficient progress made
                cond_ii = (kkt_c_new <= (beta_necessary**2) * kkt_last_restart) and (t > 0) and (kkt_c_new > kkt_c_prev) # necessary progress + stalling
                cond_iii = (t >= beta_artificial * n_iterations) # too many inner iterations

                if cond_i or cond_ii or cond_iii:
                    x_c, y_c = x_c_new, y_c_new
                    break

            kkt_c_prev = kkt_c_new # save for next iteration
        else:
            x_c, y_c = x_c_new, y_c_new

        if status: break # break out of loop if we have termination status

        # restart from candidate
        x, y = x_c, y_c

        # primal weight update
        w = primal_weight_update(x, y, x_prev, y_prev, w)

        # store previous restart start
        x_prev, y_prev = x.clone(), y.clone()

    info.update({'solve_time_sec': time.time() - start_time, 'iterations': n_iterations})
    if status in ["optimal", "iteration_limit", "time_limit"]:
        info['kkt_error_sq'] = kkt_error_sq(x, y, w).item()
        r_eq, r_ineq = b_orig - (A_orig @ x_unscaled_last), torch.clamp(h_orig - (G_orig @ x_unscaled_last), min=0.0)
        info['primal_residual'] = torch.sqrt((r_eq @ r_eq) + (r_ineq @ r_ineq)).item()
        g_orig = c_orig - (K_orig.T @ y_unscaled_last)
        info['dual_residual'] = torch.linalg.norm(g_orig - compute_lambda_for_box(x_unscaled_last, g_orig, l_orig, u_orig)).item()
        info['duality_gap'] = abs(info['primal_obj'] - info['dual_obj'])
        info['relative_gap'] = info['duality_gap'] / (1.0 + abs(info['primal_obj']) + abs(info['dual_obj']))

    if verbose:
        if status == "primal_infeasible":
            print(f"\n  Status: PRIMAL INFEASIBLE after {n_iterations} iterations in {info['solve_time_sec']:.3f}s")
            print(f"    Farkas certificate (dual ray): y = {info['ray']}")
            print(f"      K^T y ≈ 0:  ||K^T y|| = {info['dual_residual']:.3e}")
            print(f"      q^T y > 0:  q^T y = {info['dual_ray_obj']:.3e}")
            print(f"      Relative certificate quality: {info['certificate_quality']:.3e}")
        elif status == "dual_infeasible":
            print(f"\n  Status: DUAL INFEASIBLE (primal unbounded) after {n_iterations} iterations in {info['solve_time_sec']:.3f}s")
            print(f"    Unboundedness certificate (primal ray): x = {info['ray']}")
            print(f"      K x ≈ 0:  max_residual = {info['max_primal_residual']:.3e}")
            print(f"      c^T x < 0:  c^T x = {info['primal_ray_obj']:.3e}")
            print(f"      Relative certificate quality: {info['certificate_quality']:.3e}")
        elif status in ["optimal", "iteration_limit", "time_limit"]:
            if status == "optimal":
                status_msg = "converged"
            elif status == "iteration_limit":
                status_msg = f"iteration limit ({iteration_limit})"
            else:  # time_limit
                status_msg = f"time limit ({time_sec_limit:.1f}s)"
            print(f"\n  Status: {status_msg} after {n_iterations} iterations in {info['solve_time_sec']:.3f}s")
            print(f"  Primal objective: {info['primal_obj']:.6e}")
            print(f"  Dual objective: {info['dual_obj']:.6e}")
            print(f"  Duality gap: {info['duality_gap']:.6e}")

    return x_unscaled_last, y_unscaled_last, status, info