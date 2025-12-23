import torch

def solve(
    G: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    h: torch.Tensor,
    b: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    MAX_OUTER_ITERS: int = 100,
    MAX_INNER_ITERS: int = 100,
    MAX_BACKTRACK: int = 50,
    primal_weight_update_smoothing: float = 0.5,
    ruiz_iterations: int = 10,
    pock_chambolle_alpha: float = 1.0,
    eps_tol: float = 1e-8,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve a Linear Program using the Primal-Dual Hybrid Gradient (PDHG) algorithm.

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
        MAX_OUTER_ITERS: Maximum number of outer iterations (restarts)
        MAX_INNER_ITERS: Maximum number of inner PDHG iterations per restart
        MAX_BACKTRACK: Maximum backtracking steps in adaptive step size
        primal_weight_update_smoothing: Smoothing factor for primal weight updates (0-1)
        ruiz_iterations: Number of Ruiz equilibration iterations
        pock_chambolle_alpha: Pock-Chambolle rescaling parameter (0 = disable)
        eps_tol: Convergence tolerance (1e-4 for moderate, 1e-8 for high quality)
        verbose: Print detailed solver information

    Returns:
        x_sol: Primal solution (n,)
        y_sol: Dual solution (m1 + m2,) where y[:m1] are inequality duals, y[m1:] are equality duals

    Notes:
        - Internally stacks constraints as K = [G; A] and q = [h; b]
        - Applies Ruiz + Pock-Chambolle rescaling for numerical stability
        - Uses adaptive step sizing with backtracking line search
        - Implements restart schemes based on KKT progress
        - Termination criteria checked on original (unscaled) problem
    """
    # -----------------------------
    # Shape checks / setup
    # -----------------------------
    assert G.ndim == 2 and A.ndim == 2
    assert c.ndim == h.ndim == b.ndim == l.ndim == u.ndim == 1
    assert G.shape[0] == h.shape[0]
    assert A.shape[0] == b.shape[0]
    assert G.shape[1] == A.shape[1] == c.shape[0] == l.shape[0] == u.shape[0]

    device = c.device
    dtype = c.dtype
    eps_zero = 1e-12

    m1, n = G.shape
    m2 = A.shape[0]
    m = m1 + m2

    # Stack constraints: K = [G; A], q = [h; b]
    K = torch.cat([G, A], dim=0)
    q = torch.cat([h, b], dim=0)

    # Save originals (for termination checks on original problem)
    G_orig, h_orig, A_orig, b_orig = G.clone(), h.clone(), A.clone(), b.clone()
    c_orig, l_orig, u_orig = c.clone(), l.clone(), u.clone()
    K_orig, q_orig = K.clone(), q.clone()

    # -----------------------------
    # Rescaling / Preconditioning
    # -----------------------------
    constraint_rescaling = torch.ones(m, device=device, dtype=dtype)
    variable_rescaling = torch.ones(n, device=device, dtype=dtype)

    # Ruiz rescaling (L-infinity equilibration)
    for _ in range(ruiz_iterations):
        # Column rescaling: sqrt(max(|K[:,j]|, |c[j]|))
        col_rescale = torch.sqrt(torch.maximum(K.abs().max(dim=0)[0], c.abs())).clamp_min(eps_zero)
        # Row rescaling: sqrt(max(|K[i,:]|))
        row_rescale = torch.sqrt(K.abs().max(dim=1)[0]).clamp_min(eps_zero)

        # Apply rescaling
        c, l, u = c / col_rescale, l * col_rescale, u * col_rescale
        q = q / row_rescale
        K = K / row_rescale.unsqueeze(1) / col_rescale.unsqueeze(0)

        constraint_rescaling *= row_rescale
        variable_rescaling *= col_rescale

    # Pock-Chambolle rescaling (operator norm <= 1)
    if pock_chambolle_alpha > 0:
        alpha = pock_chambolle_alpha
        # Column rescaling: sqrt(sum_i |K[i,j]|^(2-alpha))
        col_rescale = torch.sqrt((K.abs() ** (2 - alpha)).sum(dim=0)).clamp_min(eps_zero)
        # Row rescaling: sqrt(sum_j |K[i,j]|^alpha)
        row_rescale = torch.sqrt((K.abs() ** alpha).sum(dim=1)).clamp_min(eps_zero)

        # Apply rescaling
        c, l, u = c / col_rescale, l * col_rescale, u * col_rescale
        q = q / row_rescale
        K = K / row_rescale.unsqueeze(1) / col_rescale.unsqueeze(0)

        constraint_rescaling *= row_rescale
        variable_rescaling *= col_rescale

    if verbose:
        print(f"Rescaling: Ruiz iters={l_inf_ruiz_iterations}, Pock-Chambolle alpha={pock_chambolle_alpha}")
        print(f"  ||K|| after rescaling: {K.norm():.3e}")

    # split scaled K, q back into G, h, A, b (for algorithm)
    G, h = K[:m1, :], q[:m1]
    A, b = K[m1:, :], q[m1:]

    # -----------------------------
    # Subprocedures
    # -----------------------------
    def proj_X(x: torch.Tensor) -> torch.Tensor:
        """Project x onto box constraints [l, u]."""
        return torch.clamp(x, l, u)

    def proj_Y(y: torch.Tensor) -> torch.Tensor:
        """Project y onto dual feasible set (y[:m1] >= 0, y[m1:] free)."""
        y2 = y.clone()
        y2[:m1] = torch.clamp(y2[:m1], min=0.0)
        return y2

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
            w_new = torch.exp(primal_weight_update_smoothing * torch.log(ratio) +
                            (1.0 - primal_weight_update_smoothing) * torch.log(w_old))
            return w_new.clamp_min(eps_zero)
        return w_old

    @torch.no_grad()
    def compute_lambda_for_box(x: torch.Tensor, g: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> torch.Tensor:
        """
        Computes lambda, the normal-cone component for box constraints at x.
        g = c - K^t y
        It takes x, g, lower, upper so that we can pass either original or scaled.
        """
        lam = torch.zeros_like(x)

        fin_l = torch.isfinite(lower_bound)
        fin_u = torch.isfinite(upper_bound)

        at_l = fin_l & (x <= lower_bound + eps_zero)
        at_u = fin_u & (x >= upper_bound - eps_zero)

        # handle numerically-tight boxes where both flags fire
        both = at_l & at_u
        if both.any():
            dl = (x - lower_bound).abs()
            du = (upper_bound - x).abs()
            at_l = (at_l & ~both) | (both & (dl <= du))
            at_u = (at_u & ~both) | (both & (du < dl))

        lam[at_l] = torch.clamp(g[at_l], min=0.0) # lambda^+ at lower bound
        lam[at_u] = torch.clamp(g[at_u], max=0.0) # lamda^- at upper bound
        return lam

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

        fin_l = torch.isfinite(l)
        fin_u = torch.isfinite(u)

        l_term = (l[fin_l] * lam_pos[fin_l]).sum()
        u_term = (u[fin_u] * lam_minus[fin_u]).sum()

        scalar = (q @ y) + l_term - u_term - (c @ x)
        term3 = scalar * scalar

        return term1 + term2 + term3

    @torch.no_grad()
    def termination_criteria(x_scaled: torch.Tensor, y_scaled: torch.Tensor) -> bool:
        """Check termination on original problem, not rescaled."""
        # Unscale solution back to original space
        x_orig = x_scaled / variable_rescaling
        y_orig = y_scaled / constraint_rescaling

        # Compute gradient and lambda in original space
        g_orig = c_orig - (K_orig.T @ y_orig)
        lam = compute_lambda_for_box(x_orig, g_orig, l_orig, u_orig)

        # Split lambda into + and -
        lam_pos = torch.clamp(lam, min=0.0)
        lam_minus = torch.clamp(-lam, min=0.0)

        fin_l = torch.isfinite(l_orig)
        fin_u = torch.isfinite(u_orig)
        l_term = (l_orig[fin_l] * lam_pos[fin_l]).sum()
        u_term = (u_orig[fin_u] * lam_minus[fin_u]).sum()

        qTy = (q_orig @ y_orig)
        cTx = (c_orig @ x_orig)

        # condition (1) primal gap
        gap_num = torch.abs(qTy + l_term - u_term - cTx)
        gap_den = 1.0 + torch.abs(qTy + l_term - u_term) + torch.abs(cTx)
        gap_ok = (gap_num / gap_den) <= eps_tol

        # condition (2) primal feasibility
        r_eq = b_orig - (A_orig @ x_orig)
        r_ineq = torch.clamp(h_orig - (G_orig @ x_orig), min=0.0)
        feas = torch.sqrt((r_eq @ r_eq) + (r_ineq @ r_ineq))
        feas_ok = feas <= eps_tol * (1.0 + torch.linalg.norm(q_orig))

        # condition (3) stationarity
        stat = torch.linalg.norm(g_orig - lam)
        stat_ok = stat <= eps_tol * (1.0 + torch.linalg.norm(c_orig))

        return bool(gap_ok and feas_ok and stat_ok)

    @torch.no_grad()
    def adaptive_step_pdhg(
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        eta_hat: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One PDHG step"""
        eta = torch.as_tensor(eta_hat, device=x.device, dtype=x.dtype).clamp_min(eps_zero)

        kp1 = float(k + 1)
        fac1 = 1.0 if k == 0 else 1.0 - (kp1 ** -0.3)
        fac2 = 1.0 + (kp1 ** -0.6)
        fac1 = torch.tensor(fac1, device=x.device, dtype=x.dtype)
        fac2 = torch.tensor(fac2, device=x.device, dtype=x.dtype)

        for _ in range(MAX_BACKTRACK):
            x_p = proj_X(x - (eta / w) * (c - (K.T @ y)))
            y_p = proj_Y(y + (eta * w) * (q - K @ (2.0 * x_p - x)))

            dx = x_p - x
            dy = y_p - y

            num = w * (dx @ dx) + (dy @ dy) / w
            denom = 2 * torch.abs(dy @ K @ dx)

            if denom <= eps_zero:
                bar_eta = torch.tensor(float("inf"), device=dx.device, dtype=dx.dtype)
            else:
                bar_eta = num / denom

            eta_p = torch.minimum(fac1 * bar_eta, fac2 * eta).clamp_min(eps_zero)

            if eta <= bar_eta:
                return x_p, y_p, eta, eta_p
            eta = eta_p

        return x_p, y_p, eta, eta


    # -----------------------------
    # Main Algorithm
    # -----------------------------

    # Initializations

    # Initialize to zeros
    x0 = proj_X(torch.zeros(n, device=device, dtype=dtype))
    y0 = torch.zeros(m, device=device, dtype=dtype)

    # step size 1/||K||_inf
    K_inf = K.abs().sum(dim=1).max().clamp_min(eps_zero)
    eta_hat = (1.0 / K_inf).to(device=device, dtype=dtype)

    w = (torch.linalg.norm(c) / torch.linalg.norm(q)).clamp_min(eps_zero)

    x, y = x0.clone(), y0.clone() # current iterate
    x_prev, y_prev = x.clone(), y.clone() # past iterate

    # initialize candidate at t=0
    x_c, y_c = x.clone(), y.clone()

    # Restart parameters (from cuPDLP.jl defaults)
    beta_sufficient = 0.2   # Sufficient progress threshold
    beta_necessary = 0.8    # Necessary progress threshold
    beta_artificial = 0.36  # Artificial restart threshold

    k_global = 0 # global step counter

    for n_outer in range(MAX_OUTER_ITERS):
        # compute KKT of last restart point with current primal weight
        kkt_last_restart = kkt_error_sq(x, y, w)

        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isnan(y).any():
            print(f"NaN detected at outer iteration {n_outer}!")
            if verbose:
                print(f"Solution in scaled space: x={x.numpy()}, y={y.numpy()}")
            x_orig, y_orig = x / variable_rescaling, y / constraint_rescaling
            if verbose:
                print(f"Solution in original space: x={x_orig.numpy()}, y={y_orig.numpy()}")
            return x_orig, y_orig
        if torch.isinf(x).any() or torch.isinf(y).any():
            print(f"Inf detected at outer iteration {n_outer}!")
            if verbose:
                print(f"Solution in scaled space: x={x.numpy()}, y={y.numpy()}")
            x_orig, y_orig = x / variable_rescaling, y / constraint_rescaling
            if verbose:
                print(f"Solution in original space: x={x_orig.numpy()}, y={y_orig.numpy()}")
            return x_orig, y_orig

        # reset averaging at start of each outer loop
        eta_sum = 0.0
        x_bar, y_bar = x.clone(), y.clone()
        kkt_c_prev = kkt_last_restart  # Initialize for first iteration

        for t in range(MAX_INNER_ITERS):
            if termination_criteria(x, y):
                # print(f"Terminated at iteration {k_global}")
                if verbose:
                    print(f"Solution in scaled space: x={x.numpy()}, y={y.numpy()}")
                x_orig, y_orig = x / variable_rescaling, y / constraint_rescaling
                if verbose:
                    print(f"Solution in original space: x={x_orig.numpy()}, y={y_orig.numpy()}")
                return x_orig, y_orig

            x, y, eta_used, eta_hat = adaptive_step_pdhg(x, y, w, eta_hat, k_global)

            # online weighted average
            eta_sum += float(eta_used)
            alpha = float(eta_used) / eta_sum
            x_bar = x_bar + alpha * (x - x_bar)
            y_bar = y_bar + alpha * (y - y_bar)

            # choose restart candidate: choose one with lower KKT
            kkt_z = kkt_error_sq(x, y, w)
            kkt_b = kkt_error_sq(x_bar, y_bar, w)
            x_c_new, y_c_new = (x, y) if (kkt_z < kkt_b) else (x_bar, y_bar)
            kkt_c_new = kkt_z if (kkt_z < kkt_b) else kkt_b

            k_global += 1

            # restart criteria
            cond_i  = (kkt_c_new <= (beta_sufficient**2) * kkt_last_restart)
            cond_ii = (kkt_c_new <= (beta_necessary**2) * kkt_last_restart) and (t > 0) and (kkt_c_new > kkt_c_prev)
            cond_iii = (t >= beta_artificial * k_global)

            kkt_c_prev = kkt_c_new # save for next iteration

            if cond_i or cond_ii or cond_iii:
                x_c, y_c = x_c_new, y_c_new
                break
        else:
            x_c, y_c = x_c_new, y_c_new

        # restart from candidate
        x, y = x_c, y_c

        # primal weight update
        w = primal_weight_update(x, y, x_prev, y_prev, w)

        # store previous restart start
        x_prev, y_prev = x.clone(), y.clone()

    # Unscale solution back to original space
    if verbose:
        print(f"\nSolution in scaled space: x={x.numpy()}, y={y.numpy()}")
    x_orig, y_orig = x / variable_rescaling, y / constraint_rescaling
    if verbose:
        print(f"Solution in original space: x={x_orig.numpy()}, y={y_orig.numpy()}")
    return x_orig, y_orig