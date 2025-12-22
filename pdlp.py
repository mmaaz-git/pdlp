import torch

"""
Consider the LP.

  min_x   c^T x
  s.t.    Gx >= h
          Ax  = b
          l <= x <= u

G: (m1,n), A: (m2,n), c: (n,), h: (m1,), b: (m2,)
K = [G; A]  (m1+m2, n)
q = [h; b]  (m1+m2,)
X = {x : l <= x <= u}
Y = {y : y[:m1] >= 0} (y[m1:] free)
"""


def solve(
    G: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    h: torch.Tensor,
    b: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    MAX_OUTER_ITERS: int = 100,
    MAX_INNER_ITERS: int = 100,
    MAX_BACKTRACK: int = 50,
    theta: float = 0.5,
):
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
    eps_bound = 1e9

    m1, n = G.shape
    m2 = A.shape[0]
    m = m1 + m2

    # Stack constraints
    K = torch.cat([G, A], dim=0) # (m, n)
    q = torch.cat([h, b], dim=0) # (m,)

    # -----------------------------
    # Projections
    # -----------------------------
    def proj_X(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, l, u)

    def proj_Y(y: torch.Tensor) -> torch.Tensor:
        y2 = y.clone()
        y2[:m1] = torch.clamp(y2[:m1], min=0.0)
        return y2

    # -----------------------------
    # Helper functions
    # -----------------------------
    @torch.no_grad()
    def initialize_primal_weight() -> torch.Tensor:
        return (torch.linalg.norm(c) / torch.linalg.norm(q)).clamp_min(eps_zero)

    @torch.no_grad()
    def primal_weight_update(
        x_new: torch.Tensor, y_new: torch.Tensor,
        x_old: torch.Tensor, y_old: torch.Tensor,
        w_old: torch.Tensor,
    ) -> torch.Tensor:
        dx = torch.linalg.norm(x_new - x_old)
        dy = torch.linalg.norm(y_new - y_old)

        if (dx > eps_zero) and (dy > eps_zero):
            ratio = (dy / dx).clamp_min(eps_zero)
            w_new = torch.exp(theta * torch.log(ratio) + (1.0 - theta) * w_old)
            return w_new.clamp_min(eps_zero)
        return w_old

    @torch.no_grad()
    def compute_lambda_for_box(x, g):
        """
        g = c - K^T y.
        λ is the normal-cone component for box constraints at x.
        """
        lam = torch.zeros_like(x)

        fin_l = torch.isfinite(l)
        fin_u = torch.isfinite(u)

        at_l = fin_l & (x <= l + eps_bound)
        at_u = fin_u & (x >= u - eps_bound)

        # handle numerically-tight boxes where both flags fire
        both = at_l & at_u
        if both.any():
            dl = (x - l).abs()
            du = (u - x).abs()
            at_l = (at_l & ~both) | (both & (dl <= du))
            at_u = (at_u & ~both) | (both & (du < dl))

        lam[at_l] = torch.clamp(g[at_l], min=0.0)  # λ^+ at lower bound
        lam[at_u] = torch.clamp(g[at_u], max=0.0)  # λ^- (negative) at upper bound
        return lam

    @torch.no_grad()
    def kkt_error_sq(x, y, w):
        """
        Eq (5): KKT_ω(z)^2
        with z=(x,y), ω=w.
        """
        w = torch.as_tensor(w, device=x.device, dtype=x.dtype).clamp_min(eps_zero)

        # primal residuals
        r_eq = b - (A @ x)
        r_ineq = torch.clamp(h - (G @ x), min=0.0)
        term1 = (w**2) * (r_eq @ r_eq + r_ineq @ r_ineq)

        # stationarity residual + box multipliers
        g = c - (K.T @ y)
        lam = compute_lambda_for_box(x, g)
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
    def get_restart_candidate(x, y, x_bar, y_bar, w):
        """
        z_c := z if KKT(z) < KKT(z_bar) else z_bar.
        (Compare squared KKT; same ordering.)
        """
        kkt_z = kkt_error_sq(x, y, w)
        kkt_b = kkt_error_sq(x_bar, y_bar, w)
        return (x, y) if (kkt_z < kkt_b) else (x_bar, y_bar)

    def should_restart(
        x_cur: torch.Tensor, y_cur: torch.Tensor,
        x_avg: torch.Tensor, y_avg: torch.Tensor,
        x_cand: torch.Tensor, y_cand: torch.Tensor,
        w_val: torch.Tensor,
        t: int, k: int,
    ) -> bool:
        # simplest: never early restart; only restart when inner loop budget ends
        # TODO
        return False

    def termination_criteria(x_cur: torch.Tensor, y_cur: torch.Tensor, k: int) -> bool:
        return False

    @torch.no_grad()
    def adaptive_step_pdhg(
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        eta_hat: torch.Tensor,
        k: int,
    ):
        """
        One PDHG step
        """
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

            dx2 = (dx * dx).sum()
            dy2 = (dy * dy).sum()
            num = w * dx2 + dy2 / w

            Kdx = K @ dx
            denom = 2.0 * (dy * Kdx).sum()

            if denom <= eps_zero:
                bar_eta = torch.tensor(float("inf"), device=dx.device, dtype=dx.dtype)
            else:
                bar_eta = num / denom

            # eta' from paper
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

    w = initialize_primal_weight()

    x, y = x0.clone(), y0.clone() # current iterate
    x_prev, y_prev = x.clone(), y.clone() # past iterate
    kkt0 = kkt_error_sq(x0, y0, w)

    # initialize candidate at t=0
    x_c, y_c = x.clone(), y.clone()
    kkt_c = kkt0

    # Restart parameters (from cuPDLP.jl defaults)
    beta_sufficient = 0.2   # Sufficient progress threshold
    beta_necessary = 0.8    # Necessary progress threshold
    beta_artificial = 0.36  # Artificial restart threshold

    k_global = 0 # global step counter

    for n_outer in range(MAX_OUTER_ITERS):
        # reset averaging at start of each outer loop
        eta_sum = 0.0
        x_bar, y_bar = x.clone(), y.clone()

        for t in range(MAX_INNER_ITERS):
            if termination_criteria(x, y, k_global):
                return x, y

            x, y, eta_used, eta_hat = adaptive_step_pdhg(x, y, w, eta_hat, k_global)

            # online weighted average
            eta_sum += float(eta_used)
            alpha = float(eta_used) / eta_sum
            x_bar = x_bar + alpha * (x - x_bar)
            y_bar = y_bar + alpha * (y - y_bar)

            # choose restart candidate z_c^{n,t+1}
            x_c_new, y_c_new = get_restart_candidate(x, y, x_bar, y_bar, w)
            kkt_c_new = kkt_error_sq(x_c_new, y_c_new, w)

            k_global += 1

            # restart criteria (matching cuPDLP.jl logic)
            cond_i  = (kkt_c_new <= (beta_sufficient**2) * kkt0)
            cond_ii = (kkt_c_new <= (beta_necessary**2) * kkt0) and (kkt_c_new > kkt_c)
            cond_iii = (t >= beta_artificial * k_global)

            # if n_outer < 3 and (cond_i or cond_ii or cond_iii):
            #     print(f"  Restart at n={n_outer}, t={t}: cond_i={cond_i}, cond_ii={cond_ii}, cond_iii={cond_iii}")
            #     print(f"    kkt_c_new={kkt_c_new.item():.3e}, kkt0={kkt0.item():.3e}, kkt_c={kkt_c.item():.3e}")

            if cond_i or cond_ii or cond_iii:
                x_c, y_c = x_c_new, y_c_new
                kkt_c = kkt_c_new
                break

        # restart from candidate
        # if n_outer < 3:  # Debug first few restarts
        #     kkt_cur = kkt_error_sq(x, y, w)
        #     kkt_avg = kkt_error_sq(x_bar, y_bar, w)
        #     kkt_cand = kkt_error_sq(x_c, y_c, w)
        #     print(f"Outer iter {n_outer}: KKT_cur={kkt_cur.item():.3e}, KKT_avg={kkt_avg.item():.3e}, KKT_cand={kkt_cand.item():.3e}")
        #     print(f"  x={x}, y={y}")
        #     print(f"  x_bar={x_bar}, y_bar={y_bar}")
        #     print(f"  Restarting to: x={x_c}, y={y_c}")

        x, y = x_c, y_c

        # primal weight update
        w = primal_weight_update(x, y, x_prev, y_prev, w)

        # store previous restart start
        x_prev, y_prev = x.clone(), y.clone()

    return x, y