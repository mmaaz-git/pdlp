"""
Problem rescaling/preconditioning matching cuPDLP.jl implementation.

Implements:
- Ruiz rescaling (L-infinity version)
- Pock-Chambolle rescaling
"""

import torch


def ruiz_rescaling(
    constraint_matrix: torch.Tensor,
    objective_vector: torch.Tensor,
    right_hand_side: torch.Tensor,
    variable_lower_bound: torch.Tensor,
    variable_upper_bound: torch.Tensor,
    num_iterations: int = 10,
) -> tuple:
    """
    Ruiz rescaling with p=Inf (matching cuPDLP.jl default).

    Iteratively rescales rows and columns of the constraint matrix to equilibrate
    their infinity norms.

    Args:
        constraint_matrix: (m, n) constraint matrix
        objective_vector: (n,) objective coefficients
        right_hand_side: (m,) RHS of constraints
        variable_lower_bound: (n,) lower bounds on variables
        variable_upper_bound: (n,) upper bounds on variables
        num_iterations: number of Ruiz iterations (default 10)

    Returns:
        Tuple of (scaled_constraint_matrix, scaled_objective_vector,
                  scaled_right_hand_side, scaled_lower_bound, scaled_upper_bound,
                  constraint_rescaling, variable_rescaling)
    """
    m, n = constraint_matrix.shape
    device = constraint_matrix.device
    dtype = constraint_matrix.dtype

    # Initialize scaling factors
    cum_constraint_rescaling = torch.ones(m, device=device, dtype=dtype)
    cum_variable_rescaling = torch.ones(n, device=device, dtype=dtype)

    # Work on copies
    K = constraint_matrix.clone()
    c = objective_vector.clone()
    q = right_hand_side.clone()
    l = variable_lower_bound.clone()
    u = variable_upper_bound.clone()

    for iteration in range(num_iterations):
        # Compute variable (column) rescaling
        # For each column: max(|K[:,j]|, |c[j]|)^0.5
        col_norms_K = K.abs().max(dim=0)[0]  # max over rows
        col_norms_c = c.abs()
        variable_rescaling = torch.sqrt(torch.maximum(col_norms_K, col_norms_c))

        # Avoid division by zero
        variable_rescaling = torch.where(
            variable_rescaling > 0,
            variable_rescaling,
            torch.ones_like(variable_rescaling)
        )

        # Compute constraint (row) rescaling
        # For each row: max(|K[i,:]|)^0.5
        if m > 0:
            row_norms = K.abs().max(dim=1)[0]  # max over columns
            constraint_rescaling = torch.sqrt(row_norms)
            constraint_rescaling = torch.where(
                constraint_rescaling > 0,
                constraint_rescaling,
                torch.ones_like(constraint_rescaling)
            )
        else:
            constraint_rescaling = torch.tensor([], device=device, dtype=dtype)

        # Apply scaling to problem data
        c = c / variable_rescaling
        l = l * variable_rescaling
        u = u * variable_rescaling

        if m > 0:
            q = q / constraint_rescaling
            # K = diag(1/constraint_rescaling) @ K @ diag(1/variable_rescaling)
            K = K / constraint_rescaling.unsqueeze(1)  # scale rows
            K = K / variable_rescaling.unsqueeze(0)    # scale columns

        # Accumulate scaling factors
        cum_constraint_rescaling = cum_constraint_rescaling * constraint_rescaling
        cum_variable_rescaling = cum_variable_rescaling * variable_rescaling

    return (K, c, q, l, u, cum_constraint_rescaling, cum_variable_rescaling)


def pock_chambolle_rescaling(
    constraint_matrix: torch.Tensor,
    objective_vector: torch.Tensor,
    right_hand_side: torch.Tensor,
    variable_lower_bound: torch.Tensor,
    variable_upper_bound: torch.Tensor,
    alpha: float = 1.0,
) -> tuple:
    """
    Pock-Chambolle rescaling (matching cuPDLP.jl default with alpha=1.0).

    Rescales the constraint matrix such that its operator norm is <= 1.

    Each column j is divided by sqrt(sum_i |K[i,j]|^(2-alpha))
    Each row i is divided by sqrt(sum_j |K[i,j]|^alpha)

    Args:
        constraint_matrix: (m, n) constraint matrix
        objective_vector: (n,) objective coefficients
        right_hand_side: (m,) RHS of constraints
        variable_lower_bound: (n,) lower bounds on variables
        variable_upper_bound: (n,) upper bounds on variables
        alpha: exponent parameter (default 1.0)

    Returns:
        Tuple of (scaled_constraint_matrix, scaled_objective_vector,
                  scaled_right_hand_side, scaled_lower_bound, scaled_upper_bound,
                  constraint_rescaling, variable_rescaling)
    """
    assert 0 <= alpha <= 2, f"alpha must be in [0, 2], got {alpha}"

    m, n = constraint_matrix.shape
    device = constraint_matrix.device
    dtype = constraint_matrix.dtype

    # Work on copies
    K = constraint_matrix.clone()
    c = objective_vector.clone()
    q = right_hand_side.clone()
    l = variable_lower_bound.clone()
    u = variable_upper_bound.clone()

    # Compute variable (column) rescaling
    # For each column: sqrt(sum_i |K[i,j]|^(2-alpha))
    variable_rescaling = torch.sqrt(
        (K.abs() ** (2 - alpha)).sum(dim=0)
    )
    variable_rescaling = torch.where(
        variable_rescaling > 0,
        variable_rescaling,
        torch.ones_like(variable_rescaling)
    )

    # Compute constraint (row) rescaling
    # For each row: sqrt(sum_j |K[i,j]|^alpha)
    constraint_rescaling = torch.sqrt(
        (K.abs() ** alpha).sum(dim=1)
    )
    constraint_rescaling = torch.where(
        constraint_rescaling > 0,
        constraint_rescaling,
        torch.ones_like(constraint_rescaling)
    )

    # Apply scaling to problem data
    c = c / variable_rescaling
    l = l * variable_rescaling
    u = u * variable_rescaling

    q = q / constraint_rescaling
    K = K / constraint_rescaling.unsqueeze(1)  # scale rows
    K = K / variable_rescaling.unsqueeze(0)    # scale columns

    return (K, c, q, l, u, constraint_rescaling, variable_rescaling)


def rescale_problem(
    constraint_matrix: torch.Tensor,
    objective_vector: torch.Tensor,
    right_hand_side: torch.Tensor,
    variable_lower_bound: torch.Tensor,
    variable_upper_bound: torch.Tensor,
    l_inf_ruiz_iterations: int = 10,
    pock_chambolle_alpha: float = 1.0,
    verbose: bool = False,
) -> tuple:
    """
    Full rescaling pipeline matching cuPDLP.jl defaults.

    1. Ruiz rescaling (10 iterations, L-infinity)
    2. Pock-Chambolle rescaling (alpha=1.0)

    Args:
        constraint_matrix: (m, n) constraint matrix [G; A] stacked
        objective_vector: (n,) objective coefficients
        right_hand_side: (m,) RHS [h; b] stacked
        variable_lower_bound: (n,) lower bounds
        variable_upper_bound: (n,) upper bounds
        l_inf_ruiz_iterations: number of Ruiz iterations (default 10)
        pock_chambolle_alpha: Pock-Chambolle alpha parameter (default 1.0)
        verbose: print rescaling info

    Returns:
        Tuple of (scaled_K, scaled_c, scaled_q, scaled_l, scaled_u,
                  constraint_rescaling, variable_rescaling)
    """
    if verbose:
        print(f"Rescaling problem:")
        print(f"  Ruiz iterations: {l_inf_ruiz_iterations}")
        print(f"  Pock-Chambolle alpha: {pock_chambolle_alpha}")

    K = constraint_matrix.clone()
    c = objective_vector.clone()
    q = right_hand_side.clone()
    l = variable_lower_bound.clone()
    u = variable_upper_bound.clone()

    m, n = K.shape
    device = K.device
    dtype = K.dtype

    constraint_rescaling = torch.ones(m, device=device, dtype=dtype)
    variable_rescaling = torch.ones(n, device=device, dtype=dtype)

    # Step 1: Ruiz rescaling (if enabled)
    if l_inf_ruiz_iterations > 0:
        K, c, q, l, u, con_rescale, var_rescale = ruiz_rescaling(
            K, c, q, l, u,
            num_iterations=l_inf_ruiz_iterations,
        )
        constraint_rescaling = constraint_rescaling * con_rescale
        variable_rescaling = variable_rescaling * var_rescale

    # Step 2: Pock-Chambolle rescaling (if alpha provided and not zero)
    if pock_chambolle_alpha is not None and pock_chambolle_alpha > 0:
        K, c, q, l, u, con_rescale2, var_rescale2 = pock_chambolle_rescaling(
            K, c, q, l, u,
            alpha=pock_chambolle_alpha,
        )
        constraint_rescaling = constraint_rescaling * con_rescale2
        variable_rescaling = variable_rescaling * var_rescale2

    if verbose:
        print(f"  Constraint matrix norm after rescaling: {K.norm():.3e}")
        print(f"  Max constraint rescaling: {constraint_rescaling.max():.3e}")
        print(f"  Max variable rescaling: {variable_rescaling.max():.3e}")

    return (K, c, q, l, u, constraint_rescaling, variable_rescaling)


def unscale_solution(
    primal_solution: torch.Tensor,
    dual_solution: torch.Tensor,
    variable_rescaling: torch.Tensor,
    constraint_rescaling: torch.Tensor,
) -> tuple:
    """
    Unscale the solution from the rescaled problem back to original space.

    Args:
        primal_solution: (n,) primal variables from rescaled problem
        dual_solution: (m,) dual variables from rescaled problem
        variable_rescaling: (n,) variable scaling factors
        constraint_rescaling: (m,) constraint scaling factors

    Returns:
        Tuple of (original_primal, original_dual)
    """
    # x_original = x_scaled / variable_rescaling (matching cuPDLP.jl)
    original_primal = primal_solution / variable_rescaling

    # y_original = y_scaled / constraint_rescaling
    original_dual = dual_solution / constraint_rescaling

    return original_primal, original_dual
