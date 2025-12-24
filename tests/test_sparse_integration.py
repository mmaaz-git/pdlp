import torch
from pdlp import solve


def test_dense_vs_sparse_small():
    """Test that sparse and dense give same result on small problem"""
    n, m1, m2 = 10, 5, 3

    # Create BOUNDED problem (to ensure finite solution)
    torch.manual_seed(42)
    G_dense = torch.randn(m1, n)
    A_dense = torch.randn(m2, n)
    c = torch.randn(n)
    h = torch.randn(m1)
    b = torch.randn(m2)
    l = torch.full((n,), -10.0)  # Bounded below
    u = torch.full((n,), 10.0)   # Bounded above

    # Solve with dense
    x_dense, y_dense, status_dense, info_dense = solve(
        G_dense, A_dense, c, h, b, l, u,
        MAX_OUTER_ITERS=50, verbose=False
    )

    # Solve with sparse
    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()
    x_sparse, y_sparse, status_sparse, info_sparse = solve(
        G_sparse, A_sparse, c, h, b, l, u,
        MAX_OUTER_ITERS=50, verbose=False
    )

    # Compare results
    assert status_dense == status_sparse, f"Status mismatch: {status_dense} vs {status_sparse}"

    x_match = torch.allclose(x_dense, x_sparse, atol=1e-4)
    y_match = torch.allclose(y_dense, y_sparse, atol=1e-4)

    assert x_match, "Primal solution mismatch"
    assert y_match, "Dual solution mismatch"

    if status_dense == "optimal":
        obj_diff = abs(info_dense['primal_obj'] - info_sparse['primal_obj'])
        assert obj_diff < 1e-4, f"Objective mismatch: {obj_diff}"


def test_very_sparse_matrix():
    """Test with very sparse matrix (1% density)"""
    n, m1, m2 = 50, 30, 20
    density = 0.01

    # Create very sparse problem
    torch.manual_seed(123)
    G_dense = torch.randn(m1, n) * (torch.rand(m1, n) < density).float()
    A_dense = torch.randn(m2, n) * (torch.rand(m2, n) < density).float()
    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()

    c = torch.randn(n)
    h = torch.randn(m1)
    b = torch.randn(m2)
    l = torch.full((n,), -10.0)
    u = torch.full((n,), 10.0)

    # This should run without errors
    x, y, status, info = solve(
        G_sparse, A_sparse, c, h, b, l, u,
        MAX_OUTER_ITERS=50, verbose=False
    )

    assert status in ["optimal", "max_iterations"], f"Unexpected status: {status}"


def test_edge_case_no_inequalities():
    """Test with no inequality constraints (m1=0)"""
    n, m2 = 10, 5

    torch.manual_seed(456)
    G_dense = torch.zeros(0, n)  # Empty
    A_dense = torch.randn(m2, n)
    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()

    c = torch.randn(n)
    h = torch.zeros(0)
    b = torch.randn(m2)
    l = torch.full((n,), -float('inf'))
    u = torch.full((n,), float('inf'))

    x, y, status, info = solve(
        G_sparse, A_sparse, c, h, b, l, u,
        MAX_OUTER_ITERS=50, verbose=False
    )

    assert status in ["optimal", "max_iterations", "primal_infeasible", "dual_infeasible"]


def test_edge_case_no_equalities():
    """Test with no equality constraints (m2=0)"""
    n, m1 = 10, 5

    torch.manual_seed(789)
    G_dense = torch.randn(m1, n)
    A_dense = torch.zeros(0, n)  # Empty
    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()

    c = torch.randn(n)
    h = torch.randn(m1)
    b = torch.zeros(0)
    l = torch.full((n,), -float('inf'))
    u = torch.full((n,), float('inf'))

    x, y, status, info = solve(
        G_sparse, A_sparse, c, h, b, l, u,
        MAX_OUTER_ITERS=50, verbose=False
    )

    assert status in ["optimal", "max_iterations", "primal_infeasible", "dual_infeasible"]
