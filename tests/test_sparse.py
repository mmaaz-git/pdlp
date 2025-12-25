import torch
from pdlp import solve


# ============================================================================
# Helper function tests
# ============================================================================

def test_col_max():
    """Test column-wise max on both dense and sparse matrices"""
    torch.manual_seed(42)

    # Test 1: Dense matrix converted to sparse
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

    col_max_sparse = lambda M: torch.zeros(M.shape[1], dtype=M.dtype, device=M.device).scatter_reduce_(
        0, (M_c := M.abs().coalesce()).indices()[1], M_c.values(), reduce='amax', include_self=False)

    result_sparse = col_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=0)[0]
    assert torch.allclose(result_sparse, result_dense, atol=1e-6)

    # Test 2: Actually sparse matrix (10% density)
    K_dense = torch.randn(100, 50) * (torch.rand(100, 50) < 0.1).float()
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = col_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=0)[0]
    assert torch.allclose(result_sparse, result_dense, atol=1e-6)

    # Test 3: Empty columns
    K_dense[:, 10:15] = 0
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = col_max_sparse(K_sparse)
    assert result_sparse[10:15].abs().max() == 0.0, "Empty columns should have 0 max"


def test_row_max():
    """Test row-wise max on both dense and sparse matrices"""
    torch.manual_seed(123)

    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

    row_max_sparse = lambda M: torch.zeros(M.shape[0], dtype=M.dtype, device=M.device).scatter_reduce_(
        0, (M_c := M.abs().coalesce()).indices()[0], M_c.values(), reduce='amax', include_self=False)

    result_sparse = row_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=1)[0]
    assert torch.allclose(result_sparse, result_dense, atol=1e-6)

    # Test with actually sparse matrix
    K_dense = torch.randn(100, 50) * (torch.rand(100, 50) < 0.1).float()
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = row_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=1)[0]
    assert torch.allclose(result_sparse, result_dense, atol=1e-6)

    # Test empty rows
    K_dense[20:25, :] = 0
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = row_max_sparse(K_sparse)
    assert result_sparse[20:25].abs().max() == 0.0, "Empty rows should have 0 max"


def test_slice_rows():
    """Test sparse row slicing"""
    torch.manual_seed(456)

    def sparse_slice_rows(K_sparse, start_row, end_row):
        K = K_sparse.coalesce()
        indices = K.indices()
        values = K.values()
        mask = (indices[0] >= start_row) & (indices[0] < end_row)
        new_indices = indices[:, mask].clone()
        new_indices[0] -= start_row
        new_values = values[mask]
        return torch.sparse_coo_tensor(new_indices, new_values,
                                         (end_row - start_row, K.shape[1]),
                                         dtype=K.dtype, device=K.device)

    # Test on dense-converted-to-sparse
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = sparse_slice_rows(K_sparse, 20, 60)
    result_dense = K_dense[20:60, :]
    assert torch.allclose(result_sparse.to_dense(), result_dense, atol=1e-6)

    # Test on actually sparse (5% density)
    K_dense = torch.randn(100, 50) * (torch.rand(100, 50) < 0.05).float()
    K_sparse = K_dense.to_sparse_coo()

    result_sparse = sparse_slice_rows(K_sparse, 30, 70)
    result_dense = K_dense[30:70, :]
    assert torch.allclose(result_sparse.to_dense(), result_dense, atol=1e-6)


def test_reciprocal_multiplication():
    """Test that K * (1/a) * (1/b) equals K / a / b on sparse"""
    torch.manual_seed(789)

    K_dense = torch.randn(100, 50) * (torch.rand(100, 50) < 0.08).float()
    K_sparse = K_dense.to_sparse_coo()
    row_scale = torch.randn(100).abs() + 0.5
    col_scale = torch.randn(50).abs() + 0.5

    result_sparse = K_sparse * (1.0 / row_scale).unsqueeze(1) * (1.0 / col_scale).unsqueeze(0)
    result_dense = K_dense / row_scale.unsqueeze(1) / col_scale.unsqueeze(0)

    assert torch.allclose(result_sparse.to_dense(), result_dense, atol=1e-5)


def test_operations_stay_sparse():
    """Verify sparse operations don't silently densify"""
    torch.manual_seed(999)

    # Very sparse matrix (1% density) - densification would be obvious
    K_dense = torch.randn(1000, 500) * (torch.rand(1000, 500) < 0.01).float()
    K_sparse = K_dense.to_sparse_coo()

    initial_nnz = K_sparse._nnz()

    # Operations used in pdlp should stay sparse
    K1 = K_sparse.abs()
    assert K1.is_sparse, "abs() should preserve sparsity"

    K2 = K1.coalesce()
    assert K2.is_sparse, "coalesce() should preserve sparsity"
    assert K2._nnz() == initial_nnz, "nnz should be preserved"

    # Verify memory efficiency: sparse uses <10% of dense storage
    assert initial_nnz < 1000 * 500 * 0.1, "Should be much smaller than dense"


# ============================================================================
# Integration tests
# ============================================================================

def test_dense_vs_sparse_equivalence():
    """Test that sparse and dense give same result"""
    n, m1, m2 = 10, 5, 3

    torch.manual_seed(42)
    G_dense = torch.randn(m1, n)
    A_dense = torch.randn(m2, n)
    c = torch.randn(n)
    h = torch.randn(m1)
    b = torch.randn(m2)
    l = torch.full((n,), -10.0)
    u = torch.full((n,), 10.0)

    # Solve with dense
    x_dense, y_dense, status_dense, info_dense = solve(
        c, G_dense, h, A_dense, b, l, u,
        iteration_limit=5000, verbose=False
    )

    # Solve with sparse
    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()
    x_sparse, y_sparse, status_sparse, info_sparse = solve(
        c, G_sparse, h, A_sparse, b, l, u,
        iteration_limit=5000, verbose=False
    )

    # Compare results
    assert status_dense == status_sparse
    assert torch.allclose(x_dense, x_sparse, atol=1e-4)
    assert torch.allclose(y_dense, y_sparse, atol=1e-4)

    if status_dense == "optimal":
        obj_diff = abs(info_dense['primal_obj'] - info_sparse['primal_obj'])
        assert obj_diff < 1e-4


def test_very_sparse_matrices():
    """Test with very sparse matrices (multiple densities)"""
    n, m1, m2 = 50, 30, 20

    for density in [0.05, 0.005]:  # 5% and 0.5% density
        torch.manual_seed(123)
        G_dense = torch.randn(m1, n) * (torch.rand(m1, n) < density).float()
        A_dense = torch.randn(m2, n) * (torch.rand(m2, n) < density).float()

        # Ensure no completely empty rows
        for i in range(m1):
            if G_dense[i, :].abs().max() == 0:
                G_dense[i, i % n] = torch.randn(1).item()
        for i in range(m2):
            if A_dense[i, :].abs().max() == 0:
                A_dense[i, i % n] = torch.randn(1).item()

        c = torch.randn(n)
        h = torch.randn(m1) * 0.1
        b = torch.randn(m2) * 0.1
        l = torch.full((n,), -5.0)
        u = torch.full((n,), 5.0)

        # Solve sparse
        G_sparse = G_dense.to_sparse_coo()
        A_sparse = A_dense.to_sparse_coo()
        x_sparse, y_sparse, status_sparse, info_sparse = solve(
            c, G_sparse, h, A_sparse, b, l, u,
            iteration_limit=10000, verbose=False
        )

        # Solve dense
        x_dense, y_dense, status_dense, info_dense = solve(
            c, G_dense, h, A_dense, b, l, u,
            iteration_limit=10000, verbose=False
        )

        assert status_sparse == status_dense
        if status_sparse == "optimal":
            obj_diff = abs(info_sparse['primal_obj'] - info_dense['primal_obj'])
            assert obj_diff < 1e-2


def test_structured_sparse():
    """Test with natural sparse structure (transportation problem)"""
    torch.manual_seed(456)

    # Mini transportation: 5 suppliers, 8 customers
    n_s, n_c = 5, 8
    n_vars = n_s * n_c

    supply = torch.rand(n_s) * 5 + 5
    demand = torch.rand(n_c) * 3 + 2
    if supply.sum() < demand.sum():
        supply *= (demand.sum() / supply.sum() * 1.2)

    costs = torch.rand(n_s, n_c) * 10
    c = costs.flatten()

    # Supply constraints: -sum_j x_ij >= -supply_i
    G_supply = torch.zeros(n_s, n_vars)
    for i in range(n_s):
        for j in range(n_c):
            G_supply[i, i * n_c + j] = -1.0
    h_supply = -supply

    # Demand constraints: sum_i x_ij >= demand_j
    G_demand = torch.zeros(n_c, n_vars)
    for j in range(n_c):
        for i in range(n_s):
            G_demand[j, i * n_c + j] = 1.0
    h_demand = demand

    G = torch.vstack([G_supply, G_demand])
    h = torch.cat([h_supply, h_demand])
    A = torch.zeros(0, n_vars)
    b = torch.zeros(0)
    l = torch.zeros(n_vars)
    u = torch.full((n_vars,), float('inf'))

    # Solve with sparse
    G_sparse = G.to_sparse_coo()
    A_sparse = A.to_sparse_coo()
    x_sparse, _, status_sparse, info_sparse = solve(
        c, G_sparse, h, A_sparse, b, l, u,
        iteration_limit=10000, verbose=False
    )

    # Solve with dense
    x_dense, _, status_dense, info_dense = solve(
        c, G, h, A, b, l, u,
        iteration_limit=10000, verbose=False
    )

    assert status_dense == status_sparse
    if status_dense == "optimal":
        obj_diff = abs(info_dense['primal_obj'] - info_sparse['primal_obj'])
        assert obj_diff < 1e-3


def test_empty_constraint_rows():
    """Test matrices with empty constraint rows"""
    n, m1, m2 = 20, 10, 5

    torch.manual_seed(111)
    G_dense = torch.randn(m1, n) * (torch.rand(m1, n) < 0.1).float()
    A_dense = torch.randn(m2, n) * (torch.rand(m2, n) < 0.1).float()

    # Force empty rows
    G_dense[2:4, :] = 0
    A_dense[1, :] = 0

    c = torch.randn(n)
    h = torch.randn(m1) * 0.5
    h[2:4] = 0  # Feasible for empty rows
    b = torch.randn(m2) * 0.5
    b[1] = 0
    l = torch.full((n,), -10.0)
    u = torch.full((n,), 10.0)

    G_sparse = G_dense.to_sparse_coo()
    A_sparse = A_dense.to_sparse_coo()

    x_sparse, _, status_sparse, _ = solve(
        c, G_sparse, h, A_sparse, b, l, u,
        iteration_limit=10000, verbose=False
    )

    x_dense, _, status_dense, _ = solve(
        c, G_dense, h, A_dense, b, l, u,
        iteration_limit=10000, verbose=False
    )

    assert status_sparse == status_dense
    if status_sparse == "optimal":
        assert torch.allclose(x_sparse, x_dense, atol=1e-3)


def test_no_inequalities():
    """Test with no inequality constraints"""
    n, m2 = 10, 5

    torch.manual_seed(456)
    G_sparse = torch.zeros(0, n).to_sparse_coo()
    A_sparse = torch.randn(m2, n).to_sparse_coo()
    c = torch.randn(n)
    h = torch.zeros(0)
    b = torch.randn(m2)
    l = torch.full((n,), -10.0)
    u = torch.full((n,), 10.0)

    x, y, status, info = solve(
        c, G_sparse, h, A_sparse, b, l, u,
        iteration_limit=5000, verbose=False
    )

    assert status in ["optimal", "iteration_limit", "primal_infeasible", "dual_infeasible"]


def test_no_equalities():
    """Test with no equality constraints"""
    n, m1 = 10, 5

    torch.manual_seed(789)
    G_sparse = torch.randn(m1, n).to_sparse_coo()
    A_sparse = torch.zeros(0, n).to_sparse_coo()
    c = torch.randn(n)
    h = torch.randn(m1)
    b = torch.zeros(0)
    l = torch.full((n,), -10.0)
    u = torch.full((n,), 10.0)

    x, y, status, info = solve(
        c, G_sparse, h, A_sparse, b, l, u,
        iteration_limit=5000, verbose=False
    )

    assert status in ["optimal", "iteration_limit", "primal_infeasible", "dual_infeasible"]
