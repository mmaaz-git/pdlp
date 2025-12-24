import torch


def test_col_max():
    """Test column-wise max for sparse and dense"""
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

    # Test implementation from Change 1
    col_max_sparse = lambda M: torch.zeros(M.shape[1], dtype=M.dtype, device=M.device).scatter_reduce_(
        0, (M_c := M.abs().coalesce()).indices()[1], M_c.values(), reduce='amax', include_self=False)

    result_sparse = col_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=0)[0]

    assert torch.allclose(result_sparse, result_dense, atol=1e-6)


def test_row_max():
    """Test row-wise max for sparse and dense"""
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

    row_max_sparse = lambda M: torch.zeros(M.shape[0], dtype=M.dtype, device=M.device).scatter_reduce_(
        0, (M_c := M.abs().coalesce()).indices()[0], M_c.values(), reduce='amax', include_self=False)

    result_sparse = row_max_sparse(K_sparse)
    result_dense = K_dense.abs().max(dim=1)[0]

    assert torch.allclose(result_sparse, result_dense, atol=1e-6)


def test_slice_rows():
    """Test sparse row slicing"""
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()

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

    result_sparse = sparse_slice_rows(K_sparse, 20, 60)
    result_dense = K_dense[20:60, :]

    assert torch.allclose(result_sparse.to_dense(), result_dense, atol=1e-6)


def test_reciprocal_multiplication():
    """Test that K * (1/a) * (1/b) equals K / a / b"""
    K_dense = torch.randn(100, 50)
    K_sparse = K_dense.to_sparse_coo()
    row_scale = torch.randn(100).abs() + 0.1  # Avoid division by zero
    col_scale = torch.randn(50).abs() + 0.1

    # Sparse version (multiplication)
    result_sparse = K_sparse * (1.0 / row_scale).unsqueeze(1) * (1.0 / col_scale).unsqueeze(0)

    # Dense version (division)
    result_dense = K_dense / row_scale.unsqueeze(1) / col_scale.unsqueeze(0)

    assert torch.allclose(result_sparse.to_dense(), result_dense, atol=1e-5)
