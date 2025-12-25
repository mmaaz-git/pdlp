"""
Compare GPU performance: Dense vs Sparse (CSR) matrix operations.
"""
import torch
import time

torch.set_default_dtype(torch.float64)

def create_transportation_matrix(n_suppliers, n_customers):
    """Create sparse transportation constraint matrix."""
    torch.manual_seed(42)
    n_vars = n_suppliers * n_customers
    n_constraints = n_suppliers + n_customers

    # Build sparse K matrix
    rows, cols, vals = [], [], []

    # Supply constraints
    for i in range(n_suppliers):
        for j in range(n_customers):
            rows.append(i)
            cols.append(i * n_customers + j)
            vals.append(-1.0)

    # Demand constraints
    for j in range(n_customers):
        for i in range(n_suppliers):
            rows.append(n_suppliers + j)
            cols.append(i * n_customers + j)
            vals.append(1.0)

    K_coo = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]),
        torch.tensor(vals),
        (n_constraints, n_vars)
    )

    K_csr = K_coo.to_sparse_csr()
    K_dense = K_coo.to_dense()

    return K_dense, K_csr, n_constraints, n_vars

def benchmark_format(K, x, y, n_iters=100, device='cuda'):
    """Benchmark matrix-vector operations."""
    K = K.to(device)
    x = x.to(device)
    y = y.to(device)

    # Warmup
    if device == 'cuda':
        for _ in range(10):
            _ = K @ x
            _ = K.T @ y
        torch.cuda.synchronize()

    # Benchmark K @ x
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        result = K @ x
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed_forward = time.time() - start

    # Benchmark K.T @ y
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        result = K.T @ y
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed_backward = time.time() - start

    return elapsed_forward, elapsed_backward

if __name__ == "__main__":
    print("Dense vs Sparse (CSR) GPU Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA GPU available. Upload to Google Colab to test!")
        exit()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Test sizes
    sizes = [
        (50, 75, "Small"),
        (100, 150, "Medium"),
        (200, 300, "Large"),
    ]

    for n_s, n_c, label in sizes:
        n_vars = n_s * n_c
        n_constraints = n_s + n_c
        nnz = 2 * n_vars
        density = nnz / (n_constraints * n_vars) * 100

        print(f"{label}: {n_s}×{n_c} = {n_vars:,} variables")
        print(f"  Constraints: {n_constraints}")
        print(f"  Nonzeros: {nnz:,}")
        print(f"  Density: {density:.4f}%")
        print()

        # Create matrices
        K_dense, K_csr, _, _ = create_transportation_matrix(n_s, n_c)
        x = torch.randn(n_vars, dtype=torch.float64)
        y = torch.randn(n_constraints, dtype=torch.float64)

        # Benchmark dense
        print("  Dense format:")
        fwd_dense, bwd_dense = benchmark_format(K_dense, x, y, n_iters=100)
        print(f"    K @ x:   {fwd_dense*1000:.2f} ms total ({fwd_dense*10:.3f} ms/iter)")
        print(f"    K.T @ y: {bwd_dense*1000:.2f} ms total ({bwd_dense*10:.3f} ms/iter)")
        total_dense = fwd_dense + bwd_dense

        # Benchmark CSR
        print("  CSR sparse format:")
        fwd_csr, bwd_csr = benchmark_format(K_csr, x, y, n_iters=100)
        print(f"    K @ x:   {fwd_csr*1000:.2f} ms total ({fwd_csr*10:.3f} ms/iter)")
        print(f"    K.T @ y: {bwd_csr*1000:.2f} ms total ({bwd_csr*10:.3f} ms/iter)")
        total_csr = fwd_csr + bwd_csr

        speedup = total_dense / total_csr
        print(f"  Speedup: {speedup:.2f}x")
        print()

    print("Note: Speedup should increase with problem size!")
