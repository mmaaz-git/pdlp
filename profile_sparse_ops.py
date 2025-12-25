"""
Profile sparse matrix operations to identify GPU bottlenecks.
"""
import torch
import time

torch.set_default_dtype(torch.float64)

def create_sparse_transportation_matrix(n_suppliers, n_customers):
    """Create sparse constraint matrix for transportation problem."""
    torch.manual_seed(42)
    n_vars = n_suppliers * n_customers
    n_constraints = n_suppliers + n_customers

    # Build sparse K matrix (constraints × variables)
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

    K = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]),
        torch.tensor(vals),
        (n_constraints, n_vars)
    )

    return K

def benchmark_sparse_ops(device_name, n_suppliers, n_customers, n_iters=100):
    """Benchmark key sparse operations."""
    device = torch.device(device_name)

    n_vars = n_suppliers * n_customers
    n_constraints = n_suppliers + n_customers
    nnz = 2 * n_vars
    density = nnz / (n_constraints * n_vars) * 100

    print(f"\n{'='*60}")
    print(f"Device: {device_name.upper()}")
    print(f"Problem: {n_suppliers}×{n_customers} = {n_vars:,} variables")
    print(f"Constraints: {n_constraints:,}")
    print(f"Nonzeros: {nnz:,} (density: {density:.4f}%)")
    print(f"{'='*60}\n")

    # Create sparse matrix and vectors
    K = create_sparse_transportation_matrix(n_suppliers, n_customers).to(device)
    x = torch.randn(n_vars, device=device, dtype=torch.float64)
    y = torch.randn(n_constraints, device=device, dtype=torch.float64)

    # Warmup
    if device.type == 'cuda':
        print("Warming up GPU...")
        for _ in range(10):
            _ = K @ x
            _ = K.T @ y
        torch.cuda.synchronize()

    # Benchmark K @ x (constraint × variable -> constraint)
    print(f"Benchmarking K @ x ({n_constraints:,} × {n_vars:,} -> {n_constraints:,}):")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        result = K @ x
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time_ms = (elapsed / n_iters) * 1000
    print(f"  Average: {avg_time_ms:.3f} ms per op")
    print(f"  Throughput: {nnz * n_iters / elapsed / 1e9:.3f} GFLOP/s")

    # Benchmark K.T @ y (variable × constraint -> variable)
    print(f"\nBenchmarking K.T @ y ({n_vars:,} × {n_constraints:,} -> {n_vars:,}):")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        result = K.T @ y
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time_ms = (elapsed / n_iters) * 1000
    print(f"  Average: {avg_time_ms:.3f} ms per op")
    print(f"  Throughput: {nnz * n_iters / elapsed / 1e9:.3f} GFLOP/s")

    # Benchmark combined operation (simulates one PDHG iteration)
    print(f"\nBenchmarking combined (K @ x + K.T @ y):")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        r1 = K @ x
        r2 = K.T @ y
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time_ms = (elapsed / n_iters) * 1000
    print(f"  Average: {avg_time_ms:.3f} ms per iteration")
    print(f"  Estimated time for 500 iters: {(avg_time_ms * 500) / 1000:.1f}s")


if __name__ == "__main__":
    print("Sparse Matrix Operation Profiler")
    print("="*60)

    # Test sizes matching the benchmark
    sizes = [
        (100, 150, "Medium"),
        (200, 300, "Large"),
        (500, 750, "Very Large"),
    ]

    for n_s, n_c, label in sizes:
        print(f"\n\n{label} Problem: {n_s}×{n_c}")
        print("="*60)

        # CPU benchmark
        benchmark_sparse_ops("cpu", n_s, n_c, n_iters=50)

        # GPU benchmark
        if torch.cuda.is_available():
            benchmark_sparse_ops("cuda", n_s, n_c, n_iters=100)
        else:
            print("\n⚠️  No CUDA GPU available")
