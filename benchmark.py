"""
Benchmark CPU vs GPU performance for PDLP solver.

Run this on Google Colab for free GPU access:
1. Go to colab.research.google.com
2. Upload this file and pdlp.py
3. Runtime -> Change runtime type -> GPU
4. Run this script
"""

import torch
import time
from pdlp import solve

torch.set_default_dtype(torch.float64)


def create_transportation_problem(n_suppliers, n_customers, sparse=False):
    """Create a transportation LP problem.

    Args:
        n_suppliers: Number of suppliers
        n_customers: Number of customers
        sparse: If True, return sparse COO tensors
    """
    torch.manual_seed(42)

    n_vars = n_suppliers * n_customers

    supply = torch.rand(n_suppliers) * 20 + 10
    demand = torch.rand(n_customers) * 15 + 5

    total_demand = demand.sum()
    total_supply = supply.sum()
    if total_supply < total_demand:
        supply = supply * (total_demand / total_supply * 1.2)

    costs = torch.zeros(n_suppliers, n_customers)
    for i in range(n_suppliers):
        for j in range(n_customers):
            costs[i, j] = torch.rand(1).item() * 5 + abs(i - j) * 0.5

    c = costs.flatten()

    if sparse:
        # Build sparse constraint matrices
        # Supply constraints: sum_j x_ij <= s_i  =>  -sum_j x_ij >= -s_i
        G_supply_rows, G_supply_cols, G_supply_vals = [], [], []
        for i in range(n_suppliers):
            for j in range(n_customers):
                idx = i * n_customers + j
                G_supply_rows.append(i)
                G_supply_cols.append(idx)
                G_supply_vals.append(-1.0)

        G_supply = torch.sparse_coo_tensor(
            torch.tensor([G_supply_rows, G_supply_cols]),
            torch.tensor(G_supply_vals),
            (n_suppliers, n_vars)
        )
        h_supply = -supply

        # Demand constraints: sum_i x_ij >= d_j
        G_demand_rows, G_demand_cols, G_demand_vals = [], [], []
        for j in range(n_customers):
            for i in range(n_suppliers):
                idx = i * n_customers + j
                G_demand_rows.append(j)
                G_demand_cols.append(idx)
                G_demand_vals.append(1.0)

        G_demand = torch.sparse_coo_tensor(
            torch.tensor([G_demand_rows, G_demand_cols]),
            torch.tensor(G_demand_vals),
            (n_customers, n_vars)
        )
        h_demand = demand

        G = torch.cat([G_supply, G_demand], dim=0).to_sparse_csr()
        h = torch.cat([h_supply, h_demand])

        A = torch.zeros(0, n_vars).to_sparse_csr()
        b = torch.tensor([])
    else:
        # Build dense constraint matrices (original code)
        G_supply = torch.zeros(n_suppliers, n_vars)
        for i in range(n_suppliers):
            for j in range(n_customers):
                idx = i * n_customers + j
                G_supply[i, idx] = -1.0
        h_supply = -supply

        G_demand = torch.zeros(n_customers, n_vars)
        for j in range(n_customers):
            for i in range(n_suppliers):
                idx = i * n_customers + j
                G_demand[j, idx] = 1.0
        h_demand = demand

        G = torch.vstack([G_supply, G_demand])
        h = torch.cat([h_supply, h_demand])

        A = torch.tensor([]).reshape(0, n_vars)
        b = torch.tensor([])

    l = torch.zeros(n_vars)
    u = torch.ones(n_vars) * float('inf')

    return G, A, c, h, b, l, u


def benchmark(device_name, n_suppliers, n_customers, n_runs=2, sparse=False):
    """Benchmark solver on specified device."""
    device = torch.device(device_name)

    n_vars = n_suppliers * n_customers

    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name.upper()}")
    print(f"Problem: {n_suppliers} suppliers × {n_customers} customers")
    print(f"Variables: {n_vars:,}")
    print(f"Constraints: {n_suppliers + n_customers:,}")
    if sparse:
        # Transportation problems are naturally sparse (0.8% density)
        nnz = 2 * n_vars  # Each var appears in exactly 2 constraints
        density = nnz / ((n_suppliers + n_customers) * n_vars)
        print(f"Format: SPARSE (density: {density*100:.4f}%)")
    else:
        print(f"Format: DENSE")
    print(f"{'='*60}")

    # Create problem on device
    G, A, c, h, b, l, u = create_transportation_problem(n_suppliers, n_customers, sparse=sparse)
    G = G.to(device)
    A = A.to(device)
    c = c.to(device)
    h = h.to(device)
    b = b.to(device)
    l = l.to(device)
    u = u.to(device)

    # Use time limit for benchmarking (10 minutes per run)
    time_limit_sec = 600  # 10 minutes
    max_iters = float('inf')  # No iteration limit, only time limit

    # Use practical tolerance for benchmarking (1e-4 achieves ~1e-5 relative gap)
    # Julia default is 1e-6 but requires more iterations for larger problems
    eps_tol = 1e-4

    print(f"Time limit: {time_limit_sec}s ({time_limit_sec/60:.0f} min), tolerance: {eps_tol:.0e}")

    # Warmup
    if device.type == 'cuda':
        print("Warming up GPU...")
        x, y, status, info = solve(c, G, h, A, b, l, u, verbose=True, iteration_limit=max_iters, time_sec_limit=time_limit_sec, eps_tol=eps_tol)
        torch.cuda.synchronize()

    times = []
    for run in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        print(f"\n--- Run {run+1}/{n_runs} ---")
        start = time.time()
        x, y, status, info = solve(c, G, h, A, b, l, u, verbose=True, iteration_limit=max_iters, time_sec_limit=time_limit_sec, eps_tol=eps_tol)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        if status in ["optimal", "iteration_limit", "time_limit"]:
            gap = abs(info['primal_obj'] - info['dual_obj'])
            print(f"  Run {run+1}: {elapsed:.3f}s - {status} (gap: {gap:.3e}, iters: {info['iterations']})")
        else:
            print(f"  Run {run+1}: {elapsed:.3f}s - {status}")

    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.3f}s")
    if status in ["optimal", "iteration_limit", "time_limit"]:
        print(f"Primal obj: {info['primal_obj']:.2f}")
        print(f"Dual obj: {info['dual_obj']:.2f}")
        print(f"Final gap: {abs(info['primal_obj'] - info['dual_obj']):.3e}")
        print(f"Total iterations: {info['iterations']}")
    else:
        print(f"Status: {status}")

    return avg_time


if __name__ == "__main__":
    print("PDLP Solver: CPU vs GPU Benchmark")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
    else:
        print("✗ No CUDA GPU available")
        print("  Run this on Google Colab for free GPU access!")

    # Test sizes: (suppliers, customers, label, use_sparse)
    # Use sparse for larger problems where it becomes necessary
    sizes = [
        (10, 15, "Tiny", False),
        (40, 60, "Small", False),
        (100, 150, "Medium-Dense", False),
        (100, 150, "Medium-Sparse", True),
        (200, 300, "Large-Sparse", True),
        (500, 750, "Very Large-Sparse", True),
    ]

    results = []

    for n_suppliers, n_customers, label, use_sparse in sizes:
        n_vars = n_suppliers * n_customers

        print(f"\n\n{'='*60}")
        print(f"{label} Problem: {n_suppliers}×{n_customers}")
        print(f"Variables: {n_vars:,}")
        print(f"{'='*60}")

        # Skip dense version for very large problems (would use too much memory)
        if not use_sparse or n_vars < 50000:
            # CPU benchmark
            cpu_time = benchmark("cpu", n_suppliers, n_customers, sparse=use_sparse)

            # GPU benchmark
            if torch.cuda.is_available():
                gpu_time = benchmark("cuda", n_suppliers, n_customers, sparse=use_sparse)
                speedup = cpu_time / gpu_time
                format_str = "Sparse" if use_sparse else "Dense"
                results.append((label, format_str, cpu_time, gpu_time, speedup))
            else:
                format_str = "Sparse" if use_sparse else "Dense"
                results.append((label, format_str, cpu_time, None, None))
        else:
            print(f"\n⚠️  Skipping CPU benchmark (would use too much memory)")
            # GPU only for very large problems
            if torch.cuda.is_available():
                gpu_time = benchmark("cuda", n_suppliers, n_customers, sparse=use_sparse)
                results.append((label, "Sparse", None, gpu_time, None))
            else:
                print(f"⚠️  No GPU available for large sparse problem")

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Problem':<20} {'Format':<10} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-"*70)

    for label, format_str, cpu_time, gpu_time, speedup in results:
        if gpu_time is not None and cpu_time is not None:
            print(f"{label:<20} {format_str:<10} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<12.2f}x")
        elif gpu_time is not None:
            print(f"{label:<20} {format_str:<10} {'N/A':<12} {gpu_time:<12.3f} {'N/A':<12}")
        else:
            print(f"{label:<20} {format_str:<10} {cpu_time:<12.3f} {'N/A':<12} {'N/A':<12}")

    if torch.cuda.is_available():
        print("\n✓ GPU benchmarking complete!")
    else:
        print("\n→ Upload to Google Colab to test GPU performance")
