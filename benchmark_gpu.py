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


def create_transportation_problem(n_suppliers, n_customers):
    """Create a transportation LP problem."""
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

    # Supply constraints: sum_j x_ij <= s_i  =>  -sum_j x_ij >= -s_i
    G_supply = torch.zeros(n_suppliers, n_vars)
    for i in range(n_suppliers):
        for j in range(n_customers):
            idx = i * n_customers + j
            G_supply[i, idx] = -1.0
    h_supply = -supply

    # Demand constraints: sum_i x_ij >= d_j
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


def benchmark(device_name, n_suppliers, n_customers, n_runs=2):
    """Benchmark solver on specified device."""
    device = torch.device(device_name)

    n_vars = n_suppliers * n_customers

    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name.upper()}")
    print(f"Problem: {n_suppliers} suppliers × {n_customers} customers")
    print(f"Variables: {n_vars}")
    print(f"Constraints: {n_suppliers + n_customers}")
    print(f"{'='*60}")

    # Create problem on device
    G, A, c, h, b, l, u = create_transportation_problem(n_suppliers, n_customers)
    G = G.to(device)
    A = A.to(device)
    c = c.to(device)
    h = h.to(device)
    b = b.to(device)
    l = l.to(device)
    u = u.to(device)

    # Scale max iterations with problem size (roughly sqrt(n))
    # But cap at reasonable values for benchmarking
    max_iters = max(100, min(500, int(200 * (n_vars / 150) ** 0.5)))
    print(f"Max iterations: {max_iters}")

    # Warmup
    if device.type == 'cuda':
        print("Warming up GPU...")
        x, y, status, info = solve(G, A, c, h, b, l, u, verbose=False, MAX_OUTER_ITERS=max_iters)
        torch.cuda.synchronize()

    times = []
    for run in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        x, y, status, info = solve(G, A, c, h, b, l, u, verbose=False, MAX_OUTER_ITERS=max_iters)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        gap = abs(info['primal_obj'] - info['dual_obj'])
        print(f"  Run {run+1}: {elapsed:.3f}s - {status} (gap: {gap:.3e})")

    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.3f}s")
    print(f"Primal obj: {info['primal_obj']:.2f}")
    print(f"Dual obj: {info['dual_obj']:.2f}")
    print(f"Final gap: {abs(info['primal_obj'] - info['dual_obj']):.3e}")

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

    # Test sizes
    sizes = [
        (10, 15, "Tiny"),
        (40, 60, "Small"),
        (100, 150, "Medium"),
        (200, 300, "Large"),
        (400, 600, "Huge"),
    ]

    results = []

    for n_suppliers, n_customers, label in sizes:
        print(f"\n\n{'='*60}")
        print(f"{label} Problem: {n_suppliers}×{n_customers}")
        print(f"{'='*60}")

        # CPU benchmark
        cpu_time = benchmark("cpu", n_suppliers, n_customers)

        # GPU benchmark
        if torch.cuda.is_available():
            gpu_time = benchmark("cuda", n_suppliers, n_customers)
            speedup = cpu_time / gpu_time
            results.append((label, cpu_time, gpu_time, speedup))
        else:
            results.append((label, cpu_time, None, None))

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Problem':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-"*60)

    for label, cpu_time, gpu_time, speedup in results:
        if gpu_time is not None:
            print(f"{label:<12} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<12.2f}x")
        else:
            print(f"{label:<12} {cpu_time:<12.3f} {'N/A':<12} {'N/A':<12}")

    if torch.cuda.is_available():
        print("\n✓ GPU benchmarking complete!")
    else:
        print("\n→ Upload to Google Colab to test GPU performance")
