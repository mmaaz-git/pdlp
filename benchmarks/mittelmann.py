"""
Benchmark MPS files from Mittelmann benchmarks on GPU.
"""

import torch
import time
import os
import sys
import urllib.request
import bz2
import gzip

# Add parent directory to path so we can import pdlp and cli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import parse_mps
from pdlp import solve


def download_mps(url, filename):
    """Download MPS file if not already present."""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists")
        return filename

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"✓ Downloaded to {filename}")
    return filename


def decompress_mps(compressed_file):
    """Decompress .bz2 or .gz MPS file."""
    if compressed_file.endswith('.bz2'):
        decompressed = compressed_file[:-4]
        if os.path.exists(decompressed):
            print(f"✓ {decompressed} already exists")
            return decompressed

        print(f"Decompressing {compressed_file}...")
        with bz2.open(compressed_file, 'rb') as f_in:
            with open(decompressed, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✓ Decompressed to {decompressed}")
        return decompressed

    elif compressed_file.endswith('.gz'):
        decompressed = compressed_file[:-3]
        if os.path.exists(decompressed):
            print(f"✓ {decompressed} already exists")
            return decompressed

        print(f"Decompressing {compressed_file}...")
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(decompressed, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✓ Decompressed to {decompressed}")
        return decompressed

    else:
        return compressed_file  # Already decompressed


def benchmark_mps(mps_file, device='cuda', time_limit=600, use_sparse=True):
    """Load and solve an MPS file with timing."""
    print("\n" + "="*80)
    print(f"BENCHMARK: {mps_file}")
    print("="*80)

    # Load problem
    print(f"\nLoading MPS file (sparse={use_sparse})...")
    load_start = time.time()
    G, A, c, h, b, l, u = parse_mps(mps_file, sparse=use_sparse)
    load_time = time.time() - load_start

    print(f"Load time: {load_time:.2f}s")
    print(f"\nProblem size:")
    print(f"  Variables: {G.shape[1]:,}")
    print(f"  Inequality constraints: {G.shape[0]:,}")
    print(f"  Equality constraints: {A.shape[0]:,}")
    print(f"  Total constraints: {G.shape[0] + A.shape[0]:,}")

    if use_sparse:
        G_nnz = G._nnz() if G.shape[0] > 0 else 0
        A_nnz = A._nnz() if A.shape[0] > 0 else 0
        total_nnz = G_nnz + A_nnz
        total_elements = (G.shape[0] + A.shape[0]) * G.shape[1]
        density = total_nnz / total_elements if total_elements > 0 else 0
        print(f"  Nonzeros: {total_nnz:,}")
        print(f"  Density: {density*100:.4f}%")

    # Move to device
    device_obj = torch.device(device)
    G = G.to(device_obj)
    A = A.to(device_obj)
    c = c.to(device_obj)
    h = h.to(device_obj)
    b = b.to(device_obj)
    l = l.to(device_obj)
    u = u.to(device_obj)

    # Solve
    print(f"\n" + "-"*80)
    print(f"SOLVING ON {device.upper()}")
    print(f"Time limit: {time_limit}s ({time_limit/60:.0f} min)")
    print(f"-"*80)

    if device == 'cuda':
        torch.cuda.synchronize()

    solve_start = time.time()
    x, y, status, info = solve(
        c, G, h, A, b, l, u,
        iteration_limit=float('inf'),
        time_sec_limit=time_limit,
        eps_tol=1e-4,
        verbose=True
    )

    if device == 'cuda':
        torch.cuda.synchronize()

    solve_time = time.time() - solve_start

    # Results
    print(f"\n" + "="*80)
    print(f"RESULTS")
    print(f"="*80)
    print(f"Status: {status}")
    print(f"Solve time: {solve_time:.2f}s ({solve_time/60:.1f} min)")
    print(f"Iterations: {info['iterations']}")

    if status in ["optimal", "iteration_limit", "time_limit"]:
        print(f"Primal objective: {info['primal_obj']:.6e}")
        print(f"Dual objective: {info['dual_obj']:.6e}")
        gap = abs(info['primal_obj'] - info['dual_obj'])
        print(f"Duality gap: {gap:.6e}")
        rel_gap = gap / (1 + abs(info['primal_obj']) + abs(info['dual_obj']))
        print(f"Relative gap: {rel_gap:.6e}")

    return {
        'file': mps_file,
        'status': status,
        'load_time': load_time,
        'solve_time': solve_time,
        'iterations': info['iterations'],
        'primal_obj': info.get('primal_obj', None),
        'dual_obj': info.get('dual_obj', None),
    }


if __name__ == "__main__":
    print("="*80)
    print("MPS BENCHMARK - Mittelmann LP Problems")
    print("="*80)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠ No GPU, using CPU")
        device = 'cpu'

    # Mittelmann benchmark problems from https://plato.asu.edu/ftp/lpopt.html
    # All problems verified to exist on server and be listed in benchmark table
    # Format: (url, compressed_filename, name, rows, cols, nnz)
    benchmarks = [
        # Small problems (< 10K rows)
        ("https://plato.asu.edu/ftp/lptestset/qap15.mps.bz2", "qap15.mps.bz2", "qap15", 6331, 22275, 110700),

        # Medium problems (10K - 100K rows)
        ("https://plato.asu.edu/ftp/lptestset/a2864.mps.bz2", "a2864.mps.bz2", "a2864", 22117, 200787, 20078717),
        ("https://plato.asu.edu/ftp/lptestset/ex10.mps.bz2", "ex10.mps.bz2", "ex10", 69609, 17680, 1179680),
        ("https://plato.asu.edu/ftp/lptestset/s82.mps.bz2", "s82.mps.bz2", "s82", 87878, 1690631, 7022608),

        # Large problems (100K - 1M rows)
        ("https://plato.asu.edu/ftp/lptestset/neos-3025225.mps.bz2", "neos-3025225.mps.bz2", "neos-3025225", 91572, 69846, 9357951),
        ("https://plato.asu.edu/ftp/lptestset/rmine15.mps.bz2", "rmine15.mps.bz2", "rmine15", 358395, 42438, 879732),

        # Very large problems (> 1M rows)
        ("https://plato.asu.edu/ftp/lptestset/dlr1.mps.bz2", "dlr1.mps.bz2", "dlr1", 1735470, 9121907, 18365107),
    ]

    print("\nAvailable benchmarks:")
    for i, (url, filename, name, rows, cols, nnz) in enumerate(benchmarks, 1):
        print(f"  {i}. {name:<20} {rows:>9,} rows × {cols:>9,} cols × {nnz:>11,} nnz")

    # Run benchmarks on both CPU and GPU
    results = []
    time_limit = 600  # 10 min per problem per device

    for url, compressed_file, name, rows, cols, nnz in benchmarks:
        try:
            # Download if needed
            compressed_path = download_mps(url, compressed_file)

            # Decompress if needed
            mps_file = decompress_mps(compressed_path)

            # Benchmark on GPU (if available)
            if device == 'cuda':
                result_gpu = benchmark_mps(mps_file, device='cuda', time_limit=time_limit, use_sparse=True)
                result_gpu['device'] = 'GPU'
                result_gpu['name'] = name
                results.append(result_gpu)

            # Benchmark on CPU
            result_cpu = benchmark_mps(mps_file, device='cpu', time_limit=time_limit, use_sparse=True)
            result_cpu['device'] = 'CPU'
            result_cpu['name'] = name
            results.append(result_cpu)

        except Exception as e:
            print(f"\n⚠ Error with {compressed_file}: {e}")
            continue

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Problem':<20} {'Device':<8} {'Status':<12} {'Solve (s)':<12} {'Iters':<10} {'Speedup':<10}")
    print("-"*80)

    # Group results by problem
    problem_results = {}
    for r in results:
        name = r['name']
        if name not in problem_results:
            problem_results[name] = {}
        problem_results[name][r['device']] = r

    for name in [b[2] for b in benchmarks]:  # Maintain order
        if name not in problem_results:
            continue

        gpu_result = problem_results[name].get('GPU')
        cpu_result = problem_results[name].get('CPU')

        if gpu_result:
            speedup = ""
            if cpu_result and gpu_result['status'] == cpu_result['status']:
                speedup = f"{cpu_result['solve_time'] / gpu_result['solve_time']:.2f}x"
            print(f"{name:<20} {'GPU':<8} {gpu_result['status']:<12} {gpu_result['solve_time']:<12.2f} {gpu_result['iterations']:<10} {speedup:<10}")

        if cpu_result:
            print(f"{name:<20} {'CPU':<8} {cpu_result['status']:<12} {cpu_result['solve_time']:<12.2f} {cpu_result['iterations']:<10} {'':<10}")

    print("\n✓ Benchmarking complete!")
