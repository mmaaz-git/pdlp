"""
Benchmark MPS files from Mittelmann benchmarks on GPU.

Usage in Colab:
1. Upload pdlp.py, load_mps.py, and this file
2. Run this script
3. It will download and solve benchmark MPS files
"""

import torch
import time
import os
import urllib.request
import bz2
import gzip

from load_mps import parse_mps
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

    # Mittelmann benchmark problems (small to large)
    # Format: (url, compressed_filename, description)
    benchmarks = [
        # Small problems (< 10k variables)
        ("https://plato.asu.edu/ftp/lp/mps/25fv47.mps.bz2", "25fv47.mps.bz2", "Small (821 vars, 0.18% dense)"),

        # Medium problems (10k-100k variables)
        ("https://plato.asu.edu/ftp/lp/mps/dfl001.mps.bz2", "dfl001.mps.bz2", "Medium (6,071 vars, 0.77% dense)"),
        ("https://plato.asu.edu/ftp/lp/mps/pilot87.mps.bz2", "pilot87.mps.bz2", "Medium (4,883 vars, 0.42% dense)"),

        # Large problems (100k-1M variables)
        ("https://plato.asu.edu/ftp/lp/mps/nug08.mps.bz2", "nug08.mps.bz2", "Large (1,632 vars, 33.8% dense)"),
    ]

    print("\nAvailable benchmarks:")
    for i, (url, filename, desc) in enumerate(benchmarks, 1):
        print(f"  {i}. {desc} - {filename}")

    # Run benchmarks
    results = []
    time_limit = 600  # 10 minutes per problem

    for url, compressed_file, desc in benchmarks:
        try:
            # Download if needed
            compressed_path = download_mps(url, compressed_file)

            # Decompress if needed
            mps_file = decompress_mps(compressed_path)

            # Benchmark
            result = benchmark_mps(mps_file, device=device, time_limit=time_limit, use_sparse=True)
            results.append(result)

        except Exception as e:
            print(f"\n⚠ Error with {compressed_file}: {e}")
            continue

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Problem':<20} {'Status':<15} {'Load (s)':<10} {'Solve (s)':<10} {'Iters':<10}")
    print("-"*80)

    for r in results:
        filename = os.path.basename(r['file'])
        print(f"{filename:<20} {r['status']:<15} {r['load_time']:<10.2f} {r['solve_time']:<10.2f} {r['iterations']:<10}")

    print("\n✓ Benchmarking complete!")
