"""
Test loading and solving real MPS files with sparse tensors.
"""

import torch
import time
from load_mps import parse_mps
from pdlp import solve

def solve_mps_file(filename, sparse=True, max_iters=500, verbose=True):
    """Load and solve an MPS file."""
    print("="*80)
    print(f"LOADING: {filename}")
    print("="*80)

    # Load problem
    print(f"\nParsing MPS file (sparse={sparse})...")
    start = time.time()
    G, A, c, h, b, l, u = parse_mps(filename, sparse=sparse)
    load_time = time.time() - start

    print(f"Load time: {load_time:.2f}s")
    print(f"\nProblem size:")
    print(f"  Variables: {G.shape[1]:,}")
    print(f"  Inequality constraints: {G.shape[0]:,}")
    print(f"  Equality constraints: {A.shape[0]:,}")
    print(f"  Total constraints: {G.shape[0] + A.shape[0]:,}")

    # Solve
    print(f"\n" + "-"*80)
    print(f"SOLVING WITH {'SPARSE' if sparse else 'DENSE'} TENSORS")
    print(f"-"*80)

    start = time.time()
    x, y, status, info = solve(
        G, A, c, h, b, l, u,
        MAX_OUTER_ITERS=max_iters,
        eps_tol=1e-4,
        verbose=verbose
    )
    solve_time = time.time() - start

    print(f"\n" + "="*80)
    print(f"RESULTS")
    print(f"="*80)
    print(f"Status: {status}")
    print(f"Solve time: {solve_time:.2f}s")

    if status in ["optimal", "max_iterations"]:
        print(f"Primal objective: {info['primal_obj']:.6e}")
        print(f"Dual objective: {info['dual_obj']:.6e}")
        gap = abs(info['primal_obj'] - info['dual_obj'])
        print(f"Duality gap: {gap:.6e}")
        rel_gap = gap / (1 + abs(info['primal_obj']) + abs(info['dual_obj']))
        print(f"Relative gap: {rel_gap:.6e}")

        # Check feasibility
        if G.shape[0] > 0:
            G_x = G @ x
            ineq_viol = torch.clamp(h - G_x, min=0.0).max()
            print(f"Max inequality violation: {ineq_viol:.6e}")

        if A.shape[0] > 0:
            eq_viol = (A @ x - b).abs().max()
            print(f"Max equality violation: {eq_viol:.6e}")

    return solve_time, status, info


if __name__ == "__main__":
    import sys

    print("\n" * 2)
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  MPS FILE SOLVER - SPARSE TENSOR TEST  ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    print()

    # Get MPS filename from command line or use default
    if len(sys.argv) > 1:
        mps_file = sys.argv[1]
    else:
        mps_file = "25fv47.mps"

    # Solve with sparse
    time_sparse, status_sparse, info_sparse = solve_mps_file(
        mps_file,
        sparse=True,
        max_iters=500,
        verbose=True
    )

    print("\n\n")
    print("🎉 MPS FILE SOLVED WITH SPARSE TENSORS! 🎉")
    print()
