#!/usr/bin/env python3
"""
PDLP Solver - Command Line Interface

A GPU-accelerated primal-dual hybrid gradient solver for linear programming.

Usage:
    python cli.py problem.mps [options]
    python cli.py problem.mps --device cuda --time-limit 3600 --output solution.sol

Exit codes:
    0 = optimal solution found
    1 = primal infeasible
    2 = dual infeasible
    3 = iteration limit reached
    4 = time limit reached
    5 = numerical error
    6 = other error
"""

import argparse
import sys
import os
import time
import torch
from pathlib import Path
from collections import defaultdict

from pdlp import solve


def parse_mps(filename, sparse=True):
    """Parse MPS file and return LP in our format.

    Args:
        filename: Path to MPS file
        sparse: If True, return sparse COO tensors for G and A
    """
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Parse sections
    rows = []  # (type, name)
    row_names = {}  # name -> index
    col_names = {}  # name -> index
    coeffs = defaultdict(dict)  # row_name -> {col_name -> value}
    rhs_vals = {}  # row_name -> rhs value
    bounds = {}  # col_name -> (lower, upper)
    obj_name = None

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue

        if line == 'NAME':
            continue
        elif line == 'OBJSENSE':
            continue
        elif line == 'MIN' or line == 'MAX':
            continue
        elif line == 'ROWS':
            section = 'ROWS'
            continue
        elif line == 'COLUMNS':
            section = 'COLUMNS'
            continue
        elif line == 'RHS':
            section = 'RHS'
            continue
        elif line == 'BOUNDS':
            section = 'BOUNDS'
            continue
        elif line == 'ENDATA':
            break

        parts = line.split()

        if section == 'ROWS':
            row_type = parts[0]
            row_name = parts[1]
            if row_type == 'N':
                obj_name = row_name
            else:
                rows.append((row_type, row_name))
                row_names[row_name] = len(rows) - 1

        elif section == 'COLUMNS':
            col_name = parts[0]
            if col_name not in col_names:
                col_names[col_name] = len(col_names)

            # Can have 2 or 4 entries: col_name row1 val1 [row2 val2]
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    row_name = parts[i]
                    value = float(parts[i+1])
                    coeffs[row_name][col_name] = value

        elif section == 'RHS':
            # RHS line: rhs_name row1 val1 [row2 val2]
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    row_name = parts[i]
                    value = float(parts[i+1])
                    rhs_vals[row_name] = value

        elif section == 'BOUNDS':
            bound_type = parts[0]
            col_name = parts[2] if len(parts) >= 3 else parts[1]

            if col_name not in bounds:
                bounds[col_name] = [0.0, float('inf')]

            if bound_type == 'LO':  # Lower bound
                bounds[col_name][0] = float(parts[3])
            elif bound_type == 'UP':  # Upper bound
                bounds[col_name][1] = float(parts[3])
            elif bound_type == 'FX':  # Fixed
                bounds[col_name][0] = float(parts[3])
                bounds[col_name][1] = float(parts[3])
            elif bound_type == 'FR':  # Free
                bounds[col_name][0] = float('-inf')
                bounds[col_name][1] = float('inf')

    # Build tensors
    n = len(col_names)
    m = len(rows)

    print(f"Building tensors: {m} constraints, {n} variables...")

    # Objective
    c = torch.zeros(n)
    if obj_name:
        for col_name, val in coeffs[obj_name].items():
            c[col_names[col_name]] = val

    # Separate E (equality) and G/L (inequality) constraints
    eq_rows = []
    ineq_rows = []
    for i, (row_type, row_name) in enumerate(rows):
        if row_type == 'E':
            eq_rows.append((i, row_name))
        else:
            ineq_rows.append((i, row_type, row_name))

    m1 = len(ineq_rows)  # inequalities
    m2 = len(eq_rows)    # equalities

    # Build G matrix (inequalities as >=)
    if sparse:
        G_rows, G_cols, G_vals = [], [], []
        h = torch.zeros(m1)
        for new_i, (old_i, row_type, row_name) in enumerate(ineq_rows):
            for col_name, val in coeffs[row_name].items():
                G_rows.append(new_i)
                G_cols.append(col_names[col_name])
                if row_type == 'L':  # a'x <= b  =>  -a'x >= -b
                    G_vals.append(-val)
                else:  # G type: a'x >= b
                    G_vals.append(val)
            rhs = rhs_vals.get(row_name, 0.0)
            h[new_i] = -rhs if row_type == 'L' else rhs

        if m1 > 0:
            G = torch.sparse_coo_tensor(
                torch.tensor([G_rows, G_cols]),
                torch.tensor(G_vals),
                (m1, n)
            )
        else:
            G = torch.zeros(0, n).to_sparse_coo()
    else:
        G = torch.zeros(m1, n)
        h = torch.zeros(m1)
        for new_i, (old_i, row_type, row_name) in enumerate(ineq_rows):
            for col_name, val in coeffs[row_name].items():
                if row_type == 'L':  # a'x <= b  =>  -a'x >= -b
                    G[new_i, col_names[col_name]] = -val
                else:  # G type: a'x >= b
                    G[new_i, col_names[col_name]] = val
            rhs = rhs_vals.get(row_name, 0.0)
            h[new_i] = -rhs if row_type == 'L' else rhs

    # Build A matrix (equalities)
    if sparse:
        A_rows, A_cols, A_vals = [], [], []
        b = torch.zeros(m2)
        for new_i, (old_i, row_name) in enumerate(eq_rows):
            for col_name, val in coeffs[row_name].items():
                A_rows.append(new_i)
                A_cols.append(col_names[col_name])
                A_vals.append(val)
            b[new_i] = rhs_vals.get(row_name, 0.0)

        if m2 > 0:
            A = torch.sparse_coo_tensor(
                torch.tensor([A_rows, A_cols]),
                torch.tensor(A_vals),
                (m2, n)
            )
        else:
            A = torch.zeros(0, n).to_sparse_coo()
    else:
        A = torch.zeros(m2, n)
        b = torch.zeros(m2)
        for new_i, (old_i, row_name) in enumerate(eq_rows):
            for col_name, val in coeffs[row_name].items():
                A[new_i, col_names[col_name]] = val
            b[new_i] = rhs_vals.get(row_name, 0.0)

    # Bounds
    l = torch.zeros(n)
    u = torch.ones(n) * float('inf')
    for col_name, idx in col_names.items():
        if col_name in bounds:
            l[idx] = bounds[col_name][0]
            u[idx] = bounds[col_name][1]

    # Print sparsity statistics
    if sparse:
        G_nnz = G._nnz() if m1 > 0 else 0
        A_nnz = A._nnz() if m2 > 0 else 0
        total_nnz = G_nnz + A_nnz
        total_elements = (m1 + m2) * n
        density = total_nnz / total_elements if total_elements > 0 else 0
        print(f"Sparsity: {G_nnz:,} + {A_nnz:,} = {total_nnz:,} nonzeros")
        print(f"Density: {density*100:.4f}% (sparsity: {(1-density)*100:.4f}%)")

    return G, A, c, h, b, l, u


def write_solution_file(x, y, status, info, output_path, problem_name):
    """
    Write solution in standard .sol format.

    Format follows the conventions used by solvers like CPLEX, Gurobi:
    - Variable values with their names (if available)
    - Objective value
    - Status information
    """
    with open(output_path, 'w') as f:
        f.write(f"=obj= {info.get('primal_obj', 'N/A')}\n")
        f.write(f"# Problem: {problem_name}\n")
        f.write(f"# Status: {status}\n")
        f.write(f"# Primal objective: {info.get('primal_obj', 'N/A')}\n")
        f.write(f"# Dual objective: {info.get('dual_obj', 'N/A')}\n")
        f.write(f"# Iterations: {info.get('iterations', 'N/A')}\n")
        f.write(f"# Solve time: {info.get('solve_time', 'N/A'):.2f}s\n")

        if status in ["optimal", "iteration_limit", "time_limit"]:
            gap = abs(info['primal_obj'] - info['dual_obj'])
            rel_gap = gap / (1 + abs(info['primal_obj']) + abs(info['dual_obj']))
            f.write(f"# Duality gap: {gap:.6e}\n")
            f.write(f"# Relative gap: {rel_gap:.6e}\n")

        f.write("\n")

        # Write variable values
        if x is not None:
            x_cpu = x.cpu().numpy() if x.is_cuda else x.numpy()
            for i, val in enumerate(x_cpu):
                # Use generic variable names x0, x1, etc.
                # In a real MPS parser, we'd preserve variable names
                f.write(f"x{i} {val:.17e}\n")


def status_to_exit_code(status):
    """Convert solver status to exit code."""
    if status == "optimal":
        return 0
    elif status == "primal_infeasible":
        return 1
    elif status == "dual_infeasible":
        return 2
    elif status == "iteration_limit":
        return 3
    elif status == "time_limit":
        return 4
    elif status == "numerical_error":
        return 5
    else:
        return 6


def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}hr"


def main():
    parser = argparse.ArgumentParser(
        description="PDLP: GPU-accelerated primal-dual hybrid gradient solver for LP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py problem.mps
  python cli.py problem.mps --device cuda --time-limit 3600
  python cli.py problem.mps --tolerance 1e-6 --output sol.txt --verbose
  python cli.py problem.mps.gz --sparse --device cpu

Modeling language integration:
  - Pyomo: Export to MPS format using `model.write('problem.mps')`
  - CVXPY: Use `problem.solve(solver='PDLP', mps_file='problem.mps')`
  - JuMP: Export to MPS and call this CLI
        """
    )

    # Required arguments
    parser.add_argument(
        'mps_file',
        type=str,
        help='Path to MPS file (supports .mps, .mps.gz, .mps.bz2)'
    )

    # Solver parameters
    parser.add_argument(
        '--tolerance', '--eps', '-e',
        type=float,
        default=1e-6,
        help='Convergence tolerance (default: 1e-4)'
    )

    parser.add_argument(
        '--time-limit', '-t',
        type=float,
        default=3600.0,
        help='Time limit in seconds (default: 3600)'
    )

    parser.add_argument(
        '--iteration-limit', '-i',
        type=int,
        default=None,
        help='Iteration limit (default: no limit)'
    )

    # Device selection
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use: cpu, cuda, or auto (default: auto)'
    )

    # Sparse format
    parser.add_argument(
        '--sparse', '-s',
        action='store_true',
        default=True,
        help='Use sparse tensor format (default: True)'
    )

    parser.add_argument(
        '--dense',
        action='store_true',
        help='Use dense tensor format (overrides --sparse)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output solution file path (default: <mps_name>.sol)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed iteration progress'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    args = parser.parse_args()

    # Validate MPS file exists
    if not os.path.exists(args.mps_file):
        print(f"Error: MPS file not found: {args.mps_file}", file=sys.stderr)
        return 6

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU", file=sys.stderr)
            device = 'cpu'

    # Determine sparse/dense format
    use_sparse = not args.dense

    # Set iteration limit
    iteration_limit = float('inf') if args.iteration_limit is None else args.iteration_limit

    # Determine output path
    if args.output is None:
        mps_path = Path(args.mps_file)
        # Remove all extensions (.mps.gz -> '')
        name = mps_path.name
        while '.' in name:
            name = name.rsplit('.', 1)[0]
        output_path = f"{name}.sol"
    else:
        output_path = args.output

    problem_name = Path(args.mps_file).name

    # Print header (unless quiet)
    if not args.quiet:
        print("="*80)
        print("PDLP Solver - GPU-accelerated Linear Programming")
        print("="*80)
        print(f"Problem: {problem_name}")
        print(f"Device: {device.upper()}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Sparse format: {use_sparse}")
        print(f"Tolerance: {args.tolerance:.0e}")
        print(f"Time limit: {format_time(args.time_limit)}")
        if args.iteration_limit is not None:
            print(f"Iteration limit: {args.iteration_limit:,}")
        print("="*80)

    # Load MPS file
    try:
        if not args.quiet:
            print(f"\nLoading MPS file...")
        load_start = time.time()
        G, A, c, h, b, l, u = parse_mps(args.mps_file, sparse=use_sparse)
        load_time = time.time() - load_start

        if not args.quiet:
            print(f"  Load time: {load_time:.2f}s")
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

    except Exception as e:
        print(f"Error loading MPS file: {e}", file=sys.stderr)
        return 6

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
    try:
        if not args.quiet:
            print(f"\n{'='*80}")
            print("SOLVING")
            print(f"{'='*80}\n")

        if device == 'cuda':
            torch.cuda.synchronize()

        solve_start = time.time()
        x, y, status, info = solve(
            c, G, h, A, b, l, u,
            iteration_limit=iteration_limit,
            time_sec_limit=args.time_limit,
            eps_tol=args.tolerance,
            verbose=args.verbose
        )

        if device == 'cuda':
            torch.cuda.synchronize()

        solve_time = time.time() - solve_start
        info['solve_time'] = solve_time

    except Exception as e:
        print(f"Error during solve: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 6

    # Print results
    if not args.quiet:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Status: {status}")
        print(f"Solve time: {format_time(solve_time)}")
        print(f"Iterations: {info['iterations']:,}")

        if status in ["optimal", "iteration_limit", "time_limit"]:
            print(f"Primal objective: {info['primal_obj']:.10e}")
            print(f"Dual objective: {info['dual_obj']:.10e}")
            gap = abs(info['primal_obj'] - info['dual_obj'])
            rel_gap = gap / (1 + abs(info['primal_obj']) + abs(info['dual_obj']))
            print(f"Duality gap: {gap:.6e}")
            print(f"Relative gap: {rel_gap:.6e}")

            if status == "optimal":
                print("\n✓ Optimal solution found!")
            elif status == "time_limit":
                print("\n⚠ Time limit reached (solution may be suboptimal)")
            elif status == "iteration_limit":
                print("\n⚠ Iteration limit reached (solution may be suboptimal)")

        elif status == "primal_infeasible":
            print("\n✗ Problem is primal infeasible")
        elif status == "dual_infeasible":
            print("\n✗ Problem is dual infeasible (unbounded)")
        else:
            print(f"\n⚠ Solver terminated with status: {status}")

    # Write solution file
    try:
        write_solution_file(x, y, status, info, output_path, problem_name)
        if not args.quiet:
            print(f"\n✓ Solution written to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not write solution file: {e}", file=sys.stderr)

    # Return appropriate exit code
    exit_code = status_to_exit_code(status)
    if not args.quiet:
        print(f"\nExit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
