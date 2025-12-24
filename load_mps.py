"""
Simple MPS file parser for LP problems.
Converts to format: G, A, c, h, b, l, u
"""
import torch
from collections import defaultdict


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
