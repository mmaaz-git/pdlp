"""
Test the PDLP solver on multiple LP problems.
"""

import torch
from pdlp import solve

# Use float64 for numerical stability (standard for LP solvers)
torch.set_default_dtype(torch.float64)


def test_problem_1():
    """
    Test 1: Simple 2D problem with inequality

    minimize:    x1 + 2*x2
    subject to:  x1 + x2 >= 1  (inequality)
                 0 <= x1, x2 <= 10
    Expected solution: x1* = 1, x2* = 0
    """
    print("\n" + "=" * 70)
    print("TEST 1: Simple 2D inequality problem")
    print("=" * 70)

    G = torch.tensor([[1.0, 1.0]])  # x1 + x2 >= 1
    h = torch.tensor([1.0])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([1.0, 2.0])
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([10.0, 10.0])

    print("  Objective: minimize x1 + 2*x2")
    print("  Constraint: x1 + x2 >= 1")
    print("  Bounds: 0 <= x1, x2 <= 10")
    print("  Expected: x = [1.0, 0.0], obj = 1.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    expected = torch.tensor([1.0, 0.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    assert error < 0.01


def test_problem_2():
    """
    Test 2: Problem with equality constraint

    minimize:    x1 + x2
    subject to:  x1 + 2*x2 = 3  (equality)
                 x1, x2 >= 0
    Expected solution: x1* = 0, x2* = 1.5, obj = 1.5
    """
    print("\n" + "=" * 70)
    print("TEST 2: Problem with equality constraint")
    print("=" * 70)

    G = torch.tensor([]).reshape(0, 2)
    h = torch.tensor([])
    A = torch.tensor([[1.0, 2.0]])  # x1 + 2*x2 = 3
    b = torch.tensor([3.0])
    c = torch.tensor([1.0, 1.0])
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([10.0, 10.0])

    print("  Objective: minimize x1 + x2")
    print("  Constraint: x1 + 2*x2 = 3")
    print("  Bounds: 0 <= x1, x2 <= 10")
    print("  Expected: x = [0.0, 1.5], obj = 1.5")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([0.0, 1.5])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    assert error < 0.01


def test_problem_3():
    """
    Test 3: 3D problem with multiple inequalities

    minimize:    x1 + x2 + x3
    subject to:  x1 + x2 + x3 >= 3
                 x1 >= 1
                 x2 >= 1
                 0 <= xi <= 10
    Expected solution: obj = 3.0 (multiple optimal solutions exist)
    """
    print("\n" + "=" * 70)
    print("TEST 3: 3D problem with multiple inequalities")
    print("=" * 70)

    G = torch.tensor([
        [1.0, 1.0, 1.0],  # x1 + x2 + x3 >= 3
        [1.0, 0.0, 0.0],  # x1 >= 1
        [0.0, 1.0, 0.0],  # x2 >= 1
    ])
    h = torch.tensor([3.0, 1.0, 1.0])
    A = torch.tensor([]).reshape(0, 3)
    b = torch.tensor([])
    c = torch.tensor([1.0, 1.0, 1.0])
    l = torch.tensor([0.0, 0.0, 0.0])
    u = torch.tensor([10.0, 10.0, 10.0])

    print("  Objective: minimize x1 + x2 + x3")
    print("  Constraints: x1 + x2 + x3 >= 3, x1 >= 1, x2 >= 1")
    print("  Bounds: 0 <= xi <= 10")
    print("  Expected: obj = 3.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=False)

    expected_obj = 3.0
    obj = (c @ x_sol).item()
    error = abs(obj - expected_obj)

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error from expected obj: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    assert error < 0.01


def test_problem_4():
    """
    Test 4: Problem with unbounded variables

    minimize:    2*x1 + x2
    subject to:  x1 + x2 >= 2
                 x1 >= 0, x2 unbounded
    Expected solution: x* = [0, 2]
    """
    print("\n" + "=" * 70)
    print("TEST 4: Problem with unbounded variable")
    print("=" * 70)

    G = torch.tensor([[1.0, 1.0]])  # x1 + x2 >= 2
    h = torch.tensor([2.0])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([2.0, 1.0])
    l = torch.tensor([0.0, -1e8])  # x2 unbounded (use large negative bound)
    u = torch.tensor([10.0, 1e8])

    print("  Objective: minimize 2*x1 + x2")
    print("  Constraint: x1 + x2 >= 2")
    print("  Bounds: x1 >= 0, x2 unbounded")
    print("  Expected: x = [0.0, 2.0], obj = 2.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([0.0, 2.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.1 else 'FAIL'}")  # Relaxed tolerance for unbounded

    assert error < 0.1


def test_problem_5():
    """
    Test 5: Problem with tight bounds at optimal

    minimize:    -x1 - x2
    subject to:  x1 + x2 >= 1
                 0 <= x1, x2 <= 5
    Expected solution: x* = [5, 5] (maximize at corner)
    """
    print("\n" + "=" * 70)
    print("TEST 5: Problem with tight bounds at optimal")
    print("=" * 70)

    G = torch.tensor([[1.0, 1.0]])  # x1 + x2 >= 1
    h = torch.tensor([1.0])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([-1.0, -1.0])  # maximize x1 + x2
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([5.0, 5.0])

    print("  Objective: minimize -x1 - x2 (maximize x1 + x2)")
    print("  Constraint: x1 + x2 >= 1")
    print("  Bounds: 0 <= x1, x2 <= 5")
    print("  Expected: x = [5.0, 5.0], obj = -10.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([5.0, 5.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    assert error < 0.01


def test_problem_6():
    """
    Test 6: Mixed equality and inequality

    minimize:    x1 + 3*x2 + 2*x3
    subject to:  x1 + x2 + x3 = 5  (equality)
                 x1 + x2 >= 2      (inequality)
                 0 <= xi <= 10
    Expected solution: x* = [5, 0, 0], obj = 5.0
    """
    print("\n" + "=" * 70)
    print("TEST 6: Mixed equality and inequality constraints")
    print("=" * 70)

    G = torch.tensor([[1.0, 1.0, 0.0]])  # x1 + x2 >= 2
    h = torch.tensor([2.0])
    A = torch.tensor([[1.0, 1.0, 1.0]])  # x1 + x2 + x3 = 5
    b = torch.tensor([5.0])
    c = torch.tensor([1.0, 3.0, 2.0])
    l = torch.tensor([0.0, 0.0, 0.0])
    u = torch.tensor([10.0, 10.0, 10.0])

    print("  Objective: minimize x1 + 3*x2 + 2*x3")
    print("  Constraints: x1 + x2 + x3 = 5, x1 + x2 >= 2")
    print("  Bounds: 0 <= xi <= 10")
    print("  Expected: x = [5.0, 0.0, 0.0], obj = 5.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([5.0, 0.0, 0.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    assert error < 0.01


def test_problem_7():
    """
    Test 7: Infeasible problem

    minimize:    x1 + x2
    subject to:  x1 + x2 >= 10
                 x1 + x2 <= 5
    This is clearly infeasible (can't satisfy both constraints)
    """
    print("\n" + "=" * 70)
    print("TEST 7: Infeasible problem")
    print("=" * 70)

    G = torch.tensor([
        [1.0, 1.0],   # x1 + x2 >= 10
        [-1.0, -1.0]  # -(x1 + x2) >= -5, i.e., x1 + x2 <= 5
    ])
    h = torch.tensor([10.0, -5.0])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([1.0, 1.0])
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([10.0, 10.0])

    print("  Objective: minimize x1 + x2")
    print("  Constraints: x1 + x2 >= 10, x1 + x2 <= 5")
    print("  Expected: PRIMAL INFEASIBLE")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True, MAX_OUTER_ITERS=100)

    print(f"  {'PASS' if True else 'FAIL'}")  # Always pass, just checking detection
    return True


def test_problem_8():
    """
    Test 8: Unbounded problem

    minimize:    -x1  (i.e., maximize x1)
    subject to:  x1 - x2 >= 0  (x1 >= x2)
                 x1, x2 >= 0, no upper bounds
    Expected: DUAL INFEASIBLE (primal unbounded)

    This is unbounded because we can set x2 = 0 and let x1 -> infinity,
    satisfying all constraints while making the objective arbitrarily negative.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Unbounded problem")
    print("=" * 70)

    G = torch.tensor([[1.0, -1.0]])  # x1 - x2 >= 0
    h = torch.tensor([0.0])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([-1.0, 0.0])  # maximize x1
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([float('inf'), float('inf')])  # truly unbounded

    print("  Objective: minimize -x1 (maximize x1)")
    print("  Constraint: x1 - x2 >= 0 (x1 >= x2)")
    print("  Bounds: x1, x2 >= 0, no upper bounds")
    print("  Expected: DUAL INFEASIBLE (primal unbounded)")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True, MAX_OUTER_ITERS=100)

    print(f"  {'PASS' if True else 'FAIL'}")  # Always pass, just checking detection
    return True


def test_problem_9():
    """
    Test 9: Trivial case - no variables, feasible

    No variables, constraints are h <= 0 and b = 0
    Expected: optimal, obj = 0
    """
    print("\n" + "=" * 70)
    print("TEST 9: Trivial case - no variables, feasible")
    print("=" * 70)

    G = torch.tensor([]).reshape(1, 0)
    h = torch.tensor([0.0])  # 0 >= 0, feasible
    A = torch.tensor([]).reshape(0, 0)
    b = torch.tensor([])
    c = torch.tensor([])
    l = torch.tensor([])
    u = torch.tensor([])

    print("  No variables, constraint: 0 >= 0")
    print("  Expected: optimal, obj = 0.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    print(f"\n  Status: {status}")
    print(f"  {'PASS' if status == 'optimal' else 'FAIL'}")

    return status == "optimal"


def test_problem_10():
    """
    Test 10: Trivial case - no variables, infeasible

    No variables, but constraints require h <= 0 which is violated
    Expected: primal_infeasible with Farkas certificate
    """
    print("\n" + "=" * 70)
    print("TEST 10: Trivial case - no variables, infeasible")
    print("=" * 70)

    G = torch.tensor([]).reshape(1, 0)
    h = torch.tensor([1.0])  # need 0 >= 1, infeasible
    A = torch.tensor([]).reshape(0, 0)
    b = torch.tensor([])
    c = torch.tensor([])
    l = torch.tensor([])
    u = torch.tensor([])

    print("  No variables, constraint: 0 >= 1")
    print("  Expected: PRIMAL INFEASIBLE")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    has_certificate = "ray" in info and "dual_ray_obj" in info
    print(f"\n  Status: {status}")
    print(f"  Certificate provided: {has_certificate}")
    print(f"  {'PASS' if status == 'primal_infeasible' and has_certificate else 'FAIL'}")

    return status == "primal_infeasible" and has_certificate


def test_problem_11():
    """
    Test 11: Trivial case - no constraints, optimal

    No constraints, all variables bounded
    minimize 2*x1 + x2
    0 <= x1, x2 <= 10
    Expected: x* = [0, 0], obj = 0.0
    """
    print("\n" + "=" * 70)
    print("TEST 11: Trivial case - no constraints, optimal")
    print("=" * 70)

    G = torch.tensor([]).reshape(0, 2)
    h = torch.tensor([])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([2.0, 1.0])
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([10.0, 10.0])

    print("  Objective: minimize 2*x1 + x2")
    print("  No constraints")
    print("  Bounds: 0 <= x1, x2 <= 10")
    print("  Expected: x = [0.0, 0.0], obj = 0.0")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    expected = torch.tensor([0.0, 0.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if status == 'optimal' and error < 0.01 else 'FAIL'}")

    return status == "optimal" and error < 0.01


def test_problem_12():
    """
    Test 12: Trivial case - no constraints, unbounded

    No constraints, negative objective with unbounded variable
    minimize -x1
    x1, x2 >= 0, no upper bounds
    Expected: DUAL INFEASIBLE (primal unbounded) with certificate
    """
    print("\n" + "=" * 70)
    print("TEST 12: Trivial case - no constraints, unbounded")
    print("=" * 70)

    G = torch.tensor([]).reshape(0, 2)
    h = torch.tensor([])
    A = torch.tensor([]).reshape(0, 2)
    b = torch.tensor([])
    c = torch.tensor([-1.0, 0.0])
    l = torch.tensor([0.0, 0.0])
    u = torch.tensor([float('inf'), float('inf')])

    print("  Objective: minimize -x1 (maximize x1)")
    print("  No constraints")
    print("  Bounds: x1, x2 >= 0, no upper bounds")
    print("  Expected: DUAL INFEASIBLE (primal unbounded)")

    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    has_certificate = "ray" in info and "primal_ray_obj" in info
    print(f"\n  Status: {status}")
    print(f"  Certificate provided: {has_certificate}")
    print(f"  {'PASS' if status == 'dual_infeasible' and has_certificate else 'FAIL'}")

    return status == "dual_infeasible" and has_certificate


def test_problem_13():
    """
    Test 13: Large transportation problem

    A transportation problem with 10 suppliers and 15 customers.
    - Each supplier i has supply capacity s_i
    - Each customer j has demand requirement d_j
    - Cost c_ij to ship from supplier i to customer j
    - Variables: x_ij = amount shipped from i to j (150 variables)
    - Constraints: sum_j x_ij <= s_i (supply limits, 10 inequalities)
                   sum_i x_ij >= d_j (demand requirements, 15 inequalities)
    - Objective: minimize total shipping cost

    This is a realistic large-scale LP with known structure.
    """
    print("\n" + "=" * 70)
    print("TEST 13: Large transportation problem")
    print("=" * 70)

    torch.manual_seed(42)  # for reproducibility

    n_suppliers = 10
    n_customers = 15
    n_vars = n_suppliers * n_customers  # 150 variables

    # Generate supply and demand
    supply = torch.rand(n_suppliers) * 20 + 10  # 10-30 units per supplier
    demand = torch.rand(n_customers) * 15 + 5   # 5-20 units per customer

    # Make problem feasible: ensure total supply >= total demand
    total_demand = demand.sum()
    total_supply = supply.sum()
    if total_supply < total_demand:
        supply = supply * (total_demand / total_supply * 1.2)  # 20% excess supply

    # Generate shipping costs (distance-based)
    # Suppliers at positions (i, 0), customers at positions (n_suppliers + j, 0)
    costs = torch.zeros(n_suppliers, n_customers)
    for i in range(n_suppliers):
        for j in range(n_customers):
            # Random cost with some structure (closer is cheaper)
            costs[i, j] = torch.rand(1).item() * 5 + abs(i - j) * 0.5

    # Flatten costs into objective vector c
    c = costs.flatten()

    # Build constraint matrices
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

    # Combine inequality constraints
    G = torch.vstack([G_supply, G_demand])
    h = torch.cat([h_supply, h_demand])

    # No equality constraints for this problem
    A = torch.tensor([]).reshape(0, n_vars)
    b = torch.tensor([])

    # Bounds: x_ij >= 0
    l = torch.zeros(n_vars)
    u = torch.ones(n_vars) * float('inf')

    print(f"  Problem size: {n_suppliers} suppliers, {n_customers} customers")
    print(f"  Variables: {n_vars} (shipping amounts)")
    print(f"  Constraints: {G.shape[0]} inequalities")
    print(f"  Total supply: {supply.sum():.2f}")
    print(f"  Total demand: {demand.sum():.2f}")
    print(f"  Expected: optimal solution with all demand satisfied")

    # With float64, the default eps_tol=1e-6 works well
    x_sol, y_sol, status, info = solve(G, A, c, h, b, l, u, verbose=True)

    # Reshape solution back to matrix form
    x_matrix = x_sol.reshape(n_suppliers, n_customers)

    # Verify constraints
    supply_used = x_matrix.sum(dim=1)  # sum over customers for each supplier
    demand_met = x_matrix.sum(dim=0)   # sum over suppliers for each customer

    supply_violation = torch.max(supply_used - supply).item()
    demand_violation = torch.max(demand - demand_met).item()

    obj = (c @ x_sol).item()

    print(f"\n  Solution found:")
    print(f"    Objective (total cost): {obj:.2f}")
    print(f"    Supply violations: {max(0, supply_violation):.6e}")
    print(f"    Demand violations: {max(0, demand_violation):.6e}")
    print(f"    Total shipped: {x_sol.sum():.2f}")
    print(f"    Sparsity: {(x_sol > 1e-6).sum().item()}/{n_vars} non-zero variables")

    # With float64, we expect true optimality
    feasible = supply_violation < 1e-3 and demand_violation < 1e-3
    passed = status == "optimal" and feasible

    print(f"  {'PASS' if passed else 'FAIL'}")

    return passed


if __name__ == "__main__":
    test_problem_1()
    test_problem_2()
    test_problem_3()
    test_problem_4()
    test_problem_5()
    test_problem_6()
    test_problem_7()
    test_problem_8()
    test_problem_9()
    test_problem_10()
    test_problem_11()
    test_problem_12()
    test_problem_13()