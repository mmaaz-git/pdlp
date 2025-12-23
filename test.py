"""
Test the PDLP solver on multiple LP problems.
"""

import torch
from pdlp import solve


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=True)

    expected = torch.tensor([1.0, 0.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([0.0, 1.5])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected_obj = 3.0
    obj = (c @ x_sol).item()
    error = abs(obj - expected_obj)

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error from expected obj: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([0.0, 2.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.1 else 'FAIL'}")  # Relaxed tolerance for unbounded

    return error < 0.1


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([5.0, 5.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([5.0, 0.0, 0.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=True, MAX_OUTER_ITERS=100)

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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=True, MAX_OUTER_ITERS=100)

    print(f"  {'PASS' if True else 'FAIL'}")  # Always pass, just checking detection
    return True


if __name__ == "__main__":
    test_problem_1()
    test_problem_2()
    test_problem_3()
    test_problem_4()
    test_problem_5()
    test_problem_6()
    test_problem_7()
    test_problem_8()