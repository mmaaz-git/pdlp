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

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

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
    Expected solution: x1* = 3, x2* = 0
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
    print("  Expected: x = [3.0, 0.0], obj = 3.0")

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([3.0, 0.0])
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
    Expected solution: x* = [1, 1, 1]
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
    print("  Expected: x = [1.0, 1.0, 1.0], obj = 3.0")

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([1.0, 1.0, 1.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
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
    Expected solution: x* = [2, 0, 3]
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
    print("  Expected: x = [2.0, 0.0, 3.0], obj = 8.0")

    x_sol, y_sol = solve(G, A, c, h, b, l, u, verbose=False)

    expected = torch.tensor([2.0, 0.0, 3.0])
    error = torch.norm(x_sol - expected).item()
    obj = (c @ x_sol).item()

    print(f"\n  Solution: x = [{x_sol[0].item():.6f}, {x_sol[1].item():.6f}, {x_sol[2].item():.6f}]")
    print(f"  Objective: {obj:.6f}")
    print(f"  Error: {error:.6e}")
    print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

    return error < 0.01


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PDLP SOLVER TEST SUITE")
    print("=" * 70)

    results = []
    results.append(("Test 1: Simple 2D inequality", test_problem_1()))
    results.append(("Test 2: Equality constraint", test_problem_2()))
    results.append(("Test 3: 3D multiple inequalities", test_problem_3()))
    results.append(("Test 4: Unbounded variable", test_problem_4()))
    results.append(("Test 5: Tight bounds at optimal", test_problem_5()))
    results.append(("Test 6: Mixed equality/inequality", test_problem_6()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
