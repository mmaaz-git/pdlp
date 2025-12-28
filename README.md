# Primal-Dual Algorithm for Linear Programming

A variant of PDHG (primal-dual hybrid gradient) for linear programming. It tries to leverage matrix operations as much as possible so that we can gain speedups from the GPU. It closely follows the papers https://arxiv.org/pdf/2311.12180 and https://arxiv.org/abs/2106.04756.

The code is meant to have a "linear flow" as much as possible, so that steps are "in line" as much as possible so the reader can see what is going on and easily map it to the math. The entire implementation is less than 600 lines (and honestly, ~250 lines of real code, i.e., not logging or docstrings, etc.), entirely in one function `solve()` in `pdlp.py`.

It supports both dense and sparse matrices, and can be used on a CPU or GPU. Just ensure your problem data is on the right device or the right format, and then PyTorch will handle the rest.

For more information, see the <a href="https://www.mmaaz.ca/writings/pdlp.html">blog post</a> I wrote about it.

## Usage

Everything is in `pdlp.py::solve()`. Suppose we have an LP of the form:

```
minimize    c^T x
subject to  G x >= h  (inequality constraints)
            A x  = b  (equality constraints)
            l <= x <= u  (variable bounds)
```

Simply pass it your problem data:

```python
x_sol, y_sol, status, info = solve(c, G, h, A, b, l, u)
```

See the docstring for more details on what it returns and what the arguments are.

There is also a light CLI in `cli.py` that can read MPS files and solve them.

```
python cli.py problem.mps
```

This also accepts various arguments -- see the docstring or `python cli.py --help`.

The `cli.py` script also contains a function `parse_mps()` that can parse MPS files and return the problem data in the format expected by `solve()`.

There are some benchmarking scripts in `benchmarks/` that can be used to test the performance of the solver. One of them, `benchmarks/transport.py`, constructs transportation problems of different sizes and solves them on CPU and GPU. I was using this while developing the solver. You can also run `benchmarks/mittelmann.py` to benchmark the solver on a random selection of Mittelmann benchmark problems (what I report below). It will download the problems, extract them, parse them, then run the solver.

## Benchmarks

Despite its simplicity, the solver is quite fast. Here are the results on some Mittelmann benchmark problems I got with an Intel Xeon CPU and an NVIDIA A100 GPU.

| Problem | Rows × Cols | NNZ (% of entries) | GPU Status | GPU Time (s) | CPU Status | CPU Time (s) |
|--------|-------------|-------------------|------------|--------------|------------|--------------|
| qap15 | 6,330 × 22,275 | 94,950 (0.0673%) | Optimal | **10.20** | Optimal | 71.97 |
| a2864 | 22,117 × 200,787 | 20,078,717 (0.4521%) | Optimal | **37.15** | Optimal | 2884.04 |
| ex10 | 69,608 × 17,680 | 1,162,000 (0.0944%) | Optimal | **1.88** | Optimal | 73.35 |
| s82 | 87,878 × 1,690,631 | 7,022,608 (0.0047%) | Time limit | 3600 | Time limit | 3601.68 |
| neos-3025225 | 91,572 × 69,846 | 9,357,951 (0.1463%) | Optimal | **137.18** | Time limit | 3602.36 |
| rmine15 | 358,395 × 42,438 | 879,732 (0.0058%) | Optimal | **57.69** | Optimal | 2396.96 |
| dlr1 | 1,735,470 × 9,142,907 | 18,365,107 (0.0001%) | Time limit | 3601.19 | Time limit | 3600 |

As you can see, the solver achieves significant speedups on a GPU -- up to 70x speedup! These runtimes are competitive with some of the solvers listed on the Mittelmann benchmark page.