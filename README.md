# Primal-Dual Algorithm for Linear Programming

A variant of PDHG (primal-dual hybrid gradient) for linear programming. It tries to leverage matrix operations as much as possible so that we can gain speedups from the GPU. It follows the paper https://arxiv.org/pdf/2311.12180.

The code is meant to have a "linear flow" as much as possible, so that steps are "in line" as much as possible so the reader can see what is going on and easily map it to the math. The entire implementation is around 500 lines (and honestly, ~350 lines of real code, i.e., not printing or docstrings, etc.), entirely in one function `solve()` in `pdlp.py`.