# Primal-Dual Algorithm for Linear Programming

A variant of PDHG (primal-dual hybrid gradient) for linear programming. It tries to leverage matrix operations as much as possible so that we can gain speedups from the GPU. It follows the paper https://arxiv.org/pdf/2311.12180.

The code is meant to have a "linear flow" as much as possible, so that steps are "in line" as much as possible so the reader can see what is going on and easily map it to the math. The entire implementation is around 500 lines (and honestly, ~350 lines of real code, i.e., not printing or docstrings, etc.), entirely in one function `solve()` in `pdlp.py`.

## some benchmarks

```
============================================================
SUMMARY
============================================================
Problem              Format     CPU (s)      GPU (s)      Speedup
----------------------------------------------------------------------
Tiny                 Dense      0.758        2.380        0.32        x
Small                Dense      2.271        5.446        0.42        x
Medium-Dense         Dense      83.180       22.410       3.71        x
Medium-Sparse        Sparse     111.592      41.982       2.66        x
Large-Sparse         Sparse     N/A          82.648       N/A
Very Large-Sparse    Sparse     N/A          600.025      N/A
```



Notes:
- divergence is a major problem.
    - CC saying its bc Julia uses averages?
- missing log in the primal weight update in the paper
- PC scaling
    - had to dig and find the paper...
- missing absolute value!

