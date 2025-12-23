# Primal-Dual Algorithm for Linear Programming

A variant of PDHG (primal-dual hybrid gradient) for linear programming. It tries to leverage matrix operations as much as possible so that we can gain speedups from the GPU. It follows the paper https://arxiv.org/pdf/2311.12180.

TODO:
- add infeasible/unbounded check
- solver should return a status field as well
-