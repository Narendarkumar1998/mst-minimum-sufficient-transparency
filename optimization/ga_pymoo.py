"""
Metaheuristics for MST with pymoo, as suggested by M. Dridi.

* ga_mst: single-objective genetic algorithm. Each solution is a binary vector
  over the admissible items (x_i = 1 means item z_i is disclosed); the objective
  is J(S) = lambda_P P(S) + lambda_C C(S) + lambda_K K(S) and the sufficiency
  requirement U(S) >= tau is a constraint, handled by pymoo's feasibility-first
  rule.
* nsga2_mst: multi-objective NSGA-II on (P(S), C(S), K(S)) under U(S) >= tau,
  which returns a set of non-dominated disclosures without fixing the weights.

All quantities use the paper's product forms, with interpretation cost
normalized by the admissible pool and cardinality by its size.
"""

from __future__ import annotations

import math
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize


class _MSTProblem(Problem):
    def __init__(self, u, p, c, admissible, tau, weights=None, multi=False):
        self.adm = np.flatnonzero(np.asarray(admissible, bool))
        self.lu = np.log1p(-np.minimum(np.asarray(u, float)[self.adm], 1 - 1e-15))
        self.lp = np.log1p(-np.minimum(np.asarray(p, float)[self.adm], 1 - 1e-15))
        c = np.asarray(c, float)
        ca = c[self.adm].sum()                      # normalization by the admissible pool D_t
        self.cn = c[self.adm] / ca if ca > 0 else np.zeros(len(self.adm))
        self.n_all = max(len(self.adm), 1)
        self.tau = tau
        self.w = weights
        self.multi = multi
        super().__init__(n_var=len(self.adm), n_obj=3 if multi else 1, n_ieq_constr=1,
                         xl=0, xu=1, vtype=bool)

    def parts(self, X):
        X = X.astype(float)
        U = 1.0 - np.exp(X @ self.lu)
        P = 1.0 - np.exp(X @ self.lp)
        C = X @ self.cn
        K = X.sum(axis=1) / self.n_all
        return U, P, C, K

    def _evaluate(self, X, out, *args, **kwargs):
        U, P, C, K = self.parts(X)
        out["G"] = (self.tau - U)[:, None]
        if self.multi:
            out["F"] = np.column_stack([P, C, K])
        else:
            lp, lc, lk = self.w
            out["F"] = (lp * P + lc * C + lk * K)[:, None]


def _operators(n):
    return dict(sampling=BinaryRandomSampling(), crossover=TwoPointCrossover(),
                mutation=BitflipMutation(prob=1.0, prob_var=1.0 / max(n, 1)),
                eliminate_duplicates=True)


def ga_mst(u, p, c, admissible, tau, weights, pop_size=100, n_gen=200, seed=0):
    """Returns (status, selected indices into the complete pool, J, runtime, evaluations)."""
    t0 = time.perf_counter()
    prob = _MSTProblem(u, p, c, admissible, tau, weights)
    if 1.0 - math.exp(prob.lu.sum()) < tau:
        return "INFEASIBLE", (), math.nan, time.perf_counter() - t0, 0
    algo = GA(pop_size=pop_size, **_operators(prob.n_var))
    res = minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    rt = time.perf_counter() - t0
    if res.X is None:
        return "NO_FEASIBLE_FOUND", (), math.nan, rt, res.algorithm.evaluator.n_eval
    x = np.asarray(res.X, bool).ravel()
    sel = tuple(int(i) for i in prob.adm[x])
    return "FEASIBLE", sel, float(res.F[0]), rt, res.algorithm.evaluator.n_eval


def nsga2_mst(u, p, c, admissible, tau, pop_size=100, n_gen=200, seed=0):
    """Returns (list of selected-index tuples, array of (P, C, K)), runtime."""
    t0 = time.perf_counter()
    prob = _MSTProblem(u, p, c, admissible, tau, multi=True)
    algo = NSGA2(pop_size=pop_size, **_operators(prob.n_var))
    res = minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    rt = time.perf_counter() - t0
    if res.X is None:
        return [], np.empty((0, 3)), rt
    X = np.atleast_2d(np.asarray(res.X, bool))
    sels = [tuple(int(i) for i in prob.adm[row]) for row in X]
    return sels, np.atleast_2d(res.F), rt
