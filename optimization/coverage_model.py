"""
MST with an interaction-aware (feature-coverage) utility, solved with SCIP.

Utility (pool-normalized weighted maximum coverage, notebook 02, Steps 124 to 129):

    U(S) = sum_f w_f max_{z in S} a_{z f}  /  sum_f w_f max_{z in D} a_{z f}

Items overlap when they cover the same decision features, so the utility of a
set is not determined by item values alone. The coverage term is linearized
exactly with auxiliary variables:

    m_f <= sum_z a_{z f} s_{z f},   sum_z s_{z f} <= 1,   0 <= s_{z f} <= x_z,
    sum_f w_f m_f >= tau * sum_f w_f max_{z in D} a_{z f}.

The objective (privacy, interpretation cost, size) is the same as for the
product-form model.
"""

from __future__ import annotations

import math
import time

from pyscipopt import Model, exp, quicksum

from scip_model import SolverResult


def solve_coverage_mst_scip(A, w, p, c, admissible, tau, weights, time_limit=60.0):
    lp, lc, lk = weights
    n_all, n_feat = len(A), len(w)
    adm = [i for i in range(n_all) if admissible[i]]
    t0 = time.perf_counter()
    denom = sum(w[f] * max(A[i][f] for i in range(n_all)) for f in range(n_feat))
    reach = sum(w[f] * max([A[i][f] for i in adm] or [0.0]) for f in range(n_feat))
    if denom <= 0 or reach / denom < tau - 1e-12:
        return SolverResult("INFEASIBLE", (), math.nan, math.nan, 0.0, time.perf_counter() - t0, 0)

    m = Model("MST-coverage")
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("numerics/feastol", 1e-9)
    m.setParam("parallel/maxnthreads", 1)
    x = {i: m.addVar(vtype="B", name=f"x{i}") for i in adm}
    mf = {}
    for f in range(n_feat):
        cov = [i for i in adm if A[i][f] > 0]
        mf[f] = m.addVar(lb=0.0, ub=max([A[i][f] for i in cov] or [0.0]), name=f"m{f}")
        s = {i: m.addVar(lb=0.0, ub=1.0, name=f"s{i}_{f}") for i in cov}
        m.addCons(mf[f] <= quicksum(A[i][f] * s[i] for i in cov))
        m.addCons(quicksum(s.values()) <= 1)
        for i in cov:
            m.addCons(s[i] <= x[i])
    m.addCons(quicksum(w[f] * mf[f] for f in range(n_feat)) >= tau * denom)

    t = m.addVar(lb=0.0, ub=1.0, name="P")
    y = m.addVar(lb=0.0, name="y")
    m.addCons(y == quicksum(-math.log1p(-p[i]) * x[i] for i in adm if p[i] < 1.0))
    m.addCons(t >= 1 - exp(-y))
    for i in adm:
        if p[i] >= 1.0:
            m.addCons(t >= x[i])
    total = sum(c)
    m.setObjective(lp * t + lc * quicksum((c[i] / total) * x[i] for i in adm)
                   + lk * quicksum(x[i] for i in adm) * (1.0 / n_all), "minimize")
    m.optimize()
    rt = time.perf_counter() - t0
    if m.getNSols() == 0:
        return SolverResult("TIME_LIMIT", (), math.nan, m.getDualbound(), math.nan, rt, int(m.getNNodes()))
    sol = m.getBestSol()
    sel = tuple(i for i in adm if m.getSolVal(sol, x[i]) > 0.5)
    obj = objective(sel, p, c, weights)
    st = m.getStatus()
    bound = m.getDualbound()
    gap = 0.0 if st == "optimal" else max(0.0, (obj - bound) / bound) if bound > 0 else math.nan
    return SolverResult("OPTIMAL" if st == "optimal" else "TIME_LIMIT", sel, obj, bound, gap, rt, int(m.getNNodes()))


def coverage_utility(sel, A, w):
    n_feat = len(w)
    denom = sum(w[f] * max(A[i][f] for i in range(len(A))) for f in range(n_feat))
    num = sum(w[f] * max([A[i][f] for i in sel] or [0.0]) for f in range(n_feat))
    return num / denom


def objective(sel, p, c, weights):
    lp, lc, lk = weights
    prod = 1.0
    for i in sel:
        prod *= (1.0 - p[i])
    total = 0
    for v in c:
        total += v
    cs = 0
    for i in sel:
        cs += c[i]
    return lp * (1.0 - prod) + lc * (cs / total) + lk * (len(sel) / len(p))
