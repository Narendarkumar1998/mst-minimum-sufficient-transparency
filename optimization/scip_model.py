"""
MST as a mixed-integer nonlinear program solved with SCIP.

Following the linearization suggested by M. Dridi:

  * decision variables x_i in {0, 1} for admissible items only (hard admissibility
    is enforced by the domain);
  * the sufficiency constraint 1 - prod(1 - u_i x_i) >= tau is written as the exact
    linear covering constraint  sum_i w_i x_i >= W_tau,  with w_i = -ln(1 - u_i)
    and W_tau = -ln(1 - tau). Weights are capped at W_tau (a standard covering-
    knapsack reduction that leaves the feasible set unchanged), which also handles
    u_i = 1 (w_i = W_tau). For tau = 1 the constraint becomes "select at least one
    item with u_i = 1";
  * the privacy term 1 - prod(1 - p_i x_i) is written as 1 - exp(-y) with
    y = sum_i pi_i x_i and pi_i = -ln(1 - p_i). Items with p_i = 1 force the
    privacy term to 1 through t >= x_i. The term is concave in y, so SCIP solves
    it to global optimality by spatial branch and bound;
  * the cost and cardinality terms are linear.

Interpretation cost is normalized by the complete candidate pool I_t and
cardinality by |I_t|, as in the paper.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from pyscipopt import Model, exp, quicksum


@dataclass
class SolverResult:
    status: str               # OPTIMAL, INFEASIBLE, TIME_LIMIT
    selected: tuple           # indices into the complete pool
    objective: float          # J of the incumbent (nan if none)
    dual_bound: float         # proven lower bound on J*
    gap: float                # (incumbent - bound) / bound, 0 when proven optimal
    runtime: float
    nodes: int


def solve_mst_scip(u, p, c, admissible, tau, weights, time_limit=60.0, gap_limit=0.0,
                   threads=1, seed=0, start=None) -> SolverResult:
    """start: optional iterable of item indices used as an initial feasible solution."""
    lp, lc, lk = weights
    n_all = len(u)
    total_cost = sum(c)
    adm = [i for i in range(n_all) if admissible[i]]
    t0 = time.perf_counter()

    # Quick infeasibility test on the admissible pool.
    prod = 1.0
    for i in adm:
        prod *= (1.0 - u[i])
    if 1.0 - prod < tau:
        return SolverResult("INFEASIBLE", (), math.nan, math.nan, 0.0, time.perf_counter() - t0, 0)

    m = Model("MST")
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", gap_limit)
    m.setParam("numerics/feastol", 1e-9)
    m.setParam("randomization/randomseedshift", seed)
    m.setParam("parallel/maxnthreads", threads)

    x = {i: m.addVar(vtype="B", name=f"x{i}") for i in adm}

    # Sufficiency: exact linear covering constraint.
    if tau >= 1.0:
        ones = [i for i in adm if u[i] >= 1.0]
        m.addCons(quicksum(x[i] for i in ones) >= 1)
    else:
        W = -math.log1p(-tau)
        w = {i: (W if u[i] >= 1.0 else min(-math.log1p(-u[i]), W)) for i in adm}
        m.addCons(quicksum(w[i] * x[i] for i in adm) >= W)

    # Privacy term P = 1 - exp(-y), with items of p_i = 1 handled separately.
    t = m.addVar(lb=0.0, ub=1.0, name="P")
    soft = [i for i in adm if p[i] < 1.0]
    y = m.addVar(lb=0.0, name="y")
    m.addCons(y == quicksum(-math.log1p(-p[i]) * x[i] for i in soft))
    m.addCons(t >= 1 - exp(-y))
    for i in adm:
        if p[i] >= 1.0:
            m.addCons(t >= x[i])

    cost_term = quicksum((c[i] / total_cost) * x[i] for i in adm) if total_cost > 0 else 0
    card_term = quicksum(x[i] for i in adm) * (1.0 / n_all)
    m.setObjective(lp * t + lc * cost_term + lk * card_term, "minimize")
    if start is not None:
        st = set(start)
        sol0 = m.createSol()
        for i in adm:
            m.setSolVal(sol0, x[i], 1.0 if i in st else 0.0)
        yv = sum(-math.log1p(-p[i]) for i in st if p[i] < 1.0)
        m.setSolVal(sol0, y, yv)
        m.setSolVal(sol0, t, 1.0 if any(p[i] >= 1.0 for i in st) else 1.0 - math.exp(-yv))
        m.addSol(sol0, free=True)
    m.optimize()

    status = m.getStatus()
    runtime = time.perf_counter() - t0
    nodes = int(m.getNNodes())
    if m.getNSols() == 0:
        st = "INFEASIBLE" if status == "infeasible" else "TIME_LIMIT"
        return SolverResult(st, (), math.nan, m.getDualbound(), math.nan, runtime, nodes)
    sol = m.getBestSol()
    sel = tuple(i for i in adm if m.getSolVal(sol, x[i]) > 0.5)
    obj = exact_objective(sel, u, p, c, weights)
    bound = m.getDualbound()
    gap = 0.0 if status == "optimal" else max(0.0, (obj - bound) / bound) if bound > 0 else math.nan
    return SolverResult("OPTIMAL" if status == "optimal" else "TIME_LIMIT", sel, obj, bound, gap, runtime, nodes)


def exact_objective(sel, u, p, c, weights):
    """J recomputed from the selected indices with the paper's product forms."""
    lp, lc, lk = weights
    prod_p = 1.0
    for i in sel:
        prod_p *= (1.0 - p[i])
    total = 0
    for v in c:
        total += v
    cs = 0
    for i in sel:
        cs += c[i]
    cn = cs / total if total != 0 else 0.0
    return lp * (1.0 - prod_p) + lc * cn + lk * (len(sel) / len(u))


def utility(sel, u):
    prod = 1.0
    for i in sel:
        prod *= (1.0 - u[i])
    return 1.0 - prod
