"""
Convergence curves of the metaheuristics and parameter sensitivity of the genetic algorithm.

Convergence: one pool of each size in {64, 256, 512} (conflict regime, replicate 0,
balanced weights, tau = 0.90); every metaheuristic runs for twice its usual budget and
records its best objective against the number of objective evaluations. The optimum
comes from the hybrid mixed-integer method.

Sensitivity: the plain and the seeded-and-repaired genetic algorithms with population
{50, 100, 200}, mutation rate {0.5, 1, 2}/m and crossover {two-point, uniform}, on
pools of 64 and 256 items (replicate 0, three regimes, balanced weights, tau = 0.90),
each with the usual budget.

Writes data/optimization/convergence_traces.csv, ga_sensitivity.csv and
figures/fig_convergence.pdf
Usage: python optimization/run_convergence.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import heuristics as h  # noqa: E402
import scip_model as sm  # noqa: E402
from run_metaheuristics import budget  # noqa: E402

SIZES = [16, 32, 64, 128, 256, 512]
W = bp.WEIGHT_PROFILES["balanced"]
TAU = 0.90


def instance(n, regime, r=0):
    si = SIZES.index(n)
    seed = bp.GLOBAL_SEED + 70_000_000 + si * 1_000_000 + r
    return bp.generate_items(n, regime, seed)


def optimum(u, p, c):
    pool = h.Pool(u, p, c, [True] * len(u), TAU, W)
    x = min((f(h.Pool(u, p, c, [True] * len(u), TAU, W)) for f in h.METHODS.values()), key=pool.J)
    s = sm.solve_mst_scip(list(u), list(p), list(c), [True] * len(u), TAU, W, time_limit=120,
                          start=tuple(int(i) for i in np.flatnonzero(x)))
    return s.objective


def convergence():
    rows = []
    for n in (64, 256, 512):
        ids, u, p, c = instance(n, "utility_burden_conflict")
        opt = optimum(u, p, c)
        for name, f in {"GA (plain)": lambda pl, b, tr: h.ga_seeded_repair(pl, b, seeded=False, use_repair=False, trace=tr),
                        "GA (seeded, repair)": lambda pl, b, tr: h.ga_seeded_repair(pl, b, trace=tr),
                        "simulated annealing": lambda pl, b, tr: h.simulated_annealing(pl, b, trace=tr),
                        "tabu search": lambda pl, b, tr: h.tabu_search(pl, b, trace=tr)}.items():
            pool = h.Pool(u, p, c, [True] * n, TAU, W)
            tr = []
            if name != "GA (plain)":            # the start solution counts as the first point
                x0 = h.start_solution(h.Pool(u, p, c, [True] * n, TAU, W))
                tr.append((1, pool.J(x0)))
            f(pool, 2 * budget(n), tr)
            for ev, J in tr:
                rows.append(dict(n_items=n, method=name, evaluations=ev, best_objective=J,
                                 gap_percent=100 * (J - opt) / opt))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data" / "optimization" / "convergence_traces.csv", index=False)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    colors = {"GA (plain)": "#9a9890", "GA (seeded, repair)": "#2a78d6",
              "simulated annealing": "#eb6834", "tabu search": "#1f9e6e"}
    for ax, n in zip(axes, (64, 256, 512)):
        for name, g in d[d.n_items == n].groupby("method"):
            g = g.sort_values("evaluations")
            ax.step(g.evaluations, g.gap_percent.clip(lower=1e-3), where="post", label=name,
                    color=colors[name], lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"{n} items", fontsize=8, loc="left")
        ax.set_xlabel("objective evaluations (log scale)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("gap to optimum, % (symlog scale)")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    return d


def sensitivity():
    rows = []
    for n in (64, 256):
        for regime in bp.INSTANCE_REGIMES:
            ids, u, p, c = instance(n, regime)
            opt = optimum(u, p, c)
            for (seeded, pop, mut, xo) in itertools.product((False, True), (50, 100, 200), (0.5, 1.0, 2.0),
                                                            ("two_point", "uniform")):
                pool = h.Pool(u, p, c, [True] * n, TAU, W)
                x = h.ga_seeded_repair(pool, budget(n), seed=0, pop_size=pop, mut_scale=mut, crossover=xo,
                                       seeded=seeded, use_repair=seeded)
                J = pool.J(x) if x is not None else np.nan
                rows.append(dict(n_items=n, regime=regime, configuration="seeded, repair" if seeded else "plain",
                                 population=pop, mutation=f"{mut}/m", crossover=xo, objective=J,
                                 gap_percent=100 * (J - opt) / opt, evaluations=pool.n_eval))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data" / "optimization" / "ga_sensitivity.csv", index=False)
    return d


if __name__ == "__main__":
    c = convergence()
    print(c.groupby(["n_items", "method"]).gap_percent.min().unstack().round(3))
    s = sensitivity()
    print(s.groupby(["configuration", "n_items"]).gap_percent.describe().round(3))
