"""
Multi-objective view of MST (suggested by M. Dridi): NSGA-II on
(P(S), C(S), K(S)) under U(S) >= tau, compared with the exact Pareto front
obtained by enumeration, and with the MST solutions of the four weight profiles.

Product-form benchmark pools, sizes 12 and 16, replicates 0-9, three regimes,
four thresholds (240 instance-threshold pairs).

Writes data/benchmark_v3/nsga2_pareto.csv and pareto_example.csv
Usage: python optimization/run_nsga2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import ga_pymoo as ga  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "benchmark_v3" / "nsga2_pareto.csv"
EX_OUT = ROOT / "data" / "benchmark_v3" / "pareto_example.csv"
SIZES = {12: 2, 16: 4}
REPLICATES = 10


def pareto_mask(F):
    """Boolean mask of non-dominated rows of F (minimization)."""
    n = len(F)
    keep = np.ones(n, bool)
    order = np.lexsort(F.T[::-1])
    F = F[order]
    best = []
    for idx in range(n):
        f = F[idx]
        dominated = False
        for g in best:
            if np.all(g <= f) and np.any(g < f):
                dominated = True
                break
        if dominated:
            keep[order[idx]] = False
        else:
            best = [g for g in best if not (np.all(f <= g) and np.any(f < g))] + [f]
    return keep


def analyse(n, si, r, regime, tau, example=False):
    ids, u, p, c = bp.generate_items(n, regime, bp.matched_seed(si, r))
    inst = m.Instance(ids, u, p, c, np.ones(n, bool))
    feas = np.flatnonzero(inst.U >= tau)
    if feas.size == 0:
        return None
    F = np.column_stack([inst.P[feas], inst.Cn[feas], inst.K[feas]])
    front = feas[pareto_mask(F)]
    front_set = set(int(x) for x in front)
    sels, FF, rt = ga.nsga2_mst(u, p, c, [True] * n, tau, pop_size=100, n_gen=200, seed=r)
    masks = {int(sum(1 << i for i in s)) for s in sels}
    found = len(masks & front_set)
    not_pareto = len(masks - front_set)
    mst_masks = {name: m.mst_exact(inst, tau, w) for name, w in bp.WEIGHT_PROFILES.items()}
    row = dict(n_items=n, regime=regime, replicate=r, tau=tau, feasible_subsets=int(feas.size),
               exact_front_size=len(front_set), nsga2_solutions=len(masks), nsga2_on_front=found,
               nsga2_not_on_front=not_pareto, front_recall=found / len(front_set),
               mst_on_front=all(v in front_set for v in mst_masks.values()),
               mst_found_by_nsga2=sum(v in masks for v in mst_masks.values()), nsga2_runtime_s=rt)
    if example:
        return row, (inst, feas, front, masks, mst_masks)
    return row


EXAMPLE = (16, 9, "utility_burden_conflict", 0.80)   # example pool drawn in figures/fig_pareto_example.pdf


def save_example(ex):
    """Every sufficient subset of the example pool with its objectives and membership flags
    (exact front, NSGA-II solution, MST optimum of each weight profile); drawn by
    optimization/make_figures_v5.py."""
    inst, feas, front, masks, mst_masks = ex
    fr = set(int(x) for x in front)
    rows = []
    for k in feas:
        k = int(k)
        rows.append(dict(mask=k, P=float(inst.P[k]), C=float(inst.Cn[k]), K=float(inst.K[k]),
                         items=int(inst.card[k]), on_front=k in fr, nsga2=k in masks,
                         **{f"mst_{name}": k == v for name, v in mst_masks.items()}))
    pd.DataFrame(rows).to_csv(EX_OUT, index=False)


def main():
    rows = []
    for n, si in SIZES.items():
        for r in range(REPLICATES):
            for regime in bp.INSTANCE_REGIMES:
                for tau in bp.TAU_LEVELS:
                    if (n, r, regime, tau) == EXAMPLE:
                        row, ex = analyse(n, si, r, regime, tau, example=True)
                        save_example(ex)
                    else:
                        row = analyse(n, si, r, regime, tau)
                    if row:
                        rows.append(row)
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.groupby("n_items")[["exact_front_size", "nsga2_solutions", "front_recall", "nsga2_not_on_front",
                                "mst_on_front", "mst_found_by_nsga2", "nsga2_runtime_s"]].mean().round(3))


if __name__ == "__main__":
    main()
