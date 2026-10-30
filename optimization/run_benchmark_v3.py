"""
Primary product-form benchmark under the revised objective (reference scales,
optimization/scales.py): 1,500 pools (sizes 8-16, 100 replicates, three regimes), four
thresholds and four weight profiles, 24,000 conditions.

Per condition: Exact-MST by enumeration with the deterministic tie-breaking of the
paper (comparison/methods.py, cross-checked against mst/mst_core.py), and the
constructive heuristics of optimization/heuristics.py; the greedy failure mechanism is
classified as in the paper (optimal, same number of items but different items, more
items, fewer items).

Writes data/benchmark_v3/product_benchmark.csv.gz
Usage: python optimization/run_benchmark_v3.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import heuristics as h  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "benchmark_v3" / "product_benchmark.csv.gz"


def mechanism(opt, x, k_opt, k_x, tol=1e-9):
    if x is None:
        return "no_solution"
    if opt <= 0 or (x - opt) / opt <= tol:
        return "optimal"
    if k_x == k_opt:
        return "same_count_different_items"
    return "more_items" if k_x > k_opt else "fewer_items"


def run(job):
    si, n, r, regime = job
    seed = bp.matched_seed(si, r)
    ids, u, p, c = bp.generate_items(n, regime, seed)
    inst = m.Instance(ids, u, p, c, np.ones(n, bool))
    rows = []
    for prof, w in bp.WEIGHT_PROFILES.items():
        J = inst.J(w)
        for tau in bp.TAU_LEVELS:
            t0 = time.perf_counter()
            s = m.mst_exact(inst, tau, w, J)
            rt = time.perf_counter() - t0
            row = dict(instance_id=f"V3_M{n}_R{r:03d}_{regime}", matched_seed=seed, n_items=n,
                       replicate=r, regime=regime, weight_profile=prof, tau=tau,
                       exact_status="INFEASIBLE" if s == m.INFEASIBLE else "OPTIMAL",
                       exact_runtime_s=rt)
            if s != m.INFEASIBLE:
                row.update(exact_objective=float(J[s]), exact_count=int(inst.card[s]),
                           exact_selected="|".join(inst.id_tuple(s)), exact_utility=float(inst.U[s]),
                           exact_privacy=float(inst.P[s]), exact_cost=float(inst.Cn[s]),
                           exact_cardinality=float(inst.K[s]))
            for name, f in h.METHODS.items():
                pool = h.Pool(u, p, c, [True] * n, tau, w)
                t0 = time.perf_counter()
                x = f(pool)
                row[f"{name}_runtime_s"] = time.perf_counter() - t0
                if x is None:
                    row[f"{name}_objective"] = np.nan
                    row[f"{name}_count"] = 0
                    continue
                row[f"{name}_objective"] = pool.J(x)
                row[f"{name}_count"] = int(x.sum())
                if name == "greedy":
                    row["greedy_selected"] = "|".join(ids[i] for i in np.flatnonzero(x))
            if s != m.INFEASIBLE:
                for name in h.METHODS:
                    row[f"{name}_mechanism"] = mechanism(row["exact_objective"], row[f"{name}_objective"],
                                                         row["exact_count"], row[f"{name}_count"])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES) for r in range(bp.REPLICATES)
            for g in bp.INSTANCE_REGIMES]
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=5)
    df = pd.DataFrame([r for rows in res for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    f = df[df.exact_status == "OPTIMAL"]
    print(f"conditions {len(df)}, infeasible {len(df) - len(f)}")
    for name in h.METHODS:
        gap = 100 * (f[f"{name}_objective"] - f.exact_objective) / f.exact_objective
        print(f"{name:22s} optimal {100 * (gap <= 1e-7).mean():6.2f}%  median {gap.median():6.3f}  "
              f"mean {gap.mean():6.3f}  max {gap.max():7.3f}  >10% {100 * (gap > 10).mean():6.2f}%")
    print(f.greedy_mechanism.value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
