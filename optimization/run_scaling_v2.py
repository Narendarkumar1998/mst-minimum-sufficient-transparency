"""
Scaling study, revised: pools of 16 to 512 items, 20 replicates per cell
(5,760 conditions), run with nothing else on the machine (2 worker processes,
one per core, single-threaded SCIP).

Per condition:
  * constructive heuristics: Dynamic Greedy-MST of the paper, truncated greedy,
    each with and without reverse deletion, and Iyer-Bilmes ISSC;
  * hybrid: SCIP on the mixed-integer nonlinear model, started from the best
    constructive solution (the reference optimum for all gaps);
  * SCIP alone (no starting solution) on replicates 0-4;
  * the exact mixed-integer linear model (product chain with McCormick
    envelopes), also started from the best constructive solution, on replicates 0-1;

Writes data/benchmark_v3/scaling.csv
Usage: python optimization/run_scaling_v2.py [--workers 2] [--replicates 20]
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
import scip_model as sm  # noqa: E402

SIZES = [16, 32, 64, 128, 256, 512]
TIME_LIMIT = 60.0
COLD_REPLICATES = 5
MILP_REPLICATES = 2
REF_ITEMS = 10.0
REF_COST = 10.0 * 0.55          # 10 items of mean cost; Beta(2,2) scaled to [0.10, 1.00]
OUT = ROOT / "data" / "benchmark_v3" / "scaling.csv"


def run(job):
    si, n, r, regime = job
    seed = bp.GLOBAL_SEED + 70_000_000 + si * 1_000_000 + r
    ids, u, p, c = bp.generate_items(n, regime, seed)
    ul, pl, cl = list(u), list(p), list(c)
    adm = [True] * n
    rows = []
    for prof, w in bp.WEIGHT_PROFILES.items():
        for tau in bp.TAU_LEVELS:
            row = dict(n_items=n, regime=regime, replicate=r, seed=seed, weight_profile=prof, tau=tau)
            best_x, best_J = None, np.inf
            for name, f in h.METHODS.items():
                pool = h.Pool(u, p, c, adm, tau, w)
                t0 = time.perf_counter()
                x = f(pool)
                rt = time.perf_counter() - t0
                J = pool.J(x) if x is not None else np.nan
                row.update({f"{name}_objective": J, f"{name}_count": int(x.sum()) if x is not None else 0,
                            f"{name}_runtime_s": rt})
                if x is not None and J < best_J:
                    best_x, best_J = x, J
            start = None if best_x is None else tuple(int(i) for i in np.flatnonzero(best_x))
            s = sm.solve_mst_scip(ul, pl, cl, adm, tau, w, time_limit=TIME_LIMIT, start=start)
            row.update(hybrid_status=s.status, hybrid_objective=s.objective, hybrid_bound=s.dual_bound,
                       hybrid_gap=s.gap, hybrid_runtime_s=s.runtime, hybrid_nodes=s.nodes,
                       hybrid_count=len(s.selected), hybrid_selected="|".join(map(str, s.selected)))
            if r < COLD_REPLICATES:
                s2 = sm.solve_mst_scip(ul, pl, cl, adm, tau, w, time_limit=TIME_LIMIT)
                row.update(scip_status=s2.status, scip_objective=s2.objective, scip_bound=s2.dual_bound,
                           scip_gap=s2.gap, scip_runtime_s=s2.runtime)
            if r < MILP_REPLICATES:
                s3 = sm.solve_mst_scip(ul, pl, cl, adm, tau, w, time_limit=TIME_LIMIT, start=start,
                                       privacy="chain")
                row.update(milp_status=s3.status, milp_objective=s3.objective, milp_bound=s3.dual_bound,
                           milp_gap=s3.gap, milp_runtime_s=s3.runtime)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=20)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(SIZES) for r in range(args.replicates)
            for g in bp.INSTANCE_REGIMES]
    jobs.sort(key=lambda j: (-j[1], j[2]))   # largest first for load balance
    done = []
    part = OUT.with_suffix(".partial.csv")
    if part.exists():                        # resume an interrupted run
        prev = pd.read_csv(part)
        done = prev.to_dict("records")
        have = set(zip(prev.n_items, prev.replicate, prev.regime))
        jobs = [j for j in jobs if (j[1], j[2], j[3]) not in have]
        print(f"resuming: {len(have)} jobs already done, {len(jobs)} left", flush=True)
    with Pool(args.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            done.extend(rows)
            print(f"{k}/{len(jobs)} jobs", flush=True)
            pd.DataFrame(done).to_csv(part, index=False)
    df = pd.DataFrame(done).sort_values(["n_items", "regime", "replicate", "weight_profile", "tau"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    part.unlink(missing_ok=True)
    print(f"conditions: {len(df)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
