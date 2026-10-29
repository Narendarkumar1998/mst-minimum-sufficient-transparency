"""
Scaling comparison of SCIP (exact, with proven bounds), Dynamic Greedy-MST and a
genetic algorithm (pymoo) on pools of 16 to 512 items.

Pools use the primary-benchmark generator with new seeds
GLOBAL_SEED + 70,000,000 + size_index * 1,000,000 + replicate.
Every condition: 3 structural regimes x 4 thresholds x 4 weight profiles.

Writes data/optimization/scaling_scip_greedy_ga.csv
Usage: python optimization/run_scaling.py [--workers 2] [--replicates 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison", "mst"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import ga_pymoo as ga  # noqa: E402
import mst_core  # noqa: E402
import scip_model as sm  # noqa: E402

SIZES = [16, 32, 64, 128, 256, 512]
TIME_LIMIT = 60.0
GA_POP, GA_GEN = 100, 200
OUT = ROOT / "data" / "optimization" / "scaling_scip_greedy_ga.csv"


def run(job):
    si, n, r, regime = job
    seed = bp.GLOBAL_SEED + 70_000_000 + si * 1_000_000 + r
    ids, u, p, c = bp.generate_items(n, regime, seed)
    u, p, c = list(u), list(p), list(c)
    items = [mst_core.InformationItem(item_id=ids[i], name=ids[i], utility=u[i], privacy=p[i],
                                      cognitive_cost=c[i]) for i in range(n)]
    adm = [True] * n
    rows = []
    for prof, w in bp.WEIGHT_PROFILES.items():
        for tau in bp.TAU_LEVELS:
            row = dict(n_items=n, regime=regime, replicate=r, seed=seed, weight_profile=prof, tau=tau)
            s = sm.solve_mst_scip(u, p, c, adm, tau, w, time_limit=TIME_LIMIT)
            row.update(scip_status=s.status, scip_objective=s.objective, scip_bound=s.dual_bound,
                       scip_gap=s.gap, scip_runtime_s=s.runtime, scip_nodes=s.nodes,
                       scip_count=len(s.selected))
            t0 = time.perf_counter()
            g = mst_core.greedy_mst(items, tau, lambda_privacy=w[0], lambda_cognitive=w[1],
                                    lambda_cardinality=w[2])
            row.update(greedy_status=g["status"], greedy_objective=g["disclosure_cost"],
                       greedy_runtime_s=time.perf_counter() - t0, greedy_count=len(g["selected_items"]))
            st, sel, J, rt, ne = ga.ga_mst(u, p, c, adm, tau, w, pop_size=GA_POP, n_gen=GA_GEN, seed=r)
            row.update(ga_status=st, ga_objective=J, ga_runtime_s=rt, ga_evaluations=ne, ga_count=len(sel))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=3)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(SIZES) for r in range(args.replicates)
            for g in bp.INSTANCE_REGIMES]
    jobs.sort(key=lambda j: -j[1])   # largest first for load balance
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows]).sort_values(["n_items", "regime", "replicate"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"conditions: {len(df)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
