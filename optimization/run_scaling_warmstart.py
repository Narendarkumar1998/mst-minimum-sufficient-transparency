"""
Re-solve the scaling conditions with SCIP started from the Dynamic Greedy-MST
disclosure (a feasible starting solution), to measure the hybrid method.

Reads  data/optimization/scaling_scip_greedy_ga.csv (for the condition list)
Writes data/optimization/scaling_scip_warmstart.csv
Usage: python optimization/run_scaling_warmstart.py [--workers 1]
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
import mst_core  # noqa: E402
import scip_model as sm  # noqa: E402

SRC = ROOT / "data" / "optimization" / "scaling_scip_greedy_ga.csv"
OUT = ROOT / "data" / "optimization" / "scaling_scip_warmstart.csv"


def run(key_rows):
    (n, regime, seed), rows = key_rows
    ids, u, p, c = bp.generate_items(int(n), regime, int(seed))
    items = [mst_core.InformationItem(item_id=ids[i], name=ids[i], utility=u[i], privacy=p[i],
                                      cognitive_cost=c[i]) for i in range(len(u))]
    out = []
    for r in rows:
        w = bp.WEIGHT_PROFILES[r["weight_profile"]]
        t0 = time.perf_counter()
        g = mst_core.greedy_mst(items, r["tau"], lambda_privacy=w[0], lambda_cognitive=w[1], lambda_cardinality=w[2])
        start = [ids.index(x) for x in g["selected_items"]]
        res = sm.solve_mst_scip(list(u), list(p), list(c), [True] * len(u), r["tau"], w, time_limit=60.0, start=start)
        out.append(dict(n_items=n, regime=regime, seed=seed, replicate=r["replicate"],
                        weight_profile=r["weight_profile"], tau=r["tau"], warm_status=res.status,
                        warm_objective=res.objective, warm_bound=res.dual_bound, warm_gap=res.gap,
                        warm_runtime_s=time.perf_counter() - t0, warm_scip_only_s=res.runtime))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    d = pd.read_csv(SRC)
    groups = [(k, g.to_dict("records")) for k, g in d.groupby(["n_items", "regime", "seed"])]
    groups.sort(key=lambda kg: -kg[0][0])
    with Pool(args.workers) as pool:
        res = pool.map(run, groups, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    print(df.groupby("n_items").agg(opt=("warm_status", lambda x: (x == "OPTIMAL").mean()),
                                     med=("warm_runtime_s", "median"), mx=("warm_runtime_s", "max")))


if __name__ == "__main__":
    main()
