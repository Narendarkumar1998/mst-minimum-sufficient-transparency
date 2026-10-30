"""
Exact mixed-integer linear model (product chain with McCormick envelopes) without a
starting solution, on replicate 0 of the scaling instances (288 conditions), to compare
with the nonlinear model without a start (scip_* columns of scaling_v2.csv).

Writes data/benchmark_v3/milp_cold.csv (resumable)
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import scip_model as sm  # noqa: E402

SIZES = [16, 32, 64, 128, 256, 512]
OUT = ROOT / "data" / "benchmark_v3" / "milp_cold.csv"


def run(job):
    si, n, r, regime = job
    seed = bp.GLOBAL_SEED + 70_000_000 + si * 1_000_000 + r
    ids, u, p, c = bp.generate_items(n, regime, seed)
    rows = []
    for prof, w in bp.WEIGHT_PROFILES.items():
        for tau in bp.TAU_LEVELS:
            s = sm.solve_mst_scip(list(u), list(p), list(c), [True] * n, tau, w, time_limit=60.0, privacy="chain")
            rows.append(dict(n_items=n, regime=regime, replicate=r, weight_profile=prof, tau=tau,
                             milpcold_status=s.status, milpcold_objective=s.objective,
                             milpcold_gap=s.gap, milpcold_runtime_s=s.runtime))
    return rows


if __name__ == "__main__":
    jobs = sorted([(si, n, 0, g) for si, n in enumerate(SIZES) for g in bp.INSTANCE_REGIMES], key=lambda j: -j[1])
    part = OUT.with_suffix(".partial.csv")
    done = []
    if part.exists():                                   # resume after an interruption
        prev = pd.read_csv(part)
        done = prev.to_dict("records")
        have = set(zip(prev.n_items, prev.regime))
        jobs = [j for j in jobs if (j[1], j[3]) not in have]
    with Pool(2) as pool:
        for rows in pool.imap_unordered(run, jobs, chunksize=1):
            done.extend(rows)
            pd.DataFrame(done).to_csv(part, index=False)
    df = pd.DataFrame(done)
    df.to_csv(OUT, index=False)
    part.unlink(missing_ok=True)
    print(df.groupby("n_items").agg(opt=("milpcold_status", lambda s: (s == "OPTIMAL").mean()),
                                     rt=("milpcold_runtime_s", "median")))
