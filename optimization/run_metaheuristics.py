"""
Metaheuristics under an equal wall-clock budget on the scaling instances
(replicates 0-4 of every size, regime, threshold and weight profile: 1,440 conditions).

Methods (optimization/heuristics.py):
  ga_plain          pymoo GA, random initialization, no repair (the configuration of the first version,
                    but with a time budget instead of a fixed number of generations)
  ga_seeded_repair  pymoo GA seeded with the constructive solution, with a repair operator
  sa                simulated annealing from the constructive solution
  tabu              tabu search from the constructive solution

Budget: max(1, 0.02 m) seconds per method and condition (1 s up to 50 items, 10.2 s at 512).
Gaps are computed later against the proven optimum of data/optimization/scaling.csv.

Writes data/optimization/metaheuristics.csv
Usage: python optimization/run_metaheuristics.py [--workers 2] [--replicates 5]
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

SIZES = [16, 32, 64, 128, 256, 512]
OUT = ROOT / "data" / "benchmark_v3" / "metaheuristics.csv"


def budget(m):
    return max(1.0, 0.02 * m)


METHODS = {
    "ga_plain": lambda pool, b, s: h.ga_seeded_repair(pool, b, seed=s, seeded=False, use_repair=False),
    "ga_seeded_repair": lambda pool, b, s: h.ga_seeded_repair(pool, b, seed=s),
    "sa": lambda pool, b, s: h.simulated_annealing(pool, b, seed=s),
    "tabu": lambda pool, b, s: h.tabu_search(pool, b, seed=s),
}


def run(job):
    si, n, r, regime = job
    seed = bp.GLOBAL_SEED + 70_000_000 + si * 1_000_000 + r
    ids, u, p, c = bp.generate_items(n, regime, seed)
    adm = [True] * n
    rows = []
    for prof, w in bp.WEIGHT_PROFILES.items():
        for tau in bp.TAU_LEVELS:
            row = dict(n_items=n, regime=regime, replicate=r, seed=seed, weight_profile=prof, tau=tau,
                       budget_s=budget(n))
            for name, f in METHODS.items():
                pool = h.Pool(u, p, c, adm, tau, w)
                t0 = time.perf_counter()
                x = f(pool, budget(n), r)
                rt = time.perf_counter() - t0
                row.update({f"{name}_objective": pool.J(x) if x is not None else np.nan,
                            f"{name}_count": int(x.sum()) if x is not None else 0,
                            f"{name}_feasible": bool(x is not None and pool.feasible(x)),
                            f"{name}_evaluations": pool.n_eval, f"{name}_runtime_s": rt})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=5)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(SIZES) for r in range(args.replicates)
            for g in bp.INSTANCE_REGIMES]
    part = OUT.with_suffix(".partial.csv")
    done = []
    if part.exists():
        prev = pd.read_csv(part)
        done = prev.to_dict("records")
        have = set(zip(prev.n_items, prev.replicate, prev.regime))
        jobs = [j for j in jobs if (j[1], j[2], j[3]) not in have]
    jobs.sort(key=lambda j: -j[1])
    with Pool(args.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            done.extend(rows)
            pd.DataFrame(done).to_csv(part, index=False)
            print(f"{k}/{len(jobs)} jobs", flush=True)
    df = pd.DataFrame(done).sort_values(["n_items", "regime", "replicate", "weight_profile", "tau"])
    df.to_csv(OUT, index=False)
    part.unlink(missing_ok=True)
    print(f"conditions: {len(df)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
