"""
Constructive heuristics on the primary product-form benchmark (24,000 conditions),
compared with the stored Exact-MST optimum.

Methods: Dynamic Greedy-MST (the stored values are re-computed and checked), the
truncated greedy, both with reverse deletion, and Iyer-Bilmes ISSC with and without
reverse deletion (optimization/heuristics.py).

Writes data/optimization/heuristics_vs_exact_benchmark.csv.gz
Usage: python optimization/run_heuristics_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import heuristics as h  # noqa: E402

STORED = ROOT / "data" / "raw" / "exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv"
OUT = ROOT / "data" / "optimization" / "heuristics_vs_exact_benchmark.csv.gz"


def main():
    d = pd.read_csv(STORED, low_memory=False)
    rows = []
    for (seed, n, regime), g in d.groupby(["matched_seed", "n_items", "regime"]):
        ids, u, p, c = bp.generate_items(int(n), regime, int(seed))
        for r in g.itertuples():
            row = dict(instance_id=r.instance_id, n_items=int(n), regime=regime, replicate=r.replicate,
                       weight_profile=r.weight_profile, tau=r.tau, exact_status=r.exact_status,
                       exact_objective=r.exact_disclosure_cost, exact_count=r.exact_selected_count,
                       stored_greedy_objective=r.greedy_disclosure_cost)
            for name, f in h.METHODS.items():
                pool = h.Pool(u, p, c, [True] * int(n), r.tau, bp.WEIGHT_PROFILES[r.weight_profile])
                t0 = time.perf_counter()
                x = f(pool)
                row[f"{name}_runtime_s"] = time.perf_counter() - t0
                row[f"{name}_objective"] = pool.J(x) if x is not None else np.nan
                row[f"{name}_count"] = int(x.sum()) if x is not None else 0
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, compression="gzip")
    f = df[df.exact_status == "OPTIMAL"]
    print("greedy reproduces stored values:", float((f.greedy_objective - f.stored_greedy_objective).abs().max()))
    for name in h.METHODS:
        gap = 100 * (f[f"{name}_objective"] - f.exact_objective) / f.exact_objective
        print(f"{name:22s} optimal {100 * (gap <= 1e-7).mean():6.2f}%  median {gap.median():6.3f}  "
              f"mean {gap.mean():6.3f}  max {gap.max():7.3f}  >10% {100 * (gap > 10).mean():6.2f}%")


if __name__ == "__main__":
    main()
