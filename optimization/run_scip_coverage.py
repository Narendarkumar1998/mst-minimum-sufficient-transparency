"""
Solve the coverage-utility benchmark with the exact SCIP linearization and compare
with the stored enumeration results.

By default the first 20 replicates of each size are used (4,800 of 24,000
conditions, all sizes, regimes, thresholds and weight profiles).

Writes data/optimization/scip_vs_exact_coverage.csv.gz
Usage: python optimization/run_scip_coverage.py [--workers 2] [--replicates 20]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimization"))
import coverage_model as cm  # noqa: E402
import coverage_pools as cp  # noqa: E402

STORED = ROOT / "data" / "coverage" / "coverage_exact_vs_greedy_main_benchmark_v1_pool_normalized.csv.gz"
OUT = ROOT / "data" / "optimization" / "scip_vs_exact_coverage.csv.gz"
W = {"balanced": (1 / 3, 1 / 3, 1 / 3), "privacy_priority": (0.60, 0.20, 0.20),
     "interpretation_priority": (0.20, 0.60, 0.20), "parsimony_priority": (0.20, 0.20, 0.60)}
NS = None


def run(group):
    global NS
    if NS is None:
        NS = cp.load()
    (seed, n, regime), rows = group
    ids, A, w, p, c = cp.coverage_pool(NS, int(n), regime, int(seed))
    out = []
    for r in rows:
        res = cm.solve_coverage_mst_scip(A, w, p, c, [True] * int(n), float(r["tau"]), W[r["weight_profile"]])
        sel = "|".join(ids[i] for i in res.selected)
        out.append({"instance_id": r["instance_id"], "n_items": int(n), "regime": regime,
                    "weight_profile": r["weight_profile"], "tau": r["tau"],
                    "exact_selected_items": r["exact_selected_items"], "exact_objective": r["exact_disclosure_cost"],
                    "greedy_objective": r["greedy_disclosure_cost"], "scip_status": res.status,
                    "scip_selected_items": sel, "scip_objective": res.objective, "scip_runtime_s": res.runtime,
                    "same_subset": sel == r["exact_selected_items"],
                    "objective_difference": res.objective - r["exact_disclosure_cost"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=20)
    args = ap.parse_args()
    d = pd.read_csv(STORED, low_memory=False)
    d = d[d.replicate < args.replicates]
    groups = [(k, g.to_dict("records")) for k, g in d.groupby(["matched_seed", "n_items", "regime"])]
    with Pool(args.workers) as pool:
        res = pool.map(run, groups, chunksize=4)
    df = pd.DataFrame([r for rows in res for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"conditions: {len(df)}; optimal: {(df.scip_status == 'OPTIMAL').mean():.4%}; "
          f"same subset: {df.same_subset.mean():.4%}; max |dJ|: {df.objective_difference.abs().max():.3e}")
    print(df.groupby("n_items").scip_runtime_s.median())


if __name__ == "__main__":
    main()
