"""
Solve all 24,000 primary-benchmark conditions with the SCIP model and compare
with the stored Exact-MST (enumeration) results.

Writes data/optimization/scip_vs_exact_benchmark.csv.gz
Usage: python optimization/run_scip_validation.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimization"))
sys.path.insert(0, str(ROOT / "comparison"))
import benchmark_pools as bp  # noqa: E402
import scip_model as sm  # noqa: E402

STORED = ROOT / "data" / "benchmark_v3" / "product_benchmark.csv.gz"
OUT = ROOT / "data" / "benchmark_v3" / "scip_vs_exact_product.csv.gz"


def run(group):
    (seed, n, regime), rows = group
    ids, u, p, c = bp.generate_items(int(n), regime, int(seed))
    u, p, c = list(u), list(p), list(c)
    out = []
    for r in rows:
        w = bp.WEIGHT_PROFILES[r["weight_profile"]]
        res = sm.solve_mst_scip(u, p, c, [True] * int(n), float(r["tau"]), w)
        sel = "|".join(ids[i] for i in res.selected)
        out.append({
            "instance_id": r["instance_id"], "n_items": int(n), "regime": regime,
            "weight_profile": r["weight_profile"], "tau": r["tau"],
            "exact_status": r["exact_status"], "exact_selected_items": r.get("exact_selected"),
            "exact_objective": r.get("exact_objective"),
            "scip_status": res.status, "scip_selected_items": sel, "scip_objective": res.objective,
            "scip_dual_bound": res.dual_bound, "scip_runtime_s": res.runtime, "scip_nodes": res.nodes,
            "same_subset": (res.status == "INFEASIBLE") if r["exact_status"] == "INFEASIBLE"
                           else sel == r["exact_selected"],
            "objective_difference": res.objective - r["exact_objective"] if r["exact_status"] == "OPTIMAL" else 0.0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    d = pd.read_csv(STORED, low_memory=False)
    groups = [((k[0], k[1], k[2]), g.to_dict("records"))
              for k, g in d.groupby(["matched_seed", "n_items", "regime"])]
    with Pool(args.workers) as pool:
        res = pool.map(run, groups, chunksize=5)
    df = pd.DataFrame([r for rows in res for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"conditions: {len(df)}")
    print(f"SCIP optimal: {(df.scip_status == 'OPTIMAL').mean():.4%}")
    print(f"same subset as enumeration: {df.same_subset.mean():.4%}")
    print(f"max |objective difference|: {df.objective_difference.abs().max():.3e}")
    print(df.groupby("n_items")[["scip_runtime_s"]].median())


if __name__ == "__main__":
    main()
