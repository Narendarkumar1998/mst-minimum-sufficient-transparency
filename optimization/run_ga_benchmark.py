"""
Genetic algorithm (pymoo) on the primary product-form benchmark, compared with the
stored Exact-MST and Dynamic Greedy-MST results.

By default the first 5 replicates of each pool size are used (1,200 of 24,000
conditions, all sizes, regimes, thresholds and weight profiles). The GA seed is
the replicate index.

Writes data/optimization/ga_vs_exact_benchmark.csv.gz
Usage: python optimization/run_ga_benchmark.py [--workers 2] [--replicates 5]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import ga_pymoo as ga  # noqa: E402

STORED = ROOT / "data" / "raw" / "exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv"
OUT = ROOT / "data" / "optimization" / "ga_vs_exact_benchmark.csv.gz"


def run(group):
    (seed, n, regime, rep), rows = group
    ids, u, p, c = bp.generate_items(int(n), regime, int(seed))
    out = []
    for r in rows:
        w = bp.WEIGHT_PROFILES[r["weight_profile"]]
        st, sel, J, rt, ne = ga.ga_mst(u, p, c, [True] * int(n), float(r["tau"]), w, seed=int(rep))
        sel_ids = "|".join(ids[i] for i in sel)
        out.append({"instance_id": r["instance_id"], "n_items": int(n), "regime": regime,
                    "weight_profile": r["weight_profile"], "tau": r["tau"],
                    "exact_status": r["exact_status"], "exact_objective": r["exact_disclosure_cost"],
                    "exact_selected_items": r["exact_selected_items"],
                    "greedy_objective": r["greedy_disclosure_cost"], "greedy_runtime_s": r["greedy_runtime_seconds"],
                    "ga_status": st, "ga_objective": J, "ga_selected_items": sel_ids,
                    "ga_runtime_s": rt, "ga_evaluations": ne})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=5)
    args = ap.parse_args()
    d = pd.read_csv(STORED, low_memory=False)
    d = d[d.replicate < args.replicates]
    groups = [(k, g.to_dict("records")) for k, g in d.groupby(["matched_seed", "n_items", "regime", "replicate"])]
    with Pool(args.workers) as pool:
        res = pool.map(run, groups, chunksize=4)
    df = pd.DataFrame([r for rows in res for r in rows])
    df["ga_gap_percent"] = 100 * (df.ga_objective - df.exact_objective) / df.exact_objective
    df["greedy_gap_percent"] = 100 * (df.greedy_objective - df.exact_objective) / df.exact_objective
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    f = df[df.exact_status == "OPTIMAL"]
    print(f"conditions: {len(df)}; feasible: {len(f)}")
    print(f"GA optimal: {(f.ga_gap_percent <= 1e-9).mean():.4%}; median/mean/max gap: "
          f"{f.ga_gap_percent.median():.4f} / {f.ga_gap_percent.mean():.4f} / {f.ga_gap_percent.max():.4f}")
    inf = df[df.exact_status != "OPTIMAL"]
    print(f"infeasible conditions: {len(inf)}; GA reported no feasible solution: "
          f"{(inf.ga_status != 'FEASIBLE').mean() if len(inf) else float('nan'):.4%}")


if __name__ == "__main__":
    main()
