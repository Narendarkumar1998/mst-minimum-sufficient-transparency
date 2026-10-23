"""
Check that the comparison code reproduces the paper's primary benchmark.

With no prohibited items, the vectorized Exact-MST of methods.py must return
the same subset and the same objective as the stored Exact-MST results of all
24,000 primary-benchmark conditions.

Usage: python comparison/validate_against_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))

import benchmark_pools as bp  # noqa: E402
import methods as m  # noqa: E402

STORED = ROOT / "data" / "raw" / "exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv"


def main():
    stored = pd.read_csv(STORED, low_memory=False)
    mismatches, max_obj_diff, checked = 0, 0.0, 0
    for (seed, n, regime), rows in stored.groupby(["matched_seed", "n_items", "regime"]):
        ids, u, p, c = bp.generate_items(int(n), regime, int(seed))
        inst = m.Instance(ids, u, p, c, np.ones(int(n), bool))
        for _, r in rows.iterrows():
            w = bp.WEIGHT_PROFILES[r["weight_profile"]]
            sel = m.mst_exact(inst, float(r["tau"]), w)
            mine = "|".join(inst.id_tuple(sel)) if sel != m.INFEASIBLE else ""
            theirs = r["exact_selected_items"] if isinstance(r["exact_selected_items"], str) else ""
            checked += 1
            if mine != theirs:
                mismatches += 1
            else:
                max_obj_diff = max(max_obj_diff, abs(float(inst.J(w)[sel]) - float(r["exact_disclosure_cost"])))
    print(f"conditions checked: {checked}")
    print(f"subset mismatches: {mismatches}")
    print(f"max objective difference on matches: {max_obj_diff:.3e}")
    return mismatches


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
