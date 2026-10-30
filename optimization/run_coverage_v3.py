"""
Coverage-utility benchmark under the revised objective (reference scales).

Pools come from the generator of notebook 02 (optimization/coverage_pools.py), unchanged:
each item covers a subset of 12 features with given strengths, and
U(S) = sum_f w_f max_{z in S} a_zf / sum_f w_f max_{z in D} a_zf.
Exact-MST enumerates all subsets (deterministic tie-breaking on (J, P, C, |S|, ids));
Dynamic Greedy-MST uses the paper's ratio dU / (dJ + 1e-12), ties to the lower index;
reverse deletion is then applied to the greedy output as a second heuristic.

Writes data/benchmark_v3/coverage_benchmark.csv.gz
Usage: python optimization/run_coverage_v3.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import coverage_pools as cp  # noqa: E402
from scales import REF_COST, REF_ITEMS  # noqa: E402

OUT = ROOT / "data" / "benchmark_v3" / "coverage_benchmark.csv.gz"
EPS = 1e-12
NS = None


def enumerate_pool(A, w, p, c):
    n, F = A.shape
    N = 1 << n
    cov = np.zeros((N, F))
    prod_p = np.ones(N)
    csum = np.zeros(N)
    card = np.zeros(N, int)
    for mask in range(1, N):
        hi = mask.bit_length() - 1
        prev = mask ^ (1 << hi)
        cov[mask] = np.maximum(cov[prev], A[hi])
        prod_p[mask] = prod_p[prev] * (1 - p[hi])
        csum[mask] = csum[prev] + c[hi]
        card[mask] = card[prev] + 1
    denom = float(w @ A.max(axis=0))
    U = (cov @ w) / denom
    return U, 1 - prod_p, csum / REF_COST, card / REF_ITEMS, card


def run(job):
    global NS
    if NS is None:
        NS = cp.load()
    si, n, r, regime = job
    seed = bp.matched_seed(si, r)
    ids, A, w, p, c = cp.coverage_pool(NS, n, regime, seed)
    A, w, p, c = np.asarray(A, float), np.asarray(w, float), np.asarray(p, float), np.asarray(c, float)
    U, P, C, K, card = enumerate_pool(A, w, p, c)
    id_order = np.argsort(np.argsort(np.array(ids)))   # rank of each item identifier
    rows = []
    for prof, lam in bp.WEIGHT_PROFILES.items():
        J = lam[0] * P + lam[1] * C + lam[2] * K
        for tau in bp.TAU_LEVELS:
            feas = np.flatnonzero(U >= tau)
            row = dict(instance_id=f"COV3_M{n}_R{r:03d}_{regime}", matched_seed=seed, n_items=n, replicate=r,
                       regime=regime, weight_profile=prof, tau=tau)
            if feas.size == 0:
                row["exact_status"] = "INFEASIBLE"
                rows.append(row)
                continue
            order = np.lexsort((card[feas], C[feas], P[feas], J[feas]))
            best = feas[order[0]]
            ties = [k for k in feas[order] if J[k] == J[best] and P[k] == P[best] and C[k] == C[best]
                    and card[k] == card[best]]
            if len(ties) > 1:
                best = min(ties, key=lambda k: sorted(ids[i] for i in range(n) if k >> i & 1))
            # Dynamic Greedy-MST
            sel = 0
            while U[sel] < tau:
                bi, bs = None, -np.inf
                for i in range(n):
                    if sel >> i & 1:
                        continue
                    nx = sel | (1 << i)
                    du = U[nx] - U[sel]
                    if du <= 0:
                        continue
                    sc = du / (J[nx] - J[sel] + EPS)
                    if sc > bs + 1e-12 or (abs(sc - bs) <= 1e-12 and bi is not None and id_order[i] < id_order[bi]):
                        bi, bs = i, sc
                sel |= 1 << bi
            g = sel
            while True:                                  # reverse deletion
                cand = [(J[g & ~(1 << i)], i) for i in range(n) if g >> i & 1 and U[g & ~(1 << i)] >= tau]
                if not cand:
                    break
                g &= ~(1 << min(cand)[1])
            jo = float(J[best])
            row.update(exact_status="OPTIMAL", exact_objective=jo, exact_count=int(card[best]),
                       exact_selected="|".join(sorted(ids[i] for i in range(n) if best >> i & 1)),
                       greedy_objective=float(J[sel]), greedy_count=int(card[sel]),
                       greedy_rd_objective=float(J[g]), greedy_rd_count=int(card[g]))
            for k in ("greedy", "greedy_rd"):
                x, kx = row[f"{k}_objective"], row[f"{k}_count"]
                row[f"{k}_mechanism"] = ("optimal" if (x - jo) / jo <= 1e-9 else
                                         "same_count_different_items" if kx == row["exact_count"] else
                                         "more_items" if kx > row["exact_count"] else "fewer_items")
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES) for r in range(bp.REPLICATES)
            for g in bp.INSTANCE_REGIMES]
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=5)
    df = pd.DataFrame([r for rows in res for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    f = df[df.exact_status == "OPTIMAL"]
    print(f"conditions {len(df)}, infeasible {len(df) - len(f)}")
    for k in ("greedy", "greedy_rd"):
        gap = 100 * (f[f"{k}_objective"] - f.exact_objective) / f.exact_objective
        print(f"{k:10s} optimal {100 * (gap <= 1e-7).mean():6.2f}%  median {gap.median():6.3f}  mean {gap.mean():6.3f}  "
              f"max {gap.max():7.3f}  >10% {100 * (gap > 10).mean():6.2f}%")
        big = f[gap > 10]
        print("   large gaps by mechanism", big[f"{k}_mechanism"].value_counts(normalize=True).round(4).to_dict())


if __name__ == "__main__":
    main()
