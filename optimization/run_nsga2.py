"""
Multi-objective view of MST (suggested by M. Dridi): NSGA-II on
(P(S), C(S), K(S)) under U(S) >= tau, compared with the exact Pareto front
obtained by enumeration, and with the MST solutions of the four weight profiles.

Product-form benchmark pools, sizes 12 and 16, replicates 0-9, three regimes,
four thresholds (240 instance-threshold pairs).

Writes data/optimization/nsga2_pareto.csv and figures/fig_pareto_example.pdf
Usage: python optimization/run_nsga2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import ga_pymoo as ga  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "optimization" / "nsga2_pareto.csv"
FIG = ROOT / "figures" / "fig_pareto_example.pdf"
SIZES = {12: 2, 16: 4}
REPLICATES = 10


def pareto_mask(F):
    """Boolean mask of non-dominated rows of F (minimization)."""
    n = len(F)
    keep = np.ones(n, bool)
    order = np.lexsort(F.T[::-1])
    F = F[order]
    best = []
    for idx in range(n):
        f = F[idx]
        dominated = False
        for g in best:
            if np.all(g <= f) and np.any(g < f):
                dominated = True
                break
        if dominated:
            keep[order[idx]] = False
        else:
            best = [g for g in best if not (np.all(f <= g) and np.any(f < g))] + [f]
    return keep


def analyse(n, si, r, regime, tau, example=False):
    ids, u, p, c = bp.generate_items(n, regime, bp.matched_seed(si, r))
    inst = m.Instance(ids, u, p, c, np.ones(n, bool))
    feas = np.flatnonzero(inst.U >= tau)
    if feas.size == 0:
        return None
    F = np.column_stack([inst.P[feas], inst.Cn[feas], inst.K[feas]])
    front = feas[pareto_mask(F)]
    front_set = set(int(x) for x in front)
    sels, FF, rt = ga.nsga2_mst(u, p, c, [True] * n, tau, pop_size=100, n_gen=200, seed=r)
    masks = {int(sum(1 << i for i in s)) for s in sels}
    found = len(masks & front_set)
    not_pareto = len(masks - front_set)
    mst_masks = {name: m.mst_exact(inst, tau, w) for name, w in bp.WEIGHT_PROFILES.items()}
    row = dict(n_items=n, regime=regime, replicate=r, tau=tau, feasible_subsets=int(feas.size),
               exact_front_size=len(front_set), nsga2_solutions=len(masks), nsga2_on_front=found,
               nsga2_not_on_front=not_pareto, front_recall=found / len(front_set),
               mst_on_front=all(v in front_set for v in mst_masks.values()),
               mst_found_by_nsga2=sum(v in masks for v in mst_masks.values()), nsga2_runtime_s=rt)
    if example:
        return row, (inst, feas, front, masks, mst_masks)
    return row


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42})
EXAMPLE = (16, 9, "utility_burden_conflict", 0.80)   # chosen as an instance whose four weight profiles give distinct MST solutions


def figure(ex, tau):
    inst, feas, front, masks, mst_masks = ex
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.scatter(inst.P[feas], inst.Cn[feas], s=4, color="#d4d2cc", label="sufficient subsets", rasterized=True)
    ax.scatter(inst.P[front], inst.Cn[front], s=14, color="#2a78d6", label="exact Pareto front")
    mk = np.array(sorted(masks))
    ax.scatter(inst.P[mk], inst.Cn[mk], s=34, facecolors="none", edgecolors="#eb6834", lw=1.0,
               label="NSGA-II solutions")
    labels = {}
    for name, sel in mst_masks.items():
        labels.setdefault(sel, []).append(name.split("_")[0])
    offsets = [(18, 26), (22, 12), (22, -14), (18, -26)]
    for k, (sel, names) in enumerate(sorted(labels.items(), key=lambda kv: inst.P[kv[0]])):
        ax.scatter(inst.P[sel], inst.Cn[sel], s=70, marker="*", color="#0b0b0b", zorder=5)
        ax.annotate(", ".join(names) + f" ({int(inst.card[sel])} items)", (inst.P[sel], inst.Cn[sel]),
                    xytext=offsets[k % 4], textcoords="offset points", fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#52514e"))
    fp, fc = inst.P[front], inst.Cn[front]
    ax.set_xlim(fp.min() - 0.02, fp.max() + 0.10)
    ax.set_ylim(fc.min() - 0.03, fc.max() + 0.18)
    ax.set_xlabel("Relative privacy exposure P(S)")
    ax.set_ylabel("Normalized interpretation cost C(S)")
    ax.set_title(f"One 16-item pool, tau = {tau}; stars: MST under each weight profile", fontsize=7.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    rows = []
    for n, si in SIZES.items():
        for r in range(REPLICATES):
            for regime in bp.INSTANCE_REGIMES:
                for tau in bp.TAU_LEVELS:
                    if (n, r, regime, tau) == EXAMPLE:
                        row, ex = analyse(n, si, r, regime, tau, example=True)
                        figure(ex, tau)
                    else:
                        row = analyse(n, si, r, regime, tau)
                    if row:
                        rows.append(row)
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.groupby("n_items")[["exact_front_size", "nsga2_solutions", "front_recall", "nsga2_not_on_front",
                                "mst_on_front", "mst_found_by_nsga2", "nsga2_runtime_s"]].mean().round(3))


if __name__ == "__main__":
    main()
