"""
Scaling figure and table for the evaluation section.

Reads  data/optimization/scaling_scip_greedy_ga.csv and scaling_scip_warmstart.csv
Writes figures/fig_scaling.pdf and tables/scaling_summary.csv
Usage: python optimization/make_scaling_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, VIOLET, TEXT2 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#52514e"


def load():
    a = pd.read_csv(ROOT / "data/optimization/scaling_scip_greedy_ga.csv")
    w = pd.read_csv(ROOT / "data/optimization/scaling_scip_warmstart.csv")
    d = a.merge(w, on=["n_items", "regime", "seed", "replicate", "weight_profile", "tau"])
    # reference optimum: proven optimum of either SCIP run
    d["opt"] = np.where(d.warm_status == "OPTIMAL", d.warm_objective,
                        np.where(d.scip_status == "OPTIMAL", d.scip_objective, np.nan))
    for k in ("greedy", "ga"):
        d[f"{k}_gap"] = 100 * (d[f"{k}_objective"] - d.opt) / d.opt
    return d


def summary(d):
    g = d.groupby("n_items")
    s = pd.DataFrame({
        "conditions": g.size(),
        "scip_proven_optimal_pct": 100 * g.scip_status.apply(lambda x: (x == "OPTIMAL").mean()),
        "scip_median_s": g.scip_runtime_s.median(),
        "scip_max_s": g.scip_runtime_s.max(),
        "hybrid_proven_optimal_pct": 100 * g.warm_status.apply(lambda x: (x == "OPTIMAL").mean()),
        "hybrid_median_s": g.warm_runtime_s.median(),
        "hybrid_max_s": g.warm_runtime_s.max(),
        "greedy_median_s": g.greedy_runtime_s.median(),
        "ga_median_s": g.ga_runtime_s.median(),
        "greedy_optimal_pct": 100 * g.greedy_gap.apply(lambda x: (x <= 1e-9).mean()),
        "greedy_median_gap_pct": g.greedy_gap.median(),
        "greedy_max_gap_pct": g.greedy_gap.max(),
        "ga_optimal_pct": 100 * g.ga_gap.apply(lambda x: (x <= 1e-9).mean()),
        "ga_median_gap_pct": g.ga_gap.median(),
        "ga_max_gap_pct": g.ga_gap.max(),
    })
    return s


def figure(d, s):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    n = s.index.to_numpy()
    for col, lab, colr, mk in [("scip_median_s", "SCIP", BLUE, "o"), ("hybrid_median_s", "greedy start + SCIP", VIOLET, "s"),
                               ("ga_median_s", "genetic algorithm", AQUA, "^"), ("greedy_median_s", "dynamic greedy", ORANGE, "D")]:
        a.plot(n, s[col], color=colr, marker=mk, ms=4, lw=1.8, label=lab)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(n, [str(int(v)) for v in n])
    a.set_xlabel("Admissible items $m$")
    a.set_ylabel("Median runtime per condition (s)")
    a.set_title("(a) Runtime", loc="left")
    a.grid(color="#e6e5e1", lw=0.5)
    a.legend(frameon=False, fontsize=7)
    for col, lab, colr, mk in [("greedy_optimal_pct", "dynamic greedy", ORANGE, "D"), ("ga_optimal_pct", "genetic algorithm", AQUA, "^")]:
        b.plot(n, s[col], color=colr, marker=mk, ms=4, lw=1.8, label=lab)
    b.set_xscale("log", base=2)
    b.set_xticks(n, [str(int(v)) for v in n])
    b.set_ylim(0, 105)
    b.set_xlabel("Admissible items $m$")
    b.set_ylabel("Conditions solved to the proven optimum (%)")
    b.set_title("(b) Heuristic quality against the proven optimum", loc="left")
    b.grid(axis="y", color="#e6e5e1", lw=0.5)
    b.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=2)
    fig.savefig(ROOT / "figures/fig_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    d = load()
    s = summary(d)
    s.round(4).to_csv(ROOT / "tables/scaling_summary.csv")
    figure(d, s)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
