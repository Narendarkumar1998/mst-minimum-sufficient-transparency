"""
Summary table and figures for the comparison with adapted baselines.

Reads  data/comparison/comparison_rows.csv.gz
Writes tables/comparison_requirements_summary.csv
       tables/comparison_by_threshold.csv
       figures/fig_comparison_requirements.pdf
       figures/fig_comparison_cost_threshold.pdf

Usage: python comparison/make_figures.py
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
sys.path.insert(0, str(ROOT / "comparison"))
import methods as m  # noqa: E402

ROWS = ROOT / "data" / "comparison" / "comparison_rows.csv.gz"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})
TEXT2 = "#52514e"
BLUE = "#2a78d6"
GREY = "#9a9893"
ORANGE, AQUA, VIOLET, RED = "#eb6834", "#1baf7a", "#6250d6", "#e34948"

ORDER = list(m.METHODS)
LABEL = {k: v[2] for k, v in m.METHODS.items()}
LABEL["MST"] = "MST (full model)"


def summary(d: pd.DataFrame) -> pd.DataFrame:
    feas = d[d.instance_feasible]
    infe = d[~d.instance_feasible]
    fv = feas[feas.valid]
    s = pd.DataFrame({
        "R1_sufficient_when_possible_pct": 100 * feas.groupby("method").sufficient.mean(),
        "R2_no_prohibited_item_pct": 100 * d.assign(a=(d.status == "INFEASIBLE") | d.admissible)
                                                .groupby("method").a.mean(),
        "R3_infeasibility_reported_pct": 100 * infe.assign(a=infe.status == "INFEASIBLE")
                                                   .groupby("method").a.mean(),
        "R1_R2_R3_all_met_pct": 100 * d.groupby("method").valid.mean(),
        "R4_not_dominated_when_valid_pct": 100 - 100 * fv.assign(
            x=fv.weakly_dominated_by_mst.astype(bool) & ~fv.same_as_mst.astype(bool))
            .groupby("method").x.mean(),
        "R1_to_R4_all_met_pct": 100 * d.assign(x=d.valid & ~(
            d.weakly_dominated_by_mst.fillna(False).astype(bool)
            & ~d.same_as_mst.fillna(True).astype(bool))).groupby("method").x.mean(),
        "median_items_valid": fv.groupby("method").selected_count.median(),
        "median_objective_excess_valid_pct": fv.groupby("method").objective_excess_pct.median(),
        "dominated_by_mst_valid_pct": 100 * fv.assign(
            x=fv.weakly_dominated_by_mst.astype(bool) & ~fv.same_as_mst.astype(bool))
            .groupby("method").x.mean(),
        "identical_to_mst_valid_pct": 100 * fv.groupby("method").same_as_mst.mean(),
        "median_repair_excess_pct": feas.groupby("method").repair_excess_pct.median(),
        "q1_repair_excess_pct": feas.groupby("method").repair_excess_pct.quantile(0.25),
        "q3_repair_excess_pct": feas.groupby("method").repair_excess_pct.quantile(0.75),
    }).reindex(ORDER)
    s.insert(0, "group", [m.METHODS[k][1] for k in s.index])
    s.insert(1, "label", [m.METHODS[k][2] for k in s.index])
    return s


def by_threshold(d: pd.DataFrame) -> pd.DataFrame:
    feas = d[d.instance_feasible]
    out = []
    for (meth, tau), g in d.groupby(["method", "tau"]):
        gf = feas[(feas.method == meth) & (feas.tau == tau)]
        out.append({
            "method": meth, "tau": tau,
            "valid_pct": 100 * g.valid.mean(),
            "prohibited_disclosed_pct": 100 * ((g.status == "DISCLOSED") & ~g.admissible.astype(bool)).mean(),
            "sufficient_when_possible_pct": 100 * gf.sufficient.mean(),
            "infeasible_conditions_pct": 100 * (~g.instance_feasible).mean(),
        })
    return pd.DataFrame(out)


def fig_requirements(s: pd.DataFrame):
    cols = ["R1_sufficient_when_possible_pct", "R2_no_prohibited_item_pct",
            "R3_infeasibility_reported_pct", "R4_not_dominated_when_valid_pct", "R1_to_R4_all_met_pct"]
    heads = ["R1\nsufficient\nwhen possible", "R2\nno prohibited\nitem",
             "R3\ninfeasibility\nreported", "R4\nnot dominated\nby MST", "R1 to R4\nall met"]
    M = s[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f4f3f0", "#a9c8ef", "#2a78d6", "#0d3a73"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=TEXT2)
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 60 else "#0b0b0b",
                    fontweight="bold" if s.index[i] == "MST" else "normal")
    ax.set_xticks(range(len(cols)), heads)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ylabels = [LABEL[k] for k in s.index]
    ax.set_yticks(range(len(ylabels)), ylabels)
    for t, k in zip(ax.get_yticklabels(), s.index):
        if k == "MST":
            t.set_fontweight("bold")
    for y in (0.5, 5.5):
        ax.axhline(y, color="white", lw=3)
    ax.axvline(3.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks(range(len(ylabels)), [("Ablation: " if m.METHODS[k][1] == "ablation" else
                                         "Baseline: " if m.METHODS[k][1] == "baseline" else "") + LABEL[k]
                                        for k in s.index])
    for t, k in zip(ax.get_yticklabels(), s.index):
        if k == "MST":
            t.set_fontweight("bold")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    fig.text(0.99, 0.01, "Percent of conditions; 24,000 conditions, 2,124 without a sufficient admissible disclosure.",
             ha="right", fontsize=6.5, color=TEXT2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_comparison_requirements.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cost_threshold(s: pd.DataFrame, t: pd.DataFrame):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) repair cost: extra objective needed to make each method's output valid
    order = s.sort_values("median_repair_excess_pct").index.tolist()
    y = np.arange(len(order))
    med = s.loc[order, "median_repair_excess_pct"].to_numpy()
    q1 = s.loc[order, "q1_repair_excess_pct"].to_numpy()
    q3 = s.loc[order, "q3_repair_excess_pct"].to_numpy()
    colors = [BLUE if k == "MST" else GREY for k in order]
    a.barh(y, med, color=colors, height=0.62)
    a.errorbar(med, y, xerr=[med - q1, q3 - med], fmt="none", ecolor="#3a3936", elinewidth=0.7, capsize=1.8)
    a.set_yticks(y, [("Ablation: " if m.METHODS[k][1] == "ablation" else "Baseline: " if m.METHODS[k][1] == "baseline" else "") + LABEL[k] for k in order])
    a.invert_yaxis()
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(0, 900)
    a.set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    a.set_xlabel("Extra disclosure cost after repair (% of MST objective)")
    a.set_title("(a) Extra cost to make each output valid", loc="left")
    for yi, v, hi in zip(y, med, q3):
        a.text(max(hi, v) * 1.25 + 0.2, yi, f"{v:.1f}", va="center", fontsize=6.8, color=TEXT2)
    a.grid(axis="x", color="#e6e5e1", lw=0.5)
    a.set_axisbelow(True)
    # (b) share of conditions meeting R1 to R3, by threshold
    show = [("MST", BLUE, "o"), ("A1_soft_admissibility", ORANGE, "s"),
            ("B4_sufficient_reason", AQUA, "^"), ("B3_parsimony_top3", VIOLET, "D")]
    for meth, col, mk in show:
        g = t[t.method == meth].sort_values("tau")
        b.plot(g.tau, g.valid_pct, color=col, lw=2, marker=mk, ms=4.5, label=LABEL[meth])
    b.set_xticks([0.70, 0.80, 0.90, 0.95])
    b.set_ylim(0, 105)
    b.set_xlabel(r"Sufficiency threshold $\tau$")
    b.set_ylabel("Conditions with R1 to R3 met (%)")
    b.set_title("(b) Requirement satisfaction by threshold", loc="left")
    b.grid(axis="y", color="#e6e5e1", lw=0.5)
    b.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGS / "fig_comparison_cost_threshold.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    d = pd.read_csv(ROWS, low_memory=False)
    for col in ("valid", "sufficient", "admissible", "instance_feasible"):
        d[col] = d[col].astype(bool)
    s = summary(d)
    t = by_threshold(d)
    TABLES.mkdir(exist_ok=True)
    s.round(4).to_csv(TABLES / "comparison_requirements_summary.csv")
    t.round(4).to_csv(TABLES / "comparison_by_threshold.csv", index=False)
    fig_requirements(s)
    fig_cost_threshold(s, t)
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
