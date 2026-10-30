"""
Figures of the revised article (one style for all; colors follow the method, never its rank).

  fig_scaling.pdf          runtime, optimality and gap against pool size (scaling.csv, metaheuristics.csv)
  fig_gap_ecdf.pdf         distribution of the relative gap on the 24,000-condition benchmark
  fig_convergence.pdf      best gap against objective evaluations (convergence_traces.csv)
  fig_pareto_example.pdf   sufficient subsets, exact Pareto front, NSGA-II and MST solutions (pareto_example.csv)
  fig_case_study.pdf       feasibility against threshold and disclosure frequency by role (case study)
  fig_robustness.pdf       true sufficiency under estimation noise (robustness.csv)

Usage: python optimization/make_figures_v5.py [name ...]
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
D3 = ROOT / "data" / "benchmark_v3"
CASE = ROOT / "data" / "case_study_lbnl"
FIG = ROOT / "figures"
W = 6.5                          # full text width in inches
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

# one color and marker per method, fixed across every figure
M = {
    "greedy": ("Greedy (paper)", "#2a78d6", "o"),
    "greedy_rd": ("Greedy + deletion", "#eb6834", "s"),
    "greedy_truncated_rd": ("Truncated greedy + deletion", "#1baf7a", "^"),
    "issc_rd": ("ISSC + deletion", "#6250d6", "v"),
    "ga_seeded_repair": ("GA (seeded, repair)", "#eda100", "D"),
    "sa": ("Simulated annealing", "#e87ba4", "P"),
    "tabu": ("Tabu search", "#008300", "X"),
    "ga_plain": ("GA (plain)", "#e34948", "h"),
    "hybrid": ("Hybrid exact (SCIP, warm start)", INK, "o"),
    "scip": ("SCIP, cold start", "#8a8983", "s"),
    "milp": ("Chain MILP, warm start", INK2, "^"),
}
ROLE = {"facility_manager": ("Facility manager", "#2a78d6", "o"),
        "energy_auditor": ("Energy auditor", "#eb6834", "s"),
        "occupant": ("Occupant", "#1baf7a", "^")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "lines.linewidth": 1.2,
    "lines.markersize": 3.6, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(x, rng, b=2000):
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


# ---------------------------------------------------------------------------------------------
def fig_scaling():
    sc = pd.read_csv(D3 / "scaling.csv")
    mh = pd.read_csv(D3 / "metaheuristics.csv")
    keys = ["n_items", "regime", "replicate", "weight_profile", "tau"]
    sizes = sorted(sc.n_items.unique())
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.35))

    # (a) runtime: median and interquartile range
    a = ax[0]
    for k, col in [("hybrid", "hybrid_runtime_s"), ("scip", "scip_runtime_s"), ("milp", "milp_runtime_s"),
                   ("greedy_rd", "greedy_rd_runtime_s")]:
        lab, c, mk = M[k]
        g = sc.dropna(subset=[col]).groupby("n_items")[col]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        a.plot(med.index, med.values, color=c, marker=mk, label=lab, ls="--" if k == "milp" else "-")
        a.fill_between(med.index, q1.values, q3.values, color=c, alpha=0.15, lw=0)
    bud = [max(1.0, 0.02 * m) for m in sizes]
    a.plot(sizes, bud, color="#b5b4ae", lw=0.9, ls=":", label="Metaheuristic budget")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(sizes, [str(s) for s in sizes])
    a.set_xlabel("Admissible items $m$")
    a.set_ylabel("Runtime per condition (s)")
    panel(a, "a", "Runtime")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6.0, handlelength=1.8,
             columnspacing=0.8)

    # (b), (c): optimality and gap on the conditions with metaheuristic runs
    m = sc.merge(mh, on=keys, how="inner")
    m = m[m.hybrid_status == "OPTIMAL"]
    meths = ["greedy", "greedy_rd", "greedy_truncated_rd", "ga_seeded_repair", "sa", "tabu"]
    for k in ["ga_seeded_repair", "sa", "tabu"]:
        m.loc[~m[f"{k}_feasible"].astype(bool), f"{k}_objective"] = np.nan
    for k in meths:
        lab, c, mk = M[k]
        gap = (100 * (m[f"{k}_objective"] - m.hybrid_objective) / m.hybrid_objective).clip(lower=0)
        opt, lo, hi, mg, glo, ghi = [], [], [], [], [], []
        for s in sizes:
            gs = gap[m.n_items == s]
            n = len(gs)
            kk = int((gs <= 1e-7).sum())                      # an infeasible return is not optimal
            o = wilson(kk, n)
            opt.append(100 * kk / n)
            lo.append(100 * o[0])
            hi.append(100 * o[1])
            gf = gs.dropna()
            mg.append(gf.mean())
            ci = boot_ci(gf, rng)
            glo.append(ci[0])
            ghi.append(ci[1])
        for a, y, l, h in ((ax[1], opt, lo, hi), (ax[2], mg, glo, ghi)):
            a.plot(sizes, y, color=c, marker=mk, label=lab)
            a.fill_between(sizes, l, h, color=c, alpha=0.15, lw=0)
    for a in ax[1:]:
        a.set_xscale("log", base=2)
        a.set_xticks(sizes, [str(s) for s in sizes])
        a.set_xlabel("Admissible items $m$")
    ax[1].set_ylabel("Conditions solved optimally (%)")
    ax[1].set_ylim(0, 102)
    panel(ax[1], "b", "Optimality (95% Wilson interval)")
    ax[2].set_ylabel("Mean relative gap (%)")
    ax[2].set_ylim(bottom=0)
    panel(ax[2], "c", "Gap (95% bootstrap interval)")
    h, l = ax[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(bottom=0.33)
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.675, 0.115), fontsize=6.3, handlelength=1.8,
               columnspacing=1.0)
    save(fig, "fig_scaling")


# ---------------------------------------------------------------------------------------------
def fig_gap_ecdf():
    b = pd.read_csv(D3 / "product_benchmark.csv.gz")
    b = b[b.exact_status == "OPTIMAL"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    for k in ["greedy", "greedy_rd", "greedy_truncated_rd", "issc_rd"]:
        lab, c, mk = M[k]
        g = np.sort((100 * (b[f"{k}_objective"] - b.exact_objective) / b.exact_objective).clip(lower=0).to_numpy())
        y = 100 * np.arange(1, len(g) + 1) / len(g)
        a.step(np.maximum(g, 1e-3), y, where="post", color=c, label=f"{lab} ({100 * (g <= 1e-7).mean():.1f}% optimal)")
    a.set_xscale("log")
    a.set_xlim(1e-2, 100)
    a.set_ylim(30, 101)
    a.set_xlabel("Relative gap to the exact optimum (%, log scale)")
    a.set_ylabel("Conditions with gap at most $x$ (%)")
    for v in (1, 10):
        a.axvline(v, color="#b5b4ae", lw=0.6, ls=":")
    panel(a, "a", f"Cumulative distribution of the gap ({len(b):,} conditions)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.3)
    # (b) failure mechanism of the paper greedy among gaps above 10%
    a = ax[1]
    big = b[(b.greedy_objective - b.exact_objective) / b.exact_objective > 0.10]
    order = ["more_items", "same_count_different_items", "fewer_items"]
    names = {"more_items": "More items than optimal", "same_count_different_items": "Same number, other items",
             "fewer_items": "Fewer items"}
    sh = big.greedy_mechanism.value_counts(normalize=True).reindex(order).fillna(0) * 100
    by = big.groupby("regime").greedy_mechanism.value_counts(normalize=True).unstack().reindex(columns=order).fillna(0) * 100
    reg = {"utility_burden_conflict": "Conflict", "independent": "Independent", "utility_burden_alignment": "Aligned"}
    rows = ["All"] + [reg.get(r, r) for r in by.index]
    vals = np.vstack([sh.values, by.values])
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    left = np.zeros(len(rows))
    for j, o in enumerate(order):
        a.barh(rows, vals[:, j], left=left, color=cols[j], height=0.62, edgecolor="white", lw=1.0, label=names[o])
        left += vals[:, j]
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("Share of greedy gaps above 10% (%)")
    a.grid(axis="x", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_title("(b) Why the paper greedy fails", loc="left", fontweight="bold", color=INK)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_gap_ecdf")


# ---------------------------------------------------------------------------------------------
def fig_convergence():
    d = pd.read_csv(D3 / "convergence_traces.csv")
    name = {"GA (plain)": "ga_plain", "GA (seeded, repair)": "ga_seeded_repair", "simulated annealing": "sa",
            "tabu search": "tabu"}
    sizes = sorted(d.n_items.unique())
    fig, ax = plt.subplots(1, len(sizes), figsize=(W, 2.1), sharey=True)
    for i, (a, n) in enumerate(zip(ax, sizes)):
        for meth, g in d[d.n_items == n].groupby("method"):
            lab, c, mk = M[name[meth]]
            g = g.sort_values("evaluations")
            ls = {"tabu": (0, (4, 2)), "sa": (0, (1.2, 1.2))}.get(name[meth], "-")
            a.step(g.evaluations, g.gap_percent.clip(lower=1e-3), where="post", color=c, label=lab, ls=ls,
                   lw=1.4 if name[meth] in ("tabu", "sa") else 1.2)
        a.set_xscale("log")
        a.set_yscale("symlog", linthresh=0.1)
        a.set_xlabel("Objective evaluations")
        panel(a, "abc"[i], f"{n} items")
    ax[0].set_ylabel("Gap to the optimum (%)")
    h, l = ax[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(bottom=0.3)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), handlelength=2.4)
    save(fig, "fig_convergence")


# ---------------------------------------------------------------------------------------------
def fig_pareto():
    e = pd.read_csv(D3 / "pareto_example.csv")
    prof = [c for c in e.columns if c.startswith("mst_")]
    pn = {"mst_balanced": ("Balanced", "s"), "mst_privacy_priority": ("Privacy priority", "^"),
          "mst_interpretation_priority": ("Interpretation priority", "v"), "mst_parsimony_priority": ("Parsimony priority", "D")}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    for i, (a, yc, yl) in enumerate(zip(ax, ["C", "K"], ["Normalized interpretation cost $C(S)$",
                                                          "Normalized cardinality $K(S)$"])):
        a.scatter(e.P, e[yc], s=2, color="#d9d8d2", lw=0, rasterized=True, label="Sufficient subsets")
        f = e[e.on_front]
        a.scatter(f.P, f[yc], s=12, color="#2a78d6", lw=0, label="Exact Pareto front")
        g = e[e.nsga2]
        a.scatter(g.P, g[yc], s=30, facecolors="none", edgecolors="#eb6834", lw=0.8, label="NSGA-II")
        for p in prof:
            r = e[e[p]]
            lab, mk = pn.get(p, (p, "o"))
            a.scatter(r.P, r[yc], s=26, marker=mk, color=INK, lw=0, zorder=5, label=f"MST, {lab.lower()}")
        a.set_xlabel("Relative privacy exposure $P(S)$")
        a.set_ylabel(yl)
        panel(a, "ab"[i], "Privacy against cost" if yc == "C" else "Privacy against size")
        lo = f.P.min()
        a.set_xlim(lo - 0.03, min(1.0, f.P.max() + 0.15))
        a.set_ylim(f[yc].min() * 0.88, f[yc].max() * 1.25)
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(bottom=0.3)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, markerscale=1.2)
    save(fig, "fig_pareto_example")


# ---------------------------------------------------------------------------------------------
def fig_case_study():
    d = pd.read_csv(CASE / "results.csv")
    it = pd.read_csv(CASE / "items.csv").drop_duplicates("item").set_index("item")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    a = ax[0]
    for r, (lab, c, mk) in ROLE.items():
        g = d[d.role == r].groupby("tau").status.apply(lambda s: 100 * (s != "INFEASIBLE").mean())
        a.plot(g.index, g.values, color=c, marker=mk, label=lab)
        nom = d[(d.role == r) & d.nominal].tau.iloc[0]
        a.scatter([nom], [g[nom]], s=46, facecolors="none", edgecolors=c, lw=1.0, zorder=5)
    a.set_xlabel(r"Sufficiency threshold $\tau$")
    a.set_ylabel("Incidents with a sufficient disclosure (%)")
    a.set_ylim(0, 102)
    panel(a, "a", "Feasibility (circled: nominal threshold)")
    a.legend(loc="upper right")
    # (b) disclosure frequency at the nominal thresholds
    a = ax[1]
    n = d[d.nominal & (d.status != "INFEASIBLE")]
    roles = list(ROLE)
    cnt = {}
    for r in roles:
        s = n[n.role == r].selected.str.split("; ").explode()
        cnt[r] = 100 * s.value_counts() / (n.role == r).sum()
    T = pd.DataFrame(cnt).fillna(0)
    T = T.loc[T.max(axis=1).sort_values(ascending=False).index[:12]]
    im = a.imshow(T.values, cmap="Blues", vmin=0, vmax=max(40, T.values.max()), aspect="auto")
    a.set_xticks(range(len(roles)), [ROLE[r][0].replace(" ", "\n") for r in roles])
    a.set_yticks(range(len(T)), [f"{i[0].upper() + i[1:]}  ($p$ = {it.loc[i, 'p']:.2f})" for i in T.index],
                 fontsize=6.4)
    permitted = {"facility_manager": None, "energy_auditor": {"occupancy"},
                 "occupant": {"hvac", "occupancy"}}             # kinds each role may not see
    for x, r in enumerate(roles):
        for y, i in enumerate(T.index):
            if permitted[r] and it.loc[i, "kind"] in permitted[r]:
                a.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor="#c9c8c2",
                                          hatch="////", lw=0))
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    for (y, x), v in np.ndenumerate(T.values):
        if v >= 5:
            a.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=6,
                   color="white" if v > 0.6 * max(40, T.values.max()) else INK)
    a.set_title("(b) Items disclosed (% of feasible incidents; hatched: not permitted)", loc="left", fontweight="bold", color=INK)
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_case_study")


# ---------------------------------------------------------------------------------------------
COMP = [("MST", "MST (complete model)"),
        ("A1_soft_admissibility", "Without hard admissibility (penalty)"),
        ("A2_utility_reward", "Without sufficiency threshold (reward)"),
        ("A3_min_cardinality", "Without cost objective (fewest items)"),
        ("A4_no_infeasibility", "Without infeasibility reporting"),
        ("B7_policy_greedy_delete", "Without exact search (truncated greedy + deletion)"),
        ("C1_model_reconciliation", "Model reconciliation"),
        ("C2_goal_obfuscation", "Goal obfuscation and legibility"),
        ("C3_role_aware_haexa", "Role-aware HAExA"),
        ("C4_parsimonious_haexa", "Parsimonious HAExA")]


def requirements(d):
    fe = d[d.instance_feasible]
    inf = d[~d.instance_feasible]
    r1 = 100 * ((fe.status == "DISCLOSED") & fe.sufficient.astype(bool)).mean()
    r2 = 100 * (d.prohibited_disclosed.fillna(0) == 0).mean()
    r3 = 100 * (inf.status == "INFEASIBLE").mean() if len(inf) else np.nan
    ok_f = (fe.status == "DISCLOSED") & fe.sufficient.astype(bool) & (fe.prohibited_disclosed.fillna(0) == 0)
    ok_i = inf.status == "INFEASIBLE"
    allm = 100 * (ok_f.sum() + ok_i.sum()) / len(d)
    return r1, r2, r3, allm


def fig_comparison():
    d = pd.read_csv(ROOT / "data" / "comparison" / "comparison_rows.csv.gz",
                    usecols=["method", "instance_feasible", "status", "sufficient", "prohibited_disclosed",
                             "objective_excess_pct", "valid"])
    rows, ex = [], []
    for k, lab in COMP:
        g = d[d.method == k]
        rows.append(requirements(g))
        e = g.loc[g.valid.astype(bool) & g.instance_feasible, "objective_excess_pct"].dropna()
        ex.append((e.median(), e.quantile(0.25), e.quantile(0.75), e.quantile(0.95), len(e)))
    V = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]), sharey=True)
    a = ax[0]
    a.imshow(V, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cols = ["R1\nsufficient", "R2\npolicy kept", "R3\ninfeasibility", "All\nthree"]
    a.set_xticks(range(4), cols, fontsize=6.6)
    a.xaxis.tick_top()
    a.set_yticks(range(len(COMP)), [lab for _, lab in COMP], fontsize=6.6)
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    for (y, x), v in np.ndenumerate(V):
        a.text(x, y, "--" if np.isnan(v) else f"{v:.1f}", ha="center", va="center", fontsize=6.2,
               color="white" if v > 60 else INK)
    for yb in (0.5, 4.5, 5.5):
        a.axhline(yb, color="white", lw=2.5)
    a.set_xlabel("Conditions meeting the requirement (%)")
    b = ax[1]
    for y, (med, q1, q3, q95, n) in enumerate(ex):
        if n == 0:
            b.text(0.02, y, "no valid output", va="center", fontsize=6.2, color=INK2, transform=b.get_yaxis_transform())
            continue
        b.plot([q1, q3], [y, y], color="#2a78d6", lw=2.2, solid_capstyle="butt")
        b.plot([q3, q95], [y, y], color="#2a78d6", lw=0.8)
        b.plot(med, y, "o", color=INK, ms=3.6)
    b.set_xscale("symlog", linthresh=1)
    b.set_xlim(-0.2, 300)
    b.grid(axis="x", color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.set_xlabel("Objective excess over MST when R1 to R3 are met (%)")
    b.tick_params(axis="y", length=0)
    b.set_title("median, interquartile range, 95th percentile", fontsize=6.4, color=INK2, loc="left")
    for yb in (0.5, 4.5, 5.5):
        b.axhline(yb, color=GRID, lw=0.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_comparison")


# ---------------------------------------------------------------------------------------------
def fig_robustness():
    r = pd.read_csv(D3 / "robustness.csv", low_memory=False)
    meths = [("exact", "Exact MST on estimates"), ("greedy", "Greedy (paper)"),
             ("greedy_truncated_rd", "Truncated greedy + deletion"), ("robust", "Robust MST (deflated $u$)")]
    dcol = {0.1: ("#86b6ef", "o", "$\\pm$10% noise"), 0.2: ("#1c5cab", "s", "$\\pm$20% noise")}
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.0), sharey=True)
    metrics = [("changed", "Selection differs from the\ntrue optimum (%)"),
               ("insufficient", "Insufficient under the\ntrue values (%)"), ("regret_pct", "Mean objective excess\nunder the true values (%)")]
    for j, (met, xl) in enumerate(metrics):
        a = ax[j]
        for y, (k, lab) in enumerate(meths):
            for dv, (c, mk, dl) in dcol.items():
                g = r[(r.delta == dv) & ~r[f"{k}_infeasible"].astype(bool)][f"{k}_{met}"].dropna().astype(float)
                yy = y + (-0.13 if dv == 0.1 else 0.13)
                if met == "regret_pct":
                    v = g.mean()
                    lo, hi = boot_ci(g, np.random.default_rng(1), 1000)
                else:
                    v = 100 * g.mean()
                    lo, hi = (100 * x for x in wilson(int(g.sum()), len(g)))
                a.plot([lo, hi], [yy, yy], color=c, lw=1.0)
                a.plot(v, yy, marker=mk, color=c, ms=4, ls="", label=dl if (j == 0 and y == 0) else None)
        a.set_xlabel(xl)
        a.grid(axis="x", color=GRID, lw=0.5)
        a.set_axisbelow(True)
        if met == "regret_pct":
            a.axvline(0, color=INK2, lw=0.6)
        else:
            a.set_xlim(0, 102)
        a.tick_params(axis="y", length=0)
    ax[0].set_yticks(range(len(meths)), [l for _, l in meths])
    ax[0].invert_yaxis()
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(top=0.86)
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.0), fontsize=6.5)
    save(fig, "fig_robustness")


ALL = {"scaling": fig_scaling, "gap_ecdf": fig_gap_ecdf, "convergence": fig_convergence,
       "pareto": fig_pareto, "comparison": fig_comparison, "case_study": fig_case_study, "robustness": fig_robustness}

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
