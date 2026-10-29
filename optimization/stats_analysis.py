"""
Statistical analysis of the heuristic and metaheuristic results.

For each comparison set it reports, per method: the share of conditions solved
optimally with a Wilson 95% interval, the median gap with its interquartile range, the
mean gap with a percentile-bootstrap 95% interval (2,000 resamples), and the maximum gap;
pairwise two-sided Wilcoxon signed-rank tests on the paired gaps with Holm correction;
and a Friedman test across methods (blocks = conditions) with Nemenyi post-hoc critical
difference (Demsar 2006), drawn as a critical-difference diagram.

Sets:
  benchmark   constructive heuristics on the 23,784 feasible product-form conditions
  scaling     constructive heuristics and metaheuristics on the 1,440 scaling conditions
              that have metaheuristic runs (pools of 16 to 512 items, replicates 0-4)

Writes tables/stats_<set>_summary.csv, tables/stats_<set>_wilcoxon.csv,
tables/stats_<set>_friedman.csv and figures/fig_cd_<set>.pdf
Usage: python optimization/stats_analysis.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "optimization"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
OPT_TOL = 1e-7          # gap in percent below which a solution counts as optimal
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268}
LABELS = {"greedy": "Greedy (paper)", "greedy_rd": "Greedy + deletion",
          "greedy_truncated": "Truncated greedy", "greedy_truncated_rd": "Truncated greedy + deletion",
          "issc": "ISSC", "issc_rd": "ISSC + deletion", "ga_plain": "GA (plain)",
          "ga_seeded_repair": "GA (seeded, repair)", "sa": "Simulated annealing", "tabu": "Tabu search"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - hw, c + hw


def boot_mean(x, rng, b=2000):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def summarize(G, rng):
    rows = []
    for k in G.columns:
        g = G[k].to_numpy()
        n = len(g)
        opt = int((g <= OPT_TOL).sum())
        lo, hi = wilson(opt, n)
        blo, bhi = boot_mean(g, rng)
        q1, q3 = np.percentile(g, [25, 75])
        rows.append(dict(method=k, label=LABELS.get(k, k), n=n, optimal_pct=100 * opt / n,
                         optimal_ci_low=100 * lo, optimal_ci_high=100 * hi, median_gap=np.median(g),
                         gap_q1=q1, gap_q3=q3, mean_gap=g.mean(), mean_ci_low=blo, mean_ci_high=bhi,
                         max_gap=g.max(), over10_pct=100 * (g > 10).mean()))
    return pd.DataFrame(rows)


def wilcoxon_holm(G):
    rows = []
    for a, b in itertools.combinations(G.columns, 2):
        d = G[a] - G[b]
        if (d != 0).sum() == 0:
            rows.append(dict(a=a, b=b, n_nonzero=0, statistic=np.nan, p=1.0, median_diff=0.0))
            continue
        res = stats.wilcoxon(G[a], G[b], zero_method="wilcox", alternative="two-sided")
        rows.append(dict(a=a, b=b, n_nonzero=int((d != 0).sum()), statistic=res.statistic, p=res.pvalue,
                         median_diff=float(np.median(d))))
    w = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(w)
    adj = np.minimum(1.0, np.maximum.accumulate(w.p.to_numpy() * (m - np.arange(m))))
    w["p_holm"] = adj
    w["significant_0.05"] = w.p_holm < 0.05
    return w


def friedman_cd(G, name):
    ranks = G.rank(axis=1, method="average")           # rank 1 = smallest gap
    mean_ranks = ranks.mean().sort_values()
    k, n = G.shape[1], G.shape[0]
    chi2, p = stats.friedmanchisquare(*[G[c] for c in G.columns])
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    pd.DataFrame(dict(method=mean_ranks.index, mean_rank=mean_ranks.values)).assign(
        friedman_chi2=chi2, friedman_p=p, nemenyi_cd=cd, n_blocks=n).to_csv(
        TAB / f"stats_{name}_friedman.csv", index=False)
    # critical-difference diagram
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                         "font.size": 7.5, "pdf.fonttype": 42})
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(6.6, 0.55 + 0.22 * k))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.85
    ax.plot([lo, hi], [y0, y0], color="#0b0b0b", lw=0.8)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color="#0b0b0b", lw=0.8)
        ax.text(t, y0 + 0.06, str(t), ha="center", va="bottom")
    ax.plot([lo, lo + cd], [0.98, 0.98], color="#eb6834", lw=1.5)
    ax.text(lo + cd / 2, 1.0, f"CD = {cd:.3f}", ha="center", va="bottom", color="#eb6834")
    items = list(mean_ranks.items())
    half = (len(items) + 1) // 2
    step = 0.7 / max(half, 1)
    for i, (mname, r) in enumerate(items):
        left = i < half
        yy = y0 - 0.08 - step * (i if left else len(items) - 1 - i)
        xx = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xx], [y0, yy, yy], color="#52514e", lw=0.6)
        ax.text(xx + (-0.02 if left else 0.02), yy, f"{LABELS.get(mname, mname)} ({r:.2f})",
                ha="right" if left else "left", va="center")
    # cliques: groups of methods whose mean ranks differ by less than CD
    vals = mean_ranks.to_numpy()
    cliques = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yy = y0 - 0.04 - 0.035 * c
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yy, yy], color="#2a78d6", lw=2.2, solid_capstyle="round")
    fig.savefig(FIG / f"fig_cd_{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return chi2, p, cd, mean_ranks


def gaps(df, opt_col, methods):
    G = pd.DataFrame({k: 100 * (df[f"{k}_objective"] - df[opt_col]) / df[opt_col] for k in methods})
    return G.clip(lower=0.0)          # tiny negative values are rounding


def analyse(name, G):
    rng = np.random.default_rng(2026)
    s = summarize(G, rng)
    s.to_csv(TAB / f"stats_{name}_summary.csv", index=False)
    w = wilcoxon_holm(G)
    w.to_csv(TAB / f"stats_{name}_wilcoxon.csv", index=False)
    chi2, p, cd, mr = friedman_cd(G, name)
    print(f"\n== {name}: {len(G)} conditions, Friedman chi2 = {chi2:.1f}, p = {p:.2e}, CD = {cd:.3f}")
    print(s[["label", "optimal_pct", "optimal_ci_low", "optimal_ci_high", "median_gap", "gap_q1", "gap_q3",
             "mean_gap", "mean_ci_low", "mean_ci_high", "max_gap"]].round(2).to_string(index=False))
    print(mr.round(3).to_string())
    print(w[~w["significant_0.05"]][["a", "b", "p_holm"]].to_string(index=False))


def main():
    cons = ["greedy", "greedy_rd", "greedy_truncated", "greedy_truncated_rd", "issc", "issc_rd"]
    b = pd.read_csv(DATA / "heuristics_vs_exact_benchmark.csv.gz")
    b = b[b.exact_status == "OPTIMAL"]
    analyse("benchmark", gaps(b, "exact_objective", cons))
    sc = pd.read_csv(DATA / "scaling_v2.csv")
    mh = pd.read_csv(DATA / "metaheuristics_v2.csv")
    keys = ["n_items", "regime", "replicate", "weight_profile", "tau"]
    m = sc.merge(mh, on=keys, how="inner")
    m = m[m.hybrid_status == "OPTIMAL"]
    meta = ["ga_plain", "ga_seeded_repair", "sa", "tabu"]
    for k in meta:                              # an infeasible return counts as the worst possible gap
        m.loc[~m[f"{k}_feasible"].astype(bool), f"{k}_objective"] = np.nan
    G = gaps(m, "hybrid_objective", cons + meta)
    G = G.fillna(G.max().max())
    analyse("scaling", G)


if __name__ == "__main__":
    main()
