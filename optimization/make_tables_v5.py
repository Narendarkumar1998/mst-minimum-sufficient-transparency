"""
LaTeX tables of the Online Resource, generated directly from the result files.

Writes tables/latex/tab_S_*.tex
Usage: python optimization/make_tables_v5.py [name ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "data" / "benchmark_v3"
CASE = ROOT / "data" / "case_study_lbnl"
TAB = ROOT / "tables"
OUT = TAB / "latex"
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"greedy": "Greedy (paper)", "greedy_rd": "Greedy + deletion", "greedy_truncated": "Truncated greedy",
       "greedy_truncated_rd": "Truncated greedy + deletion", "issc": "ISSC", "issc_rd": "ISSC + deletion",
       "ga_plain": "GA (plain)", "ga_seeded_repair": "GA (seeded, repair)", "sa": "Simulated annealing",
       "tabu": "Tabu search"}
REG = {"independent": "Independent", "utility_burden_alignment": "Alignment", "utility_burden_conflict": "Conflict"}


def write(name, caption, label, colspec, header, rows, note=None, size="\\footnotesize"):
    lines = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r + " \\\\" if not r.startswith("\\") else r for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += [f"\\par\\smallskip\\parbox{{0.95\\textwidth}}{{\\scriptsize {note}}}"]
    lines += ["\\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(lines))
    print("wrote", name)


def f(x, d=1):
    return "--" if pd.isna(x) else f"{x:,.{d}f}"


def gap(df, k, opt="exact_objective"):
    return (100 * (df[f"{k}_objective"] - df[opt]) / df[opt]).clip(lower=0)


# ---------------------------------------------------------------------------------------------
def scaling():
    sc = pd.read_csv(D3 / "scaling.csv")
    mc = pd.read_csv(D3 / "milp_cold.csv") if (D3 / "milp_cold.csv").exists() else None
    rows = []
    for n, g in sc.groupby("n_items"):
        def ex(col_s, col_t):
            x = g.dropna(subset=[col_s])
            return f"{(x[col_s] == 'OPTIMAL').sum()}/{len(x)}", f(x[col_t].median(), 2), f(x[col_t].max(), 1)
        h = ex("hybrid_status", "hybrid_runtime_s")
        s = ex("scip_status", "scip_runtime_s")
        m = ex("milp_status", "milp_runtime_s")
        if mc is not None:
            x = mc[mc.n_items == n]
            c = (f"{(x.milpcold_status == 'OPTIMAL').sum()}/{len(x)}", f(x.milpcold_runtime_s.median(), 2))
        else:
            c = ("--", "--")
        o = g[g.hybrid_status == "OPTIMAL"]
        gg, gr = gap(o, "greedy", "hybrid_objective"), gap(o, "greedy_rd", "hybrid_objective")
        rows.append(f"{n} & {h[0]} & {h[1]} & {h[2]} & {s[0]} & {s[1]} & {m[0]} & {m[1]} & {c[0]} & {c[1]} & "
                    f"{f(100 * (gg <= 1e-7).mean())} & {f(gg.mean())} & {f(gr.mean())} & "
                    f"{f(1000 * g.greedy_runtime_s.median(), 2)}")
    write("tab_S_scaling",
          "Scaling experiment (20 replicates, 960 conditions per pool size). Exact methods: conditions with proven "
          "optimality within 60~s and median runtime in seconds (hybrid: also maximum). Greedy: share of conditions "
          "solved optimally, mean gap in percent, mean gap after reverse deletion, and median runtime in milliseconds.",
          "tab:S-scaling", "@{}r ccc cc cc cc cccc@{}",
          "$m$ & \\multicolumn{3}{c}{Hybrid} & \\multicolumn{2}{c}{SCIP, cold} & \\multicolumn{2}{c}{MILP, warm} & "
          "\\multicolumn{2}{c}{MILP, cold} & \\multicolumn{4}{c}{Greedy} \\\\ & opt. & med. & max & opt. & med. & opt. & med. & "
          "opt. & med. & opt.\\,\\% & gap & +del. & ms", rows, size="\\scriptsize")


def stats(name, caption, label):
    s = pd.read_csv(TAB / f"stats_{name}_summary.csv")
    fr = pd.read_csv(TAB / f"stats_{name}_friedman.csv").set_index("method")
    rows = []
    for _, r in s.sort_values("method", key=lambda c: c.map(fr.mean_rank)).iterrows():
        rows.append(f"{r.label} & {f(r.optimal_pct)} [{f(r.optimal_ci_low)}, {f(r.optimal_ci_high)}] & "
                    f"{f(r.median_gap, 2)} [{f(r.gap_q1, 2)}, {f(r.gap_q3, 2)}] & "
                    f"{f(r.mean_gap, 2)} [{f(r.mean_ci_low, 2)}, {f(r.mean_ci_high, 2)}] & {f(r.max_gap)} & "
                    f"{fr.loc[r.method, 'mean_rank']:.2f}")
    x = fr.iloc[0]
    write(f"tab_S_stats_{name}", caption + f" Friedman $\\chi^2 = {x.friedman_chi2:,.0f}$, $p < 10^{{-15}}$; "
          f"Nemenyi critical difference {x.nemenyi_cd:.3f} on {int(x.n_blocks):,} conditions.", label,
          "@{}l cccc c@{}", "Method & Optimal, \\% [95\\% CI] & Median gap [IQR] & Mean gap [95\\% CI] & Max gap & Mean rank",
          rows)


def ga_sensitivity():
    s = pd.read_csv(D3 / "ga_sensitivity.csv")
    rows = []
    for (cfg, n), g in s.groupby(["configuration", "n_items"]):
        by = g.groupby(["population", "mutation", "crossover"]).gap_percent.mean()
        rows.append(f"{cfg} & {n} & {f(100 * (g.gap_percent <= 1e-7).mean())} & {f(g.gap_percent.mean(), 2)} & "
                    f"{f(g.gap_percent.max(), 1)} & {f(by.min(), 2)} & {f(by.max(), 2)}")
    write("tab_S_ga_sensitivity",
          "Sensitivity of the genetic algorithm (36 settings: population 50, 100, 200; mutation rate 0.5, 1, 2 per $m$; "
          "two-point or uniform crossover) on three pools per size, balanced weights, $\\tau = 0.90$, usual budget. "
          "Gap in percent to the proven optimum; the last two columns give the mean gap of the best and worst setting.",
          "tab:S-ga-sensitivity", "@{}lc ccc cc@{}",
          "Configuration & $m$ & Optimal, \\% & Mean gap & Max gap & Best setting & Worst setting", rows)


def greedy_strata():
    b = pd.read_csv(D3 / "product_benchmark.csv.gz")
    b = b[b.exact_status == "OPTIMAL"]
    rows = []
    for key, lab in (("regime", "Regime"), ("tau", "Threshold"), ("weight_profile", "Weights"), ("n_items", "Pool size")):
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        for v, g in b.groupby(key):
            gg, gt = gap(g, "greedy"), gap(g, "greedy_truncated_rd")
            name = REG.get(v, str(v).replace("_priority", "").replace("_", " ")) if key != "tau" else f"{v:.2f}"
            rows.append(f"\\quad {name} & {len(g):,} & {f(100 * (gg <= 1e-7).mean())} & {f(gg.mean(), 2)} & "
                        f"{f(100 * (gg > 10).mean())} & {f(100 * (gt <= 1e-7).mean())} & {f(gt.mean(), 2)}")
    write("tab_S_greedy_strata",
          "Paper greedy and truncated greedy with reverse deletion on the 23{,}784 feasible benchmark conditions, by "
          "stratum: share of conditions solved optimally, mean gap in percent, and share of gaps above 10\\%.",
          "tab:S-greedy-strata", "@{}l r ccc cc@{}",
          "Stratum & $N$ & \\multicolumn{3}{c}{Greedy (paper)} & \\multicolumn{2}{c}{Truncated + deletion} \\\\ "
          "& & opt.\\,\\% & mean & $>$10\\,\\% & opt.\\,\\% & mean", rows)


def coverage():
    c = pd.read_csv(D3 / "coverage_benchmark.csv.gz")
    c = c[c.exact_status == "OPTIMAL"]
    rows = []
    for n, g in c.groupby("n_items"):
        a, b = gap(g, "greedy"), gap(g, "greedy_rd")
        rows.append(f"{n} & {len(g):,} & {f(100 * (a <= 1e-7).mean())} & {f(a.median(), 2)} & {f(a.mean(), 2)} & "
                    f"{f(a.max())} & {f(100 * (b <= 1e-7).mean())} & {f(b.mean(), 2)} & {f(b.max())}")
    s = pd.read_csv(D3 / "scip_vs_exact_coverage.csv.gz")
    write("tab_S_coverage",
          "Coverage-utility benchmark (24{,}000 conditions, all feasible): paper greedy and greedy with reverse "
          f"deletion against enumeration, by pool size (gaps in percent). The exact linearization solved with SCIP "
          f"returned the enumeration subset in {s.same_subset.sum():,} of {len(s):,} stratified conditions.",
          "tab:S-coverage", "@{}rr cccc ccc@{}",
          "$M$ & $N$ & \\multicolumn{4}{c}{Greedy (paper)} & \\multicolumn{3}{c}{Greedy + deletion} \\\\ "
          "& & opt.\\,\\% & median & mean & max & opt.\\,\\% & mean & max", rows)


def admissibility():
    a = pd.read_csv(D3 / "admissibility.csv.gz")
    a["inf"] = a.exact_status == "INFEASIBLE"
    rows = []
    for rho, g in a.groupby("rho"):
        inf = g[g.inf]
        fe = g[~g.inf]
        gg, gt = gap(fe, "greedy"), gap(fe, "greedy_truncated_rd")
        rel = inf.release_needed.dropna()
        rows.append(f"{int(100 * rho)}\\,\\% & {len(g):,} & {f(100 * g.inf.mean())} & "
                    f"{f(100 * inf.release_possible.mean()) if len(inf) else '--'} & "
                    f"{f(rel.median(), 0) if len(rel) else '--'} ({f(rel.max(), 0) if len(rel) else '--'}) & "
                    f"{f(inf.shortfall.median(), 3) if len(inf) else '--'} & {f(100 * (gg <= 1e-7).mean())} & "
                    f"{f(100 * (gt <= 1e-7).mean())}")
    write("tab_S_admissibility",
          "Benchmark pools with a share $\\rho$ of items prohibited at random (sizes 8 to 16, 25 replicates, 6{,}000 "
          "conditions per level): share of infeasible conditions; among them, the share for which releasing prohibited "
          "items would restore feasibility, the median (maximum) number of items to release, and the median utility "
          "shortfall; among feasible ones, the share solved optimally by the paper greedy and by the truncated greedy "
          "with reverse deletion.", "tab:S-admissibility", "@{}rr c ccc cc@{}",
          "$\\rho$ & $N$ & Infeasible, \\% & Releasable, \\% & Release & Shortfall & Greedy opt. & Trunc.+del. opt.",
          rows)


def case_items():
    it = pd.read_csv(CASE / "items.csv")
    t = it.groupby(["item", "kind"]).agg(p=("p", "first"), c=("c", "first"), r2=("r2", "max")).reset_index()
    t = t.sort_values(["kind", "item"])
    rows = [f"{r.item[0].upper() + r.item[1:]} & {r.kind} & {f(r.p, 3)} & {f(r.c, 2)} & {f(r.r2, 2)}"
            for r in t.itertuples()]
    q = pd.read_csv(CASE / "fit_quality.csv")
    write("tab_S_case_items",
          "Candidate items of the case study (LBNL Building~59). $p$: measured privacy exposure ($R^2$ of a floor's "
          "occupant count on the item, net of hour-of-week means, maximum of two floors, capped at 0.95); $c$: modeled "
          "interpretation cost before role multipliers; $R^2$: largest univariate $R^2$ of the item on the daily "
          "deviation of one of the five targets. Ten-fold cross-validated $R^2$ of the ridge fits used for the "
          "utilities: " + ", ".join(f"{r.target} {r.cv_r2:.2f}" for r in q.itertuples()) + ".",
          "tab:S-case-items", "@{}l l ccc@{}", "Item & Kind & $p$ & $c$ & $R^2$", rows)


def robustness():
    r = pd.read_csv(D3 / "robustness.csv")
    lab = {"exact": "Exact MST on estimates", "greedy": "Greedy (paper)",
           "greedy_truncated_rd": "Truncated greedy + deletion", "robust": "Robust MST"}
    rows = []
    for k, l in lab.items():
        for dv, g in r.groupby("delta"):
            ok = ~g[f"{k}_infeasible"].astype(bool)
            gg = g[ok]
            rows.append(f"{l} & $\\pm${int(100 * dv)}\\,\\% & {f(100 * gg[f'{k}_changed'].astype(float).mean())} & "
                        f"{f(100 * gg[f'{k}_insufficient'].astype(float).mean(), 2)} & {f(100 * (~ok).mean(), 2)} & "
                        f"{f(gg[f'{k}_regret_pct'].mean(), 2)} & {f(gg[f'{k}_regret_pct'].quantile(0.95), 1)}")
    n = r.groupby("delta").size().iloc[0]
    write("tab_S_robustness",
          f"Selection on noisy estimates, evaluated on the true values ({n:,} feasible conditions per noise level: 150 "
          "pools, 20 noise draws, four thresholds, four weight profiles). Changed: selection differs from the true "
          "optimum; insufficient: true utility below the threshold; infeasible: no selection returned on the estimates; "
          "excess: objective under the true values relative to the true optimum, in percent (mean and 95th percentile).",
          "tab:S-robustness", "@{}l c cccc c@{}",
          "Method & Noise & Changed, \\% & Insufficient, \\% & Infeasible, \\% & Mean excess & P95 excess", rows)


ALL = {"scaling": scaling,
       "stats": lambda: (stats("benchmark", "Constructive heuristics on the 23{,}784 feasible benchmark conditions "
                               "(gaps in percent).", "tab:S-stats-benchmark"),
                         stats("scaling", "Constructive heuristics and metaheuristics on the 1{,}439 scaling "
                               "conditions of replicates 0 to 4 with a proven optimum (gaps in percent; an infeasible "
                               "return counts as the largest observed gap).", "tab:S-stats-scaling")),
       "ga_sensitivity": ga_sensitivity, "greedy_strata": greedy_strata, "coverage": coverage,
       "admissibility": admissibility, "case_items": case_items, "robustness": robustness}

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
