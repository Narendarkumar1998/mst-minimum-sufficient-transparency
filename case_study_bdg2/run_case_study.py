"""
Case study on real building data: MST on candidate pools built from the
Building Data Genome 2 (BDG2) data set (Miller et al., Scientific Data 7, 368, 2020),
site "Bobcat" (36 education, office, residence and other buildings; hourly
electricity, chilled-water and hot-water meters and weather, 2016-2017).

Scenario. Each meter is reported by a remote agent. When the daily electricity use
of a target building deviates strongly from its baseline, the assistant agent must
decide what to disclose to three stakeholders who oversee the alert: the facility
manager, an external energy auditor and an occupant of the building. Each
candidate item is a piece of evidence about the anomalous day.

Item values.
  u_i  data-driven: share of the variance of the target's daily electricity deviation
       explained by the item's own daily series over the two years (R^2 of a
       univariate least-squares fit), so u_i measures how much the item explains
       the kind of deviation being reported;
  p_i  modeled by item class (residence-hall meters reveal residents' presence);
  c_i  modeled by representation (a scalar deviation, a load profile, a comparison);
  admissibility, threshold and weights by role.

Writes data/case_study_bdg2/results.csv, items.csv and walkthrough.csv
Usage: python case_study_bdg2/run_case_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimization"))
import heuristics as h  # noqa: E402
import scip_model as sm  # noqa: E402

DATA = ROOT / "data" / "case_study_bdg2"
TARGETS = ["Bobcat_education_Miki", "Bobcat_education_Alissa", "Bobcat_education_Angela",
           "Bobcat_education_Marisha", "Bobcat_science_Tammy", "Bobcat_education_Dylan"]
INCIDENTS_PER_TARGET = 10
N_PEER_ELEC, N_PEER_CW = 15, 5
WEATHER = ["airTemperature", "dewTemperature", "cloudCoverage", "windSpeed", "precipDepth1HR"]

# modeled privacy exposure and interpretation cost by item class
CLASS_P = {"own_meter": 0.05, "own_profile": 0.08, "weather": 0.01, "calendar": 0.01,
           "campus": 0.02, "peer_elec": 0.06, "peer_cw": 0.04, "residence": 0.25}
CLASS_C = {"own_meter": 0.20, "own_profile": 0.50, "weather": 0.15, "calendar": 0.05,
           "campus": 0.15, "peer_elec": 0.30, "peer_cw": 0.30, "residence": 0.30}

ROLES = {
    # admissible classes, threshold, weights (P, C, K), cost multiplier for meter items
    "facility_manager": dict(classes=set(CLASS_P), tau=0.40, w=(0.2, 0.4, 0.4), cmult=1.0),
    "energy_auditor": dict(classes=set(CLASS_P) - {"residence", "own_profile"}, tau=0.30,
                           w=(1 / 3, 1 / 3, 1 / 3), cmult=1.0),
    "occupant": dict(classes={"own_meter", "weather", "calendar", "campus"}, tau=0.10,
                     w=(0.2, 0.6, 0.2), cmult=1.5),
}
# reference scales: a typical disclosure of 10 items of typical cost 0.25
CASE_REF_ITEMS = 10.0
CASE_REF_COST = 10.0 * 0.25
TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]     # thresholds swept for every role


def daily(name, how="sum"):
    d = pd.read_csv(DATA / f"bobcat_{name}.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
    g = d.resample("D")
    return (g.sum(min_count=20) if how == "sum" else g.mean())


def deviation(s):
    """Relative deviation from the median of the same weekday in the two weeks before
    and the two weeks after (the day itself excluded)."""
    out = pd.Series(np.nan, index=s.index)
    for wd in range(7):
        m = s.index.dayofweek == wd
        x = s[m]
        nb = pd.concat([x.shift(k) for k in (-2, -1, 1, 2)], axis=1)
        base = nb.median(axis=1, skipna=True).where(nb.notna().sum(axis=1) >= 2)
        out[m] = (x - base) / base.abs()
    return out.replace([np.inf, -np.inf], np.nan)


def fit(y, x):
    """Univariate least squares of y on x over the two years: (R^2, slope, intercept)."""
    ok = y.notna() & x.notna()
    if ok.sum() < 60 or x[ok].std() == 0:
        return 0.0, 0.0, 0.0
    b, a = np.polyfit(x[ok], y[ok], 1)
    c = np.corrcoef(y[ok], x[ok])[0, 1]
    return (float(c * c) if np.isfinite(c) else 0.0), float(b), float(a)


def r2(y, x):
    return fit(y, x)[0]


def day_utility(y, x, day):
    """u_i(t) = R^2_i times the share of the day's deviation that the item's fit
    accounts for, clipped to [0, 1]; an item pointing the other way is worth nothing."""
    R2, b, a = fit(y, x)
    if not (np.isfinite(y.get(day, np.nan)) and np.isfinite(x.get(day, np.nan))) or y[day] == 0:
        return 0.0
    share = float(np.clip((b * x[day]) / y[day], 0.0, 1.0))
    return R2 * share


DRY = "--dry" in sys.argv


def main():
    elec, cw, hw = daily("electricity"), daily("chilledwater"), daily("hotwater")
    hourly_e = pd.read_csv(DATA / "bobcat_electricity.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
    weather = pd.read_csv(DATA / "bobcat_weather.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
    wd = weather[WEATHER].resample("D").agg({"airTemperature": "mean", "dewTemperature": "mean",
                                            "cloudCoverage": "mean", "windSpeed": "mean",
                                            "precipDepth1HR": "sum"})
    w_anom = wd - wd.rolling(15, center=True, min_periods=7).mean()
    meta = pd.read_csv(DATA / "bobcat_metadata.csv").set_index("building_id")
    lodging = [b for b in meta.index if "_lodging_" in b and b in elec.columns]

    dev_e = elec.apply(deviation)
    dev_cw = cw.apply(deviation)
    dev_hw = hw.apply(deviation)
    night = hourly_e.between_time("00:00", "04:59").resample("D").mean()
    dev_night = night.apply(deviation)
    campus = dev_e[[b for b in dev_e.columns if b not in lodging]].median(axis=1)
    weekend = pd.Series((dev_e.index.dayofweek >= 5).astype(float), index=dev_e.index)

    rows, item_rows, walk = [], [], []
    for target in TARGETS:
        y = dev_e[target]
        z = (y - y.median()) / (1.4826 * (y - y.median()).abs().median())
        cand_days = z.abs().dropna().sort_values(ascending=False)
        days = []
        for d in cand_days.index:                         # largest anomalies, at least 7 days apart
            if all(abs((d - e).days) >= 7 for e in days):
                days.append(d)
            if len(days) == INCIDENTS_PER_TARGET:
                break
        # candidate series and their data-driven utilities (one fit per target)
        series = {}
        if target in dev_cw:
            series[("own_meter", "chilled water")] = dev_cw[target]
        if target in dev_hw:
            series[("own_meter", "hot water")] = dev_hw[target]
        series[("own_profile", "night base load")] = dev_night[target]
        for v in WEATHER:
            series[("weather", v)] = w_anom[v]
        series[("calendar", "weekend")] = weekend
        series[("campus", "campus median electricity")] = campus
        others = [b for b in dev_e.columns if b != target and b not in lodging]
        peer_r2 = sorted(((r2(y, dev_e[b]), b) for b in others), reverse=True)[:N_PEER_ELEC]
        for _, b in peer_r2:
            series[("peer_elec", f"electricity {b.split('_')[-1]}")] = dev_e[b]
        cw_others = [b for b in dev_cw.columns if b != target]
        for _, b in sorted(((r2(y, dev_cw[b]), b) for b in cw_others), reverse=True)[:N_PEER_CW]:
            series[("peer_cw", f"chilled water {b.split('_')[-1]}")] = dev_cw[b]
        for b in lodging:
            series[("residence", f"electricity {b.split('_')[-1]} (residence)")] = dev_e[b]
        keys = list(series)
        cls = [k[0] for k in keys]
        for k in keys:
            item_rows.append(dict(target=target, item_class=k[0], item=k[1], r2=r2(y, series[k]),
                                  p=CLASS_P[k[0]], c=CLASS_C[k[0]]))
        for day in days:
            u = np.array([day_utility(y, series[k], day) for k in keys])
            for role, R0 in ROLES.items():
              for tau in TAU_GRID:
                  R = dict(R0, tau=tau)
                  adm = [c in R["classes"] for c in cls]
                  p = np.array([CLASS_P[c] for c in cls])
                  c = np.array([CLASS_C[c] * (R["cmult"] if c in ("own_meter", "own_profile", "peer_elec",
                                                                    "peer_cw", "residence") else 1.0)
                                for c in cls])
                  row = dict(target=target, day=day.date(), z=float(z[day]), role=role, tau=R["tau"],
                             nominal=bool(abs(tau - R0["tau"]) < 1e-9),
                             n_items=len(keys), n_admissible=int(sum(adm)))
                  if DRY:
                      rows.append(dict(row, max_admissible_utility=1 - float(np.prod([1 - u[i] for i in range(len(u)) if adm[i]])),
                                       n_positive=int((u > 0.01).sum()), u_max=float(u.max())))
                      continue
                  best, best_J, res = None, np.inf, {}
                  for name, f in h.METHODS.items():
                      pl = h.Pool(u, p, c, adm, R["tau"], R["w"], cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS)
                      t0 = time.perf_counter()
                      x = f(pl)
                      res[name] = (pl.J(x) if x is not None else np.nan, time.perf_counter() - t0)
                      if x is not None and pl.J(x) < best_J:
                          best, best_J = pl.to_indices(x), pl.J(x)
                  s = sm.solve_mst_scip(list(u), list(p), list(c), adm, R["tau"], R["w"], start=best,
                                         cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS)
                  row.update(status=s.status, objective=s.objective, runtime_s=s.runtime,
                             n_selected=len(s.selected),
                             selected="; ".join(f"{keys[i][1]}" for i in s.selected),
                             selected_classes="|".join(sorted({keys[i][0] for i in s.selected})),
                             utility=1 - float(np.prod([1 - u[i] for i in s.selected])) if s.selected else np.nan,
                             max_admissible_utility=1 - float(np.prod([1 - u[i] for i in range(len(u)) if adm[i]])))
                  for name, (J, rt) in res.items():
                      row[f"{name}_objective"] = J
                      row[f"{name}_runtime_s"] = rt
                  if s.status == "INFEASIBLE":
                      # diagnostic: prohibited items of highest utility whose release restores feasibility
                      prod = np.prod([1 - u[i] for i in range(len(u)) if adm[i]])
                      rel = []
                      for i in sorted((i for i in range(len(u)) if not adm[i]), key=lambda i: -u[i]):
                          if 1 - prod >= R["tau"]:
                              break
                          prod *= 1 - u[i]
                          rel.append(keys[i][1])
                      ok = 1 - prod >= R["tau"]
                      row["release_needed"] = len(rel) if ok else np.nan
                      row["release_items"] = "; ".join(rel) if ok else ""
                  rows.append(row)
    df = pd.DataFrame(rows)
    if DRY:
        print(df.groupby("role")[["n_admissible", "max_admissible_utility", "n_positive", "u_max"]].describe().T.round(3))
        return
    df.to_csv(DATA / "results.csv", index=False)
    pd.DataFrame(item_rows).to_csv(DATA / "items.csv", index=False)
    print(df.groupby("role").agg(n=("status", "size"), infeasible=("status", lambda s: (s == "INFEASIBLE").mean()),
                                 adm=("n_admissible", "median"), sel=("n_selected", "median"),
                                 rt=("runtime_s", "median")))
    print(df.n_items.describe())


if __name__ == "__main__":
    main()
