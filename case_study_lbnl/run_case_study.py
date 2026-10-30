"""
Case study on measured building data: MST on candidate pools built from the LBNL
Building 59 data set (Luo et al., Scientific Data 9, 156, 2022; Dryad doi:10.7941/D1N33Q,
CC BY 4.0), an office building in Berkeley with end-use electricity, four rooftop units,
a heat pump, zone temperatures and setpoints, weather and camera occupant counts.
Study period 2018-06-01 to 2019-02-28 (case_study_lbnl/prepare_data.py).

Scenario. Each subsystem is reported by a remote agent. When the daily energy of an
end use (HVAC north or south, plug loads north or south, lighting south) deviates
strongly from its baseline on a working day, the assistant agent decides what to disclose to three
stakeholders who oversee the alert: the facility manager, an external energy auditor
and an occupant. Each candidate item is the daily deviation of one point.

Item values.
  u_i  measured: the item's share of the part of the incident day's deviation that a
       ridge fit of the target's daily deviation on all items (fitted without the
       incident day) explains, from a linear attribution (linear SHAP); the utilities of
       an incident sum to the explained share (attribution_utilities);
  p_i  measured: how much the item reveals about the presence of people, the R^2 of the
       camera occupant count of a floor on the item's hourly series after removing the
       hour-of-week means of both (the building schedule is public knowledge), maximum
       over the two monitored floors, capped at 0.95;
  c_i  modeled: interpretation cost by kind of point (the only modeled item value),
       tested against a uniform cost;
  admissibility, threshold and weights by role.

Writes data/case_study_lbnl/results.csv, items.csv, fit_quality.csv and cost_sensitivity.csv
Usage: python case_study_lbnl/run_case_study.py
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

DATA = ROOT / "data" / "case_study_lbnl"
TARGETS = {"hvac_N": "HVAC north", "hvac_S": "HVAC south", "mels_N": "plug loads north",
           "mels_S": "plug loads south", "lig_S": "lighting south"}
ENERGY = list(TARGETS)
INCIDENTS_PER_TARGET = 12
OCC = ["occ_third_south", "occ_fourth_south"]
P_CAP = 0.95

# item kind of every point: (kind, readable name, daily aggregation)
POINTS = {**{e: ("energy", TARGETS[e], "sum") for e in ENERGY},
          **{f"rtu_00{k}_fltrd_sa_flow_tn": ("hvac", f"rooftop unit {k} supply airflow", "mean") for k in range(1, 5)},
          **{f"rtu_00{k}_sa_temp": ("hvac", f"rooftop unit {k} supply air temperature", "mean") for k in range(1, 5)},
          **{f"rtu_00{k}_oadmpr_pct": ("hvac", f"rooftop unit {k} outdoor-air damper", "mean") for k in range(1, 5)},
          "hp_hws_temp": ("hvac", "heat-pump hot-water temperature", "mean"),
          "indoor_temp_mean": ("comfort", "indoor temperature", "mean"),
          "cooling_setpoint_mean": ("comfort", "zone cooling setpoints", "mean"),
          "air_temp_set_1": ("weather", "outdoor temperature", "mean"),
          "relative_humidity_set_1": ("weather", "outdoor humidity", "mean"),
          "solar_radiation_set_1": ("weather", "solar radiation", "mean"),
          "occ_third_south": ("occupancy", "occupant count, third floor", "mean"),
          "occ_fourth_south": ("occupancy", "occupant count, fourth floor", "mean"),
          "weekend": ("calendar", "weekend or holiday", "mean")}

# modeled interpretation cost by kind (a scalar daily summary of one point)
KIND_C = {"energy": 0.20, "hvac": 0.30, "comfort": 0.15, "weather": 0.10, "occupancy": 0.15, "calendar": 0.05}
ROLES = {
    # admissible kinds, threshold, weights (P, C, K), cost multiplier for technical HVAC points
    "facility_manager": dict(kinds=set(KIND_C), tau=0.30, w=(0.2, 0.4, 0.4), hvac_mult=1.0),
    "energy_auditor": dict(kinds=set(KIND_C) - {"occupancy"}, tau=0.25, w=(1 / 3, 1 / 3, 1 / 3), hvac_mult=1.0),
    "occupant": dict(kinds={"energy", "comfort", "weather", "calendar"}, tau=0.10, w=(0.2, 0.6, 0.2),
                     hvac_mult=2.0),
}
CASE_REF_ITEMS = 10.0           # reference scales: a typical disclosure of 10 items of cost 0.2
CASE_REF_COST = 10.0 * 0.20
TAU_GRID = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
US_HOLIDAYS = ["2018-07-04", "2018-09-03", "2018-11-12", "2018-11-22", "2018-11-23", "2018-12-24", "2018-12-25",
               "2018-12-31", "2019-01-01", "2019-01-21", "2019-02-18"]   # University of California holidays


def deviation(s, relative):
    """Deviation from the median of the same weekday in the two weeks before and after."""
    out = pd.Series(np.nan, index=s.index)
    for wd in range(7):
        m = s.index.dayofweek == wd
        x = s[m]
        nb = pd.concat([x.shift(k) for k in (-2, -1, 1, 2)], axis=1)
        base = nb.median(axis=1, skipna=True).where(nb.notna().sum(axis=1) >= 2)
        out[m] = (x - base) / base.abs() if relative else x - base
    return out.replace([np.inf, -np.inf], np.nan)


def fit(y, x):
    ok = y.notna() & x.notna()
    if ok.sum() < 60 or x[ok].std() == 0:
        return 0.0, 0.0
    b, _ = np.polyfit(x[ok], y[ok], 1)
    c = np.corrcoef(y[ok], x[ok])[0, 1]
    return (float(c * c) if np.isfinite(c) else 0.0), float(b)


def day_utility(y, x, day):
    R2, b = fit(y, x)
    if not (np.isfinite(y.get(day, np.nan)) and np.isfinite(x.get(day, np.nan))) or y[day] == 0:
        return 0.0
    return R2 * float(np.clip(b * x[day] / y[day], 0.0, 1.0))


def attribution_utilities(y, X, day, alpha=1.0):
    """u_i for one incident: linear attribution of the day's deviation (linear SHAP,
    Lundberg and Lee 2017). A ridge regression of the target's daily deviation on all item
    deviations (standardized, missing deviations set to zero) is fitted on every other day
    of the study period; item i's contribution on the incident day is
    phi_i = b_i (x_i(day) - mean x_i). The share of the day's deviation the fit explains,
    e = clip(sum_i phi_i / (y(day) - mean y), 0, 1), is divided among the items that push
    in the direction of the deviation in proportion to their contribution, so that
    u_i = e phi_i^+ / sum_j phi_j^+ and sum_i u_i = e."""
    ok = y.notna()
    Xf = X.fillna(0.0)
    train = ok & (y.index != day)
    mu, sd = Xf[train].mean(), Xf[train].std().replace(0, 1.0)
    Z = ((Xf - mu) / sd).to_numpy()
    Zt, yt = Z[train.to_numpy()], y[train].to_numpy()
    ym = yt.mean()
    A = Zt.T @ Zt + alpha * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Zt.T @ (yt - ym))
    zd = Z[X.index.get_loc(day)]
    target = y[day] - ym
    phi = b * zd
    if target == 0:
        return np.zeros(len(b))
    phi = phi * np.sign(target)                  # positive = pushes towards the observed deviation
    pos = np.clip(phi, 0.0, None)
    explained = float(np.clip(phi.sum() / abs(target), 0.0, 1.0))
    return pos / pos.sum() * explained if pos.sum() > 0 else np.zeros(len(b))


def cv_r2(y, X, alpha=1.0, folds=10):
    """Ten-fold cross-validated R^2 of the ridge fit used for the attribution."""
    ok = y.notna().to_numpy()
    Z = X.fillna(0.0).to_numpy()[ok]
    yy = y.to_numpy()[ok]
    idx = np.arange(len(yy)) % folds
    pred = np.empty_like(yy)
    for f in range(folds):
        tr, te = idx != f, idx == f
        mu, sd = Z[tr].mean(0), Z[tr].std(0)
        sd[sd == 0] = 1.0
        Zt = (Z[tr] - mu) / sd
        b = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(Z.shape[1]), Zt.T @ (yy[tr] - yy[tr].mean()))
        pred[te] = yy[tr].mean() + ((Z[te] - mu) / sd) @ b
    return 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def privacy(hourly):
    """p_i: R^2 of a floor's occupant count on the item, both net of hour-of-week means."""
    how = hourly.index.dayofweek * 24 + hourly.index.hour
    resid = hourly - hourly.groupby(how).transform("mean")
    p = {}
    for col in POINTS:
        vals = []
        for o in OCC:
            ok = resid[col].notna() & resid[o].notna()
            if ok.sum() < 500 or resid.loc[ok, col].std() == 0:
                vals.append(0.0)
                continue
            r = np.corrcoef(resid.loc[ok, col], resid.loc[ok, o])[0, 1]
            vals.append(float(r * r) if np.isfinite(r) else 0.0)
        p[col] = min(P_CAP, max(vals))
    return p


def solve(u, p, c, adm, tau, w):
    best, best_J, res = None, np.inf, {}
    for name, f in h.METHODS.items():
        pl = h.Pool(u, p, c, adm, tau, w, cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS)
        t0 = time.perf_counter()
        x = f(pl)
        res[name] = (pl.J(x) if x is not None else np.nan, time.perf_counter() - t0)
        if x is not None and pl.J(x) < best_J:
            best, best_J = pl.to_indices(x), pl.J(x)
    s = sm.solve_mst_scip(list(u), list(p), list(c), adm, tau, w, start=best,
                           cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS)
    return s, res


def main():
    hourly = pd.read_csv(DATA / "bldg59_hourly.csv.gz", parse_dates=["date"]).set_index("date")
    day_index = hourly.resample("D").mean().index
    hol = pd.DatetimeIndex(US_HOLIDAYS)
    weekend_d = pd.Series(((day_index.dayofweek >= 5) | day_index.isin(hol)).astype(float), index=day_index)
    hourly["weekend"] = ((hourly.index.dayofweek >= 5) | hourly.index.normalize().isin(hol)).astype(float)

    daily = {}
    for col, (kind, _, agg) in POINTS.items():
        if col == "weekend":
            daily[col] = weekend_d
            continue
        g = hourly[col].resample("D")
        daily[col] = g.sum(min_count=18) if agg == "sum" else g.mean()
    dev = {}
    for col, (kind, _, _) in POINTS.items():
        if kind == "weather":
            dev[col] = daily[col] - daily[col].rolling(15, center=True, min_periods=7).mean()
        elif kind == "calendar":
            dev[col] = daily[col]
        else:
            dev[col] = deviation(daily[col], relative=False)
    target_dev = {e: deviation(daily[e], relative=False) for e in ENERGY}   # kWh above or below baseline
    P = privacy(hourly)

    rows, item_rows, sens, fit_quality = [], [], [], []
    for target in ENERGY:
        y = target_dev[target]
        z = (y - y.median()) / (1.4826 * (y - y.median()).abs().median())
        days = []
        for d in z.abs().dropna().sort_values(ascending=False).index:
            if weekend_d[d] == 1:                       # alerts on working days only
                continue
            if all(abs((d - e).days) >= 7 for e in days):
                days.append(d)
            if len(days) == INCIDENTS_PER_TARGET:
                break
        keys = [k for k in POINTS if k != target]
        kinds = [POINTS[k][0] for k in keys]
        p = np.array([P[k] for k in keys])
        for k in keys:
            item_rows.append(dict(target=TARGETS[target], point=k, item=POINTS[k][1], kind=POINTS[k][0],
                                  r2=fit(y, dev[k])[0], p=P[k], c=KIND_C[POINTS[k][0]]))
        X = pd.DataFrame({k: dev[k] for k in keys})
        fit_quality.append(dict(target=TARGETS[target], cv_r2=cv_r2(y, X)))
        for day in days:
            u = attribution_utilities(y, X, day)
            for role, R0 in ROLES.items():
                adm = [kd in R0["kinds"] for kd in kinds]
                c = np.array([KIND_C[kd] * (R0["hvac_mult"] if kd == "hvac" else 1.0) for kd in kinds])
                for tau in TAU_GRID:
                    s, res = solve(u, p, c, adm, tau, R0["w"])
                    row = dict(target=TARGETS[target], day=day.date(), z=float(z[day]), role=role, tau=tau,
                               nominal=bool(abs(tau - R0["tau"]) < 1e-9), n_items=len(keys),
                               n_admissible=int(sum(adm)), status=s.status, objective=s.objective,
                               runtime_s=s.runtime, n_selected=len(s.selected),
                               selected="; ".join(POINTS[keys[i]][1] for i in s.selected),
                               selected_kinds="|".join(sorted({kinds[i] for i in s.selected})),
                               utility=1 - float(np.prod([1 - u[i] for i in s.selected])) if s.selected else np.nan,
                               privacy=1 - float(np.prod([1 - p[i] for i in s.selected])) if s.selected else np.nan,
                               max_admissible_utility=1 - float(np.prod([1 - u[i] for i in range(len(u)) if adm[i]])),
                               max_pool_utility=1 - float(np.prod(1 - u)))
                    for name, (J, rt) in res.items():
                        row[f"{name}_objective"] = J
                        row[f"{name}_runtime_s"] = rt
                    if s.status == "INFEASIBLE":
                        prod = np.prod([1 - u[i] for i in range(len(u)) if adm[i]])
                        rel = []
                        for i in sorted((i for i in range(len(u)) if not adm[i]), key=lambda i: -u[i]):
                            if 1 - prod >= tau:
                                break
                            prod *= 1 - u[i]
                            rel.append(POINTS[keys[i]][1])
                        ok = 1 - prod >= tau
                        row["release_needed"] = len(rel) if ok else np.nan
                        row["release_items"] = "; ".join(rel) if ok else ""
                    rows.append(row)
                    # sensitivity of the nominal selection to the modeled cost: uniform cost 0.2
                    if row["nominal"] and s.status != "INFEASIBLE":
                        s2, _ = solve(u, p, np.full(len(keys), 0.2), adm, tau, R0["w"])
                        sens.append(dict(target=TARGETS[target], day=day.date(), role=role,
                                         same_selection=sorted(s.selected) == sorted(s2.selected),
                                         jaccard=len(set(s.selected) & set(s2.selected)) /
                                         max(1, len(set(s.selected) | set(s2.selected))),
                                         n_nominal=len(s.selected), n_uniform=len(s2.selected)))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "results.csv", index=False)
    pd.DataFrame(item_rows).to_csv(DATA / "items.csv", index=False)
    pd.DataFrame(fit_quality).to_csv(DATA / "fit_quality.csv", index=False)
    print(pd.DataFrame(fit_quality).round(3).to_string(index=False))
    sd = pd.DataFrame(sens)
    sd.to_csv(DATA / "cost_sensitivity.csv", index=False)
    nom = df[df.nominal]
    print(nom.groupby("role").agg(n=("status", "size"), infeasible=("status", lambda s: (s == "INFEASIBLE").mean()),
                                  adm=("n_admissible", "median"), sel=("n_selected", "median"),
                                  rt=("runtime_s", "median"), rtmax=("runtime_s", "max")))
    print(sd.groupby("role")[["same_selection", "jaccard"]].mean())
    print(pd.Series(P).round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
