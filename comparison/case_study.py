"""
Worked case study on the smart-building pools of the SARL execution.

For each role in one scenario and threshold, shows what MST and each adapted
baseline would disclose. Pools, attributes and admissibility flags are those
of sarl/src/main/java/org/mst/sarl/core/MSTService.java; weights are balanced.

Writes tables/comparison_case_study.csv
Usage: python comparison/case_study.py [--scenario HVAC_FAULT] [--tau 0.9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))
import methods as m  # noqa: E402

# (id, utility, privacy, interpretation cost, admissible), copied from MSTService.java
POOLS = {
    ("HVAC_FAULT", "OCCUPANT"): [("zone_temperature", .70, .08, .20, True), ("comfort_deviation", .68, .05, .18, True), ("maintenance_staff_identity", .40, .45, .30, False)],
    ("HVAC_FAULT", "FACILITY_MANAGER"): [("hvac_fault_code", .82, .15, .30, True), ("airflow_anomaly", .58, .10, .24, True), ("zone_occupancy_count", .45, .25, .35, True)],
    ("HVAC_FAULT", "SECURITY"): [("safety_interlock_status", .76, .08, .20, True), ("affected_access_zone", .55, .16, .22, True), ("occupant_identity_trace", .42, .60, .40, False)],
    ("COMFORT_DEVIATION", "OCCUPANT"): [("zone_temperature", .72, .08, .18, True), ("comfort_target_difference", .64, .06, .18, True), ("neighbour_zone_occupancy", .42, .48, .34, False)],
    ("COMFORT_DEVIATION", "FACILITY_MANAGER"): [("temperature_trend", .76, .10, .24, True), ("hvac_operating_mode", .66, .12, .26, True), ("aggregated_occupancy_level", .48, .22, .30, True)],
    ("COMFORT_DEVIATION", "SECURITY"): [("safety_threshold_status", .78, .07, .18, True), ("affected_zone", .44, .12, .18, True), ("occupant_identity", .35, .58, .36, False)],
    ("SECURITY_ALARM", "OCCUPANT"): [("alarm_status", .74, .07, .18, True), ("affected_access_zone", .58, .12, .20, True), ("camera_identity_trace", .46, .62, .40, False)],
    ("SECURITY_ALARM", "FACILITY_MANAGER"): [("alarm_status", .72, .08, .18, True), ("affected_building_zone", .60, .14, .22, True), ("device_health_summary", .52, .12, .24, True)],
    ("SECURITY_ALARM", "SECURITY"): [("alarm_status", .88, .10, .18, True), ("affected_access_zone", .62, .16, .20, True), ("camera_event_summary", .54, .24, .26, True)],
}
BALANCED = (1 / 3, 1 / 3, 1 / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="HVAC_FAULT")
    ap.add_argument("--tau", type=float, default=0.90)
    args = ap.parse_args()
    rows = []
    for (scenario, role), items in POOLS.items():
        inst = m.Instance([i[0] for i in items], [i[1] for i in items], [i[2] for i in items],
                          [i[3] for i in items], [i[4] for i in items])
        for tau in (0.70, 0.80, 0.90, 0.95):
            for name, (fn, group, label) in m.METHODS.items():
                sel = fn(inst, tau, BALANCED)
                if sel == m.INFEASIBLE:
                    out = dict(outcome="INFEASIBLE", items="", utility=None, sufficient=None, prohibited=0)
                else:
                    out = dict(outcome="DISCLOSED", items=", ".join(inst.id_tuple(sel)),
                               utility=round(float(inst.U[sel]), 4), sufficient=bool(inst.U[sel] >= tau),
                               prohibited=int(inst.n_prohibited[sel]))
                rows.append(dict(scenario=scenario, role=role, tau=tau, method=name, label=label, **out))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "tables" / "comparison_case_study.csv", index=False)
    show = df[(df.scenario == args.scenario) & (df.tau == args.tau)]
    print(show[["role", "label", "outcome", "items", "utility", "prohibited"]].to_string(index=False))


if __name__ == "__main__":
    main()
