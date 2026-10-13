# Data

All files are frozen outputs of the analysis described in the paper. Column
names follow the implementation: `cognitive` refers to modeled interpretation
cost, `privacy` to relative privacy exposure, and `disclosure_cost` to the
objective J(S).

## raw/

| File | Rows | Description |
|---|---|---|
| `exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv` | 24,000 | Authoritative primary benchmark: 5 pool sizes x 100 repetitions x 3 regimes x 4 thresholds x 4 weight profiles. |
| `exact_vs_greedy_main_benchmark_v2_marginal_preserving_analysis_ready.csv` | 24,000 | Same benchmark with derived gap, recovery, and mechanism columns. |
| `corrected_v2_greedy_failure_mechanism_rows.csv` | | Per-condition failure-mechanism classification. |
| `corrected_v2_algorithm_comparison_long.csv` | 96,000 | Dynamic greedy, static ratio, utility-first, and lowest-cost-first heuristics. |
| `corrected_v2_runtime_instance_level.csv`, `corrected_v2_runtime_matched_group_level.csv` | | Exact and greedy runtimes for M = 8 to 16. |
| `greedy_scalability_v2.csv` | 24,000 | Dynamic Greedy-MST for M = 16 to 256. |
| `greedy_scalability_v2_matched_group_runtime.csv` | | Matched-group runtimes used for Table 6. |
| `corrected_generator_*`, `synthetic_regime_validation.csv` | | Generator settings and association validation. |
| `progressive_mst_*`, `normalized_progressive_mst_sequence.csv`, `normalized_mst_property_validation.csv` | | Progressive-MST validation sequence and property checks. |
| `exact_vs_greedy_counterexample.csv` | | The greedy counterexample (Example 1). |
| `mst_core_automated_validation.csv` | | Automated checks of the core module. |

## sarl/

`sarl_mst_multiscenario_sweep.csv` contains the 36 scenario, role, and
threshold conditions of the SARL execution (26 feasible, 10 infeasible). The
`results_sarl_*` files are the independent source-to-result reconstruction
and audits.

## audits/

Row, column, denominator, and integrity audits accepted before results were
reported (`final_computational_analysis_audit_v1.csv` summarizes them).

Bootstrap replicate files (about 70 MB) are not included here; the intervals
derived from them are in `tables/`. They can be provided on request.
