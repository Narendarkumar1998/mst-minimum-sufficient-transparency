# Minimum Sufficient Transparency (MST)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23344151.svg)](https://doi.org/10.5281/zenodo.23344151)

Code and data for the article

> N. Kumar, Y. Mualla, V. Hilaire, S. Galland.
> **Enough, but No More: Minimum Sufficient Transparency for Human Oversight of Multi-Agent Systems.**
> Submitted to *Complex & Intelligent Systems* (2026).

MST is the selection capability of the assistant agent in a human-supervised
multi-agent system. For each stakeholder role it selects the subset of
admissible information items that reaches a sufficiency threshold
$U(S) \ge \tau$ while minimizing a weighted combination of relative privacy
exposure, normalized interpretation cost, and size. When no admissible subset
is sufficient, it reports `INFEASIBLE` instead of crossing the disclosure policy.

## Repository structure

| Folder | Content |
|---|---|
| `mst/` | Reference Python implementation (`mst_core.py`): exact enumeration, Dynamic Greedy-MST, set metrics, and optimality gaps. Standard library only. (Progressive-MST is also included; it is not part of the current article.) |
| `optimization/` | Mixed-integer model solved with SCIP, genetic algorithm and NSGA-II with pymoo, the coverage-utility model, and the scripts for the solver validation, scaling, and Pareto experiments (see `optimization/README.md`). |
| `comparison/` | Comparison of MST with the four closest approaches, ablations, and further baselines, plus a smart-building case study (see `comparison/README.md`). |
| `notebooks/` | Jupyter notebooks that generate the benchmarks and analyses (see `notebooks/README.md`), and `regenerate_coverage_benchmark.py`, which reruns the coverage-utility benchmark. |
| `sarl/` | SARL and Java multi-agent implementation: role agents request disclosures from an exact MST service (see `sarl/README.md`, including the reconstruction notice). |
| `data/raw/` | Product-form benchmark (24,000 conditions) and related result files (see `data/README.md`). |
| `data/coverage/` | Coverage-utility benchmark (24,000 conditions) and its audits. |
| `data/optimization/` | Solver validation, scaling, genetic-algorithm, and Pareto results. |
| `data/comparison/` | One row per method and condition for the comparison. |
| `data/benchmark_v3/` | **Results of the revised article** (cost and size normalized by fixed reference scales, `optimization/scales.py`): product and coverage benchmarks, SCIP validation, scaling (20 replicates), metaheuristics, convergence and GA sensitivity, chain MILP, robustness, admissibility levels, NSGA-II. |
| `case_study_lbnl/`, `data/case_study_lbnl/` | Case study on the public LBNL Building 59 data set (Luo et al., *Scientific Data* 9, 156, 2022; Dryad doi:10.7941/D1N33Q, CC BY 4.0): preprocessing of the raw files to hourly data, and the case study itself. |
| `data/sarl/` | The 36-condition SARL execution output and its reconstruction audits. |
| `data/audits/` | Integrity and consistency audits. |
| `tables/` | Summary tables, bootstrap intervals, and contrasts. |
| `figures/` | Paper figures, including the MST flowchart and its draw.io source. |
| `docs/` | Analysis freeze notes and the software and hardware environment. |

## Quick start

```bash
git clone https://github.com/Narendarkumar1998/mst-minimum-sufficient-transparency.git
cd mst-minimum-sufficient-transparency
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The greedy counterexample from the paper (Example 1):

```python
from mst import InformationItem, exact_mst, greedy_mst

items = [
    InformationItem("a", "item a", utility=0.85, privacy=0.0, cognitive_cost=0.70),
    InformationItem("b", "item b", utility=0.70, privacy=0.0, cognitive_cost=0.60),
    InformationItem("c", "item c", utility=0.70, privacy=0.0, cognitive_cost=0.60),
]
# tau = 0.90, interpretation cost only
print(exact_mst(items, 0.90, lambda_privacy=0, lambda_cognitive=1, lambda_cardinality=0))
# -> selected ('b', 'c'), disclosure_cost ~ 0.6316
print(greedy_mst(items, 0.90, lambda_privacy=0, lambda_cognitive=1, lambda_cardinality=0))
# -> selected ('a', 'b'), disclosure_cost ~ 0.6842
```

## Reproducing the revised article

All results of the revised article are in `data/benchmark_v3/`, `data/comparison/` and `data/case_study_lbnl/`.

```bash
python optimization/run_benchmark_v3.py      # product benchmark, 24,000 conditions
python optimization/run_coverage_v3.py       # coverage benchmark
python optimization/run_scip_validation.py && python optimization/run_scip_coverage.py
python optimization/run_scaling_v2.py --replicates 20
python optimization/run_metaheuristics.py && python optimization/run_convergence.py
python optimization/run_milp_cold.py && python optimization/run_robustness.py
python optimization/run_admissibility.py && python optimization/run_nsga2.py
python comparison/run_comparison.py && python comparison/sensitivity.py
python case_study_lbnl/prepare_data.py "<path to Bldg59_clean data>"   # only to rebuild the hourly file
python case_study_lbnl/run_case_study.py
python optimization/stats_analysis.py
python optimization/make_figures_v5.py && python optimization/make_tables_v5.py
```

## Reproducing the first version

| Paper result | Files | Script |
|---|---|---|
| Product-form benchmark, 24,000 conditions, greedy quality and failure mechanisms (Sections 7.2 to 7.4) | `data/raw/exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv`, `data/raw/corrected_v2_greedy_failure_mechanism_rows.csv`, `tables/corrected_v2_*` | notebook 02 |
| SCIP agrees with enumeration on all 24,000 conditions (Section 7.2) | `data/optimization/scip_vs_exact_benchmark.csv.gz` | `optimization/run_scip_validation.py` |
| Scaling to 512 items: SCIP, greedy start + SCIP, greedy, genetic algorithm (Table 1, Fig. 2) | `data/optimization/scaling_scip_greedy_ga.csv`, `data/optimization/scaling_scip_warmstart.csv`, `tables/scaling_summary.csv`, `figures/fig_scaling.pdf` | `optimization/run_scaling.py`, `run_scaling_warmstart.py`, `make_scaling_figure.py` |
| Genetic algorithm on the benchmark (Section 7.3) | `data/optimization/ga_vs_exact_benchmark.csv.gz` | `optimization/run_ga_benchmark.py` |
| Coverage-utility benchmark (Section 7.5) | `data/coverage/coverage_exact_vs_greedy_main_benchmark_v1_pool_normalized.csv.gz`, `tables/coverage_*` | `notebooks/regenerate_coverage_benchmark.py` |
| SCIP on the coverage utility (Section 7.5) | `data/optimization/scip_vs_exact_coverage.csv.gz` | `optimization/run_scip_coverage.py` |
| NSGA-II Pareto fronts (Section 7.6, Fig. 3) | `data/optimization/nsga2_pareto.csv`, `figures/fig_pareto_example.pdf` | `optimization/run_nsga2.py` |
| Comparison with the closest approaches and ablations (Section 8, Fig. 4, Table 2) | `data/comparison/comparison_rows.csv.gz`, `tables/comparison_*.csv`, `figures/fig_comparison_*.pdf` | `comparison/*.py` |
| SARL execution, 36 conditions (Section 6, Online Resource S3) | `data/sarl/sarl_mst_multiscenario_sweep.csv` | `sarl/` |

The product-form and coverage pools were generated with base seed `2026`.
The original benchmark runtimes were measured on Windows 11, CPython 3.12.2
(`docs/software_hardware_environment.csv`); the solver, scaling, and
genetic-algorithm runs on a 2-core Linux machine with Python 3.13, SCIP 10.0
(PySCIPOpt 6.3), and pymoo 0.6.2. Runtimes are not hardware-independent.

## Scope of the evidence

Utilities, privacy exposures, and interpretation costs are controlled model
parameters, not measurements from human participants. The sufficiency
threshold is a modeled requirement, relative privacy exposure is not a formal
privacy guarantee, and no approximation ratio is claimed for the greedy
procedure. See `docs/computational_analysis_freeze_notes_v1.txt`.

## Citation

See `CITATION.cff`. Please cite the article once published, and this archived release as:

> N. Kumar, Y. Mualla, V. Hilaire, S. Galland. *Minimum Sufficient Transparency (MST): code and data*, v1.0.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.23344151

## License

Code: MIT. Data, tables, and figures: CC BY 4.0. See `LICENSE`.

## Contact

Narendar Kumar, CIAD UR 7533, Université de Technologie de Belfort-Montbéliard
(UTBM), Belfort, France. narendar.kumar@utbm.fr
