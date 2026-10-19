# Minimum Sufficient Transparency (MST)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23344151.svg)](https://doi.org/10.5281/zenodo.23344151)

Code and data for the article

> N. Kumar, Y. Mualla, V. Hilaire, S. Galland.
> **Enough, but No More: Minimum Sufficient Transparency for Human Oversight of Multi-Agent Systems.**
> Submitted to *Autonomous Agents and Multi-Agent Systems* (2026).

MST is the selection capability of the assistant agent in a human-supervised
multi-agent system. For each stakeholder role it selects the subset of
admissible information items that reaches a sufficiency threshold
$U(S) \ge \tau$ while minimizing a weighted combination of relative privacy
exposure, normalized interpretation cost, and size. When no admissible subset
is sufficient, it reports `INFEASIBLE` instead of crossing the disclosure policy.

## Repository structure

| Folder | Content |
|---|---|
| `mst/` | Authoritative Python implementation (`mst_core.py`): Exact-MST, Dynamic Greedy-MST, Progressive-MST, set metrics, and optimality gaps. Standard library only. |
| `notebooks/` | Jupyter notebooks that generate the benchmarks, analyses, and figures (see `notebooks/README.md` for the run order and setup). |
| `sarl/` | SARL and Java multi-agent implementation: role agents request disclosures from an Exact-MST service (see `sarl/README.md`, including the reconstruction notice). |
| `data/raw/` | Frozen result datasets behind the paper (see `data/README.md`). |
| `data/sarl/` | The 36-condition SARL execution output and its reconstruction audits. |
| `data/audits/` | Integrity and consistency audits accepted before reporting results. |
| `tables/` | Summary tables, bootstrap intervals, and contrasts reported in the paper. |
| `figures/` | Figures 1 to 4 of the paper (PDF and PNG). |
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

## Reproducing the paper

| Paper result | File |
|---|---|
| Primary benchmark, 24,000 exact and greedy comparisons (Tables 4 to 6, Figs. 1 to 3) | `data/raw/exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv` |
| Failure mechanisms and large-gap attribution (Table 5, Fig. 2) | `data/raw/corrected_v2_greedy_failure_mechanism_rows.csv`, `tables/corrected_v2_greedy_failure_mechanism_summary.csv` |
| Threshold sensitivity and bootstrap contrasts (Table 6, Fig. 3) | `tables/corrected_v2_threshold_*`, `tables/corrected_v2_paired_threshold_*` |
| Dynamic versus static ranking, 96,000 rows | `data/raw/corrected_v2_algorithm_comparison_long.csv`, `tables/corrected_v2_dynamic_*` |
| Scalability to 256 candidates (Table 7, Fig. 4) | `data/raw/greedy_scalability_v2.csv`, `data/raw/greedy_scalability_v2_matched_group_runtime.csv` |
| Progressive-MST validation | `data/raw/progressive_mst_validation_sequence.csv`, `data/raw/normalized_progressive_mst_sequence.csv` |
| Synthetic generator validation | `data/raw/corrected_generator_validation_summary.csv`, `data/raw/synthetic_regime_validation.csv` |
| SARL execution, 36 conditions (Table 8) | `data/sarl/sarl_mst_multiscenario_sweep.csv` |

All synthetic candidate pools were generated with random seed `2026`.
Runtimes were measured on Windows 11, CPython 3.12.2, Intel 4 cores / 8
threads, 15.65 GiB RAM (`docs/software_hardware_environment.csv`) and are
not hardware-independent guarantees.

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
