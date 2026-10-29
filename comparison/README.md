# Comparison with the closest approaches, ablations and adapted baselines

This folder compares MST with (i) the four closest approaches in the paper's
positioning table, (ii) ablations, each removing one component of the model, and
(iii) adapted baselines for the remaining families of related work. Every method
runs on the same candidate pools as the paper's primary benchmark: 1,500 pools, 4 thresholds and 4 weight profiles,
giving 24,000 conditions.

## Candidate pools

`benchmark_pools.py` regenerates the primary-benchmark pools with the same
generator (`marginal_preserving_v2`) and the same seeds. `validate_against_paper.py`
confirms that, with every item admissible, Exact-MST here returns the stored
subset in all 24,000 conditions (objective difference at most 2.2e-16).

The comparison adds one thing: in each matched pool, ceil(0.25 n) item positions
are marked as prohibited by the disclosure policy, chosen at random with a seed
derived from the matched seed. In 2,124 conditions, no admissible subset reaches
the threshold, so the correct outcome there is to report infeasibility.

## Methods

**The four closest approaches** in the paper's positioning table, each reduced to its
selection rule and applied to the same candidate items:

| Key | Approach | Selection rule applied here |
|---|---|---|
| C1 | Model reconciliation (Chakraborti et al. 2017) | smallest sufficient subset (minimally complete explanation), no policy, no privacy term |
| C2 | Goal obfuscation and legibility (Kulkarni et al. 2020) | least interpretation cost reaching the threshold with privacy exposure within a budget of 0.35; failure reported otherwise; no item-level policy |
| C3 | Role-aware HAExA (Kumar et al. 2026) | every item the role is permitted to see |
| C4 | Parsimonious HAExA (Mualla et al. 2022) | the three items of highest utility |

**Ablations**, each removing one component of MST:

| Key | Component removed |
|---|---|
| A1 | hard admissibility, replaced by a penalty of 1 per prohibited item |
| A2 | sufficiency threshold, replaced by a utility reward, J(S) minus U(S) |
| A3 | disclosure-cost objective, replaced by the number of items |
| A4 | infeasibility reporting: every admissible item is disclosed instead |
| A5 | exact search, replaced by Dynamic Greedy-MST |

**Further baselines** for the remaining families: full disclosure (B1), greedy
submodular cover by utility per interpretation cost (B5), and highest utility within
a normalized cost budget of 0.25 (B6).

## Requirements measured

- **R1:** the disclosure is sufficient whenever a sufficient admissible disclosure exists.
- **R2:** no prohibited item is disclosed.
- **R3:** infeasibility is reported when no admissible disclosure is sufficient.
- **R4:** a valid disclosure is not weakly dominated by MST on privacy exposure,
  interpretation cost and size together. This criterion does not use the objective weights.

A further measure, the **repair cost**, takes each output, removes prohibited items,
adds admissible items by the Dynamic Greedy-MST rule until it is sufficient, and
reports the extra objective relative to MST.

## Run

```bash
python comparison/validate_against_paper.py   # about 4 min
python comparison/run_comparison.py           # about 4 min with 2 workers
python comparison/make_figures.py
python comparison/case_study.py               # smart-building pools of the SARL run
python comparison/sensitivity.py              # free parameters of A1, C2, C4 and B6, about 8 min
```

## Outputs

| File | Content |
|---|---|
| `data/comparison/comparison_rows.csv.gz` | 312,000 rows, one per method and condition |
| `tables/comparison_requirements_summary.csv` | R1 to R4, items, objective excess, repair cost per method |
| `tables/comparison_by_threshold.csv` | the same measures by threshold |
| `tables/comparison_case_study.csv` | disclosures of every method on the nine SARL pools |
| `tables/comparison_parameter_sensitivity.csv` | baseline results under other parameter values |
| `figures/fig_comparison_closest.pdf` | MST and the four closest approaches on R1 to R4 |
| `figures/fig_comparison_ablations.pdf` | MST and the five ablations on R1 to R4 |
| `figures/fig_comparison_requirements.pdf` | all thirteen methods on R1 to R4 |
| `figures/fig_comparison_cost_threshold.pdf` | repair cost of every method, and MST and the closest approaches by threshold |

The baselines are adaptations of each family's main idea to disclosure
selection, not reimplementations of the cited systems, which solve related but
different problems.

Results in this folder were produced with Python 3, numpy 2.5.3, pandas 3.0.5 and matplotlib 3.11.2; the code also runs with the versions in `requirements.txt`.
