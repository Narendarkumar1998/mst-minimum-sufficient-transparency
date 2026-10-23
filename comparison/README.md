# Comparison with ablations and adapted baselines

This folder compares MST with (i) ablations, each removing one component of the
model, and (ii) adapted baselines, one for each family of related work in the
paper's positioning table. Every method runs on the same candidate pools as the
paper's primary benchmark: 1,500 pools, 4 thresholds and 4 weight profiles,
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

| Key | Method | What it represents |
|---|---|---|
| MST | Exact-MST | the full model |
| A1 | hard admissibility replaced by a penalty of 1 per prohibited item | soft privacy penalties |
| A2 | sufficiency threshold replaced by a utility reward, J(S) minus U(S) | utility and cost trade-offs |
| A3 | objective replaced by the number of items | minimal sufficient subsets inside the policy |
| A4 | when infeasible, disclose every admissible item | systems that always answer |
| A5 | Dynamic Greedy-MST | heuristic search |
| B1 | full disclosure | maximal transparency |
| B2 | every admissible item | role-aware policy filtering |
| B3 | three items of highest utility | parsimonious explanation |
| B4 | smallest sufficient subset, ignoring the policy and cost | sufficient reasons and abductive explanations |
| B5 | greedy cover by utility per interpretation cost | submodular set cover |
| B6 | highest utility within a normalized cost budget of 0.25 | budgeted coverage |

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
python comparison/sensitivity.py              # free parameters of A1, B3 and B6, about 6 min
```

## Outputs

| File | Content |
|---|---|
| `data/comparison/comparison_rows.csv.gz` | 288,000 rows, one per method and condition |
| `tables/comparison_requirements_summary.csv` | R1 to R4, items, objective excess, repair cost per method |
| `tables/comparison_by_threshold.csv` | the same measures by threshold |
| `tables/comparison_case_study.csv` | disclosures of every method on the nine SARL pools |
| `tables/comparison_parameter_sensitivity.csv` | baseline results under other parameter values |
| `figures/fig_comparison_requirements.pdf` | requirement satisfaction per method |
| `figures/fig_comparison_cost_threshold.pdf` | repair cost and requirement satisfaction by threshold |

The baselines are adaptations of each family's main idea to disclosure
selection, not reimplementations of the cited systems, which solve related but
different problems.

Results in this folder were produced with Python 3, numpy 2.5.3, pandas 3.0.5 and matplotlib 3.11.2; the code also runs with the versions in `requirements.txt`.
