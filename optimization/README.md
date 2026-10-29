# Solvers and optimization experiments

Additions made in response to the optimization review of the article (exact
mixed-integer model, metaheuristics, Pareto view).

| File | Content |
|---|---|
| `scip_model.py` | MST as a mixed-integer nonlinear program: exact linear covering constraint for sufficiency (logarithms, weights capped at the requirement), privacy term 1 − exp(−y) with y linear, solved with SCIP; optional starting solution (hybrid method) |
| `coverage_model.py` | The same model with the interaction-aware coverage utility, linearized exactly with auxiliary variables |
| `coverage_pools.py` | Loads the coverage instance generator from notebook 02 without running the notebook |
| `ga_pymoo.py` | Genetic algorithm (objective J, sufficiency as a constraint) and NSGA-II on (P, C, K) |
| `run_scip_validation.py` | SCIP versus enumeration on all 24,000 product-form conditions |
| `run_scip_coverage.py` | SCIP versus enumeration on 4,800 coverage-utility conditions |
| `run_scaling.py` | SCIP, greedy and genetic algorithm on pools of 16 to 512 items (864 conditions) |
| `run_scaling_warmstart.py` | SCIP started from the greedy disclosure on the same conditions |
| `make_scaling_figure.py` | Table and figure of the scaling results |
| `run_ga_benchmark.py` | Genetic algorithm versus enumeration on 1,200 product-form conditions |
| `run_nsga2.py` | NSGA-II versus the exact Pareto front on 240 instance and threshold pairs |

Install the extra packages with `pip install pyscipopt pymoo`. Results are written to
`data/optimization/`, `tables/` and `figures/`.
