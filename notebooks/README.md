# Notebooks

These are the notebooks in which `mst_core.py` was developed and the paper's
results were produced. They are the working record of the analysis: cells
were executed incrementally, and outputs have been cleared for the repository.

| Notebook | Content | Main outputs |
|---|---|---|
| `01_core_implementation_and_validation.ipynb` | Development and automated validation of the core module, the greedy counterexample, and Progressive-MST property checks | `mst_core_automated_validation.csv`, `exact_vs_greedy_counterexample.csv`, `progressive_mst_*` |
| `02_benchmark_generation_and_statistical_analysis.ipynb` | Synthetic generator and its validation, the 24,000-condition primary benchmark, matched-group bootstrap analyses, and the analysis freeze | `exact_vs_greedy_main_benchmark_v2_marginal_preserving.csv`, `corrected_generator_*`, `corrected_v2_*` tables, freeze notes |
| `03_runtime_scalability_and_results_audits.ipynb` | Four-heuristic comparison, failure mechanisms, runtime, large-pool scalability, and the audits behind the results section, including the SARL source-to-result reconstruction | `corrected_v2_algorithm_comparison_long.csv`, `greedy_scalability_v2.csv`, `results_*` audits |
| `04_methodology_audits.ipynb` | Audits of tolerances, weight profiles, recovery definitions, and the software and hardware environment | `methodology_*` files |
| `05_figures.ipynb` | Figures 1 to 4 | `fig1.pdf` to `fig4.pdf` and PNG versions |

## Running the notebooks

The notebooks use the original project folder names. Create a workspace with
that layout and point the notebooks to it:

```bash
python notebooks/setup_workspace.py workspace
export MST_PROJECT_ROOT=$PWD/workspace      # Windows: set MST_PROJECT_ROOT=%CD%\workspace
jupyter lab notebooks/
```

Some cells also refer to an exploratory second utility model ("coverage")
that is not reported in the paper; its outputs are not included in this
repository, and those cells can be skipped.
