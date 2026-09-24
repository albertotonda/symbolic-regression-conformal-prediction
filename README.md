# Symbolic Regression of Confidence Intervals for Conformal Prediction 

Code and results for the paper "Symbolic Regression of Confidence Intervals for Conformal Prediction", submitted to the EA 2024 conference, https://ea2024.inria.fr/

Usage: install the Python packages found inside `requirements.txt`, then `uv run src/run_sigma_sr.py` or `uv run src/run_interval_sr.py` to compare the methodologies on data sets in OpenML-CTR23.

`src/analysis/` contains post-hoc analysis scripts, also run from inside `src/`:
- `python -m analysis.check_pareto_optimality <results_folder>` checks Pareto optimality between methods for a given results folder produced by `run_experiments.py`.
- `python analysis/backfill_testing_features.py <results_folder>` rebuilds `testing_features.csv` (normalized test features, used by the dashboard's worst-slab coverage) for runs made before `run_sigma_sr.py` saved it.
- `python -m analysis.statistics_openml_ctr23` computes descriptive statistics (missing data, categorical features, baseline R2/MSE) for the OpenML-CTR23 suite.