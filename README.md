# Symbolic Regression of Confidence Intervals for Conformal Prediction 

Code and results for the paper "Symbolic Regression of Confidence Intervals for Conformal Prediction", submitted to the EA 2024 conference, https://ea2024.inria.fr/

Usage: install the Python packages found inside `src/requirements.txt`, then from inside `src/` run `python run_experiments.py` to compare the methodologies on data sets in OpenML-CTR23.

`src/analysis/` contains post-hoc analysis scripts, also run from inside `src/`:
- `python -m analysis.check_pareto_optimality <results_folder>` checks Pareto optimality between methods for a given results folder produced by `run_experiments.py`.
- `python -m analysis.statistics_openml_ctr23` computes descriptive statistics (missing data, categorical features, baseline R2/MSE) for the OpenML-CTR23 suite.