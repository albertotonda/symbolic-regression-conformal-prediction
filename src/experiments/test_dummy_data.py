import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from types import SimpleNamespace

from crepes import WrapRegressor, ConformalRegressor
from crepes.extras import DifficultyEstimator, binning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from pysr import PySRRegressor

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data import split_and_normalize_data
from evaluate import evaluate_and_plot_method, log_equations, plot_pareto, setup_results_folder, translations
from cp_methods import fit_difficulty_estimator, compute_normalized_intervals, find_bin_thresholds_with_min_size
from losses import bin_crossfit_loss_julia

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style='darkgrid')
print('All imports OK')

random_seed = 42
confidence = 0.95
rng = np.random.default_rng(random_seed)

results_folder = setup_results_folder("test-sigma-dummy", random_seed)
task_folder = os.path.join(results_folder, "synthetic_noise_test")
os.makedirs(task_folder, exist_ok=True)

dataset = SimpleNamespace(name="synthetic_noise_test")

# 1. Generate a synthetic dataset with a known, closed-form heteroscedastic
# noise law.
#
# x1, x2 drive the residual noise scale via sigma_true; x3, x4 drive the
# mean function f. Keeping these feature sets disjoint means any pattern the
# base regressor's OOB residuals show w.r.t. x1, x2 is purely the designed-in
# noise, not leakage from the regressor's own fit error on f.
#
# sigma_true = exp(x1 + b * x2) is strictly positive by construction (no
# additive floor needed). The bin_crossfit loss below already has the tree
# predict log(sigma) and applies sigma = exp(log_sigma) itself, so the SR
# tree's actual target is log(sigma_true) = x1 + b * x2 -- a plain linear
# combination, discoverable with just "+"/"*", no unary operators required.
# This is the simplest possible positive-control case: if SR still can't
# recover it from raw features, that's strong evidence of the noise-floor/
# smoothing problem rather than of this particular functional form being
# too hard to search for.
#
# Constants tuned (see NOTES.md) so the base regressor lands at a realistic,
# non-degenerate R² (~0.44) with a clearly detectable but noisy relationship
# between |residual| and sigma_true (corr ~0.55) -- neither an unlearnable
# pure-noise dataset nor a trivial perfect fit.
n_samples = 5000

x1 = rng.uniform(-1, 1, n_samples)
x2 = rng.uniform(-1, 1, n_samples)
x3 = rng.uniform(-1, 1, n_samples)
x4 = rng.uniform(-1, 1, n_samples)

a_noise = 2
b_noise = 0.5
sigma_true = np.exp(a_noise * x1 + b_noise * x2)
f_mean = 2 * x3 - 1.5 * x4 + x3 * x4

y_data = f_mean + rng.normal(0, sigma_true, n_samples)
X_data = np.column_stack([x1, x2, x3, x4])
feature_names = ["x1", "x2", "x3", "x4"]

print(f"std(f_mean)  = {f_mean.std():.4f}")
print(f"sigma_true   : min={sigma_true.min():.4f} mean={sigma_true.mean():.4f} max={sigma_true.max():.4f}")

df_X = pd.DataFrame(X_data, columns=feature_names)
df_y = pd.Series(y_data, name="y")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(df_y.values, bins=60)
ax.set_xlabel('Target value')
ax.set_ylabel('Count')
ax.set_title(f"Distribution of target in '{dataset.name}'")
plt.savefig(os.path.join(task_folder, "target_distribution.png"))
plt.close(fig)

# 2. Split & Normalize (same 50/25/25 split as run_sigma_sr.py)
X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test = split_and_normalize_data(df_X, df_y, random_seed)

print(f'Proper training : {X_prop_train.shape[0]} samples')
print(f'Calibration     : {X_cal.shape[0]} samples')
print(f'Test            : {X_test.shape[0]} samples')

# 3. Train Base Regressor (Random Forest)
print("Training base regressor...")
base_regressor = WrapRegressor(
    RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed)
)
base_regressor.fit(X_prop_train, y_prop_train)

y_test_pred = base_regressor.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
print(f'R² on test set: {r2:.4f}')

# 4. Compute other Conformal Predictors for comparison
learner_prop = base_regressor.learner
sigmas_train = {}
sigmas_cal = {}
sigmas_test = {}
sigmas_comp = {}
conf_intervals = {}

y_pred_oob = learner_prop.oob_prediction_
residuals_prop_oob = y_prop_train - y_pred_oob

# Standard CP
print("Computing CI for SCP...")
base_regressor.calibrate(X_cal, y_cal)
sigmas_comp["conformal_predictor"] = np.ones(len(X_cal))
conf_intervals["conformal_predictor"] = base_regressor.predict_int(X_test, confidence=confidence)

# KNN distance
# de.apply(X) on a real X doesn't depend on the oob flag (only the no-arg
# apply() used for the training-set augmentation column does), so a single
# fit serves both compute_normalized_intervals and augmentation instead of
# fitting the same KNN index twice.
print("Computing CI for knn_dist NCP...")
de_knn_dist = fit_difficulty_estimator(X_prop_train, "knn_dist", oob=True)
intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, confidence)
conf_intervals["normalized_cp_knn_dist"] = intervals
sigmas_comp["normalized_cp_knn_dist"] = comp_sigma_cal
sigmas_train["knn_dist"] = de_knn_dist.apply()
sigmas_cal["knn_dist"] = comp_sigma_cal
sigmas_test["knn_dist"] = de_knn_dist.apply(X_test)

# KNN std
print("Computing CI for knn_std NCP...")
de_knn_std = fit_difficulty_estimator(X_prop_train, "knn_std", y_prop_train=y_prop_train, oob=True)
intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, confidence)
conf_intervals["normalized_cp_knn_std"] = intervals
sigmas_comp["normalized_cp_knn_std"] = comp_sigma_cal
sigmas_train["knn_std"] = de_knn_std.apply()
sigmas_cal["knn_std"] = comp_sigma_cal
sigmas_test["knn_std"] = de_knn_std.apply(X_test)

# KNN out-of-bag residuals
print("Computing CI for knn_res NCP...")
de_knn_res = fit_difficulty_estimator(X_prop_train, "knn_res", y_prop_train=y_prop_train, learner_prop=learner_prop, oob=True)
intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, confidence)
conf_intervals["normalized_cp_knn_res"] = intervals
sigmas_comp["normalized_cp_knn_res"] = comp_sigma_cal
sigmas_train["knn_res"] = de_knn_res.apply()
sigmas_cal["knn_res"] = comp_sigma_cal
sigmas_test["knn_res"] = de_knn_res.apply(X_test)

# Random Forest variance
# unlike the KNN estimators above, de.apply(X) for the variance estimator
# DOES depend on the oob flag (the oob branch expects X sized to the
# training set), so cal/test must keep using the separate non-oob fit.
print("Computing CI for var NCP...")
de_var = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop)
intervals, comp_sigma_cal = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, confidence)
conf_intervals["normalized_cp_norm_var"] = intervals
sigmas_comp["normalized_cp_norm_var"] = comp_sigma_cal
de_var_oob = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop, oob=True)
sigmas_train["var"] = de_var_oob.apply()
sigmas_cal["var"] = comp_sigma_cal  # same de_var.apply(X_cal) already computed above
sigmas_test["var"] = de_var.apply(X_test)

# Mondrian CP using variance
print("Computing CI for MCP...")
min_points = int(1 / (1 - confidence) - 1) + 1
bin_thresholds = find_bin_thresholds_with_min_size(sigmas_comp["normalized_cp_norm_var"], min_points, random_seed)
number_of_bins = len(bin_thresholds) - 1
print(f"Number of Mondrian bins: {number_of_bins}")

# the "mc" argument for calibrate()/predict_int() internally takes X as its
# only parameter; reuse the variance sigmas already computed above for
# X_cal/X_test instead of recomputing a full RF-variance pass over them.
sigma_var_cache = {id(X_cal): sigmas_comp["normalized_cp_norm_var"], id(X_test): sigmas_test["var"]}


def mondrian_categories(X):
    sigmas = sigma_var_cache.get(id(X))
    if sigmas is None:
        sigmas = de_var.apply(X)
    return binning(sigmas, bins=bin_thresholds, seed=random_seed)


regressor_mond = WrapRegressor(learner_prop)
regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
sigmas_comp["mondrian_cp"] = np.ones(len(X_cal))
conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=confidence)

# 5. Symbolic Regression as an estimator for sigma(x)

# 5.1. Augment input using the NCP sigmas computed above (all already fit
# with oob=True for this purpose)
X_train_sr = np.zeros((X_prop_train.shape[0], len(sigmas_train)), dtype=np.float32)
X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_cal)), dtype=np.float32)
X_test_sr = np.zeros((X_test.shape[0], len(sigmas_test)), dtype=np.float32)
for i, key in enumerate(sigmas_train.keys()):
    X_train_sr[:, i] = sigmas_train[key]
    X_cal_sr[:, i] = sigmas_cal[key]
    X_test_sr[:, i] = sigmas_test[key]
X_train_sr = np.concatenate((X_prop_train, X_train_sr), axis=1)
X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

y_raw_residual = residuals_prop_oob

# 5.2. Bin-crossfit loss (see NOTES.md), same Julia source as run_sigma_sr.py
sigma_losses = [
    ("bin_crossfit", dict(loss_function=bin_crossfit_loss_julia(confidence)), y_raw_residual),
]

for loss_name, loss_kwargs, y_train_sr in sigma_losses:
    sigma_predictor = PySRRegressor(
        model_selection="score",
        tournament_selection_n=15,
        populations=31,
        population_size=50,
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "log", "exp"],
        nested_constraints={
            "sin": {"cos": 0, "sin": 0},
            "cos": {"cos": 0, "sin": 0},
            "log": {"log": 0},
            "exp": {"exp": 0}},
        temp_equation_file=True,
        verbosity=1,
        random_state=random_seed,
        **loss_kwargs,
    )
    sigma_predictor.fit(X_train_sr, y_train_sr)
    log_equations(sigma_predictor, task_folder, f"symbolic_regression_{loss_name}")

    de_sr = DifficultyEstimator()
    de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.predict(X)), scaler=True)

    sigmas_cal_sr = de_sr.apply(X_cal_sr)
    sigmas_test_sr = de_sr.apply(X_test_sr)

    cr_sr = ConformalRegressor()
    cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

    cp_key = f"symbolic_regression_{loss_name}"
    conf_intervals[cp_key] = cr_sr.predict_int(
        learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=confidence
    )
    sigmas_comp[cp_key] = sigmas_cal_sr

# per-method CI plot + coverage/amplitude stats
ci_means = {}
ci_medians = {}
coverages = {}
for method, intervals in conf_intervals.items():
    ci_means[method], ci_medians[method], coverages[method] = evaluate_and_plot_method(method, intervals, y_test, y_test_pred, dataset, task_folder)

fig, ax = plot_pareto(list(conf_intervals.keys()), ci_medians, coverages, translations=translations)
ax.set_title(f"Performance of conformal prediction methods on dataset \"{dataset.name}\"")
plt.savefig(os.path.join(task_folder, "pareto.png"), dpi=300)
plt.close(fig)

# 6. Save results
results_dictionary = {"task_id": ["synthetic"], "dataset_name": [dataset.name], "r2": [r2]}
for method in ci_means.keys():
    results_dictionary[f"{method}_mean"] = [ci_means[method]]
    results_dictionary[f"{method}_median"] = [ci_medians[method]]
    results_dictionary[f"{method}_coverage"] = [coverages[method]]

df_results = pd.DataFrame.from_dict(results_dictionary)
df_results.to_csv(os.path.join(results_folder, "results.csv"), index=False)
