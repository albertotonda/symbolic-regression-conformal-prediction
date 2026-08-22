# %% [markdown]
# # Symbolic Regression for Conformal Prediction — Sandbox

# %% [markdown]
# ## 0. Imports & Setup

# %%
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2

from collections import defaultdict

from crepes import WrapRegressor, ConformalRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer, binning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pysr import PySRRegressor

# make src/ (this file's parent's parent) importable, so this script can be
# run directly (e.g. `uv run src/experiments/run_sigma_sr.py`) regardless of
# the current working directory
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data import load_and_preprocess_openml_task, get_benchmark_task_ids
from evaluate import evaluate_and_plot_method, log_equations, plot_pareto, setup_results_folder, translations

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style='darkgrid')
print('All imports OK')

# %% [markdown]
# ## 1. Loop over all OpenML-CTR23 tasks
#
# Same walkthrough as before (load data, split, train base regressor,
# compute conformal predictors), repeated for every task in the suite. For
# the symbolic-regression sigma estimator (section 5), we now also try the
# candidate loss functions discussed in NOTES.md (2026-08-14 entry): MAE,
# pinball/quantile loss, a scale-invariant log-variance loss, and Gaussian
# NLL.

# %%*
TASKS_TOO_GOOD = [361236, 361247, 361252, 361254, 361256, 361257, 361268, 361617]
TASKS_TOO_BAD = [361243, 361244, 361261, 361618, 361619]
task_ids = get_benchmark_task_ids(353, TASKS_TOO_GOOD, TASKS_TOO_BAD)

random_seed = 42
confidence = 0.95

# results-sigma-sr-<seed>_<timestamp>/<dataset_name>/..., same structure as
# run_interval_sr.py's results folder
results_folder = setup_results_folder("sigma-sr", random_seed)

# accumulated across all tasks, same as run_interval_sr.py's results_dictionary
results_dictionary = defaultdict(list, {"task_id": [], "dataset_name": [], "r2": []})
last_task_methods = []

for task_id in task_ids:

    # ## 1. Load & Explore a Dataset
    df_X, df_y, task = load_and_preprocess_openml_task(task_id)
    dataset = task.get_dataset()
    task_folder = os.path.join(results_folder, dataset.name)
    os.makedirs(task_folder, exist_ok=True)

    print(f"Dataset  : {dataset.name}")
    print(f"Samples  : {df_X.shape[0]}")
    print(f"Features : {df_X.shape[1]}")
    print(f"Target   : min={df_y.min():.2f}  max={df_y.max():.2f}  mean={df_y.mean():.2f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(df_y.values, bins=60)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Count')
    ax.set_title(f"Distribution of target in '{dataset.name}'")
    plt.savefig(os.path.join(task_folder, "target_distribution.png"))
    plt.close(fig)

    # ## 2. Split & Normalize
    #
    # Conformal prediction requires a calibration set separate from training.
    # We use a 50 / 25 / 25 split:
    # - Proper training set (X_prop_train): trains the base regressor
    # - Calibration set (X_cal): calibrates conformal predictors (never seen
    #   by the base regressor during training)
    # - Test set (X_test): evaluates coverage and interval width
    #
    # We standardize X and y so that difficulty estimates and the SR loss
    # function are scale-invariant.
    X = df_X.values
    y = df_y.values
    feature_names = list(df_X.columns)

    X_prop_train, X_test, y_prop_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True, random_state=random_seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test, y_test, test_size=0.5, shuffle=True, random_state=random_seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_prop_train = scaler_X.fit_transform(X_prop_train)
    X_cal        = scaler_X.transform(X_cal)
    X_test       = scaler_X.transform(X_test)

    y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1, 1)).ravel()
    y_cal        = scaler_y.transform(y_cal.reshape(-1, 1)).ravel()
    y_test       = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f'Proper training : {X_prop_train.shape[0]} samples')
    print(f'Calibration     : {X_cal.shape[0]} samples')
    print(f'Test            : {X_test.shape[0]} samples')

    # ## 3. Train Base Regressor (Random Forest)
    #
    # crepes.WrapRegressor wraps any sklearn-compatible regressor and adds
    # conformal prediction methods (.calibrate() and .predict_int()).
    #
    # We use oob_score=True so the Random Forest stores out-of-bag (OOB)
    # predictions — predictions made on each training sample by trees that
    # did not train on it. These are used later as one of the difficulty
    # estimators.
    print("Training base regressor...")
    base_regressor = WrapRegressor(
        RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed)
    )
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')

    # ## 4. Compute other Conformal Predictors for comparison
    def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
        """
        Calibrate a normalized conformal regressor using an already-fitted
        DifficultyEstimator. Returns the confidence intervals for the test set,
        together with the difficulty estimates on the calibration and test sets.
        """
        sigmas_cal = de.apply(X_cal)

        regressor_norm = WrapRegressor(learner_prop)
        regressor_norm.calibrate(X_cal, y_cal, de=de)

        intervals = regressor_norm.predict_int(X_test, confidence=confidence)

        return intervals, sigmas_cal

    def _find_bin_thresholds_with_min_size(sigmas_cal_var, min_points, random_seed):
        """
        crepes.extras.binning's min_size parameter requests bins=len(values)//
        min_size equal-frequency bins, but with tied/duplicated difficulty
        scores (common for the ensemble-variance estimator) pd.qcut can still
        leave a handful of bins a few points short of min_size. So, rather than
        trusting the requested bin count outright, verify the actual per-bin
        counts and back off the number of bins until every one of them holds at
        least min_points calibration points.
        """
        number_of_bins = len(sigmas_cal_var) // min_points
        while number_of_bins > 1:
            assigned_bins, bin_thresholds = binning(
                sigmas_cal_var, bins=number_of_bins, seed=random_seed)
            counts = np.bincount(assigned_bins.astype(int))
            if counts.min() >= min_points:
                return bin_thresholds
            number_of_bins -= 1
        return np.array([-np.inf, np.inf])

    learner_prop = base_regressor.learner
    sigmas = {}
    conf_intervals = {}

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    # Standard CP
    print("Computing CI for SCP...")
    base_regressor.calibrate(X_cal, y_cal)
    sigmas["conformal_predictor"] = np.ones(len(X_cal))
    conf_intervals["conformal_predictor"] = base_regressor.predict_int(X_test, confidence=confidence)

    # KNN distance
    print("Computing CI for knn_dist NCP...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, scaler=True)
    conf_intervals["normalized_cp_knn_dist"], sigmas["normalized_cp_knn_dist"] = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN std
    print("Computing CI for knn_std NCP...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
    conf_intervals["normalized_cp_knn_std"], sigmas["normalized_cp_knn_std"] = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN out-of-bag residuals
    print("Computing CI for knn_res NCP...")
    de_knn_res = DifficultyEstimator()
    de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
    conf_intervals["normalized_cp_knn_res"], sigmas["normalized_cp_knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, confidence)

    # Random Forest variance
    print("Computing CI for var NCP...")
    de_var = DifficultyEstimator()
    de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
    conf_intervals["normalized_cp_norm_var"], sigmas["normalized_cp_norm_var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, confidence)

    # Mondrian CP using variance
    print("Computing CI for MCP...")
    min_points = int(1 / (1-confidence) - 1) + 1
    bin_thresholds = _find_bin_thresholds_with_min_size(sigmas["normalized_cp_norm_var"], min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas["mondrian_cp"] = np.ones(len(X_cal))
    conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=confidence)

    # ## 5. Symbolic Regression as an estimator for sigma(x)
    #
    # The idea is that instead of using Symbolic Regression to directly
    # predict confidence bounds, we use it as a difficulty estimator to use
    # in Normalized Conformal Prediction (NCP). Since we need extra data to
    # train the symbolic regressor, we can make use of Random Forest's
    # out-of-bag predictions.

    # ### 5.1. Computing all NPS sigma on OOB predictions
    print("Computing NCP sigmas for data augmentation...")
    sigmas_train = {}
    sigmas_cal = {}
    sigmas_test = {}

    # KNN distance
    de_knn_dist_oob = DifficultyEstimator()
    de_knn_dist_oob.fit(X=X_prop_train, scaler=True, oob=True)
    sigmas_train["knn_dist"] = de_knn_dist_oob.apply()
    sigmas_cal["knn_dist"] = de_knn_dist_oob.apply(X_cal)
    sigmas_test["knn_dist"] = de_knn_dist_oob.apply(X_test)

    # KNN std
    de_knn_std_oob = DifficultyEstimator()
    de_knn_std_oob.fit(X=X_prop_train, y=y_prop_train, scaler=True, oob=True)
    sigmas_train["knn_std"] = de_knn_std_oob.apply()
    sigmas_cal["knn_std"] = de_knn_std_oob.apply(X_cal)
    sigmas_test["knn_std"] = de_knn_std_oob.apply(X_test)

    # KNN out-of-bag residuals
    de_knn_res_oob = DifficultyEstimator()
    de_knn_res_oob.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True, oob=True)
    sigmas_train["knn_res"] = de_knn_res_oob.apply()
    sigmas_cal["knn_res"] = de_knn_res_oob.apply(X_cal)
    sigmas_test["knn_res"] = de_knn_res_oob.apply(X_test)

    # Random Forest variance
    de_var_oob = DifficultyEstimator()
    de_var_oob.fit(X=X_prop_train, learner=learner_prop, scaler=True, oob=True)
    sigmas_train["var"] = de_var_oob.apply()
    sigmas_cal["var"] = de_var.apply(X_cal) # For cal and test, use default (no oob) version, otherwise same oob trees are used instead of full model
    sigmas_test["var"] = de_var.apply(X_test)

    # augment input using sigmas
    X_train_sr = np.zeros((X_prop_train.shape[0], len(sigmas_train)), dtype=np.float32)
    X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_cal)), dtype=np.float32)
    X_test_sr = np.zeros((X_test.shape[0], len(sigmas_test)), dtype=np.float32)
    for i, key in enumerate(sigmas_train.keys()):
            X_train_sr[:,i] = sigmas_train[key]
            X_cal_sr[:,i] = sigmas_cal[key]
            X_test_sr[:,i] = sigmas_test[key]
    X_train_sr = np.concatenate((X_prop_train, X_train_sr), axis=1)
    X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
    X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

    y_log_abs_residual = np.log(np.abs(residuals_prop_oob))
    y_raw_residual = residuals_prop_oob

    # ### 5.2. Try the candidate fitness functions discussed in NOTES.md

    # Two candidate losses that target the actual downstream conformal
    # objective instead of regressing against a noisy single-point proxy
    # (see NOTES.md 2026-08-21 entry / Claude memory sigma-sr-theory):
    # both are scale-invariant by construction (a uniform blow-up of sigma
    # changes neither the simulated-calibration width nor a pairwise
    # ranking), so unlike the direct-bound predictor's asymmetric loss they
    # don't need a hand-added term to prevent degenerate huge intervals.
    mean_width_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual (not its log)
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        sigma = exp.(log_sigma)
        if any(sigma .<= zero(T)) || !all(isfinite.(sigma))
            return L(Inf)
        end

        # simulate the real conformal calibration step (empirical 95% quantile
        # of normalized residuals) on this batch, then score the resulting
        # mean interval width -- literally the downstream deliverable, not a
        # proxy for it. Manual sort-based quantile since Statistics.quantile
        # may not be in scope inside PySR's custom-loss eval context.
        scores = abs.(dataset.y) ./ sigma
        n = dataset.n
        sorted_scores = sort(scores)
        idx = clamp(ceil(Int, 0.95 * n), 1, n)
        q_hat = L(sorted_scores[idx])

        widths = q_hat .* sigma
        return L(sum(widths) / n)
    end
    """

    pairwise_ranking_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual (not its log)
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        if !all(isfinite.(log_sigma))
            return L(Inf)
        end

        # only the RELATIVE ordering of sigma across points matters for
        # downstream conformal efficiency (a constant rescale cancels out at
        # calibration), so reward correctly ranking pairs of points by true
        # residual size instead of matching a noisy pointwise target.
        # Consecutive-row pairing keeps this O(n) and deterministic across
        # every fitness evaluation (rows are already shuffled by the
        # upstream train/cal/test split, so this is as good as random
        # pairing without the run-to-run noise random sampling would add).
        # Pairs are weighted by |delta_resid| so near-ties (unreliable,
        # mostly-noise comparisons) contribute little.
        n = dataset.n
        npairs = div(n, 2)
        eps = L(1e-6)
        # NOTE: margin must be > 0. At margin=0, a constant tree gives
        # delta_sigma=0 for every pair, so hinge=0 for every pair and the
        # loss is EXACTLY 0 -- the global minimum, trivially and immediately
        # achieved by any constant. That's a degenerate optimum, not a
        # search-budget problem: no formula can ever score better than the
        # constant's 0, so there is zero selection pressure to leave it.
        # margin=0.1 makes a constant score exactly 0.1 (bad but beatable),
        # only reachable by 0 through genuine, sufficiently-separated ranking.
        margin = L(0.1)

        total_loss = zero(L)
        total_weight = zero(L)
        for k in 1:npairs
            i = 2k - 1
            j = 2k
            delta_resid = log(abs(L(dataset.y[j])) + eps) - log(abs(L(dataset.y[i])) + eps)
            delta_sigma = L(log_sigma[j]) - L(log_sigma[i])

            s = sign(delta_resid)
            weight = abs(delta_resid)

            hinge = max(zero(L), margin - s * delta_sigma)
            total_loss += weight * hinge
            total_weight += weight
        end

        return total_weight == zero(L) ? zero(L) : total_loss / total_weight
    end
    """

    sigma_losses = [
        # ("mae", dict(elementwise_loss="L1DistLoss()"), y_log_abs_residual),
        # ("mean_width", dict(loss_function=mean_width_loss_julia), y_raw_residual),
        ("pairwise_rank", dict(loss_function=pairwise_ranking_loss_julia), y_raw_residual),
    ]

    for loss_name, loss_kwargs, y_train_sr in sigma_losses:
        sigma_predictor = PySRRegressor(
            model_selection="score",
            tournament_selection_n=15, # default 15
            populations=31, # default 31
            population_size=30, # must be >= topn:=12 (default 27)
            niterations=100, # default 100
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "log", "exp"],
            # nested_constraints={
            #     "sin": {"cos": 0, "sin": 0}, 
            #     "cos": {"cos": 0, "sin": 0},
            #     "log": {"log": 0},
            #     "exp": {"exp": 0}},
            temp_equation_file=True, # does not clutter directory with temporary files
            verbosity=1, # can also be set to 0, it should be ok
            random_state=random_seed,
            **loss_kwargs,
        )
        sigma_predictor.fit(X_train_sr, y_train_sr)
        log_equations(sigma_predictor, task_folder, f"symbolic_regression_{loss_name}")

        de_sr = DifficultyEstimator()
        de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.predict(X)), scaler=True)

        # WrapRegressor.calibrate()/.predict_int() feed the SAME X to both the
        # wrapped learner (needs the raw data) and de.apply() (needs the
        # sigma-augmented columns) — incompatible here, so calibrate manually via
        # the lower-level ConformalRegressor instead.
        sigmas_cal_sr = de_sr.apply(X_cal_sr)
        sigmas_test_sr = de_sr.apply(X_test_sr)

        cr_sr = ConformalRegressor()
        cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

        cp_key = f"symbolic_regression_{loss_name}"
        conf_intervals[cp_key] = cr_sr.predict_int(
            learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=confidence
        )
        sigmas[cp_key] = sigmas_cal_sr

    # per-method CI plot + coverage/amplitude stats, same helper
    # run_interval_sr.py uses (stats get appended into results_dictionary)
    for method, intervals in conf_intervals.items():
        evaluate_and_plot_method(method, intervals, y_test, y_test_pred,
                                  dataset, task_folder, results_dictionary)

    results_dictionary["task_id"].append(task_id)
    results_dictionary["dataset_name"].append(dataset.name)
    results_dictionary["r2"].append(r2)
    last_task_methods = list(conf_intervals.keys())

    # per-task Pareto plot across all methods computed for this task
    fig, ax = plot_pareto(last_task_methods, results_dictionary, translations=translations)
    ax.set_title(f"Performance of conformal prediction methods on dataset \"{dataset.name}\"")
    plt.savefig(os.path.join(task_folder, "pareto.png"), dpi=300)
    plt.close(fig)

# %% [markdown]
# ## 6. Save results and plot the global Pareto front, across all tasks

# %%
df_results = pd.DataFrame.from_dict(results_dictionary)
df_results.to_csv(os.path.join(results_folder, "results.csv"), index=False)

fig, ax = plot_pareto(last_task_methods, results_dictionary, translations=translations, all_results=True)
ax.set_title("Performance of conformal prediction methods on selected CTR-23 datasets")
plt.savefig(os.path.join(results_folder, "pareto.png"), dpi=300)
plt.close(fig)
