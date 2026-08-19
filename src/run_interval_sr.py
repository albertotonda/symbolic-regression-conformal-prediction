# -*- coding: utf-8 -*-
"""
Entry point and full experiment logic. For each OpenML-CTR23 task:
- split training/calibration/test (data.py)
- test different conformal predictors
  -- standard conformal predictor
  -- normalized conformal predictors (several difficulty-estimation strategies)
  -- Mondrian conformal predictor
- for each candidate set of confidence intervals, check tightness and coverage
- symbolic regression
  -- uses as features all statistics computed by the normalized estimators
  -- plus feature values of the original problem

Everything needed to run and understand one experiment lives in this file,
grouped into three sections below: conformal predictors, symbolic
regression, and orchestration (the part that is actually "run"). Config
(config.py), data loading (data.py) and result plotting/statistics
(evaluate.py) are kept separate since they're reused as-is by the
post-hoc analysis scripts in analysis/.
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg") # headless batch script, only ever saves figures to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, binning

from sklearn.metrics import r2_score

from pysr import PySRRegressor

from config import REGRESSOR_MODELS, parse_cli_config, save_config_snapshot
from data import get_benchmark_task_ids, prepare_task_data
from evaluate import evaluate_and_plot_method, log_equations, plot_pareto, setup_results_folder, translations


# ---------------------------------------------------------------------------
# Conformal predictors: fitting/calibrating the standard conformal regressor,
# normalized conformal regressors (one per difficulty-estimation strategy),
# and Mondrian conformal regressors.
# ---------------------------------------------------------------------------

# maps each difficulty-estimation strategy (as used internally, and as a key
# into the sigmas_cal/sigmas_test dicts consumed by symbolic regression) to
# the task_results key its normalized CP intervals are stored under
NORMALIZED_CP_RESULT_KEYS = {
    "knn_dist": "normalized_cp_knn_dist",
    "knn_std": "normalized_cp_knn_std",
    "knn_oob_res": "normalized_cp_knn_res",
    "ensemble_var": "normalized_cp_norm_var",
}


def train_base_regressor(X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test,
                          regressor_class, regressor_params, random_seed):
    """
    Train the base regressor, wrap it as a conformal regressor, and
    calibrate the standard conformal predictor.
    """
    print("Training regressor...")
    base_regressor = regressor_class(**regressor_params)
    if "random_state" in base_regressor.get_params():
         base_regressor.set_params(random_state=random_seed)
    regressor = WrapRegressor(base_regressor)
    regressor.fit(X_prop_train, y_prop_train)

    # get predictions for the test set and calibration set from the learner
    y_cal_pred = regressor.predict(X_cal)
    y_test_pred = regressor.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)

    print("Calibrating conformal regressor...")
    regressor.calibrate(X_cal, y_cal)

    return regressor, y_cal_pred, y_test_pred, r2_test


def fit_difficulty_estimators(X_prop_train, y_prop_train, learner_prop, config):
    """
    Fit one DifficultyEstimator per normalization strategy applicable to the
    current predictor model. KNN-distance and KNN-on-target-std apply to any
    predictor; OOB-residuals and ensemble-variance require an out-of-bag,
    inspectable ensemble, so they're only fit for RandomForestRegressor.

    Returns a dict {sigma_key: DifficultyEstimator}, keyed the same way as
    NORMALIZED_CP_RESULT_KEYS.
    """
    difficulty_estimators = {}

    # distance of KNN in feature space, default k=25
    print("Normalizing confidence intervals using KNN for difficulty estimation...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, k=config.ncp_knn_k, scaler=True)
    difficulty_estimators["knn_dist"] = de_knn_dist

    # standard deviation of KNN in target space
    print("Now normalizing using standard deviations...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, k=config.ncp_knn_k, scaler=True)
    difficulty_estimators["knn_std"] = de_knn_std

    # a third way of normalizing, using absolute residuals; it does not work
    # for XGBoost, because only Random Forest has out-of-bag predictions for
    # each individual learner
    if config.predictor_model == "RandomForestRegressor":
        print("Now normalizing using OOB predictions of each estimator...")
        residuals_prop_oob = y_prop_train - learner_prop.oob_prediction_
        de_knn_res = DifficultyEstimator()
        de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, k=config.ncp_knn_k, scaler=True)
        difficulty_estimators["knn_oob_res"] = de_knn_res

        # a fourth way: using the variance of each element of the ensemble;
        # XGBoostRegressor doesn't expose the individual estimators, so this
        # is also RandomForest-only
        print("Now normalizing using variance of the estimators...")
        de_var = DifficultyEstimator()
        de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
        difficulty_estimators["ensemble_var"] = de_var

    return difficulty_estimators


def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
    """
    Calibrate a normalized conformal regressor using an already-fitted
    DifficultyEstimator. Returns the confidence intervals for the test set,
    together with the difficulty estimates on the calibration and test sets
    (needed later as features for the symbolic regression step).
    """
    sigmas_cal = de.apply(X_cal)

    regressor_norm = WrapRegressor(learner_prop)
    regressor_norm.calibrate(X_cal, y_cal, de=de)

    sigmas_test = de.apply(X_test)
    intervals = regressor_norm.predict_int(X_test, confidence=confidence)

    return intervals, sigmas_cal, sigmas_test


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


def compute_mondrian_intervals(learner_prop, de_var, sigmas_cal_var, X_cal, y_cal,
                                X_test, confidence, random_seed):
    """
    Calibrate a Mondrian conformal regressor. Bin boundaries are computed
    directly from the calibration set's own ensemble-variance difficulty
    scores, using the largest number of equal-sized bins for which every
    bin actually holds at least the minimum number of calibration points
    required for a finite conformal quantile at this confidence level.
    """
    # minimal number of data points per bin is n >= 1/(1-confidence) - 1;
    # +1 as a safety margin, since crepes' own check on the calibration side
    # (base.py: int((1-confidence)*(n+1))-1 >= 0) can trip at the exact
    # boundary count due to floating-point rounding of (1-confidence) for
    # typical confidence levels (e.g. 1-0.9 != 0.1 exactly in binary float)
    min_points = int(1 / (1-confidence) - 1) + 1

    bin_thresholds = _find_bin_thresholds_with_min_size(sigmas_cal_var, min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    intervals_mond = regressor_mond.predict_int(X_test, confidence=confidence)

    return intervals_mond, number_of_bins


# ---------------------------------------------------------------------------
# Symbolic regression: predicting confidence-interval amplitude via PySR,
# from point predictions, difficulty estimates, and the original features.
# ---------------------------------------------------------------------------

SR_MODEL_FILENAME = "symbolic_regression_cp.pk"
SR_FEATURES_FILENAME = "sr_features.json"

# unfortunately, using pysr we can define a custom loss function...in Julia.
# since the syntax is different, we can only use a string that is then passed
# to the Julia interpreter internally inside the PySRRegressor object
loss_function_julia_penalize_smaller = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}

    # get predicted values for the current tree
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # 'flag' == false means that evaluating the tree caused an error
    if !flag
        return L(Inf)
    end

    result = 0.0
    coverage = 0.0
    coverage_penalty = 100.0

    # instead of just having a sum of squared means, we penalize more heavily
    # samples for which the predictions are inferior to 'y' (here the difference
    # between the true value and the predicted value)
    for i in 1:length(dataset.y)
        if (prediction[i] < dataset.y[i])
            result += 10 * (prediction[i] - dataset.y[i])^2
        else
            result += (prediction[i] - dataset.y[i])^2
            coverage += 1
        end
    end

    if ((coverage / dataset.n) < 0.95)
        # penalty is equal to the difference between complete coverage and current result * weight
        result += (0.95 - coverage/dataset.n) * dataset.n * coverage_penalty
    end

    return result / dataset.n
end
"""


def run_symbolic_regression(X_cal, X_test, y_cal, y_test, y_cal_pred, y_test_pred,
                             sigmas_cal, sigmas_test,
                             feature_names, task_folder, config, random_seed):
    """
    Train a PySRRegressor to predict the amplitude of a confidence interval
    around the point prediction, using point predictions, difficulty
    estimates, and the original features. Uses a custom Julia loss function
    (loss_function_julia_penalize_smaller) that penalizes intervals that do
    not cover the true value more heavily than it penalizes wide intervals.
    """
    # step 1: prepare data sets with all sigmas and stuff on calibration set
    # and test set; these will be a special version, just for symbolic regression
    X_train_sr = np.zeros((y_cal.shape[0], len(sigmas_cal)+1), dtype=np.float32)
    X_test_sr = np.zeros((y_test.shape[0], len(sigmas_cal)+1), dtype=np.float32)

    # add point predictions
    X_train_sr[:,0] = y_cal_pred
    X_test_sr[:,0] = y_test_pred

    for i, key in enumerate(sigmas_cal.keys()):
        X_train_sr[:,i+1] = sigmas_cal[key]
        X_test_sr[:,i+1] = sigmas_test[key]

    # TODO: information used by the Mondrian conformal predictors is not immediately
    # applicable, unless I use something about the bins? to be explored
    # finally, add the feature information from the original data set
    X_train_sr = np.concatenate((X_train_sr, X_cal), axis=1)
    X_test_sr = np.concatenate((X_test_sr, X_test), axis=1)

    # finally, we need a target (y) for our problem of confidence interval
    # regression; we can obtain that by computing the absolute difference
    # between the y_true and the y_pred for a dataset
    y_train_sr = abs(y_cal - y_cal_pred)

    # now, for the more complex part: we can use a PySRRegressor, but we
    # need to change the fitness function! the fitness function is described
    # as a string (lines of Julia), defined above
    ci_regressor = PySRRegressor(
        tournament_selection_n=config.sr_tournament_selection_n,
        population_size=config.sr_population_size, # must be >= topn (default 12)
        niterations=config.sr_niterations,
        binary_operators=config.sr_binary_operators,
        unary_operators=config.sr_unary_operators,
        loss_function=loss_function_julia_penalize_smaller,
        temp_equation_file=True, # does not clutter directory with temporary files
        verbosity=1, # can also be set to 0, it should be ok
        random_state=random_seed,
        deterministic=True,
        parallelism="serial"
        )

    print("Running symbolic regression...")
    ci_regressor.fit(X_train_sr, y_train_sr)
    log_equations(ci_regressor, task_folder, "symbolic_regression_cp")

    print("Now computing confidence intervals for conformal set...")
    ci_amplitude_cal = ci_regressor.predict(X_train_sr)

    print(f"Number of CI with undercoverage in calibration set: {sum(ci_amplitude_cal < y_train_sr)}")

    print("And computing confidence intervals for test set...")
    ci_amplitude_test = ci_regressor.predict(X_test_sr)
    ci_test = np.zeros((y_test.shape[0], 2))
    for i in range(0, y_test.shape[0]):
        ci_test[i,0] = y_test_pred[i] - ci_amplitude_test[i]
        ci_test[i,1] = y_test_pred[i] + ci_amplitude_test[i]

    # save the predictor as a pickle file
    with open(os.path.join(task_folder, SR_MODEL_FILENAME), "wb") as fp:
        pickle.dump(ci_regressor, fp)

    # save the full list of SR feature names for this task (fixed synthetic
    # columns + this dataset's own feature names, in the order used above)
    with open(os.path.join(task_folder, SR_FEATURES_FILENAME), "w") as fp:
        json.dump({"sr_features": list(sigmas_cal.keys()) + feature_names}, fp, indent=2)

    return ci_test


# ---------------------------------------------------------------------------
# Orchestration: run every method above on every task, save results as we go.
# ---------------------------------------------------------------------------

def run_single_task(task_id, config, random_seed, results_folder, results_dictionary):
    """
    Run every conformal prediction method being compared on a single task:
    data prep, base regressor, standard/normalized/Mondrian/symbolic-
    regression CP, evaluation, and a per-task Pareto plot. New rows are
    appended in place into results_dictionary (a collections.defaultdict(list)).

    Returns the list of method keys evaluated for this task (used by the
    caller for the final, all-tasks Pareto plot).
    """
    # placeholders to concatenate all sigmas for symbolic regression
    sigmas_cal = {}
    sigmas_test = {}

    (X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test,
     feature_names, dataset, task_folder) = prepare_task_data(
         task_id, results_folder, random_seed)

    regressor, y_cal_pred, y_test_pred, r2_test = train_base_regressor(
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test,
        REGRESSOR_MODELS[config.predictor_model], config.predictor_params, random_seed)

    task_results = {}

    # Standard CP
    task_results["conformal_predictor"] = regressor.predict_int(X_test, confidence=config.confidence_level)

    # now we need to access the wrapped learner to re-use it for the other
    # conformal predictors, but it's not difficult
    learner_prop = regressor.learner

    # normalized CP: one variant per difficulty-estimation strategy applicable
    # to this predictor model (see fit_difficulty_estimators above)
    difficulty_estimators = fit_difficulty_estimators(X_prop_train, y_prop_train, learner_prop, config)
    for sigma_key, de in difficulty_estimators.items():
        intervals, s_cal, s_test = compute_normalized_intervals(
            de, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
        task_results[NORMALIZED_CP_RESULT_KEYS[sigma_key]] = intervals
        sigmas_cal[sigma_key] = s_cal
        sigmas_test[sigma_key] = s_test

    # Mondrian CP, binned on RF ensemble-variance
    if config.predictor_model == "RandomForestRegressor":
        de_mond = difficulty_estimators["ensemble_var"]
        sigmas_cal_mond = sigmas_cal["ensemble_var"]
        intervals_mond, number_of_bins = compute_mondrian_intervals(
            learner_prop, de_mond, sigmas_cal_mond, X_cal, y_cal, X_test,
            config.confidence_level, random_seed)
        task_results["mondrian_cp"] = intervals_mond
        results_dictionary["mondrian_bins"].append(number_of_bins)

    # proposed approach: symbolic regression intervals, using all sigmas
    task_results["symbolic_regression_cp"] = run_symbolic_regression(
        X_cal, X_test, y_cal, y_test, y_cal_pred, y_test_pred,
        sigmas_cal, sigmas_test,
        feature_names, task_folder, config, random_seed)

    # post-processing of the results for the different confidence intervals
    # statistics we are interested in: coverage, mean size, median size
    for method, confidence_intervals in task_results.items():
        evaluate_and_plot_method(method, confidence_intervals, y_test, y_test_pred,
                                  dataset, task_folder, results_dictionary)

    results_dictionary["task_id"].append(task_id)
    results_dictionary["dataset_name"].append(dataset.name)
    results_dictionary["r2"].append(r2_test)

    # also plot a Pareto-like scheme for this task
    fig, ax = plot_pareto([k for k in task_results], results_dictionary, translations=translations)
    ax.set_title("Performance of conformal prediction methods on dataset \"%s\"" % dataset.name)
    plt.savefig(os.path.join(task_folder, "pareto.png"), dpi=300)
    plt.close(fig)

    return list(task_results.keys())


def run_all_tasks(config, random_seed=42):
    results_folder = setup_results_folder("interval-sr", random_seed)

    # set up plotting
    sns.set_theme(style='darkgrid')

    # get task_id for all tasks in the benchmark suite
    task_ids = get_benchmark_task_ids(config.suite_id, config.tasks_too_good, config.tasks_too_bad)

    # data structure to store the results; pre-seed the leading columns so
    # the CSV has a stable, readable column order regardless of which
    # per-method keys get added first
    results_dictionary = defaultdict(list, {"task_id": [], "dataset_name": [], "r2": []})

    save_config_snapshot(config, random_seed, results_folder)

    # start the loop, for every task
    last_task_methods = []
    for task_id in task_ids:
        last_task_methods = run_single_task(
            task_id, config, random_seed, results_folder, results_dictionary)

        # save global dictionary of results as DataFrame after every task,
        # so a crash partway through doesn't lose already-computed results
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, config.results_csv_name), index=False)

    # and now, a global Pareto front plot
    fig, ax = plot_pareto(last_task_methods, results_dictionary, translations=translations, all_results=True)
    ax.set_title("Performance of conformal prediction methods on selected CTR-23 datasets")
    plt.savefig(os.path.join(results_folder, "pareto.png"), dpi=300)

    # TODO more informative: how many times a conformal predictor is Pareto-optimal?
    # it has to be done data set by data set, and maybe I could write a specific
    # post-processing script


if __name__ == "__main__":
    config = parse_cli_config()

    # let's run several experiments in a row, with different random seeds
    for random_seed in config.random_seeds:
        run_all_tasks(config, random_seed)
