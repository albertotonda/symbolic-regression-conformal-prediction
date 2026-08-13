# -*- coding: utf-8 -*-
"""
Entry point. For each OpenML-CTR23 task:
- split training/calibration/test
- test different conformal predictors
  -- standard conformal predictor
  -- normalized conformal predictors (several difficulty-estimation strategies)
  -- Mondrian conformal predictor
- for each candidate set of confidence intervals, check tightness and coverage
- symbolic regression
  -- uses as features all statistics computed by the normalized estimators
  -- plus feature values of the original problem
"""

import matplotlib
matplotlib.use("Agg") # headless batch script, only ever saves figures to file
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import seaborn as sns

from collections import defaultdict
from datetime import datetime

from config import REGRESSOR_MODELS, parse_cli_config
from data import get_benchmark_task_ids, prepare_task_data
from conformal import (train_base_regressor, fit_difficulty_estimators,
                        compute_normalized_intervals, select_mondrian_source,
                        compute_mondrian_intervals, NORMALIZED_CP_RESULT_KEYS)
from symbolic_regression import run_symbolic_regression
from evaluate import evaluate_and_plot_method, plot_pareto, translations


def _setup_results_folder(random_seed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = "results-%d_%s" % (random_seed, timestamp)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


def _save_config_snapshot(config, random_seed, results_folder):
    # save experiment config for tracability
    with open(os.path.join(results_folder, "config.json"), "w") as fp:
        json.dump({**config.model_dump(), "random_seed": random_seed}, fp, indent=2)


def run_task_pipeline(task_id, config, random_seed, results_folder, results_dictionary):
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
    # to this predictor model (see conformal.fit_difficulty_estimators)
    difficulty_estimators = fit_difficulty_estimators(X_prop_train, y_prop_train, learner_prop, config)
    for sigma_key, de in difficulty_estimators.items():
        intervals, s_cal, s_test = compute_normalized_intervals(
            de, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
        task_results[NORMALIZED_CP_RESULT_KEYS[sigma_key]] = intervals
        sigmas_cal[sigma_key] = s_cal
        sigmas_test[sigma_key] = s_test

    # Mondrian CP, binned on whichever difficulty estimator applies: RF
    # ensemble-variance by default, KNN-distance as an experimental
    # alternative (config.use_alt_mondrian) for other predictor models
    mondrian_source = select_mondrian_source(config, difficulty_estimators, sigmas_cal)
    if mondrian_source is not None:
        de_mond, sigmas_cal_mond = mondrian_source
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


def run_experiment(config, random_seed=42):
    results_folder = _setup_results_folder(random_seed)

    # set up plotting
    sns.set_theme(style='darkgrid')

    # get task_id for all tasks in the benchmark suite
    task_ids = get_benchmark_task_ids(config.suite_id, config.tasks_too_good, config.tasks_too_bad)

    # data structure to store the results; pre-seed the leading columns so
    # the CSV has a stable, readable column order regardless of which
    # per-method keys get added first
    results_dictionary = defaultdict(list, {"task_id": [], "dataset_name": [], "r2": []})

    _save_config_snapshot(config, random_seed, results_folder)

    # start the loop, for every task
    last_task_methods = []
    for task_id in task_ids:
        last_task_methods = run_task_pipeline(
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
        run_experiment(config, random_seed)
