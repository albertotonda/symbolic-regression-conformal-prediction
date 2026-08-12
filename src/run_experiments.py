# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:15:37 2024

@author: Alberto
"""

import matplotlib
matplotlib.use("Agg") # headless batch script, only ever saves figures to file
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import json
import pandas as pd
import pickle
import seaborn as sns
import warnings

from datetime import datetime

from crepes import WrapRegressor
from crepes.extras import MondrianCategorizer, DifficultyEstimator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from pysr import PySRRegressor

from pydantic import BaseModel, Field, field_validator

# local library
from common import (load_and_preprocess_openml_task, plot_confidence_intervals,
                     plot_pareto, translations, loss_function_julia_penalize_smaller)

# regressor models selectable via the "predictor_model" config key or --predictor-model overwrite
REGRESSOR_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor
}

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


class Config(BaseModel):
    """All experiment settings; loaded from a JSON file and overwritable from the CLI."""
    random_seeds: list[int]
    suite_id: int
    tasks_too_good: list[int]
    tasks_too_bad: list[int]
    results_csv_name: str
    confidence_level: float = Field(gt=0)
    max_mondrian_bins: int = Field(gt=1)
    predictor_model: str
    predictor_params: dict # at the moment, params are always those for random forests, might need better solution later
    ncp_knn_k: int = Field(gt=0)
    sr_tournament_selection_n: int = Field(gt=0, default=15)
    sr_population_size: int = Field(ge=12) # must be >= topn (default 12)
    sr_niterations: int = Field(gt=0)
    sr_binary_operators: list[str]
    sr_unary_operators: list[str]

    @field_validator("predictor_model")
    @classmethod
    def predictor_model_is_known(cls, v):
        if v not in REGRESSOR_MODELS:
            raise ValueError("must be one of %s" % list(REGRESSOR_MODELS))
        return v


def load_config(config_path, cli_overrides=None):
    """
    Load the base config from `config_path`, then apply any CLI overrides
    (non-None values) on top; keys not overridden keep the config file's
    value. Raises a pydantic ValidationError if the merged config is invalid.
    """
    print("Loading config...")
    with open(config_path) as fp:
        raw = json.load(fp)

    raw.update({k: v for k, v in (cli_overrides or {}).items() if v is not None})

    return Config(**raw)


def get_benchmark_task_ids(suite_id, tasks_too_good, tasks_too_bad):
    """
    Get the task_ids for all tasks in the benchmark suite, removing the ones
    for which we already know performance is too good or too bad.
    """
    suite = openml.study.get_suite(suite_id)
    task_ids = [t for t in suite.tasks]

    # remove task_ids that for which we had results that are too good or too bad
    task_ids = [t for t in task_ids if t not in tasks_too_bad and t not in tasks_too_good]

    print("After removing data sets with low or high performance, I am left with %d tasks!" % len(task_ids))

    return task_ids


def prepare_task_data(task_id, results_folder, random_seed):
    """
    Download and pre-process a task, split it into training/calibration/test
    sets, and normalize features and target.
    """
    print("Downloading and pre-processing task %d..." % (task_id))
    df_X, df_y, task = load_and_preprocess_openml_task(task_id)

    # get names for features and target
    feature_names = [c for c in df_X.columns]

    # get actual numpy values
    X = df_X.values
    y = df_y.values

    # get dataset name and create task folder
    dataset = task.get_dataset()
    task_folder = os.path.join(results_folder, dataset.name)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    print("Starting work on dataset \"%s\" for task %d..." % (dataset.name, task_id))

    # training/test split and normalization; 50/25/25 split
    X_prop_train, X_test, y_prop_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        shuffle=True, random_state=random_seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                                shuffle=True, random_state=random_seed)

    # even if normalizing is not really necessary, we do it anyways
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_prop_train = scaler_X.fit_transform(X_prop_train)
    X_cal = scaler_X.transform(X_cal)
    X_test = scaler_X.transform(X_test)

    y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1,1)).ravel()
    y_cal = scaler_y.transform(y_cal.reshape(-1,1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1,1)).ravel()

    print("Training set: %d samples" % X_prop_train.shape[0])
    print("Calibration set: %d samples" % X_cal.shape[0])
    print("Test set: %d samples" % X_test.shape[0])

    return (X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test,
            feature_names, dataset, task_folder)


def train_base_regressor(X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test,
                          regressor_class, regressor_params, random_seed):
    """
    Train the base regressor, wrap it as a conformal regressor, and
    calibrate the standard conformal predictor.
    """
    print("Training regressor...")
    regressor = WrapRegressor(regressor_class(random_state=random_seed, **regressor_params))
    regressor.fit(X_prop_train, y_prop_train)

    # get predictions for the test set and calibration set from the learner
    y_cal_pred = regressor.predict(X_cal)
    y_test_pred = regressor.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)

    print("Calibrating conformal regressor...")
    regressor.calibrate(X_cal, y_cal)

    return regressor, y_cal_pred, y_test_pred, r2_test


def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
    """
    Calibrate a normalized conformal regressor using an already-fitted
    DifficultyEstimator. Returns the confidence intervals for the test set,
    together with the difficulty estimates on the calibration and test sets
    (needed later as features for the symbolic regression step).
    """
    sigmas_cal = de.apply(X_cal)

    regressor_norm = WrapRegressor(learner_prop)
    # deprecated:
    # regressor_norm.calibrate(X_cal, y_cal, sigmas=sigmas_cal)
    regressor_norm.calibrate(X_cal, y_cal, de=de)

    sigmas_test = de.apply(X_test)
    intervals = regressor_norm.predict_int(X_test, confidence=confidence)

    return intervals, sigmas_cal, sigmas_test


def compute_mondrian_intervals(learner_prop, de_var, X_prop_train, X_cal, y_cal, X_test, max_bins):
    """
    Calibrate a Mondrian conformal regressor. Mondrian conformal predictors
    only work if there are enough values to bin; but "enough values" is
    dependent on the number of bins, so we iterate, reducing the number of
    bins, until either it works or the number of bins goes down to 1.
    """
    number_of_bins = max_bins
    keep_iterating = True

    while keep_iterating and number_of_bins > 1:

        # capture a warning that can happen during binning
        with warnings.catch_warnings(record=True):
            # this line makes raising warning the same as raising exceptions
            warnings.simplefilter("error")

            try:
                # bins_cal, bin_thresholds = binning(sigmas_cal_var, bins=number_of_bins)
                # regressor_mond = WrapRegressor(learner_prop)
                # regressor_mond.calibrate(X_cal, y_cal, bins=bins_cal)

                # bins_test = binning(sigmas_test_var, bins=bin_thresholds)
                # intervals_mond = regressor_mond.predict_int(X_test, bins=bins_test)

                # keep_iterating = False
                mc = MondrianCategorizer()
                mc.fit(X=X_prop_train, de=de_var, no_bins=number_of_bins)
                regressor_mond = WrapRegressor(learner_prop)
                regressor_mond.calibrate(X_cal, y_cal, mc=mc)
                intervals_mond = regressor_mond.predict_int(X_test)
                keep_iterating = False

            # TODO: check if condition is still valid
            except UserWarning as w:
                print(w)
                print("UserWarning raised, the bins do not contain enough samples, retrying...")
                number_of_bins -= 1

        # check: if the confidence intervals do not contain any '-inf', '+inf'
        # we stop; otherwise, reduce number of bins and iterate
        # TODO: now, this does not work as intended, because some of the bins
        # might be empty (!) so in the conformal set we will have no
        # infinite confidence intervals, but they might appear in the test set;
        # the code below does not work, the proper way of dealing with this
        # is instead to capture the UserWarning as an error, and act
        # accordingly; see the code above for catching UserWarning

        #if np.isfinite(intervals_mond).any() :
        #    keep_iterating = False
        #    print("Found non-infinite confidence intervals for Mondrian conformal predictors for %d bins, stopping" %
        #          number_of_bins)
        #else :
        #    print("Found infinite confidence intervals for Mondrian conformal predictor at %d bins, iterating..."
        #          % number_of_bins)
        #    number_of_bins -= 1

    return intervals_mond, number_of_bins


def run_symbolic_regression(X_cal, X_test, y_cal, y_test, y_cal_pred, y_test_pred,
                             sigmas_cal, sigmas_test,
                             feature_names, task_folder, config, random_seed):
    """
    Train a PySRRegressor to predict the amplitude of a confidence interval
    around the point prediction, using point predictions, difficulty
    estimates, and the original features. Uses a custom Julia loss function
    (loss_function_julia_penalize_smaller, defined in common.py) that
    penalizes intervals that do not cover the true value more heavily than
    it penalizes wide intervals.
    """
    # step 1: prepare data sets with all sigmas and stuff on calibration set
    # and test set; these will be a special version, just for symbolic regression
    X_train_sr = np.zeros((y_cal.shape[0], len(sigmas_cal)+1), dtype=np.float32)
    X_test_sr = np.zeros((y_test.shape[0], len(sigmas_cal)+1), dtype=np.float32)

    # TODO: are sigmas still relevant?

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
    # as a string (lines of Julia), imported from common.py
    ci_regressor = PySRRegressor(
        tournament_selection_n=config.sr_tournament_selection_n,
        population_size=config.sr_population_size, # must be >= topn (default 12)
        niterations=config.sr_niterations,
        binary_operators=config.sr_binary_operators,
        unary_operators=config.sr_unary_operators,
        loss_function=loss_function_julia_penalize_smaller, # defined as a string in common.py
        temp_equation_file=True, # does not clutter directory with temporary files
        verbosity=1, # can also be set to 0, it should be ok
        random_state=random_seed,
        deterministic=True,
        parallelism="serial"
        )

    print("Running symbolic regression...")
    ci_regressor.fit(X_train_sr, y_train_sr)

    print("Now computing confidence intervals for conformal set...")
    ci_amplitude_cal = ci_regressor.predict(X_train_sr)

    print("And computing confidence intervals for test set...")
    ci_amplitude_test = ci_regressor.predict(X_test_sr)
    ci_test = np.zeros((y_test.shape[0], 2))
    for i in range(0, y_test.shape[0]):
        ci_test[i,0] = y_test_pred[i] - ci_amplitude_test[i]
        ci_test[i,1] = y_test_pred[i] + ci_amplitude_test[i]

    # save the predictor as a pickle file
    with open(os.path.join(task_folder, "symbolic_regression_cp.pk"), "wb") as fp:
        pickle.dump(ci_regressor, fp)

    # save the full list of SR feature names for this task (fixed synthetic
    # columns + this dataset's own feature names, in the order used above)
    with open(os.path.join(task_folder, "sr_features.json"), "w") as fp:
        json.dump({"sr_features": list(sigmas_cal.keys()) + feature_names}, fp, indent=2)

    return ci_test


def evaluate_and_plot_method(method, confidence_intervals, y_test, y_test_pred,
                              dataset, task_folder, results_dictionary):
    """
    Compute coverage/amplitude statistics for one set of confidence
    intervals, store them in results_dictionary, and save a plot of the
    intervals for this method.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    # this expression below is a bit of a mess, but it's 1 if the measured
    # value falls within the confidence intervals, and 0 otherwise (summed up, divided by n_samples)
    coverage = np.sum([1 if (y_test[i] >= confidence_intervals[i,0] and
                           y_test[i] <= confidence_intervals[i,1]) else 0
                    for i in range(len(y_test))])/len(y_test)

    # add results to global dictionary of results
    key_mean = method + "_mean"
    key_median = method + "_median"
    key_coverage = method + "_coverage"

    if key_mean not in results_dictionary:
        results_dictionary[key_mean] = []
        results_dictionary[key_median] = []
        results_dictionary[key_coverage] = []

    results_dictionary[key_mean].append(ci_amplitude_mean)
    results_dictionary[key_median].append(ci_amplitude_median)
    results_dictionary[key_coverage].append(coverage)

    # plot time! it would be nice to have a classic plot with CI
    # BUT ALSO a plot Pareto-front style, using (for example)
    # median and coverage; just take a few points
    fig, ax = plot_confidence_intervals(y_test[:20], y_test_pred[:20],
                                        confidence_intervals[:20])

    title = "%s on data set \"%s\" (coverage=%.4f, median=%.2f)" % (translations[method], dataset.name, coverage, ci_amplitude_median)
    ax.set_title(title)

    plt.savefig(os.path.join(task_folder, method + ".png"), dpi=300)
    plt.close(fig)


def run_experiment(config, random_seed = 42):

    # for each dataset
    # - split training/calibration/test
    # - test different conformal predictors
    #   -- regular conformal predictor
    #   -- normalized conformal predictors (N versioni)
    #   -- Mondrian conformal predictors
    # - check for each candidate set of confidence intervals, the tightness and whether the point is inside
    # - Symbolic Regression
    #   -- use as features all statistics computed by normalized and Mondrian
    #   -- plus feature values of the original problem
    #   -- plus (predicted?) value of the target?

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = "results-%d_%s" % (random_seed, timestamp)

    # filtering warnings is usually bad, but here I am getting lots of annoying
    # FutureWarnings on stuff I cannot modify (it's inside other functions), so
    # I am going to filter them
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    # set up plotting
    sns.set_theme(style='darkgrid')

    # get task_id for all tasks in the benchmark suite
    task_ids = get_benchmark_task_ids(config.suite_id, config.tasks_too_good, config.tasks_too_bad)

    # create data structures to store the results
    results_dictionary = {
        'task_id' : [], 'dataset_name' : [], 'r2' : [], 'mondrian_bins' : [],
                          }

    # prepare directory for the results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # save experiment config for tracability
    with open(os.path.join(results_folder, "config.json"), "w") as fp:
            json.dump({**config.model_dump(), "random_seed": random_seed}, fp, indent=2)

    # start the loop, for every task
    for task_index, task_id in enumerate(task_ids):

        # data structure for results related to this task
        task_results = {}

        # placeholders to concatenate all sigmas for symbolic regression
        sigmas_cal = {}
        sigmas_test = {}

        # get the task
        (X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test,
         feature_names, dataset, task_folder) = prepare_task_data(
             task_id, results_folder, random_seed)

        regressor, y_cal_pred, y_test_pred, r2_test = train_base_regressor(
            X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test,
            REGRESSOR_MODELS[config.predictor_model], config.predictor_params, random_seed)

        # Standard CP
        task_results["conformal_predictor"] = regressor.predict_int(X_test, confidence=config.confidence_level)

        # now we need to access the wrapped learner to re-use it for the other
        # conformal predictors, but it's not difficult
        learner_prop = regressor.learner

        # distance of KNN in feature space, default k=25
        print("Normalizing confidence intervals using KNN for difficulty estimation...")
        de_knn = DifficultyEstimator()
        de_knn.fit(X=X_prop_train, k=config.ncp_knn_k, scaler=True)
        (intervals_norm_knn_dist, sigmas_cal_knn_dist, sigmas_test_knn_dist) = compute_normalized_intervals(
            de_knn, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
        
        task_results["normalized_cp_knn_dist"] = intervals_norm_knn_dist
        sigmas_cal["knn_dist"] = sigmas_cal_knn_dist
        sigmas_test["knn_dist"] = sigmas_test_knn_dist

        # standard deviation of KNN in target space
        print("Now normalizing using standard deviations...")
        de_knn_std = DifficultyEstimator()
        de_knn_std.fit(X=X_prop_train, y=y_prop_train, k=config.ncp_knn_k, scaler=True)
        (intervals_norm_knn_std, sigmas_cal_knn_std, sigmas_test_knn_std) = compute_normalized_intervals(
            de_knn_std, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
        
        task_results["normalized_cp_knn_std"] = intervals_norm_knn_std
        sigmas_cal["knn_std"] = sigmas_cal_knn_std
        sigmas_test["knn_std"] = sigmas_test_knn_std

        # a third way of normalizing, using absolute residuals; it does not work
        # for XGBoost, because only Random Forest has out-of-bag predictions for
        # each individual learner...but it's a cool idea! maybe I should go back
        # and pick RandomForest as the estimator
        if config.predictor_model == "RandomForestRegressor":
            print("Now normalizing using OOB predictions of each estimator...")
            oob_predictions = regressor.learner.oob_prediction_
            residuals_prop_oob = y_prop_train - oob_predictions
            de_knn_res = DifficultyEstimator()
            de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, k=config.ncp_knn_k, scaler=True)
            (intervals_norm_knn_res, sigmas_cal_knn_res, sigmas_test_knn_res) = compute_normalized_intervals(
                de_knn_res, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
            
            task_results["normalized_cp_knn_res"] = intervals_norm_knn_res
            sigmas_cal["knn_oob_res"] = sigmas_cal_knn_res
            sigmas_test["knn_oob_res"] = sigmas_test_knn_res

        # a fourth way: using the variance of each element of the ensemble (!)
        # but we need to check whether XGBoost can actually deal with this;
        # update IT CAN'T, because the XGBoostRegressor object does not have
        # the ._regressor part
        if config.predictor_model == "RandomForestRegressor":
            print("Now normalizing using variance of the estimators...")
            de_var = DifficultyEstimator()
            de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
            (intervals_norm_var, sigmas_cal_var, sigmas_test_var) = compute_normalized_intervals(
                de_var, learner_prop, X_cal, y_cal, X_test, config.confidence_level)
            
            task_results["normalized_cp_norm_var"] = intervals_norm_var
            sigmas_cal["ensemble_var"] = sigmas_cal_var
            sigmas_test["ensemble_var"] = sigmas_test_var

        # Mondrian conformal regressor; in the original version, it is using
        # sigmas_cal_var, but for XGBoost I don't have it... :-D
        # so, in the end we ARE switching back to Random Forest
        if config.predictor_model == "RandomForestRegressor":
            print("Now calibrating a Mondrian regressor...")

            # here we might need to perform a few iterations; basically Mondrian
            # conformal predictors only work if there are enough values to bin;
            # but "enough values" is dependent on the number of bins, so we can
            # iterate until either the number of bins goes to 1, or until the
            # size of the confidence intervals is not infinite
            intervals_mond, number_of_bins = compute_mondrian_intervals(
                learner_prop, de_var, X_prop_train, X_cal, y_cal, X_test, config.max_mondrian_bins)

            task_results["mondrian_cp"] = intervals_mond
            results_dictionary["mondrian_bins"].append(number_of_bins)

        # proposed approach: symbolic regression intervals, using all sigmas
        ci_test = run_symbolic_regression(
            X_cal, X_test, y_cal, y_test, y_cal_pred, y_test_pred,
            sigmas_cal, sigmas_test,
            feature_names, task_folder, config, random_seed)

        task_results["symbolic_regression_cp"] = ci_test

        # post-processing of the results for the different confidence intervals
        # statistics we are interested in: coverage, mean size, median size
        for method, confidence_intervals in task_results.items():
            evaluate_and_plot_method(method, confidence_intervals, y_test, y_test_pred,
                                      dataset, task_folder, results_dictionary)

        # add other necessary details for the row in the results dictionary
        results_dictionary["task_id"].append(task_id)
        results_dictionary["dataset_name"].append(dataset.name)
        results_dictionary["r2"].append(r2_test)

        # also plot a Pareto-like scheme
        fig, ax = plot_pareto([k for k in task_results], results_dictionary, translations=translations)
        ax.set_title("Performance of conformal prediction methods on dataset \"%s\"" % dataset.name)
        plt.savefig(os.path.join(task_folder, "pareto.png"), dpi=300)
        plt.close(fig)

        # save global dictionary of results as DataFrame
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, config.results_csv_name), index=False)

    # and now, a global Pareto front plot
    fig, ax = plot_pareto([k for k in task_results], results_dictionary, translations=translations, all_results=True)
    ax.set_title("Performance of conformal prediction methods on selected CTR-23 datasets")
    plt.savefig(os.path.join(results_folder, "pareto.png"), dpi=300)

    # TODO more informative: how many times a conformal predictor is Pareto-optimal?
    # it has to be done data set by data set, and maybe I could write a specific
    # post-processing script

    return


if __name__ == "__main__":
    import argparse
    from typing import get_origin

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default_config.json",
                         help="config file name inside the configs/ directory (default: default_config.json)")

    # one CLI flag per config field, so any setting can be overwritten;
    # unset flags default to None and are ignored by load_config
    for name, model_field in Config.model_fields.items():
        flag = "--" + name.replace("_", "-")
        if name == "predictor_model":
            parser.add_argument(flag, choices=list(REGRESSOR_MODELS))
        elif get_origin(model_field.annotation) in (list, dict):
            parser.add_argument(flag, type=json.loads, metavar="JSON",
                                 help="JSON value, e.g. %s '[1, 2, 3]'" % flag)
        else:
            parser.add_argument(flag, type=model_field.annotation)

    args = vars(parser.parse_args())
    config = load_config(os.path.join(CONFIG_DIR, args.pop("config")), args)

    # let's run several experiments in a row, with different random seeds
    for random_seed in config.random_seeds:
        run_experiment(config, random_seed)
