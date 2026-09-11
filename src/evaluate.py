# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics used to compare conformal prediction methods. Plotting itself
lives in plotting.py.
"""

import os

from datetime import datetime

import numpy as np

from plotting import save_confidence_interval_plot


def setup_results_folder(prefix, random_seed):
    """
    Create (and return the path to) a fresh timestamped results folder,
    named results-<prefix>-<random_seed>_<timestamp>, in the current
    working directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = "results-%s-%d_%s" % (prefix, random_seed, timestamp)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


def log_equations(sr_model, task_folder, label):
    """
    Print the full set of candidate equations found by a fitted
    PySRRegressor (its Pareto front of complexity vs. loss), marking the one
    actually selected per sr_model.model_selection, and save the same table
    as a CSV file (<label>_equations.csv) in task_folder.
    """
    equations = sr_model.equations_.copy()
    chosen_index = sr_model.get_best().name
    equations["chosen"] = [True if idx == chosen_index else False for idx in equations.index]

    print("\n%s: candidate equations (model_selection=%r)" % (label, sr_model.model_selection))
    print(equations[["complexity", "loss", "score", "equation", "chosen"]].to_string(index=False))

    equations.to_csv(os.path.join(task_folder, "%s_equations.csv" % label), index=False)


def evaluate_and_plot_method(method, confidence_intervals, y_test, y_test_pred,
                              dataset, task_folder):
    """
    Compute coverage/amplitude statistics for one set of confidence
    intervals and save a plot of the intervals for this method.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    coverage = np.mean((y_test >= confidence_intervals[:,0]) & (y_test <= confidence_intervals[:,1]))

    save_confidence_interval_plot(
        method, y_test, y_test_pred, confidence_intervals,
        dataset.name, coverage, ci_amplitude_median,
        os.path.join(task_folder, method + ".png"))

    return ci_amplitude_mean, ci_amplitude_median, coverage
