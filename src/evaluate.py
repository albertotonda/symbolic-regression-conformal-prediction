# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics and plots used to compare conformal prediction methods.
"""

import os

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


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


# this is used to translate internal naming convention to readable strings
# for the plots
translations = {
    "conformal_predictor" : "Standard conformal predictor",
    "normalized_cp_knn_dist" : "CP with intervals normalized using KNN on distance",
    "normalized_cp_knn_std" : "CP with intervals normalized using KNN on standard deviation",
    "normalized_cp_knn_res" : "CP with intervals normalized using KNN on OOB residuals",
    "normalized_cp_norm_var" : "CP with intervals normalized using variance of ensemble predictors",
    "mondrian_cp" : "Mondrian CP",
    "symbolic_regression_cp" : "Symbolic Regression CP"
    }

def plot_confidence_intervals(y, y_pred, y_pred_ci) :
    """

    """
    # sort y_test values from small to big, along with y_pred_ci
    # using a list is pretty slow, there is probably a smarter way to do this
    # with numpy arrays, but the data set sizes should be small, so who cares
    y_and_ci = []
    for i in range(0, len(y)) :
        y_and_ci.append([y[i], y_pred[i], y_pred_ci[i]])
    y_and_ci = sorted(y_and_ci, key=lambda x : x[0])

    fig, ax = plt.subplots()#figsize=(10,8))

    # plot measured values and point predictions for y
    x = range(0, len(y))
    ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
    ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")

    # visualize corresponding confidence intervals around point predictions
    ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][1] for x in y_and_ci], color='orange', alpha=0.3)

    ax.set_xlabel("Samples sorted by increasing value of target")
    ax.set_ylabel("Value of target y")
    ax.legend(loc='best')

    return fig, ax

def plot_pareto(methods, results_dictionary, translations=None, all_results=False) :

    fig, ax = plt.subplots(figsize=(10,8))

    for method in methods :

        # get the information related to coverage
        key_coverage = method + "_coverage"
        x = results_dictionary[key_coverage]

        # get information on median (or mean)
        key_median = method + "_median"
        y = results_dictionary[key_median]

        if all_results == False :
            x = x[-1]
            y = y[-1]

        if translations is not None :
            method = translations[method]

        ax.scatter(x, y, label=method)

    # invert x-axis, so that the plot is more readable
    ax.invert_xaxis()

    ax.set_xlabel("coverage on the test set")
    ax.set_ylabel("median amplitude of the confidence intervals")
    ax.legend(loc='best')

    return fig, ax


def evaluate_and_plot_method(method, confidence_intervals, y_test, y_test_pred,
                              dataset, task_folder, results_dictionary):
    """
    Compute coverage/amplitude statistics for one set of confidence
    intervals, store them in results_dictionary (expected to be a
    collections.defaultdict(list)), and save a plot of the intervals for
    this method.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    # this expression below is a bit of a mess, but it's 1 if the measured
    # value falls within the confidence intervals, and 0 otherwise (summed up, divided by n_samples)
    coverage = np.sum([1 if (y_test[i] >= confidence_intervals[i,0] and
                           y_test[i] <= confidence_intervals[i,1]) else 0
                    for i in range(len(y_test))])/len(y_test)

    # add results to global dictionary of results
    results_dictionary[method + "_mean"].append(ci_amplitude_mean)
    results_dictionary[method + "_median"].append(ci_amplitude_median)
    results_dictionary[method + "_coverage"].append(coverage)

    # plot time! it would be nice to have a classic plot with CI
    # BUT ALSO a plot Pareto-front style, using (for example)
    # median and coverage; just take a few points
    fig, ax = plot_confidence_intervals(y_test[:20], y_test_pred[:20],
                                        confidence_intervals[:20])

    title = "%s on data set \"%s\" (coverage=%.4f, median=%.2f)" % (translations[method], dataset.name, coverage, ci_amplitude_median)
    ax.set_title(title)

    plt.savefig(os.path.join(task_folder, method + ".png"), dpi=300)
    plt.close(fig)
