# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics used to compare conformal prediction methods. Plotting itself
lives in plotting.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def compute_ci_stats(confidence_intervals, y_test):
    """
    Compute mean/median interval amplitude and empirical coverage for one
    set of confidence intervals.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    coverage = np.mean((y_test >= confidence_intervals[:,0]) & (y_test <= confidence_intervals[:,1]))

    return ci_amplitude_mean, ci_amplitude_median, coverage


def score_equation_complexity(hof):
    """
    Complexity of the equation PySR's model_selection="score" picks: the
    largest -d(log loss)/d(complexity) between consecutive HoF entries.
    """
    hof = hof.sort_index()
    loss = hof["Loss"].to_numpy(dtype=float)
    complexity = hof.index.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = -np.log(loss[1:] / loss[:-1]) / np.diff(complexity)
    score = np.where(loss[1:] > 0, score, np.inf)
    score = np.concatenate([[0.0], score])
    return int(hof.index[np.argmax(score)])


def concordance_index(score, target):
    """
    Harrell's concordance index between a score and a target, over the pairs
    with different targets; pairs tied on the score count 0.5. Computed from
    Kendall's tau-b in O(n log n), as C = 0.5 + (P - Q) / (2 * pairs).
    """
    n_pairs = len(score) * (len(score) - 1) / 2
    score_ties = (pd.Series(score).value_counts() ** 2 - pd.Series(score).value_counts()).sum() / 2
    target_ties = (pd.Series(target).value_counts() ** 2 - pd.Series(target).value_counts()).sum() / 2
    if score_ties == n_pairs:
        return 0.5
    tau_b = kendalltau(score, target).statistic
    concordant_minus_discordant = tau_b * np.sqrt((n_pairs - score_ties) * (n_pairs - target_ties))
    return 0.5 + concordant_minus_discordant / (2 * (n_pairs - target_ties))
