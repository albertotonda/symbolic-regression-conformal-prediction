# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics used to compare conformal prediction methods. Plotting itself
lives in plotting.py.
"""

import numpy as np


def compute_ci_stats(confidence_intervals, y_test):
    """
    Compute mean/median interval amplitude and empirical coverage for one
    set of confidence intervals.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    coverage = np.mean((y_test >= confidence_intervals[:,0]) & (y_test <= confidence_intervals[:,1]))

    return ci_amplitude_mean, ci_amplitude_median, coverage
