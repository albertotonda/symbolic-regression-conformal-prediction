# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics used to compare conformal prediction methods. Plotting itself
lives in plotting.py.
"""

import numpy as np
import pandas as pd

from crepes.extras import binning

from utils.cp_methods import find_bin_thresholds_with_min_size


def compute_ci_stats(confidence_intervals, y_test):
    """
    Compute mean/median interval amplitude and empirical coverage for one
    set of confidence intervals.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    coverage = np.mean((y_test >= confidence_intervals[:,0]) & (y_test <= confidence_intervals[:,1]))

    return ci_amplitude_mean, ci_amplitude_median, coverage


def compute_binned_ci_stats(sigmas, confidence_intervals, y_test, min_points, random_seed, max_bins=10):
    """
    Bin test points into quantile bins of `sigmas` (a per-point difficulty
    estimate) then compute empirical coverage, mean and median
    interval width within each bin.

    Returns a DataFrame with one row per bin:
    `bin`, `coverage`, `median_width`, `mean_width`, `count`.
    """
    min_points = max(min_points, len(sigmas) // max_bins)
    bin_thresholds = find_bin_thresholds_with_min_size(sigmas, min_points, random_seed)
    assigned_bins = binning(sigmas, bins=bin_thresholds, seed=random_seed).astype(int)

    widths = confidence_intervals[:, 1] - confidence_intervals[:, 0]
    covered = (y_test >= confidence_intervals[:, 0]) & (y_test <= confidence_intervals[:, 1])

    rows = []
    for b in sorted(np.unique(assigned_bins)):
        mask = assigned_bins == b
        rows.append({
            "bin": b,
            "coverage": covered[mask].mean(),
            "median_width": np.median(widths[mask]),
            "mean_width": np.mean(widths[mask]),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)
