# -*- coding: utf-8 -*-
"""
Turning a set of confidence intervals into the coverage/amplitude
statistics used to compare conformal prediction methods. Plotting itself
lives in plotting.py.
"""

import os

from datetime import datetime

import numpy as np
import pandas as pd

from crepes.extras import binning

from utils.cp_methods import find_bin_thresholds_with_min_size


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
    equations["chosen"] = (equations.index == sr_model.get_best().name)

    print("\n%s: candidate equations (model_selection=%r)" % (label, sr_model.model_selection))
    print(equations[["complexity", "loss", "score", "equation", "chosen"]].to_string(index=False))

    equations.to_csv(os.path.join(task_folder, "%s_equations.csv" % label), index=False)


def compute_ci_stats(confidence_intervals, y_test):
    """
    Compute mean/median interval amplitude and empirical coverage for one
    set of confidence intervals.
    """
    ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
    ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
    coverage = np.mean((y_test >= confidence_intervals[:,0]) & (y_test <= confidence_intervals[:,1]))

    return ci_amplitude_mean, ci_amplitude_median, coverage


def compute_binned_coverage_width(sigmas, confidence_intervals, y_test, min_points, random_seed, max_bins=10):
    """
    Bin test points into quantile bins of `sigmas` (a per-point difficulty
    estimate) then compute empirical coverage and median
    interval width within each bin.

    Returns a DataFrame with one row per bin:
    `bin`, `coverage`, `median_width`, `count`.
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


def melt_results_for_cross_dataset_plot(df_results, methods):
    """
    Reshape the wide per-dataset results table (one row per dataset, with
    `<method>_median`/`<method>_coverage` columns, as written to
    results.csv) into a tidy long DataFrame with one row per
    (dataset, method): `dataset_name`, `method`, `coverage`, `median`. Used
    for a cross-dataset Pareto scatter (plotting.plot_cross_dataset_pareto).
    """
    rows = []
    for _, row in df_results.iterrows():
        for method in methods:
            median_col, coverage_col = f"{method}_median", f"{method}_coverage"
            if median_col not in row or coverage_col not in row:
                continue
            rows.append({
                "dataset_name": row["dataset_name"],
                "method": method,
                "median": row[median_col],
                "coverage": row[coverage_col],
            })
    return pd.DataFrame(rows)
