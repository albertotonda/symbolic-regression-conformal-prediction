# -*- coding: utf-8 -*-
"""
Per-method performance statistics for the Method Comparison page's
Statistics tab: coverage validity, width rank among valid methods, Winkler
(interval) score and width-coverage correlation. y is z-scored per dataset
during preprocessing, so widths and scores are already comparable across
datasets.
"""

import numpy as np
import pandas as pd


def coverage_tolerance(n, alpha):
    """Two binomial standard errors of the empirical coverage of n points
    at target coverage 1 - alpha."""
    return 2 * np.sqrt(alpha * (1 - alpha) / n)


def interval_score(y, lower, upper, alpha):
    """Per-point Winkler score: width plus (2/alpha) times the distance by
    which y falls outside [lower, upper]. Lower is better."""
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    return (upper - lower) + (2 / alpha) * (below + above)


def width_coverage_corr(width, covered):
    """Pearson correlation between interval width and the coverage
    indicator; NaN if either is constant (widths equal up to float
    rounding count as constant). Near 0 means coverage does not depend on
    width, which conditional coverage requires."""
    width = np.asarray(width, dtype=float)
    covered = np.asarray(covered, dtype=float)
    if np.ptp(width) <= 1e-9 * max(np.abs(width).max(), 1.0) or np.ptp(covered) == 0:
        return float("nan")
    return float(np.corrcoef(width, covered)[0, 1])


def dataset_stats(testing, intervals, alpha):
    """One row per method for one dataset.

    `testing` has `y` indexed by test-point index; `intervals` is long
    format (method, index, lower_bound, upper_bound). Methods are valid if
    their coverage is within `coverage_tolerance` of 1 - alpha; only valid
    methods get a `width_rank` (1 = narrowest median width).
    """
    y = testing["y"]
    tolerance = coverage_tolerance(len(y), alpha)

    rows = {}
    for method, group in intervals.groupby("method", sort=False):
        bounds = group.set_index("index").reindex(y.index)
        lower, upper = bounds["lower_bound"].to_numpy(), bounds["upper_bound"].to_numpy()
        width = upper - lower
        covered = (y.to_numpy() >= lower) & (y.to_numpy() <= upper)
        coverage = covered.mean()
        score = interval_score(y, lower, upper, alpha)
        rows[method] = {
            "coverage": coverage,
            "valid": abs(coverage - (1 - alpha)) <= tolerance,
            "median_width": np.median(width),
            "mean_width": width.mean(),
            "winkler": score.mean(),
            "median_winkler": np.median(score),
            "width_coverage_corr": width_coverage_corr(width, covered),
        }

    stats = pd.DataFrame.from_dict(rows, orient="index")
    stats["width_rank"] = stats["median_width"].where(stats["valid"]).rank(method="average")
    return stats


def summarize(per_dataset):
    """Aggregate a long per-(dataset, method) table from `dataset_stats`
    (with a `method` column) into one row per method. Width and Winkler
    columns use the median over datasets, so a single dataset with a few
    huge intervals can't dominate."""
    grouped = per_dataset.groupby("method", sort=False)
    return pd.DataFrame({
        "n_datasets": grouped.size(),
        "n_valid": grouped["valid"].sum(),
        "mean_rank": grouped["width_rank"].mean(),
        "median_rank": grouped["width_rank"].median(),
        "mean_width": grouped["mean_width"].median(),
        "median_width": grouped["median_width"].median(),
        "winkler": grouped["winkler"].median(),
        "median_winkler": grouped["median_winkler"].median(),
        "width_coverage_corr": grouped["width_coverage_corr"].mean(),
    })
