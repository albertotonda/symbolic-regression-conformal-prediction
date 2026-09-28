# -*- coding: utf-8 -*-
"""
Conditional-coverage helpers shared by the Method Comparison and Hall of
Fame pages: sliding-window coverage along any 1-D ordering, worst-group
coverage over equal-count bins of a score, and decile binning.
"""

import numpy as np
import pandas as pd

def sliding_coverage(order_values, covered, window):
    """Sort points by `order_values`, then return (sorted values, centered
    rolling mean of the coverage indicator)."""
    order_values = np.asarray(order_values)
    order = np.argsort(order_values, kind="stable")
    covered_sorted = np.asarray(covered, dtype=float)[order]
    smoothed = pd.Series(covered_sorted).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return order_values[order], smoothed


def split_halves(n, seed=0):
    """Boolean mask selecting the search half; its complement is the evaluation half."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[: n // 2]] = True
    return mask


def worst_bin(order_values, covered, min_fraction=0.2, seed=0):
    """Worst-group coverage over equal-count bins of a 1-D score, each
    holding about `min_fraction` of the points. The worst bin is picked on
    the search half and its coverage reported on the evaluation half.
    Returns (held-out coverage, bin index, n bins),
    or NaN coverage if the score is constant."""
    order_values = np.asarray(order_values, dtype=float)
    covered = np.asarray(covered, dtype=bool)
    if np.unique(order_values).size <= 1:
        return float("nan"), None, 0

    n_bins = max(1, int(round(1 / min_fraction)))
    ranks = np.argsort(np.argsort(order_values, kind="stable"), kind="stable")
    bin_id = np.minimum(ranks * n_bins // len(ranks), n_bins - 1)
    search = split_halves(len(covered), seed=seed)

    def _bin_cov(mask):
        return np.array([covered[mask & (bin_id == b)].mean() if (mask & (bin_id == b)).any() else np.inf
                         for b in range(n_bins)])

    worst = int(np.argmin(_bin_cov(search)))
    heldout = _bin_cov(~search)[worst]
    return float(heldout) if np.isfinite(heldout) else float("nan"), worst, n_bins


def decile_ids(values, n_bins=10):
    """Equal-count bin index (0 .. n_bins-1) of each point by rank of
    `values`, or None if `values` is constant (ranks would be arbitrary)."""
    values = np.asarray(values, dtype=float)
    if np.unique(values).size <= 1:
        return None
    return pd.qcut(pd.Series(values).rank(method="first"), n_bins, labels=False).to_numpy()
