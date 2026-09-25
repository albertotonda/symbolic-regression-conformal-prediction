# -*- coding: utf-8 -*-
"""
Conditional-coverage helpers shared by the Method Comparison and Hall of
Fame pages: sliding-window coverage along any 1-D ordering, and
worst-slab coverage (Cauchois et al., 2021; Romano et al., 2020).

Worst-slab coverage searches random unit directions v in feature space for
the slab {x : a <= v.x <= b}, holding at least a fraction `delta` of the
points, where coverage is lowest. The search runs on one half of the test
set and the slab's coverage is measured on the other half, so the reported
value isn't optimistically low from searching and scoring on the same
points.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_DIRECTIONS = 1000
N_GRID = 50  # candidate slab endpoints per direction, at evenly spaced ranks
_CHUNK = 100  # directions projected/sorted at once, bounds memory on large test sets


@dataclass
class Slab:
    direction: np.ndarray  # unit vector in feature space
    lower: float  # slab bounds on the projection X @ direction
    upper: float
    search_coverage: float  # coverage on the search half (optimistic)
    heldout_coverage: float  # coverage on the evaluation half
    heldout_size: int  # evaluation-half points inside the slab


def sliding_coverage(order_values, covered, window):
    """Sort points by `order_values`, then return (sorted values, centered
    rolling mean of the coverage indicator)."""
    order_values = np.asarray(order_values)
    order = np.argsort(order_values, kind="stable")
    covered_sorted = np.asarray(covered, dtype=float)[order]
    smoothed = pd.Series(covered_sorted).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return order_values[order], smoothed


def random_directions(n_features, n_directions=N_DIRECTIONS, seed=0):
    """Unit vectors drawn uniformly on the sphere, shape (n_directions, n_features)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_directions, n_features))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def split_halves(n, seed=0):
    """Boolean mask selecting the search half; its complement is the evaluation half."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[: n // 2]] = True
    return mask


def _search(X, covered_by_key, directions, delta):
    """For each key, the lowest-coverage slab over all `directions`, among
    slabs holding at least ceil(delta * n) points. Slabs are rank intervals
    [start, end) of the sorted projection, on a shared grid of ranks.
    Returns {key: (direction index, lower, upper, coverage)}."""
    n = X.shape[0]
    min_size = max(1, int(np.ceil(delta * n)))
    grid = np.unique(np.linspace(0, n, N_GRID + 1).round().astype(int))
    starts, ends = np.meshgrid(grid, grid, indexing="ij")
    valid = (ends - starts) >= min_size
    starts, ends = starts[valid], ends[valid]
    sizes = (ends - starts)[:, None]

    best = {key: (np.inf, None) for key in covered_by_key}
    for c0 in range(0, directions.shape[0], _CHUNK):
        chunk = directions[c0:c0 + _CHUNK]
        projections = X @ chunk.T  # (n, chunk)
        order = np.argsort(projections, axis=0)
        for key, covered in covered_by_key.items():
            cum = np.vstack([np.zeros(chunk.shape[0]), np.cumsum(covered[order], axis=0)])
            cov = (cum[ends] - cum[starts]) / sizes  # (n_candidates, chunk)
            cand, d = np.unravel_index(np.argmin(cov), cov.shape)
            if cov[cand, d] < best[key][0]:
                sorted_proj = projections[order[:, d], d]
                best[key] = (
                    cov[cand, d],
                    (c0 + d, sorted_proj[starts[cand]], sorted_proj[ends[cand] - 1]),
                )
    return {key: (d, lo, hi, cov) for key, (cov, (d, lo, hi)) in best.items()}


def worst_slabs(X, covered_by_key, delta=0.2, n_directions=N_DIRECTIONS, seed=0):
    """Worst-slab coverage for several methods on the same test set.

    `covered_by_key` maps a method (or equation) key to its per-point
    coverage indicator. Every key is searched over the same random
    directions and the same search/evaluation split. Returns {key: Slab}.
    """
    X = np.asarray(X, dtype=float)
    covered_by_key = {k: np.asarray(v, dtype=bool) for k, v in covered_by_key.items()}
    search = split_halves(X.shape[0], seed=seed)

    directions = random_directions(X.shape[1], n_directions, seed=seed)
    found = _search(
        X[search], {k: v[search].astype(float) for k, v in covered_by_key.items()}, directions, delta,
    )

    slabs = {}
    for key, (d, lower, upper, search_cov) in found.items():
        v = directions[d]
        proj_eval = X[~search] @ v
        in_slab = (proj_eval >= lower) & (proj_eval <= upper)
        covered_eval = covered_by_key[key][~search]
        heldout = covered_eval[in_slab].mean() if in_slab.any() else float("nan")
        slabs[key] = Slab(v, float(lower), float(upper), float(search_cov), float(heldout), int(in_slab.sum()))
    return slabs


def worst_bin(order_values, covered, min_fraction=0.2, seed=0):
    """Worst-group coverage over equal-count bins of a 1-D score, each
    holding about `min_fraction` of the points. The worst bin is picked on
    the search half and its coverage reported on the evaluation half (same
    split as `worst_slabs`). Returns (held-out coverage, bin index, n bins),
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
