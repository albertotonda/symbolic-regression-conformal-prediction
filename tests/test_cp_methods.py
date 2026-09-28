import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.cp_methods import mondrian_min_bin_size, find_bin_thresholds_with_min_size
from crepes.extras import binning


@pytest.mark.parametrize("confidence, expected", [(0.8, 5), (0.9, 10), (0.95, 20), (0.99, 100)])
def test_mondrian_min_bin_size(confidence, expected):
    assert mondrian_min_bin_size(confidence) == expected


def test_bin_thresholds_respect_min_size():
    sigmas = np.random.default_rng(0).exponential(size=1000)
    min_points = mondrian_min_bin_size(0.95)
    thresholds = find_bin_thresholds_with_min_size(sigmas, min_points, random_seed=42)
    counts = np.bincount(binning(sigmas, bins=thresholds, seed=42).astype(int))
    assert counts.min() >= min_points
    assert len(counts) == 1000 // min_points


def test_normalized_intervals_match_wrap_regressor():
    from crepes import WrapRegressor
    from sklearn.ensemble import RandomForestRegressor
    from utils.cp_methods import fit_difficulty_estimator, compute_normalized_intervals
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] + rng.normal(size=300) * np.abs(X[:, 1])
    learner = RandomForestRegressor(n_estimators=20, random_state=0).fit(X[:100], y[:100])
    de = fit_difficulty_estimator(X[:100], "knn_dist")
    intervals, _, _ = compute_normalized_intervals(de, learner, X[100:200], y[100:200], X[200:], 0.9)
    wrapped = WrapRegressor(learner)
    wrapped.calibrate(X[100:200], y[100:200], de=de)
    assert np.allclose(intervals, wrapped.predict_int(X[200:], confidence=0.9))
