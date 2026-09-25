# -*- coding: utf-8 -*-
"""Unit tests for dashboard/conditional_coverage.py. Run with:
`uv run --no-project --with pytest --with pandas --with numpy pytest dashboard/tests`
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_coverage as cc  # noqa: E402


def test_sliding_coverage_sorts_and_smooths():
    values = np.array([3.0, 1.0, 2.0, 4.0])
    covered = np.array([1, 0, 1, 1])
    sorted_values, smoothed = cc.sliding_coverage(values, covered, window=1)
    np.testing.assert_array_equal(sorted_values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(smoothed, [0.0, 1.0, 1.0, 1.0])


def test_split_halves_is_a_half_split():
    mask = cc.split_halves(101, seed=1)
    assert mask.sum() == 50


def test_worst_bin_finds_miscovered_top_bin():
    rng = np.random.default_rng(4)
    score = rng.random(5000)
    covered = ~((score > 0.8) & (rng.random(5000) < 0.5))
    heldout, worst, n_bins = cc.worst_bin(score, covered, min_fraction=0.2, seed=0)
    assert n_bins == 5
    assert worst == 4
    assert abs(heldout - 0.5) < 0.06


def test_worst_bin_constant_score_is_nan():
    heldout, worst, n_bins = cc.worst_bin(np.ones(100), np.ones(100, dtype=bool))
    assert np.isnan(heldout) and worst is None


def test_decile_ids_orders_by_rank_and_skips_constant():
    ids = cc.decile_ids(np.arange(100)[::-1], n_bins=10)
    assert ids[0] == 9 and ids[-1] == 0
    assert np.bincount(ids).tolist() == [10] * 10
    assert cc.decile_ids(np.ones(50)) is None
