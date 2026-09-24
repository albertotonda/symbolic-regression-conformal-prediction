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


def test_random_directions_are_unit_and_reproducible():
    a = cc.random_directions(5, 20, seed=3)
    b = cc.random_directions(5, 20, seed=3)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)
    np.testing.assert_array_equal(a, b)


def test_split_halves_is_a_half_split():
    mask = cc.split_halves(101, seed=1)
    assert mask.sum() == 50


def test_worst_slab_finds_planted_miscovered_region():
    # coverage fails only where feature 0 is high; feature 1 is irrelevant
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 2))
    covered = ~((X[:, 0] > 0.8) & (rng.random(4000) < 0.6))
    slab = cc.worst_slabs(X, {"m": covered}, delta=0.2, n_directions=200, seed=0)["m"]

    assert abs(slab.direction[0]) > 0.9  # direction aligned with feature 0
    assert slab.heldout_coverage < 0.8  # well below marginal (~0.87)
    assert slab.heldout_size >= 0.15 * 2000


def test_worst_slab_is_near_marginal_for_uniform_miscoverage():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4000, 3))
    covered = rng.random(4000) < 0.9
    slab = cc.worst_slabs(X, {"m": covered}, delta=0.2, n_directions=200, seed=0)["m"]
    # held-out coverage of a searched slab on pure noise stays near 0.9
    assert abs(slab.heldout_coverage - 0.9) < 0.04
    # the search half is optimistic (lower), held-out corrects it
    assert slab.search_coverage <= slab.heldout_coverage + 0.02


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


def test_right_sizing_on_oracle_width_follows_diagonal():
    rng = np.random.default_rng(5)
    sigma = np.exp(rng.uniform(-1, 1, 20000))
    abs_residual = np.abs(rng.standard_normal(20000)) * sigma
    width = 2 * 1.96 * sigma  # oracle 95% interval for Gaussian noise
    half, q, sizes = cc.right_sizing_bins(width, abs_residual, 0.95, n_bins=10)
    assert len(half) == 10 and sizes.sum() == 20000
    np.testing.assert_allclose(q, half, rtol=0.1)


def test_right_sizing_groups_few_distinct_widths_by_value():
    width = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    half, q, sizes = cc.right_sizing_bins(width, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.5)
    np.testing.assert_array_equal(half, [0.5, 1.0])
    np.testing.assert_array_equal(sizes, [2, 3])


def test_right_sizing_treats_float_noise_as_one_width():
    width = 2.0 + np.array([0.0, 1e-12, -1e-12, 2e-12])
    half, q, sizes = cc.right_sizing_bins(width, np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert len(half) == 1 and sizes[0] == 4
