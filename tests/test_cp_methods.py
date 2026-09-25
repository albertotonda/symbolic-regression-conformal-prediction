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
