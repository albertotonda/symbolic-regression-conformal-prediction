# -*- coding: utf-8 -*-
"""Tests for the Julia sigma-SR losses in src/utils/losses.py, evaluated
directly in PySR's Julia session on synthetic heteroscedastic data where
the true sigma is known (log sigma = x0). Run with:
`.venv/bin/python -m pytest tests` (first run starts Julia, ~1 min).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.losses import bin_crossfit_loss_julia, pinball_loss_julia  # noqa: E402

N = 4000
CONFIDENCE = 0.95


@pytest.fixture(scope="module")
def julia():
    from pysr import jl
    jl.seval("using SymbolicRegression")
    jl.seval("options = Options(binary_operators=[+, *])")
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1.5, 1.5, N)
    residuals = rng.standard_normal(N) * np.exp(x0)
    jl.X = x0[None, :].astype(np.float64)
    jl.y = residuals.astype(np.float64)
    jl.seval("dataset = Dataset(Array(X), Vector(y))")
    return jl, x0, residuals


def _loss(julia, tree, lambda_cov=500.0, seed=0):
    jl = julia[0]
    jl.seval(bin_crossfit_loss_julia(CONFIDENCE, lambda_cov, seed=seed))
    return float(jl.seval(f"eval_loss({tree}, dataset, options)"))


X0 = "Node{Float64}(feature=1)"
ORACLE = X0
CONSTANT = "Node{Float64}(val=0.0)"
FLATTENED = f"({X0} * 0.01)"  # same ranking as the oracle, almost no spread
SCALED = f"({X0} + {np.log(10)})"  # oracle sigma * 10


def test_oracle_beats_constant(julia):
    assert _loss(julia, ORACLE) < _loss(julia, CONSTANT)


def test_loss_is_invariant_to_sigma_scale(julia):
    assert _loss(julia, SCALED) == pytest.approx(_loss(julia, ORACLE), rel=1e-6)


def test_flattened_sigma_is_not_rescued_by_its_ranking(julia):
    # a per-bin quantile would make this nearly as good as the oracle; with
    # one global quantile its sigma bins follow the true difficulty but its
    # widths barely vary, so the per-bin coverage penalty is large -- worse
    # than a constant sigma, whose bins are arbitrary and so look calibrated
    oracle, flattened, constant = _loss(julia, ORACLE), _loss(julia, FLATTENED), _loss(julia, CONSTANT)
    assert flattened > oracle * 1.5
    assert flattened > constant


def test_width_term_matches_global_normalized_cp(julia):
    # with lambda_cov = 0 the loss is the mean half-width q * sigma
    _, x0, residuals = julia
    sigma = np.exp(x0)
    scores = np.sort(np.abs(residuals) / sigma)
    q = scores[int(np.ceil(CONFIDENCE * (N + 1))) - 1]
    expected = np.mean(q * sigma)
    assert _loss(julia, ORACLE, lambda_cov=0.0) == pytest.approx(expected, rel=0.05)


def test_fold_split_is_fixed_by_seed(julia):
    assert _loss(julia, ORACLE, seed=1) == _loss(julia, ORACLE, seed=1)


def test_pinball_minimizer_is_the_target_quantile(julia):
    # the best constant prediction under pinball loss at level tau is the
    # tau quantile of the targets (checks the sign convention)
    jl, x0, residuals = julia
    jl.seval(pinball_loss_julia(CONFIDENCE))
    jl.t = np.log(np.abs(residuals) + 1e-8)
    grid = np.linspace(-3, 4, 701)
    jl.grid = grid
    losses = np.asarray(jl.seval("[sum(pinball.(c, Vector(t))) for c in Vector(grid)]"))
    expected = np.quantile(np.log(np.abs(residuals) + 1e-8), CONFIDENCE)
    assert abs(grid[np.argmin(losses)] - expected) < 0.02
