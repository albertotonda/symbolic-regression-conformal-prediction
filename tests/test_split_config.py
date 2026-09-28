import os
import sys

import numpy as np
import pandas as pd
import pytest
from box import Box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.data import split_and_normalize_data_with_sr
from utils.config import validate_sigma_sr_config, build_predictor


def _data(n=200):
    rng = np.random.default_rng(0)
    df_X = pd.DataFrame(rng.normal(size=(n, 3)))
    df_y = pd.Series(np.arange(n, dtype=float))
    return df_X, df_y


def _config(model, sr_train, params=None):
    return Box({
        "predictor_model": model,
        "predictor_params": params or {},
        "split": {"train": 50, "sr_train": sr_train, "cal": 50 - sr_train - 10, "test": 10},
    })


def test_split_sizes_and_disjoint():
    df_X, df_y = _data()
    splits = split_and_normalize_data_with_sr(df_X, df_y, {"train": 50, "sr_train": 20, "cal": 20, "test": 10}, 42)
    sizes = {name: len(y) for name, (_, y) in splits.items()}
    assert sizes == {"train": 100, "sr_train": 40, "cal": 40, "test": 20}
    # y is a unique id per row, normalized by an affine map, so distinct values mean disjoint splits
    all_y = np.concatenate([y for _, y in splits.values()])
    assert len(np.unique(np.round(all_y, 8))) == 200
    X_train, _ = splits["train"]
    assert np.allclose(X_train.mean(axis=0), 0)


def test_split_zero_sr_train():
    df_X, df_y = _data()
    splits = split_and_normalize_data_with_sr(df_X, df_y, {"train": 50, "sr_train": 0, "cal": 25, "test": 25}, 42)
    assert splits["sr_train"] == (None, None)
    assert len(splits["train"][1]) + len(splits["cal"][1]) + len(splits["test"][1]) == 200


def test_split_must_sum_to_100():
    df_X, df_y = _data()
    with pytest.raises(ValueError):
        split_and_normalize_data_with_sr(df_X, df_y, {"train": 50, "sr_train": 20, "cal": 20, "test": 20}, 42)


@pytest.mark.parametrize("model", ["SVR", "XGBRegressor", "LinearRegression"])
def test_oob_requires_random_forest(model):
    with pytest.raises(ValueError):
        validate_sigma_sr_config(_config(model, sr_train=0))
    validate_sigma_sr_config(_config(model, sr_train=20))


def test_unknown_model():
    with pytest.raises(ValueError):
        validate_sigma_sr_config(_config("Foo", sr_train=20))


def test_build_predictor_oob_and_params():
    params = {"RandomForestRegressor": {"n_estimators": 7}}
    rf = build_predictor(_config("RandomForestRegressor", 0, params), random_seed=3)
    assert rf.oob_score and rf.n_estimators == 7 and rf.random_state == 3
    rf = build_predictor(_config("RandomForestRegressor", 20, params), random_seed=3)
    assert rf.oob_score
    # a model without a predictor_params entry uses its defaults
    build_predictor(_config("SVR", 20, params), random_seed=3)
