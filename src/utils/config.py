# -*- coding: utf-8 -*-
"""
Experiment configuration: the Config schema, the regressor registry it
validates against, and loading/CLI-parsing helpers built on top of it.
"""

from box import Box
import yaml
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

def load_config(experiment, config_name = "default_config"):
    with open(os.path.join(CONFIG_DIR, experiment, f"{config_name}.yaml"), "r") as f:
        config = Box(yaml.safe_load(f))

    return config


def dump_config(config, folder):
    config.to_yaml(os.path.join(folder, "config.yaml"))


PREDICTOR_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "SVR": SVR,
    "XGBRegressor": XGBRegressor,
    "LinearRegression": LinearRegression,
}

def validate_sigma_sr_config(config):
    """
    Check the predictor/split combination: sr_train == 0 trains the SR on
    out-of-bag residuals, which only a RandomForestRegressor provides.
    """
    if config.predictor_model not in PREDICTOR_MODELS:
        raise ValueError(f"Unknown predictor_model {config.predictor_model}, expected one of {list(PREDICTOR_MODELS)}")
    if config.split.sr_train == 0 and config.predictor_model != "RandomForestRegressor":
        raise ValueError(f"split.sr_train = 0 (out-of-bag SR training) requires RandomForestRegressor, got {config.predictor_model}")


def build_predictor(config, random_seed):
    """
    Instantiate predictor_model with its predictor_params entry. Out-of-bag
    predictions are enabled when the SR is trained on them (sr_train == 0).
    """
    params = dict(config.predictor_params.get(config.predictor_model) or {})
    predictor = PREDICTOR_MODELS[config.predictor_model](**params)
    if "random_state" in predictor.get_params():
        predictor.set_params(random_state=random_seed)
    if config.split.sr_train == 0:
        predictor.set_params(oob_score=True)
    return predictor
