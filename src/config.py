# -*- coding: utf-8 -*-
"""
Experiment configuration: the Config schema, the regressor registry it
validates against, and loading/CLI-parsing helpers built on top of it.
"""

import argparse
import json
import os

from pydantic import BaseModel, Field, field_validator
from typing import Any, get_origin

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

from xgboost import XGBRegressor

# regressor models selectable via the "predictor_model" config key or --predictor-model overwrite
REGRESSOR_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "SVR": SVR,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso
}

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


class Config(BaseModel):
    """All experiment settings; loaded from a JSON file and overwritable from the CLI."""
    random_seeds: list[int]
    suite_id: int
    tasks_too_good: list[int]
    tasks_too_bad: list[int]
    results_csv_name: str
    confidence_level: float = Field(gt=0)
    predictor_model: str
    predictor_params : dict[str, Any] # at the moment, params in the json are always for random forests, might need better solution later
    ncp_knn_k: int = Field(gt=0)
    sr_tournament_selection_n: int = Field(gt=0, default=15)
    sr_population_size: int = Field(ge=12) # must be >= topn (default 12)
    sr_niterations: int = Field(gt=0)
    sr_binary_operators: list[str]
    sr_unary_operators: list[str]
    use_alt_mondrian: bool = Field(default=False)

    @field_validator("predictor_model")
    @classmethod
    def predictor_model_is_known(cls, v):
        if v not in REGRESSOR_MODELS:
            raise ValueError("must be one of %s" % list(REGRESSOR_MODELS))
        return v


def load_config(config_path, cli_overrides=None):
    """
    Load the base config from `config_path`, then apply any CLI overrides
    (non-None values) on top; keys not overridden keep the config file's
    value. Raises a pydantic ValidationError if the merged config is invalid.
    """
    print("Loading config...")
    with open(config_path) as fp:
        raw = json.load(fp)

    raw.update({k: v for k, v in (cli_overrides or {}).items() if v is not None})

    return Config(**raw)


def parse_cli_config():
    """
    Build a CLI parser with one flag per Config field (so any setting can be
    overwritten), plus --config to pick the config file, then load and
    return the resulting Config. Unset flags default to None and are
    ignored by load_config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default_config.json",
                         help="config file name inside the configs/ directory (default: default_config.json)")

    for name, model_field in Config.model_fields.items():
        flag = "--" + name.replace("_", "-")
        if name == "predictor_model":
            parser.add_argument(flag, choices=list(REGRESSOR_MODELS))
        elif get_origin(model_field.annotation) in (list, dict):
            parser.add_argument(flag, type=json.loads, metavar="JSON",
                                 help="JSON value, e.g. %s '[1, 2, 3]'" % flag)
        else:
            parser.add_argument(flag, type=model_field.annotation)

    args = vars(parser.parse_args())
    return load_config(os.path.join(CONFIG_DIR, args.pop("config")), args)
