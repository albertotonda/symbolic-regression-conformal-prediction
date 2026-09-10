# -*- coding: utf-8 -*-
"""
Experiment configuration: the Config schema, the regressor registry it
validates against, and loading/CLI-parsing helpers built on top of it.
"""

from box import Box
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

def load_config(experiment, config_name = "default_config"):
    with open(os.path.join(CONFIG_DIR, experiment, f"{config_name}.yaml"), "r") as f:
        config = Box(yaml.safe_load(f))

    return config


def dump_config(config, folder):
    config.to_yaml(os.path.join(folder, "config.yaml"))