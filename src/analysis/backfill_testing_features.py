# -*- coding: utf-8 -*-
"""
Rebuild `testing_features.csv` (normalized test-set features) for runs made
before `run_sigma_sr.py` saved it. Reloads each dataset from OpenML and
redoes the same seeded train/calibration/test split, then checks the
rebuilt test targets against the run's own `testing_data.csv` before
writing, so a mismatched split is skipped instead of saved.

Usage:
    uv run src/analysis/backfill_testing_features.py results-sigma-sr-full
    uv run src/analysis/backfill_testing_features.py <run> --seed 42 --overwrite
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.data import load_and_preprocess_openml_task, split_and_normalize_data  # noqa: E402


def run_seed(run_path: Path, seed_arg):
    if seed_arg is not None:
        return seed_arg
    with open(run_path / "config.yaml") as f:
        seeds = yaml.safe_load(f)["random_seeds"]
    if len(seeds) != 1:
        raise SystemExit(f"config.yaml lists several seeds {seeds}; pass the run's seed with --seed")
    return seeds[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    seed = run_seed(run_path, args.seed)
    results = pd.read_csv(run_path / "results.csv")

    for task_id, name in zip(results["task_id"], results["dataset_name"]):
        dataset_dir = run_path / name
        out_path = dataset_dir / "testing_features.csv"
        testing_path = dataset_dir / "testing_data.csv"
        if out_path.exists() and not args.overwrite:
            print(f"{name}: already present, skipped")
            continue
        if not testing_path.exists():
            print(f"{name}: no testing_data.csv, skipped")
            continue

        dataset = load_and_preprocess_openml_task(int(task_id))
        _, _, X_test, _, _, y_test = split_and_normalize_data(dataset.df_X, dataset.df_y, seed)

        y_saved = pd.read_csv(testing_path, index_col="index")["y"].to_numpy()
        if len(y_saved) != len(y_test) or not np.allclose(y_saved, y_test):
            print(f"{name}: rebuilt split does not match testing_data.csv, skipped")
            continue

        pd.DataFrame(X_test, columns=dataset.df_X.columns).to_csv(out_path, index_label="index")
        print(f"{name}: wrote {X_test.shape[0]} x {X_test.shape[1]}")


if __name__ == "__main__":
    main()
