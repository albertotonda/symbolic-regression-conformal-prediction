# -*- coding: utf-8 -*-
"""
Compare the features used by the SR difficulty equation that PySR's "score"
model selection picks, in a run_sigma_sr.py results folder, with those found
by classical feature selection on the same target, the absolute residuals of
the SR training set.

For each dataset, the SR training data comes from sr_training_data.csv, or
is rebuilt for older runs by downloading the OpenML task and retraining the
base regressor with the run's config and seed (then cached to that file).

Outputs, for each loss:
- <dataset>/feature_selection_<loss>.csv: per-feature scores/ranks/selections
  of every selector, plus which features the SR equation uses
- feature_selection_summary_<loss>.csv: one row per dataset x selector, with
  the SR and selector feature sets and their overlap, and how well the
  equation's sigma ranks held-out calibration residuals (Spearman)
"""

import os
import re
import sys

import argparse
import numpy as np
import pandas as pd

from box import Box
from scipy.stats import spearmanr
import yaml

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.config import build_predictor
from utils.data import load_and_preprocess_openml_task, split_and_normalize_data_with_sr
from utils.feature_selection import equation_features, select_equation, run_selectors, compare_with_equation

SR_TRAINING_FILE = "sr_training_data.csv"


def run_seed(results_folder, config):
    """Seed of a run: the number following results-sigma-sr- in the folder name, else the first config seed."""
    match = re.search(r"results-sigma-sr-(\d+)", os.path.basename(os.path.normpath(results_folder)))
    return int(match.group(1)) if match else config.random_seeds[0]


def rebuild_sr_training_data(task_id, config, random_seed, testing_data=None):
    """
    Recompute the SR training inputs and residuals as run_sigma_sr.py does
    (same split, base regressor and seed). If the run's testing_data.csv is
    given, check the retrained regressor reproduces its test predictions.
    """
    if any(config.data_augmentation.values()):
        raise NotImplementedError("rebuilding SR training data with data augmentation is not supported")

    dataset = load_and_preprocess_openml_task(task_id)
    splits = split_and_normalize_data_with_sr(dataset.df_X, dataset.df_y, config.split, random_seed)
    X_train, y_train = splits["train"]

    # n_jobs doesn't change a seeded forest's predictions
    predictor = build_predictor(config, random_seed)
    if "n_jobs" in predictor.get_params():
        predictor.set_params(n_jobs=-1)
    predictor.fit(X_train, y_train)

    if testing_data is not None:
        y_test_pred = predictor.predict(splits["test"][0])
        if not np.allclose(y_test_pred, testing_data["y_pred"].to_numpy(), atol=1e-6):
            print(f"  WARNING: rebuilt regressor does not reproduce the run's test predictions")

    if config.split.sr_train == 0:
        X_sr = X_train
        residuals = y_train - predictor.oob_prediction_
    else:
        X_sr, y_sr = splits["sr_train"]
        residuals = y_sr - predictor.predict(X_sr)

    df = pd.DataFrame(X_sr, columns=[f"x{j}" for j in range(X_sr.shape[1])])
    df["residual"] = residuals
    return df


def target_suffix(target):
    """Output file suffix: none for the default |residual| target."""
    return "" if target == "abs" else f"_{target}"


def load_sr_training_data(task_folder, task_id, config, random_seed):
    path = os.path.join(task_folder, SR_TRAINING_FILE)
    if os.path.exists(path):
        return pd.read_csv(path)
    print("  rebuilding SR training data...")
    testing_path = os.path.join(task_folder, "testing_data.csv")
    testing_data = pd.read_csv(testing_path) if os.path.exists(testing_path) else None
    df = rebuild_sr_training_data(task_id, config, random_seed, testing_data)
    df.to_csv(path, index=False)
    return df


def analyse_dataset(task_folder, df_sr, loss_names, random_seed, max_samples, rfe_step, sfs_tol, target="abs"):
    """Run the selectors once, then compare them with each loss's "score" equation.
    The selectors' target is |residual| ("abs") or log|residual| ("log_abs", the SR's own target)."""
    X = df_sr.drop(columns="residual").to_numpy()
    y = np.abs(df_sr["residual"].to_numpy())
    if target == "log_abs":
        y = np.log(y + 1e-8)
    if max_samples and len(X) > max_samples:
        idx = np.random.default_rng(random_seed).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    df_selection = run_selectors(X, y, random_seed, rfe_step=rfe_step, sfs_tol=sfs_tol)

    summaries = {}
    for loss_name in loss_names:
        df_hof = pd.read_csv(os.path.join(task_folder, f"hof_{loss_name}.csv"))
        eq = df_hof.loc[select_equation(df_hof, "score")]
        sr_features = equation_features(eq["Equation"])

        df_features = df_selection.copy()
        df_features["sr_used"] = df_features.index.isin(sr_features)
        df_features.to_csv(os.path.join(task_folder, f"feature_selection_{loss_name}{target_suffix(target)}.csv"), index=False)

        # rank agreement between the equation's sigma and |residual| on held-out calibration data
        sigmas_cal = pd.read_csv(os.path.join(task_folder, f"hof_sigmas_cal_{loss_name}.csv"))[str(eq["Complexity"])]
        residuals_cal = pd.read_csv(os.path.join(task_folder, "calibration_data.csv"))["residuals"]
        eq_spearman_cal = spearmanr(sigmas_cal, np.abs(residuals_cal)).statistic

        df_cmp = compare_with_equation(df_selection, sr_features)
        df_cmp.insert(0, "complexity", eq["Complexity"])
        df_cmp.insert(1, "equation", eq["Equation"])
        df_cmp.insert(2, "eq_spearman_cal", eq_spearman_cal)
        summaries[loss_name] = df_cmp
    return summaries


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_folder", help="results folder produced by run_sigma_sr.py")
    parser.add_argument("--seed", type=int, default=None, help="run seed (default: parsed from the folder name)")
    parser.add_argument("--datasets", nargs="*", default=None, help="dataset names to analyse (default: all)")
    parser.add_argument("--min-r2", type=float, default=0.3, help="skip datasets whose base regressor test R² is below this")
    parser.add_argument("--max-r2", type=float, default=0.99, help="skip datasets whose base regressor test R² is above this")
    parser.add_argument("--max-samples", type=int, default=0, help="subsample size for selectors, 0 (default) for all points")
    parser.add_argument("--rfe-step", type=int, default=1, help="features removed per RFE iteration")
    parser.add_argument("--resume", action="store_true", help="keep the existing summaries and skip datasets already in them")
    parser.add_argument("--sfs-tol", type=float, default=1e-3, help="minimum CV R² gain to add a feature in forward selection")
    parser.add_argument("--target", choices=["abs", "log_abs"], default="abs",
                        help="selectors' target: |residual| (default) or log|residual|; log_abs outputs get a _log_abs suffix")
    args = parser.parse_args()

    with open(os.path.join(args.results_folder, "config.yaml")) as f:
        config = Box(yaml.safe_load(f))
    random_seed = args.seed if args.seed is not None else run_seed(args.results_folder, config)
    df_results = pd.read_csv(os.path.join(args.results_folder, "results.csv"))

    def summary_path(loss_name):
        return os.path.join(args.results_folder, f"feature_selection_summary_{loss_name}{target_suffix(args.target)}.csv")

    # a dataset is done once its rows are in the summary of every loss
    all_summaries = {loss_name: [] for loss_name in config.loss_functions}
    done = set()
    if args.resume:
        done = set(df_results["dataset_name"])
        for loss_name in config.loss_functions:
            if os.path.exists(summary_path(loss_name)):
                df_existing = pd.read_csv(summary_path(loss_name))
                all_summaries[loss_name].append(df_existing)
                done &= set(df_existing["dataset_name"])
            else:
                done = set()

    for task_id, name, r2 in zip(df_results["task_id"], df_results["dataset_name"], df_results["r2"]):
        if args.datasets and name not in args.datasets:
            continue
        # residuals carry little difficulty information when the base regressor is near-perfect or poor
        if not args.min_r2 <= r2 <= args.max_r2:
            print(f"{name}: skipped (R² = {r2:.3f})")
            continue
        if name in done:
            print(f"{name}: already done")
            continue
        print(f"{name} (task {task_id}, seed {random_seed})")
        task_folder = os.path.join(args.results_folder, name)

        df_sr = load_sr_training_data(task_folder, task_id, config, random_seed)
        summaries = analyse_dataset(
            task_folder, df_sr, config.loss_functions, random_seed, args.max_samples, args.rfe_step, args.sfs_tol,
            target=args.target,
        )

        # rewrite the summaries after each dataset so partial results survive an interruption
        for loss_name, df_summary in summaries.items():
            df_summary.insert(0, "dataset_name", name)
            df_summary.insert(1, "r2", r2)
            df_summary.insert(2, "n_features", df_sr.shape[1] - 1)
            all_summaries[loss_name] = [
                df[df["dataset_name"] != name] for df in all_summaries[loss_name]
            ] + [df_summary]
            pd.concat(all_summaries[loss_name], ignore_index=True).to_csv(summary_path(loss_name), index=False)
