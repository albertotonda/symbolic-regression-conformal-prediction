"""
Compare density/outlier-detection scores with the SR difficulty score of an
existing run_sigma_sr.py run. For each dataset, the splits are rebuilt from
the run's config and seed, the estimators are fit on the base regressor's
training features, and the Spearman correlation between each rarity score
and the sigma of the PySR "score" equation is computed on cal + test.
"""

import os
import re
import sys

import argparse
import numpy as np
import pandas as pd
from box import Box
from scipy.stats import spearmanr

from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, LocalOutlierFactor

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.data import load_and_preprocess_openml_task, split_and_normalize_data_with_sr

DENSITY_METHODS = ("kde", "isolation_forest", "lof", "gmm")
GMM_MAX_COMPONENTS = 10


def score_equation_complexity(hof):
    """
    Complexity of the equation PySR's model_selection="score" picks: the
    largest -d(log loss)/d(complexity) between consecutive HoF entries.
    """
    hof = hof.sort_index()
    loss = hof["Loss"].to_numpy(dtype=float)
    complexity = hof.index.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = -np.log(loss[1:] / loss[:-1]) / np.diff(complexity)
    score = np.where(loss[1:] > 0, score, np.inf)
    score = np.concatenate([[0.0], score])
    return int(hof.index[np.argmax(score)])


def fit_rarity_scorers(X_train, k, random_seed):
    """
    Fit each density/outlier estimator on X_train and return, per method, a
    function mapping X to a rarity score (higher = rarer). Every estimator
    exposes a "normality" score (log density, log likelihood, or anomaly
    score where higher = more normal), so rarity is its negation.
    """
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(X_train)
    iforest = IsolationForest(n_estimators=200, random_state=random_seed).fit(X_train)
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X_train)

    # number of components chosen by BIC; reg_covar keeps covariances
    # invertible on constant or integer-coded features
    gmms = [
        GaussianMixture(n_components=n, reg_covar=1e-4, random_state=random_seed).fit(X_train)
        for n in range(1, GMM_MAX_COMPONENTS + 1)
    ]
    gmm = min(gmms, key=lambda g: g.bic(X_train))

    return {
        "kde": lambda X: -kde.score_samples(X),
        "isolation_forest": lambda X: -iforest.score_samples(X),
        "lof": lambda X: -lof.score_samples(X),
        "gmm": lambda X: -gmm.score_samples(X),
    }


def compare_dataset(task_folder, task_id, config, loss_name, random_seed):
    """
    Rarity scores and SR sigma on cal + test for one dataset, and the
    Spearman correlation of each method's score with the SR sigma.
    """
    dataset = load_and_preprocess_openml_task(task_id)
    splits = split_and_normalize_data_with_sr(dataset.df_X, dataset.df_y, config.split, random_seed)
    X_prop_train, _ = splits["train"]
    X_eval = {"cal": splits["cal"][0], "test": splits["test"][0]}

    # the rebuilt splits must match the run's, checked on the saved targets
    for split, file in (("cal", "calibration_data.csv"), ("test", "testing_data.csv")):
        y_saved = pd.read_csv(os.path.join(task_folder, file))["y"].to_numpy()
        if not np.allclose(splits[split][1], y_saved):
            raise ValueError(f"{dataset.name}: rebuilt {split} split doesn't match the run's")

    hof = pd.read_csv(os.path.join(task_folder, f"hof_{loss_name}.csv"), index_col="Complexity")
    complexity = score_equation_complexity(hof)

    scorers = fit_rarity_scorers(X_prop_train, config.ncp_knn_k, random_seed)

    frames = []
    for split in ("cal", "test"):
        sigmas = pd.read_csv(os.path.join(task_folder, f"hof_sigmas_{split}_{loss_name}.csv"), index_col="index")
        df = pd.DataFrame({"split": split, "sr_sigma": sigmas[str(complexity)].to_numpy()})
        for method, scorer in scorers.items():
            df[method] = scorer(X_eval[split])
        frames.append(df)
    df_scores = pd.concat(frames, ignore_index=True)

    # NaN when the SR sigma is constant (complexity-1 equation)
    spearman = {method: spearmanr(df_scores[method], df_scores["sr_sigma"]).statistic for method in DENSITY_METHODS}
    return df_scores, complexity, spearman


def seed_from_folder(results_folder):
    """Seed in a run folder name, results-sigma-sr-<seed>[_<timestamp>|-<suffix>]."""
    match = re.search(r"results-sigma-sr-(\d+)", os.path.basename(os.path.normpath(results_folder)))
    if match is None:
        raise ValueError(f"Can't read the seed from {results_folder}, pass it with --seed")
    return int(match.group(1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results", required=True, help="run_sigma_sr.py results folder")
    parser.add_argument("--seed", type=int, default=None, help="run seed, read from the folder name by default")
    parser.add_argument("--loss", default=None, help="SR loss to compare, the run's first loss by default")
    args = parser.parse_args()

    config = Box.from_yaml(filename=os.path.join(args.results, "config.yaml"))
    random_seed = args.seed if args.seed is not None else seed_from_folder(args.results)
    loss_name = args.loss or config.loss_functions[0]

    df_results = pd.read_csv(os.path.join(args.results, config.results_csv_name))
    rows = []
    for task_id, dataset_name in zip(df_results["task_id"], df_results["dataset_name"]):
        print(f"{dataset_name}...", flush=True)
        task_folder = os.path.join(args.results, dataset_name)
        df_scores, complexity, spearman = compare_dataset(
            task_folder, task_id, config, loss_name, random_seed
        )
        df_scores.to_csv(os.path.join(task_folder, "density_scores.csv"), index_label="index")
        rows.append({"dataset_name": dataset_name, "sr_complexity": complexity, "n_points": len(df_scores), **spearman})

        df_spearman = pd.DataFrame(rows)
        df_spearman.to_csv(os.path.join(args.results, "density_spearman.csv"), index=False)

    print(df_spearman.round(2).to_string(index=False))
    print("\nMedian Spearman:")
    print(df_spearman[list(DENSITY_METHODS)].median().round(2).to_string())
