# -*- coding: utf-8 -*-
"""
Feature selection on the SR training target (absolute residuals) and
comparison with the features used by Hall-of-Fame equations.

Selectors return either a ranking (one importance score per feature, higher
is more important) or a selected subset. Equations name features x0..x{p-1},
matching the columns of the SR training inputs.
"""

import re

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

RANKING_METHODS = ["spearman", "mutual_info", "permutation_rf"]
SUBSET_METHODS = ["rfecv_rf", "sfs_forward_rf"]


def equation_features(equation):
    """Indices of the x<i> variables used in a PySR equation string."""
    return sorted({int(i) for i in re.findall(r"\bx(\d+)\b", str(equation))})


def select_equation(df_hof, model_selection):
    """
    Index (row label) of the equation PySR would pick with `model_selection`
    ("accuracy", "best" or "score"), from a HoF table with Complexity and
    Loss columns. Mirrors pysr.sr.idx_model_selection / calculate_scores.
    """
    df = df_hof.sort_values("Complexity")
    if model_selection == "accuracy":
        return df["Loss"].idxmin()

    # negated derivative of log-loss with respect to complexity
    loss = df["Loss"].to_numpy(dtype=float)
    complexity = df["Complexity"].to_numpy(dtype=float)
    scores = np.zeros(len(df))
    for i in range(1, len(df)):
        if loss[i] > 0:
            scores[i] = -np.log(loss[i] / loss[i - 1]) / (complexity[i] - complexity[i - 1])
        else:
            scores[i] = np.inf
    scores = pd.Series(scores, index=df.index)

    if model_selection == "score":
        return scores.idxmax()
    if model_selection == "best":
        return scores[df["Loss"] <= 1.5 * df["Loss"].min()].idxmax()
    raise ValueError(f"Unknown model_selection {model_selection}")


def _forest(n_estimators, random_seed, min_samples_leaf=5):
    return RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=random_seed
    )


def spearman_scores(X, y):
    """Absolute Spearman correlation of each feature with y (0 for constant features)."""
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) > 0:
            scores[j] = abs(spearmanr(X[:, j], y).statistic)
    return scores


def mutual_info_scores(X, y, random_seed):
    return mutual_info_regression(X, y, random_state=random_seed)


def permutation_rf_scores(X, y, random_seed, n_estimators=100, cv=5, n_repeats=5):
    """Permutation importance of a random forest, measured on held-out CV folds and averaged."""
    scores = np.zeros(X.shape[1])
    for train_idx, test_idx in KFold(cv, shuffle=True, random_state=random_seed).split(X):
        model = _forest(n_estimators, random_seed).fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            model, X[test_idx], y[test_idx], n_repeats=n_repeats, random_state=random_seed, n_jobs=-1
        )
        scores += result.importances_mean / cv
    return scores


def rfecv_rf(X, y, random_seed, n_estimators=100, cv=5, step=1):
    """Recursive feature elimination with CV-chosen size. Returns (selected mask, ranking)."""
    selector = RFECV(
        _forest(n_estimators, random_seed), step=step, min_features_to_select=1,
        cv=KFold(cv, shuffle=True, random_state=random_seed), scoring="r2",
    ).fit(X, y)
    return selector.support_, selector.ranking_


def sfs_forward_rf(X, y, random_seed, n_estimators=50, cv=5, tol=1e-3):
    """Forward stepwise selection, stopping when the CV R² gain drops below tol. Returns selected mask."""
    selector = SequentialFeatureSelector(
        _forest(n_estimators, random_seed), n_features_to_select="auto", tol=tol, direction="forward",
        cv=KFold(cv, shuffle=True, random_state=random_seed), scoring="r2",
    ).fit(X, y)
    return selector.get_support()


def scores_to_ranks(scores):
    """Rank 1 for the highest score; ties share their average rank."""
    return pd.Series(-np.asarray(scores, dtype=float)).rank(method="average").to_numpy()


def compare_sets(sr_features, method_features):
    """Overlap size and Jaccard index between the SR feature set and a selector's set."""
    sr, method = set(sr_features), set(method_features)
    overlap = len(sr & method)
    union = len(sr | method)
    return {"n_overlap": overlap, "jaccard": overlap / union if union else np.nan}


def format_features(features):
    """Space-separated x<i> names, e.g. "x2 x5"."""
    return " ".join(f"x{j}" for j in sorted(features))


def run_selectors(X, y, random_seed, rfe_step=1, sfs_tol=1e-3):
    """
    Run every selector on (X, y). Returns a per-feature DataFrame with
    <method>_score and <method>_rank columns for ranking methods, and
    <method>_selected (plus rfecv_rf_rank) for subset methods.
    """
    df = pd.DataFrame({"feature": [f"x{j}" for j in range(X.shape[1])]})

    scores = {
        "spearman": spearman_scores(X, y),
        "mutual_info": mutual_info_scores(X, y, random_seed),
        "permutation_rf": permutation_rf_scores(X, y, random_seed),
    }
    for method in RANKING_METHODS:
        df[f"{method}_score"] = scores[method]
        df[f"{method}_rank"] = scores_to_ranks(scores[method])

    support, ranking = rfecv_rf(X, y, random_seed, step=rfe_step)
    df["rfecv_rf_selected"] = support
    df["rfecv_rf_rank"] = ranking
    df["sfs_forward_rf_selected"] = sfs_forward_rf(X, y, random_seed, tol=sfs_tol)
    return df


def compare_with_equation(df_selection, sr_features):
    """
    One row per selector comparing it with the SR feature set. Ranking
    methods are compared through their top-k features, k being the number
    of SR features; subset methods through their own selection.
    """
    k = len(sr_features)
    rows = []
    for method in RANKING_METHODS + SUBSET_METHODS:
        if method in RANKING_METHODS:
            ranks = df_selection[f"{method}_rank"].to_numpy()
            method_features = list(np.argsort(ranks, kind="stable")[:k])
        else:
            method_features = list(np.flatnonzero(df_selection[f"{method}_selected"].to_numpy()))

        row = {"method": method, "k_sr": k, "k_method": len(method_features)}
        row.update(compare_sets(sr_features, method_features))
        row["sr_features"] = format_features(sr_features)
        row["method_features"] = format_features(method_features)
        rows.append(row)
    return pd.DataFrame(rows)
