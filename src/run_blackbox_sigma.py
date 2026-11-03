"""
Black-box difficulty scores for normalized CP, compared with the SR sigma of
an existing run_sigma_sr.py run (OOB variant, sr_train = 0). For each dataset,
the splits and the base regressor's out-of-bag residuals are rebuilt from the
run's config and seed, then two black-box sigmas are learned on them:

- xgb_quantile: XGBoost with the pinball loss at the run's confidence on
  log|residual|, the SR's own target and loss, deployed as sigma = exp(model)
  like the SR equation. Hyperparameters are picked by 5-fold CV pinball loss.
- rf_abs: random forest on |residual| (squared error), the usual learned sigma
  of normalized CP, min-max scaled by crepes' DifficultyEstimator.

Both are calibrated with normalized CP on the run's calibration set. Every
method of the run (the SR one being its "score" equation) is then scored on
the test set: coverage, median/mean width, Winkler score, width-coverage
correlation and concordance index of the width with |residual|.

The features of the SR equation are also compared with the top-k features of
xgb_quantile by mean |SHAP| (XGBoost's exact TreeSHAP), k = number of SR
features.

Outputs:
- <dataset>/blackbox_sigmas.csv, <dataset>/blackbox_intervals.csv
- <dataset>/shap_xgb_quantile.csv: mean |SHAP| per feature
- blackbox_metrics.csv: one row per dataset x method
- blackbox_shap.csv: one row per dataset, SR vs SHAP features
"""

import os
import re
import sys

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from box import Box

from crepes import ConformalRegressor
from crepes.extras import DifficultyEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV, KFold

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.config import build_predictor
from utils.data import load_and_preprocess_openml_task, split_and_normalize_data_with_sr
from utils.evaluate import concordance_index, score_equation_complexity

SR_TRAINING_FILE = "sr_training_data.csv"
SR_LOSS = "pinball"
XGB_GRID = {
    "max_depth": [2, 4, 6],
    "n_estimators": [100, 300, 600],
    "min_child_weight": [1, 20],
}


def seed_from_folder(results_folder):
    """Seed in a run folder name, results-sigma-sr-<seed>[_<timestamp>|-<suffix>]."""
    match = re.search(r"results-sigma-sr-(\d+)", os.path.basename(os.path.normpath(results_folder)))
    if match is None:
        raise ValueError(f"Can't read the seed from {results_folder}, pass it with --seed")
    return int(match.group(1))


def sr_training_data(task_folder, splits, config, random_seed):
    """
    SR inputs and out-of-bag residuals of the base regressor on the training
    set, from the cached sr_training_data.csv or by retraining the regressor.
    """
    path = os.path.join(task_folder, SR_TRAINING_FILE)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.drop(columns="residual").to_numpy(), df["residual"].to_numpy()

    print("  rebuilding out-of-bag residuals...", flush=True)
    X_train, y_train = splits["train"]
    # n_jobs doesn't change a seeded forest's predictions
    predictor = build_predictor(config, random_seed)
    predictor.set_params(n_jobs=-1)
    predictor.fit(X_train, y_train)
    residuals = y_train - predictor.oob_prediction_

    df = pd.DataFrame(X_train, columns=[f"x{j}" for j in range(X_train.shape[1])])
    df["residual"] = residuals
    df.to_csv(path, index=False)
    return X_train, residuals


def fit_xgb_quantile(X, log_abs_residuals, quantile, random_seed):
    """XGBoost quantile model of log|residual|, tuned by CV pinball loss."""
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile, learning_rate=0.05,
        tree_method="hist", n_jobs=1, random_state=random_seed,
    )
    scorer = make_scorer(mean_pinball_loss, alpha=quantile, greater_is_better=False)
    search = GridSearchCV(model, XGB_GRID, scoring=scorer, cv=KFold(5, shuffle=True, random_state=random_seed), n_jobs=-1)
    search.fit(X, log_abs_residuals)
    return search.best_estimator_, search.best_params_, -search.best_score_


def normalized_intervals(residuals_cal, sigmas_cal, y_test_pred, sigmas_test, confidence):
    regressor = ConformalRegressor()
    regressor.fit(residuals=residuals_cal, sigmas=sigmas_cal)
    return regressor.predict_int(y_hat=y_test_pred, sigmas=sigmas_test, confidence=confidence)


def interval_metrics(y, lower, upper, abs_residuals, alpha):
    """Test-set scores of one method's intervals."""
    width = upper - lower
    covered = (y >= lower) & (y <= upper)
    winkler = width + (2 / alpha) * (np.clip(lower - y, 0, None) + np.clip(y - upper, 0, None))
    # NaN for constant widths (standard CP)
    constant = np.ptp(width) <= 1e-9 * max(np.abs(width).max(), 1.0)
    wc_corr = np.nan if constant or covered.all() else np.corrcoef(width, covered)[0, 1]
    return {
        "coverage": covered.mean(),
        "median_width": np.median(width),
        "mean_width": width.mean(),
        "winkler": winkler.mean(),
        "width_coverage_corr": wc_corr,
        "cindex": concordance_index(width, abs_residuals),
    }


def run_dataset(task_folder, task_id, config, random_seed):
    dataset = load_and_preprocess_openml_task(task_id)
    splits = split_and_normalize_data_with_sr(dataset.df_X, dataset.df_y, config.split, random_seed)
    X_cal, X_test = splits["cal"][0], splits["test"][0]

    df_cal = pd.read_csv(os.path.join(task_folder, "calibration_data.csv"))
    df_test = pd.read_csv(os.path.join(task_folder, "testing_data.csv"))
    # the rebuilt splits must match the run's, checked on the saved targets
    if not (np.allclose(splits["cal"][1], df_cal["y"]) and np.allclose(splits["test"][1], df_test["y"])):
        raise ValueError(f"{dataset.name}: rebuilt splits don't match the run's")

    X_sr, residuals_sr = sr_training_data(task_folder, splits, config, random_seed)
    residuals_cal = df_cal["residuals"].to_numpy()
    y_test, y_test_pred = df_test["y"].to_numpy(), df_test["y_pred"].to_numpy()

    sigmas, intervals = {}, {}

    xgb_model, xgb_params, xgb_cv_loss = fit_xgb_quantile(
        X_sr, np.log(np.abs(residuals_sr) + 1e-8), config.confidence, random_seed
    )
    # same deployment as the SR equation: exp of the predicted log, clipped
    sigmas["xgb_quantile"] = {
        split: np.exp(np.clip(xgb_model.predict(X), -50.0, 50.0)) for split, X in (("cal", X_cal), ("test", X_test))
    }

    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=random_seed).fit(X_sr, np.abs(residuals_sr))
    de_rf = DifficultyEstimator()
    de_rf.fit(X_sr, f=rf.predict, scaler=True)
    sigmas["rf_abs"] = {"cal": de_rf.apply(X_cal), "test": de_rf.apply(X_test)}

    for method, s in sigmas.items():
        intervals[method] = normalized_intervals(residuals_cal, s["cal"], y_test_pred, s["test"], config.confidence)

    pd.concat(
        {method: pd.DataFrame({"split": np.repeat(["cal", "test"], [len(X_cal), len(X_test)]),
                               "sigma": np.concatenate([s["cal"], s["test"]])}) for method, s in sigmas.items()},
        names=["method", "index"],
    ).to_csv(os.path.join(task_folder, "blackbox_sigmas.csv"))
    pd.concat(
        {method: pd.DataFrame(arr, columns=["lower_bound", "upper_bound"]) for method, arr in intervals.items()},
        names=["method", "index"],
    ).to_csv(os.path.join(task_folder, "blackbox_intervals.csv"))

    # the run's methods; the SR one is its "score" equation, like in the paper
    df_run = pd.read_csv(os.path.join(task_folder, "methods_intervals.csv"))
    for method, df in df_run.groupby("method", sort=False):
        if not method.startswith("sr_"):
            intervals[method] = df[["lower_bound", "upper_bound"]].to_numpy()
    df_hof = pd.read_csv(os.path.join(task_folder, f"hof_{SR_LOSS}.csv"), index_col="Complexity")
    complexity = score_equation_complexity(df_hof)
    df_hof_int = pd.read_csv(os.path.join(task_folder, f"hof_intervals_{SR_LOSS}.csv"))
    intervals["sr_score"] = df_hof_int[df_hof_int["complexity"] == complexity][["lower_bound", "upper_bound"]].to_numpy()

    alpha = 1 - config.confidence
    abs_residuals = np.abs(y_test - y_test_pred)
    metrics = [
        {"method": method, **interval_metrics(y_test, arr[:, 0], arr[:, 1], abs_residuals, alpha)}
        for method, arr in intervals.items()
    ]

    # SR features vs top-k mean |SHAP| of the XGBoost quantile model (last column is the bias)
    contribs = xgb_model.get_booster().predict(xgb.DMatrix(X_sr), pred_contribs=True)[:, :-1]
    df_shap = pd.DataFrame({"feature": [f"x{j}" for j in range(X_sr.shape[1])], "mean_abs_shap": np.abs(contribs).mean(axis=0)})
    equation = df_hof.loc[complexity, "Equation"]
    sr_features = sorted(set(re.findall(r"\bx\d+\b", equation)), key=lambda f: int(f[1:]))
    df_shap["sr_used"] = df_shap["feature"].isin(sr_features)
    df_shap.to_csv(os.path.join(task_folder, "shap_xgb_quantile.csv"), index=False)

    k = len(sr_features)
    shap_ranked = df_shap.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()
    shap_row = {
        "sr_complexity": complexity, "equation": equation, "k": k,
        "sr_features": " ".join(sr_features), "shap_features": " ".join(shap_ranked[:k]),
        "n_overlap": len(set(sr_features) & set(shap_ranked[:k])),
        "shap_top1_in_sr": shap_ranked[0] in sr_features if k else np.nan,
        "xgb_cv_pinball": xgb_cv_loss, **{f"xgb_{p}": v for p, v in xgb_params.items()},
    }
    return metrics, shap_row


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--results", required=True, help="run_sigma_sr.py results folder (sr_train = 0)")
    parser.add_argument("--seed", type=int, default=None, help="run seed, read from the folder name by default")
    parser.add_argument("--datasets", nargs="*", default=None, help="dataset names to run (default: all)")
    args = parser.parse_args()

    config = Box.from_yaml(filename=os.path.join(args.results, "config.yaml"))
    if config.split.sr_train != 0:
        raise ValueError("only runs with out-of-bag SR training (sr_train = 0) are supported")
    random_seed = args.seed if args.seed is not None else seed_from_folder(args.results)

    df_results = pd.read_csv(os.path.join(args.results, config.results_csv_name))
    metric_rows, shap_rows = [], []
    for task_id, name, r2 in zip(df_results["task_id"], df_results["dataset_name"], df_results["r2"]):
        if args.datasets and name not in args.datasets:
            continue
        print(f"{name}...", flush=True)
        metrics, shap_row = run_dataset(os.path.join(args.results, name), task_id, config, random_seed)
        metric_rows += [{"dataset_name": name, "r2": r2, **m} for m in metrics]
        shap_rows.append({"dataset_name": name, "r2": r2, **shap_row})

        # rewritten after each dataset so partial results survive an interruption
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.results, "blackbox_metrics.csv"), index=False)
        pd.DataFrame(shap_rows).to_csv(os.path.join(args.results, "blackbox_shap.csv"), index=False)

    df_metrics = pd.DataFrame(metric_rows)
    print(df_metrics.groupby("method")[["coverage", "median_width", "winkler", "cindex"]].median().round(3).to_string())
