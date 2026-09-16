import os
import sys

os.environ["PYSR_RECORDER"] = "true"
os.environ["PYSR_RECORDER_FILE"] = "history.jsonl"

import argparse
import numpy as np
import pandas as pd
import sympy

from collections import defaultdict

from crepes import WrapRegressor, ConformalRegressor
from crepes.extras import DifficultyEstimator, binning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from pysr import PySRRegressor
from pysr.julia_import import SymbolicRegression, jl
from pysr.julia_helpers import jl_array

import openml

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.utils import setup_results_folder
from utils.data import load_and_preprocess_openml_task, split_and_normalize_data
from utils.evaluate import compute_ci_stats, compute_binned_ci_stats
from utils.plotting import (
    plot_confidence_intervals, plot_pareto, plot_target_distribution, plot_pareto_fronts,
    plot_binned_sigma_metric, plot_sigma_distributions,
    plot_equation_performance_vs_complexity, plot_sigma_vs_residuals,
)
from utils.cp_methods import fit_difficulty_estimator, compute_normalized_intervals, find_bin_thresholds_with_min_size
from utils.losses import bin_crossfit_loss_julia
from utils.config import load_config, dump_config

# Keys must match custom operator names in sr_params.unary_operators.
SIGMA_SR_EXTRA_SYMPY_MAPPINGS = {
    "logm": lambda x: sympy.log(sympy.Abs(x) + 1e-8),
}

def extract_all_equations(model) -> pd.DataFrame:
    """Every individual in the final population(s) of a fitted PySRRegressor,
    as a flat DataFrame with `output_index`, `complexity`, `loss`, `cost`, `equation`.
    """
    populations, _hof = model.julia_state_
    options = model.julia_options_
    variable_names = jl_array([str(v) for v in model.feature_names_in_])

    nout = getattr(model, "nout_", 1)
    rows = []
    for j in range(nout):
        for pop in populations[j]:
            for member in pop.members:
                rows.append(
                    {
                        "output_index": j,
                        "complexity": int(
                            SymbolicRegression.compute_complexity(member, options)
                        ),
                        "loss": float(member.loss),
                        "cost": float(member.cost),
                        "equation": str(
                            SymbolicRegression.string_tree(
                                member.tree, options, variable_names=variable_names
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compute_pareto_fronts(df: pd.DataFrame, n_fronts: int = 1) -> list:
    """Peel successive Pareto fronts out of a DataFrame of equations
    (needs `complexity`, `loss`, `cost` columns -- one output at a time).

    Matches SymbolicRegression.jl's own dominance rule in pure pandas: the
    best-cost individual at each complexity, then only those whose loss beats
    every smaller-complexity survivor (front 1); peeling those off and
    repeating gives front 2, 3, ...
    """
    remaining = (
        df.loc[df.groupby("complexity")["cost"].idxmin()]
        .sort_values("complexity")
        .reset_index(drop=True)
    )

    fronts = []
    for _ in range(n_fronts):
        if remaining.empty:
            break
        running_min = remaining["loss"].shift().cummin().fillna(np.inf)
        on_front = remaining["loss"] < running_min
        fronts.append(remaining[on_front].reset_index(drop=True))
        remaining = remaining[~on_front]
    return fronts


def run_single_task(dataset, task_folder, config, random_seed):

    X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test = split_and_normalize_data(dataset.df_X, dataset.df_y, random_seed)

    # crepes.WrapRegressor wraps any sklearn-compatible regressor and adds
    # conformal prediction methods (.calibrate() and .predict_int()).
    print("Training base regressor...")
    base_regressor = WrapRegressor(
        RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed)
    )
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')

    learner_prop = base_regressor.learner
    sigmas_train_oob = {}
    sigmas_cal_oob = {}
    sigmas_test_oob = {}
    sigmas_comp = {} # sigmas on test set for comparison
    conf_intervals = {}

    # Standard CP
    base_regressor.calibrate(X_cal, y_cal)
    sigmas_comp["standard_cp"] = np.ones(len(X_test))
    conf_intervals["standard_cp"] = base_regressor.predict_int(X_test, confidence=config.confidence)

    # KNN distance
    # de.apply(X) on a real X doesn't depend on the oob flag for KNN-based methods, so a
    # single fit serves both compute_normalized_intervals and augmentation.
    augment_knn_dist = config.data_augmentation.sigma_knn_dist
    de_knn_dist = fit_difficulty_estimator(X_prop_train, "knn_dist", oob=augment_knn_dist)
    conf_intervals["knn_dist"], sigma_cal_knn, sigmas_comp["knn_dist"] = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, config.confidence)
    if augment_knn_dist:
        sigmas_train_oob["knn_dist"] = de_knn_dist.apply()
        sigmas_cal_oob["knn_dist"] = sigma_cal_knn
        sigmas_test_oob["knn_dist"] = sigmas_comp["knn_dist"]

    # KNN std
    augment_knn_std = config.data_augmentation.sigma_knn_std
    de_knn_std = fit_difficulty_estimator(X_prop_train, "knn_std", y_prop_train=y_prop_train, oob=augment_knn_std)
    conf_intervals["knn_std"], sigma_cal_std, sigmas_comp["knn_std"] = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, config.confidence)
    if augment_knn_std:
        sigmas_train_oob["knn_std"] = de_knn_std.apply()
        sigmas_cal_oob["knn_std"] = sigma_cal_std
        sigmas_test_oob["knn_std"] = sigmas_comp["knn_std"]

    # KNN out-of-bag residuals
    augment_knn_res = config.data_augmentation.sigma_knn_res
    de_knn_res = fit_difficulty_estimator(X_prop_train, "knn_res", y_prop_train=y_prop_train, learner_prop=learner_prop, oob=augment_knn_res)
    conf_intervals["knn_res"], sigma_cal_res, sigmas_comp["knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, config.confidence)
    if augment_knn_res:
        sigmas_train_oob["knn_res"] = de_knn_res.apply()
        sigmas_cal_oob["knn_res"] = sigma_cal_res
        sigmas_test_oob["knn_res"] = sigmas_comp["knn_res"]

    # Random Forest variance
    # unlike the KNN estimators above, de.apply(X) for the variance estimator
    # DOES depend on the oob flag (the oob branch expects X sized to the
    # training set), so cal/test must keep using the separate non-oob fit.
    de_var = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop)
    conf_intervals["var"], sigma_cal_var, sigmas_comp["var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, config.confidence)
    if config.data_augmentation.sigma_var:
        de_var_oob = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop, oob=True)
        sigmas_train_oob["var"] = de_var_oob.apply()
        sigmas_cal_oob["var"] = sigma_cal_var
        sigmas_test_oob["var"] = sigmas_comp["var"]

    # Mondrian CP using variance
    min_points = int(1 / (1-config.confidence) - 1) + 1
    bin_thresholds = find_bin_thresholds_with_min_size(sigma_cal_var, min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate()/predict_int() internally takes X as
    # its only parameter; reuse the variance sigmas already computed above
    # for X_cal and X_test instead of recomputing a full RF-variance pass
    # over them.
    def mondrian_categories(X):
        sigmas = de_var.apply(X)
        return binning(sigmas, bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas_comp["mondrian_cp"] = np.ones(len(X_test))
    conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=config.confidence)

    # augment input using sigmas
    X_train_sr = np.zeros((X_prop_train.shape[0], len(sigmas_train_oob)), dtype=np.float32)
    X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_cal_oob)), dtype=np.float32)
    X_test_sr = np.zeros((X_test.shape[0], len(sigmas_test_oob)), dtype=np.float32)
    for i, key in enumerate(sigmas_train_oob.keys()):
            X_train_sr[:,i] = sigmas_train_oob[key]
            X_cal_sr[:,i] = sigmas_cal_oob[key]
            X_test_sr[:,i] = sigmas_test_oob[key]
    X_train_sr = np.concatenate((X_prop_train, X_train_sr), axis=1)
    X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
    X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    abs_res_test = np.abs(y_test_pred - y_test)

    sigma_losses = {
        "bin_crossfit": (dict(loss_function=bin_crossfit_loss_julia(config.confidence, config.lambda_cov)), residuals_prop_oob),
    }

    for loss_name in config.loss_functions:
        loss_kwargs, y_train_sr = sigma_losses[loss_name]
        sigma_predictor = PySRRegressor(
            model_selection="score",
            tournament_selection_n=15,
            populations=config.sr_params.npopulations, # default 31
            population_size=config.sr_params.population_size, # must be >= topn:=12 (default 27)
            niterations=config.sr_params.niterations, # default 100
            binary_operators=config.sr_params.binary_operators,
            unary_operators=config.sr_params.unary_operators,
            nested_constraints=config.sr_params.nested_constraints,
            extra_sympy_mappings=SIGMA_SR_EXTRA_SYMPY_MAPPINGS,
            verbosity=1,
            random_state=random_seed,
            output_directory=task_folder,
            run_id="checkpoints",
            tempdir=task_folder,
            **loss_kwargs,
        )

        sigma_predictor.fit(X_train_sr, y_train_sr)

        # Hall of Fame equations
        df_hof = pd.read_csv(sigma_predictor.get_equation_file())
        df_hof["Chosen"] = (df_hof.index == sigma_predictor.get_best().name) # get_best method returns best equation according to model_selection parameter
        df_hof["sigmas"] = pd.Series([None] * len(df_hof), index=df_hof.index, dtype=object)
        df_hof.set_index("Complexity", inplace=True)

        equation_binned_stats = {}
        for idx in range(len(df_hof)):
            def sigma_f(X, idx=idx):
                # equation predicts log(sigma); an equation using its own "exp"
                # node can overflow once exponentiated here, so clip in
                # log-space first.
                log_sigma = sigma_predictor.equations_["lambda_format"][idx](X)
                log_sigma = np.nan_to_num(log_sigma, nan=50.0)
                return np.exp(np.clip(log_sigma, -50.0, 50.0))

            de = DifficultyEstimator()
            de.fit(X_train_sr, f=sigma_f)
            sigmas_cal_sr = de.apply(X_cal_sr)
            sigmas_test_sr = de.apply(X_test_sr)

            # WrapRegressor.calibrate()/.predict_int() feed the SAME X to both the
            # wrapped learner (needs the raw data) and de.apply() (needs the
            # sigma-augmented columns) — incompatible here, so calibrate manually via
            # the lower-level ConformalRegressor instead.
            cr = ConformalRegressor()
            cr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)
    
            ci_intervals = cr.predict_int(
                learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=config.confidence
            )

            if df_hof.iloc[idx]["Chosen"]:
                conf_intervals[f"sr_{loss_name}"] = ci_intervals
                sigmas_comp[f"sr_{loss_name}"] = sigmas_test_sr

            ci_mean, ci_median, coverage = compute_ci_stats(ci_intervals, y_test)
            row_label = df_hof.index[idx]
            df_hof.loc[row_label, "ci_median"] = ci_median
            df_hof.loc[row_label, "ci_mean"] = ci_mean
            df_hof.loc[row_label, "coverage"] = coverage
            df_hof.at[row_label, "sigmas"] = list(sigmas_test_sr)
            equation_binned_stats[row_label] = compute_binned_ci_stats(
                sigmas_test_sr, ci_intervals, y_test, min_points, random_seed)

        df_hof.to_csv(sigma_predictor.get_equation_file())

        # Total equations at the end of the evolution (different from HOF!)
        df_equations = extract_all_equations(sigma_predictor)
        df_equations.to_csv(os.path.join(task_folder, "checkpoints", "equations_all.csv"))
        fronts = compute_pareto_fronts(df_equations, n_fronts=3)
        plot_pareto_fronts(fronts, os.path.join(task_folder, "pareto_fronts.png"))

        # complexity vs. coverage/width across every Hall-of-Fame equation for this loss
        plot_equation_performance_vs_complexity(
            df_hof, loss_name, dataset.name,
            os.path.join(task_folder, f"equation_performance_{loss_name}.png"))

        # sigma-binned coverage/width across every Hall-of-Fame equation for this loss
        chosen = [k for k in df_hof.index if df_hof.loc[k]["Chosen"]]
        plot_binned_sigma_metric(
            binned_stats=equation_binned_stats,
            metric="coverage",
            save_path=os.path.join(task_folder, f"equation_binned_sigma_coverage_{loss_name}.png"),
            highlighted_keys=chosen,
            use_complexity=True
            )
        plot_binned_sigma_metric(
            binned_stats=equation_binned_stats,
            metric="median_width",
            save_path=os.path.join(task_folder, f"equation_binned_sigma_median_width_{loss_name}.png"),
            highlighted_keys=chosen,
            use_complexity=True
            )

        # per-equation sigmas vs. base learner's absolute residuals,
        # across every Hall-of-Fame equation for this loss
        plot_sigma_vs_residuals(
            df_hof, abs_res_test, loss_name, dataset.name,
            os.path.join(task_folder, f"sigmas_vs_residuals_{loss_name}.png"))

    ci_means = {}
    ci_medians = {}
    coverages = {}

    # per-method coverage/amplitude stats + CI plot
    for method, intervals in conf_intervals.items():
        ci_means[method], ci_medians[method], coverages[method] = compute_ci_stats(intervals, y_test)
        plot_confidence_intervals(
            method, y_test, y_test_pred, intervals, dataset.name,
            coverages[method], ci_medians[method],
            os.path.join(task_folder, method + ".png"))

    # per-task Pareto plot across all methods computed for this task
    plot_pareto(
        list(conf_intervals.keys()), ci_medians, coverages,
        title=f"Performance of conformal prediction methods on dataset \"{dataset.name}\"",
        save_path=os.path.join(task_folder, "pareto.png"))

    # size-stratified coverage/width, for methods with a real fitted difficulty
    # estimator (sigmas_test_comp is keyed by difficulty-estimator name,
    # mapped to its CP method via SIGMA_TEST_KEY_TO_METHOD)
    binned_stats = {}
    for method, sigma_arr in sigmas_comp.items():
        if method in conf_intervals:
            binned_stats[method] = compute_binned_ci_stats(
                sigma_arr, conf_intervals[method], y_test, min_points, random_seed)
    if binned_stats:
        plot_binned_sigma_metric(
            binned_stats=binned_stats,
            metric="coverage",
            save_path=os.path.join(task_folder, "method_binned_sigma_coverage.png"),
            )
        plot_binned_sigma_metric(
            binned_stats=binned_stats,
            metric="median_width",
            save_path=os.path.join(task_folder, "method_binned_sigma_median_width.png"),
            )

    # marginal difficulty-score distributions
    plot_sigma_distributions(
        sigmas_comp, f"Difficulty-score distributions on \"{dataset.name}\"",
        os.path.join(task_folder, "sigma_distributions.png")
        )

    return ci_means, ci_medians, coverages, r2


def iter_datasets(config):
    """
    Yield one preprocessed Dataset at a time, so tasks are downloaded and
    run one-by-one instead of buffering the whole suite in memory upfront.
    """
    if config.data_source == "openml":
        suite = openml.study.get_suite(config.openml_params.suite_id)
        for task_id in suite.tasks:
            yield load_and_preprocess_openml_task(task_id)

    elif config.data_source == "synthetic":
        # TODO: implement synthetic dataset
        print("Synthetic dataset not implemented yet")


def run_all_tasks(config, random_seed):

    results_folder = setup_results_folder("sigma-sr", random_seed)
    results_dictionary = defaultdict(list, {"task_id": [], "dataset_name": [], "r2": []})

    # Save config for tracing
    dump_config(config, results_folder)

    for dataset in iter_datasets(config):
        print(dataset)

        task_folder = os.path.join(results_folder, dataset.name)
        os.makedirs(task_folder, exist_ok=True)

        ci_means, ci_medians, coverages, r2 = run_single_task(dataset, task_folder, config, random_seed)

        results_dictionary["task_id"].append(dataset.id)
        results_dictionary["dataset_name"].append(dataset.name)
        results_dictionary["r2"].append(r2)

        for method in ci_means.keys():
            results_dictionary[f"{method}_mean"].append(ci_means[method])
            results_dictionary[f"{method}_median"].append(ci_medians[method])
            results_dictionary[f"{method}_coverage"].append(coverages[method])

        plot_target_distribution(dataset.df_y.values, dataset.name, os.path.join(task_folder, "target_distribution.png"))

        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, "results.csv"), index=False)

        #TODO: remove for all datasets
        break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, default="default_config")
    args = parser.parse_args()
    config = load_config("sigma-sr", args.config)

    for random_seed in config.random_seeds:
        run_all_tasks(config, random_seed)
