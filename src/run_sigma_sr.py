import os
import sys

os.environ["PYSR_RECORDER"] = "true"
os.environ["PYSR_RECORDER_FILE"] = "history.jsonl"

import argparse
import numpy as np
import pandas as pd

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

from data import load_and_preprocess_openml_task, split_and_normalize_data
from evaluate import evaluate_and_plot_method, log_equations, setup_results_folder
from plotting import save_method_pareto_plot, plot_target_distribution, plot_pareto_fronts
from cp_methods import fit_difficulty_estimator, compute_normalized_intervals, find_bin_thresholds_with_min_size
from losses import bin_crossfit_loss_julia
from config import load_config, dump_config

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
    sigmas_train = {}
    sigmas_cal = {}
    sigmas_test = {}
    sigmas_comp = {}
    conf_intervals = {}

    # Standard CP
    base_regressor.calibrate(X_cal, y_cal)
    sigmas_comp["conformal_predictor"] = np.ones(len(X_cal))
    conf_intervals["conformal_predictor"] = base_regressor.predict_int(X_test, confidence=config.confidence)

    # KNN distance
    # de.apply(X) on a real X doesn't depend on the oob flag for KNN-based methods, so a
    # single fit serves both compute_normalized_intervals and augmentation.
    augment_knn_dist = config.data_augmentation.sigma_knn_dist
    de_knn_dist = fit_difficulty_estimator(X_prop_train, "knn_dist", oob=augment_knn_dist)
    intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, config.confidence)
    conf_intervals["normalized_cp_knn_dist"] = intervals
    sigmas_comp["normalized_cp_knn_dist"] = comp_sigma_cal
    if augment_knn_dist:
        sigmas_train["knn_dist"] = de_knn_dist.apply()
        sigmas_cal["knn_dist"] = comp_sigma_cal
        sigmas_test["knn_dist"] = de_knn_dist.apply(X_test)

    # KNN std
    augment_knn_std = config.data_augmentation.sigma_knn_std
    de_knn_std = fit_difficulty_estimator(X_prop_train, "knn_std", y_prop_train=y_prop_train, oob=augment_knn_std)
    intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, config.confidence)
    conf_intervals["normalized_cp_knn_std"] = intervals
    sigmas_comp["normalized_cp_knn_std"] = comp_sigma_cal
    if augment_knn_std:
        sigmas_train["knn_std"] = de_knn_std.apply()
        sigmas_cal["knn_std"] = comp_sigma_cal
        sigmas_test["knn_std"] = de_knn_std.apply(X_test)

    # KNN out-of-bag residuals
    augment_knn_res = config.data_augmentation.sigma_knn_res
    de_knn_res = fit_difficulty_estimator(X_prop_train, "knn_res", y_prop_train=y_prop_train, learner_prop=learner_prop, oob=augment_knn_res)
    intervals, comp_sigma_cal = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, config.confidence)
    conf_intervals["normalized_cp_knn_res"] = intervals
    sigmas_comp["normalized_cp_knn_res"] = comp_sigma_cal
    if augment_knn_res:
        sigmas_train["knn_res"] = de_knn_res.apply()
        sigmas_cal["knn_res"] = comp_sigma_cal
        sigmas_test["knn_res"] = de_knn_res.apply(X_test)

    # Random Forest variance
    # unlike the KNN estimators above, de.apply(X) for the variance estimator
    # DOES depend on the oob flag (the oob branch expects X sized to the
    # training set), so cal/test must keep using the separate non-oob fit.
    de_var = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop)
    intervals, comp_sigma_cal = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, config.confidence)
    conf_intervals["normalized_cp_norm_var"] = intervals
    sigmas_comp["normalized_cp_norm_var"] = comp_sigma_cal
    if config.data_augmentation.sigma_var:
        de_var_oob = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop, oob=True)
        sigmas_train["var"] = de_var_oob.apply()
        sigmas_cal["var"] = comp_sigma_cal # For cal and test, use default (no oob) version, otherwise same oob trees are used instead of full model
        sigmas_test["var"] = de_var.apply(X_test)

    # Mondrian CP using variance
    min_points = int(1 / (1-config.confidence) - 1) + 1
    bin_thresholds = find_bin_thresholds_with_min_size(sigmas_comp["normalized_cp_norm_var"], min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate()/predict_int() internally takes X as
    # its only parameter; reuse the variance sigmas already computed above
    # for X_cal (and X_test, when the var augmentation ran) instead of
    # recomputing a full RF-variance pass over them.
    sigma_var_cache = {id(X_cal): sigmas_comp["normalized_cp_norm_var"]}
    if "var" in sigmas_test:
        sigma_var_cache[id(X_test)] = sigmas_test["var"]

    def mondrian_categories(X):
        sigmas = sigma_var_cache.get(id(X))
        if sigmas is None:
            sigmas = de_var.apply(X)
        return binning(sigmas, bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas_comp["mondrian_cp"] = np.ones(len(X_cal))
    conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=config.confidence)

    # augment input using sigmas
    X_train_sr = np.zeros((X_prop_train.shape[0], len(sigmas_train)), dtype=np.float32)
    X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_cal)), dtype=np.float32)
    X_test_sr = np.zeros((X_test.shape[0], len(sigmas_test)), dtype=np.float32)
    for i, key in enumerate(sigmas_train.keys()):
            X_train_sr[:,i] = sigmas_train[key]
            X_cal_sr[:,i] = sigmas_cal[key]
            X_test_sr[:,i] = sigmas_test[key]
    X_train_sr = np.concatenate((X_prop_train, X_train_sr), axis=1)
    X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
    X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    y_raw_residual = residuals_prop_oob

    sigma_losses = {
        "bin_crossfit": (dict(loss_function=bin_crossfit_loss_julia(config.confidence)), y_raw_residual),
    }

    for loss_name in config.loss_functions:
        loss_kwargs, y_train_sr = sigma_losses[loss_name]
        sigma_predictor = PySRRegressor(
            model_selection="score",
            tournament_selection_n=15, # default 15
            populations=config.sr_params.npopulations, # default 31
            population_size=config.sr_params.population_size, # must be >= topn:=12 (default 27)
            niterations=config.sr_params.niterations, # default 100
            binary_operators=config.sr_params.binary_operators,
            unary_operators=config.sr_params.unary_operators,
            nested_constraints=config.sr_params.nested_constraints,
            verbosity=1, # can also be set to 0, it should be ok
            random_state=random_seed,
            output_directory=task_folder,
            run_id="checkpoints",
            tempdir=task_folder,
            **loss_kwargs,
        )


        sigma_predictor.fit(X_train_sr, y_train_sr)

        # Hall of Fame equations
        # de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.equations_["lambda_format"][0](X)))

        df_hof = pd.read_csv(sigma_predictor.get_equation_file())
        df_hof["Chosen"] = (df_hof.index == sigma_predictor.get_best().name)
        df_hof.to_csv(sigma_predictor.get_equation_file())

        # Total equations at the end of the evolution (different from HOF!)
        df_equations = extract_all_equations(sigma_predictor)
        df_equations.to_csv(os.path.join(task_folder, "checkpoints", "equations_all.csv"))
        fronts = compute_pareto_fronts(df_equations, n_fronts=3)
        plot_pareto_fronts(fronts, os.path.join(task_folder, "pareto_fronts.png"))

        de_sr = DifficultyEstimator()
        de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.predict(X)), scaler=True)
        sigmas_cal_sr = de_sr.apply(X_cal_sr)
        sigmas_test_sr = de_sr.apply(X_test_sr)

        # WrapRegressor.calibrate()/.predict_int() feed the SAME X to both the
        # wrapped learner (needs the raw data) and de.apply() (needs the
        # sigma-augmented columns) — incompatible here, so calibrate manually via
        # the lower-level ConformalRegressor instead.
        cr_sr = ConformalRegressor()
        cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

        conf_intervals[f"symbolic_regression_{loss_name}"] = cr_sr.predict_int(
            learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=config.confidence
        )
        sigmas_comp[f"symbolic_regression_{loss_name}"] = sigmas_cal_sr

    ci_means = {}
    ci_medians = {}
    coverages = {}

    # per-method CI plot + coverage/amplitude stats
    for method, intervals in conf_intervals.items():
        ci_means[method], ci_medians[method], coverages[method] = evaluate_and_plot_method(method, intervals, y_test, y_test_pred, dataset, task_folder)

    # per-task Pareto plot across all methods computed for this task
    save_method_pareto_plot(
        list(conf_intervals.keys()), ci_medians, coverages,
        title=f"Performance of conformal prediction methods on dataset \"{dataset.name}\"",
        save_path=os.path.join(task_folder, "pareto.png"))

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

        # Plot target distribution
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
