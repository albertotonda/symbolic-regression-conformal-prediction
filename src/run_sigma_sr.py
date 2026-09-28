import os
import sys

import argparse
import numpy as np
import pandas as pd
import sympy

from collections import defaultdict

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, binning

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score

from pysr import PySRRegressor
from pysr.logger_specs import TensorBoardLoggerSpec

import openml

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.utils import setup_results_folder, fit_with_early_stopping, read_tensorboard_scalar, redirect_output_to_file
from utils.data import load_and_preprocess_openml_task, split_and_normalize_data_with_sr
from utils.evaluate import compute_ci_stats
from utils.cp_methods import fit_difficulty_estimator, compute_normalized_intervals, find_bin_thresholds_with_min_size, mondrian_min_bin_size
from utils.losses import bin_crossfit_loss_julia, pinball_loss_julia
from utils.config import load_config, dump_config, validate_sigma_sr_config, build_predictor

# Keys must match custom operator names in sr_params.unary_operators.
SIGMA_SR_EXTRA_SYMPY_MAPPINGS = {
    "logm": lambda x: sympy.log(sympy.Abs(x) + 1e-8),
}

def run_single_task(dataset, task_folder, config, random_seed):

    splits = split_and_normalize_data_with_sr(dataset.df_X, dataset.df_y, config.split, random_seed)
    X_prop_train, y_prop_train = splits["train"]
    X_cal, y_cal = splits["cal"]
    X_test, y_test = splits["test"]

    # with no SR split, the SR is trained on the base regressor's out-of-bag
    # residuals over the training set; otherwise on residuals over the SR split
    use_oob = config.split.sr_train == 0
    # ensemble variance (var, mondrian_cp) needs the individual trees
    is_forest = config.predictor_model == "RandomForestRegressor"

    # Store y_cal, y_test, y_cal_pred, y_test_pred, residuals_cal, residuals_pred for saving
    calibration_data = {}
    testing_data = {}
    calibration_data["y"] = y_cal
    testing_data["y"] = y_test

    # crepes.WrapRegressor wraps any sklearn-compatible regressor and adds
    # conformal prediction methods (.calibrate() and .predict_int()).
    print(f"Training base regressor ({config.predictor_model})...")
    base_regressor = WrapRegressor(build_predictor(config, random_seed))
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    calibration_data["y_pred"] = y_cal_pred
    testing_data["y_pred"] = y_test_pred
    calibration_data["residuals"] = y_cal - y_cal_pred
    testing_data["residuals"] = y_test - y_test_pred

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')

    learner_prop = base_regressor.learner

    # SR training inputs and residuals, never seen by the base regressor
    if use_oob:
        X_sr, y_sr = X_prop_train, y_prop_train
        residuals_sr = y_prop_train - learner_prop.oob_prediction_
    else:
        X_sr, y_sr = splits["sr_train"]
        residuals_sr = y_sr - learner_prop.predict(X_sr)

    sigmas_sr = {} # For data augmentation
    sigmas_cal, sigmas_test, conf_intervals = {}, {}, {}

    # Standard CP
    base_regressor.calibrate(X_cal, y_cal)
    sigmas_cal["standard_cp"] = np.ones(len(X_cal))
    sigmas_test["standard_cp"] = np.ones(len(X_test))
    conf_intervals["standard_cp"] = base_regressor.predict_int(X_test, confidence=config.confidence)

    # KNN distance and KNN std, fitted on the training set
    # de.apply(X) on a real X doesn't depend on the oob flag for KNN-based methods, so a
    # single fit serves both compute_normalized_intervals and augmentation.
    # With an SR split, augmentation applies the estimator to X_sr directly.
    for key, y_fit in (("knn_dist", None), ("knn_std", y_prop_train)):
        augment = config.data_augmentation[f"sigma_{key}"]
        de = fit_difficulty_estimator(X_prop_train, key, y_prop_train=y_fit, oob=augment and use_oob)
        conf_intervals[key], sigmas_cal[key], sigmas_test[key] = compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, config.confidence)
        if augment:
            sigmas_sr[key] = de.apply() if use_oob else de.apply(X_sr)

    # KNN residuals, fitted on the SR residuals (out-of-bag or SR split);
    # oob=True makes augmentation leave each point out of its own neighbours
    augment_knn_res = config.data_augmentation.sigma_knn_res
    de_knn_res = fit_difficulty_estimator(X_sr, "knn_res", residuals=residuals_sr, oob=augment_knn_res)
    conf_intervals["knn_res"], sigmas_cal["knn_res"], sigmas_test["knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, config.confidence)
    if augment_knn_res:
        sigmas_sr["knn_res"] = de_knn_res.apply()

    if is_forest:
        # Random Forest variance
        # unlike the KNN estimators above, de.apply(X) for the variance estimator
        # DOES depend on the oob flag (the oob branch expects X sized to the
        # training set)
        de_var = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop)
        conf_intervals["var"], sigmas_cal["var"], sigmas_test["var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, config.confidence)
        if config.data_augmentation.sigma_var:
            if use_oob:
                de_var_oob = fit_difficulty_estimator(X_prop_train, "var", learner_prop=learner_prop, oob=True)
                sigmas_sr["var"] = de_var_oob.apply()
            else:
                sigmas_sr["var"] = de_var.apply(X_sr)

        # Mondrian CP using variance
        min_points = mondrian_min_bin_size(config.confidence)
        bin_thresholds = find_bin_thresholds_with_min_size(sigmas_cal["var"], min_points, random_seed)
        number_of_bins = len(bin_thresholds) - 1
        print(f"Number of Mondrian bins: {number_of_bins}")

        # the "mc" argument for calibrate()/predict_int() takes X as its only
        # parameter, so the variance sigmas are recomputed from X
        def mondrian_categories(X):
            sigmas = de_var.apply(X)
            return binning(sigmas, bins=bin_thresholds, seed=random_seed)

        regressor_mond = WrapRegressor(learner_prop)
        regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
        sigmas_cal["mondrian_cp"] = np.ones(len(X_cal))
        sigmas_test["mondrian_cp"] = np.ones(len(X_test))
        conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=config.confidence)
    else:
        if config.data_augmentation.sigma_var:
            print(f"sigma_var augmentation needs RandomForestRegressor, skipping for {config.predictor_model}")
        print(f"var and mondrian_cp need RandomForestRegressor, skipping for {config.predictor_model}")

    # augment input using sigmas
    X_train_sr = np.zeros((X_sr.shape[0], len(sigmas_sr)), dtype=np.float32)
    X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_sr)), dtype=np.float32)
    X_test_sr = np.zeros((X_test.shape[0], len(sigmas_sr)), dtype=np.float32)
    for i, key in enumerate(sigmas_sr.keys()):
            X_train_sr[:,i] = sigmas_sr[key]
            X_cal_sr[:,i] = sigmas_cal[key]
            X_test_sr[:,i] = sigmas_test[key]
    X_train_sr = np.concatenate((X_sr, X_train_sr), axis=1)
    X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
    X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

    y_log_abs_residual_sr = np.log(np.abs(residuals_sr) + 1e-8)

    sigma_losses = {
        "bin_crossfit": (dict(loss_function=bin_crossfit_loss_julia(config.confidence, config.lambda_cov, seed=random_seed)), residuals_sr),
        "pinball": (dict(elementwise_loss=pinball_loss_julia(config.confidence)), y_log_abs_residual_sr),
    }

    for loss_name in config.loss_functions:
        loss_kwargs, y_train_sr = sigma_losses[loss_name]
        tb_log_dir = os.path.join(task_folder, "tb_logs", loss_name)
        sigma_predictor = PySRRegressor(
            model_selection="accuracy", # more complex equation means more expressivity of sigma
            tournament_selection_n=15,
            populations=config.sr_params.npopulations, # default 31
            population_size=config.sr_params.population_size, # must be >= topn:=12 (default 27)
            niterations=config.sr_params.niterations, # default 100
            binary_operators=config.sr_params.binary_operators,
            unary_operators=config.sr_params.unary_operators,
            nested_constraints=config.sr_params.nested_constraints,
            extra_sympy_mappings=SIGMA_SR_EXTRA_SYMPY_MAPPINGS,
            verbosity=1,
            input_stream="devnull", # disable input reading for tmux
            random_state=random_seed,
            output_directory=task_folder,
            run_id="checkpoints",
            tempdir=task_folder,
            logger_spec=TensorBoardLoggerSpec(log_dir=tb_log_dir, log_interval=1, overwrite=True),
            **loss_kwargs,
        )

        if config.sr_params.early_stop:
            fit_with_early_stopping(
                sigma_predictor, X_train_sr, y_train_sr,
                chunk_size=config.sr_params.early_stop_chunk_size,
                patience=config.sr_params.early_stop_patience,
                min_relative_improvement=config.sr_params.early_stop_min_improvement,
            )
        else:
            sigma_predictor.fit(X_train_sr, y_train_sr)

        steps, losses = read_tensorboard_scalar(tb_log_dir, "search/data/summaries/min_loss")
        pd.DataFrame({"step": steps, "loss": losses}).to_csv(
            os.path.join(task_folder, f"loss_curve_{loss_name}.csv"), index=False
        )

        # Hall of Fame equations
        df_hof = pd.read_csv(sigma_predictor.get_equation_file())
        df_hof["Chosen"] = (df_hof.index == sigma_predictor.get_best().name) # get_best method returns best equation according to model_selection parameter
        df_hof.set_index("Complexity", inplace=True)

        sigmas_cal_hof = {}
        sigmas_test_hof = {}
        conf_intervals_hof = {}
        for idx in range(len(df_hof)):
            complexity = df_hof.index[idx]

            def sigma_f(X, idx=idx):
                # equation predicts log(sigma); an equation using its own "exp"
                # node can overflow once exponentiated here, so clip in
                # log-space first.
                log_sigma = sigma_predictor.equations_["lambda_format"][idx](X)
                log_sigma = np.nan_to_num(log_sigma, nan=50.0)
                return np.exp(np.clip(log_sigma, -50.0, 50.0))

            # deploy exp(equation) exactly as the loss scored it: no min-max
            # scaling (it shifts sigma, changing its ratios) and no beta offset
            de_hof = DifficultyEstimator()
            de_hof.fit(X_train_sr, f=sigma_f, scaler=False, beta=0)
            conf_intervals_hof[complexity], sigmas_cal_hof[complexity], sigmas_test_hof[complexity] = compute_normalized_intervals(
                de=de_hof,
                learner_prop=learner_prop, 
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test,
                confidence=config.confidence,
                X_cal_de=X_cal_sr,
                X_test_de=X_test_sr)

            if df_hof.iloc[idx]["Chosen"]:
                conf_intervals[f"sr_{loss_name}"] = conf_intervals_hof[complexity]
                sigmas_cal[f"sr_{loss_name}"] = sigmas_cal_hof[complexity]
                sigmas_test[f"sr_{loss_name}"] = sigmas_test_hof[complexity]

            ci_mean, ci_median, coverage = compute_ci_stats(conf_intervals_hof[complexity], y_test)
            df_hof.loc[complexity, "ci_median"] = ci_median
            df_hof.loc[complexity, "ci_mean"] = ci_mean
            df_hof.loc[complexity, "coverage"] = coverage

        df_sigma_cal_hof = pd.DataFrame.from_dict(sigmas_cal_hof)
        df_sigma_test_hof = pd.DataFrame.from_dict(sigmas_test_hof)
        df_intervals_hof = pd.concat({complexity: pd.DataFrame(arr) for complexity, arr in conf_intervals_hof.items()}, axis=0)
        df_sigma_cal_hof.to_csv(os.path.join(task_folder, f"hof_sigmas_cal_{loss_name}.csv"), index_label="index")
        df_sigma_test_hof.to_csv(os.path.join(task_folder, f"hof_sigmas_test_{loss_name}.csv"), index_label="index")
        df_intervals_hof.to_csv(os.path.join(task_folder, f"hof_intervals_{loss_name}.csv"), header=["lower_bound", "upper_bound"], index_label=["complexity", "index"])
        df_hof.to_csv(os.path.join(task_folder, f"hof_{loss_name}.csv"))


    for regressor_name in config.extra_regressors.keys():
        if not config.extra_regressors[regressor_name].activate:
            continue

        match regressor_name:
            case "random_forest":
                print("Training random forest regressor for sigma...")
                regressor = WrapRegressor(
                    RandomForestRegressor(n_estimators=config.extra_regressors.random_forest.n_estimators, random_state=random_seed)
                )
            case "extra_trees":
                print("Training extra trees regressor for sigma...")
                regressor = WrapRegressor(
                    ExtraTreesRegressor(n_estimators=config.extra_regressors.extra_trees.n_estimators, random_state=random_seed)
                )
            case _:
                print(f"Regressor {regressor_name} not implemented, skipping...")
                continue

        regressor.fit(X_train_sr, np.abs(residuals_sr))

        de = DifficultyEstimator()
        de.fit(X_train_sr, f=lambda X: regressor.predict(X), scaler=True)
        conf_intervals[regressor_name], sigmas_cal[regressor_name], sigmas_test[regressor_name] = compute_normalized_intervals(
            de=de,
            learner_prop=learner_prop, 
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            confidence=config.confidence,
            X_cal_de=X_cal_sr,
            X_test_de=X_test_sr)

            
    # Save everything to csv
    df_sigmas_cal_per_method = pd.DataFrame.from_dict(sigmas_cal)
    df_sigmas_test_per_method = pd.DataFrame.from_dict(sigmas_test)
    df_intervals = pd.concat({method: pd.DataFrame(arr) for method, arr in conf_intervals.items()}, axis=0)
    df_sigmas_cal_per_method.to_csv(os.path.join(task_folder, "methods_sigmas_cal.csv"), index_label="index")
    df_sigmas_test_per_method.to_csv(os.path.join(task_folder, "methods_sigmas_test.csv"), index_label="index")
    df_intervals.to_csv(os.path.join(task_folder, "methods_intervals.csv"), header=["lower_bound", "upper_bound"],  index_label=["method", "index"])

    df_calibration = pd.DataFrame.from_dict(calibration_data)
    df_testing = pd.DataFrame.from_dict(testing_data)
    df_calibration.to_csv(os.path.join(task_folder, "calibration_data.csv"), index_label="index")
    df_testing.to_csv(os.path.join(task_folder, "testing_data.csv"), index_label="index")

    ci_means, ci_medians, coverages = {}, {}, {}
    for method in conf_intervals.keys():
        ci_mean, ci_median, coverage = compute_ci_stats(conf_intervals[method], y_test)
        ci_medians[method] = ci_median
        ci_means[method] = ci_mean
        coverages[method] = coverage

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
    redirect_output_to_file(os.path.join(results_folder, "run.log"))
    results_dictionary = defaultdict(list, {"task_id": [], "dataset_name": [], "r2": []})

    # Save config for tracing
    dump_config(config, results_folder)

    for dataset in iter_datasets(config):

        # optional subset of datasets by name; empty or missing runs all
        if config.get("datasets") and dataset.name not in config.datasets:
            continue
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

        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, "results.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, default="default_config")
    args = parser.parse_args()
    config = load_config("sigma-sr", args.config)
    validate_sigma_sr_config(config)

    for random_seed in config.random_seeds:
        run_all_tasks(config, random_seed)
