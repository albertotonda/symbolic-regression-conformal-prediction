# %% [markdown]
# # Symbolic Regression for Conformal Prediction — Sandbox

# %% [markdown]
# ## 0. Imports & Setup

# %%
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer, binning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pysr import PySRRegressor

from data import load_and_preprocess_openml_task, get_benchmark_task_ids
from evaluate import plot_confidence_intervals, plot_pareto, translations

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style='darkgrid')
print('All imports OK')

# %% [markdown]
# ## 1. Loop over all OpenML-CTR23 tasks
#
# Same walkthrough as before (load data, split, train base regressor,
# compute conformal predictors), repeated for every task in the suite. For
# the symbolic-regression sigma estimator (section 5), we now also try the
# candidate loss functions discussed in NOTES.md (2026-08-14 entry): MAE,
# pinball/quantile loss, a scale-invariant log-variance loss, and Gaussian
# NLL.

# %%
task_ids = get_benchmark_task_ids(353, [], [])
TASKS_TOO_GOOD = [361236, 361247, 361252, 361254, 361256, 361257, 361268, 361617]
TASKS_TOO_BAD = [361243, 361244, 361261, 361618, 361619]
task_ids = [id for id in task_ids if id not in TASKS_TOO_BAD and id not in TASKS_TOO_GOOD]

random_seed = 42
confidence = 0.95

for task_id in task_ids:

    # ## 1. Load & Explore a Dataset
    df_X, df_y, task = load_and_preprocess_openml_task(task_id)
    dataset = task.get_dataset()

    print(f"Dataset  : {dataset.name}")
    print(f"Samples  : {df_X.shape[0]}")
    print(f"Features : {df_X.shape[1]}")
    print(f"Target   : min={df_y.min():.2f}  max={df_y.max():.2f}  mean={df_y.mean():.2f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(df_y.values, bins=60)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Count')
    ax.set_title(f"Distribution of target in '{dataset.name}'")
    plt.savefig(f"src/{dataset.name}.png")
    plt.close(fig)

    # ## 2. Split & Normalize
    #
    # Conformal prediction requires a calibration set separate from training.
    # We use a 50 / 25 / 25 split:
    # - Proper training set (X_prop_train): trains the base regressor
    # - Calibration set (X_cal): calibrates conformal predictors (never seen
    #   by the base regressor during training)
    # - Test set (X_test): evaluates coverage and interval width
    #
    # We standardize X and y so that difficulty estimates and the SR loss
    # function are scale-invariant.
    X = df_X.values
    y = df_y.values
    feature_names = list(df_X.columns)

    X_prop_train, X_test, y_prop_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True, random_state=random_seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test, y_test, test_size=0.5, shuffle=True, random_state=random_seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_prop_train = scaler_X.fit_transform(X_prop_train)
    X_cal        = scaler_X.transform(X_cal)
    X_test       = scaler_X.transform(X_test)

    y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1, 1)).ravel()
    y_cal        = scaler_y.transform(y_cal.reshape(-1, 1)).ravel()
    y_test       = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f'Proper training : {X_prop_train.shape[0]} samples')
    print(f'Calibration     : {X_cal.shape[0]} samples')
    print(f'Test            : {X_test.shape[0]} samples')

    # ## 3. Train Base Regressor (Random Forest)
    #
    # crepes.WrapRegressor wraps any sklearn-compatible regressor and adds
    # conformal prediction methods (.calibrate() and .predict_int()).
    #
    # We use oob_score=True so the Random Forest stores out-of-bag (OOB)
    # predictions — predictions made on each training sample by trees that
    # did not train on it. These are used later as one of the difficulty
    # estimators.
    print("Training base regressor...")
    base_regressor = WrapRegressor(
        RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed)
    )
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')

    # ## 4. Compute other Conformal Predictors for comparison
    def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
        """
        Calibrate a normalized conformal regressor using an already-fitted
        DifficultyEstimator. Returns the confidence intervals for the test set,
        together with the difficulty estimates on the calibration and test sets.
        """
        sigmas_cal = de.apply(X_cal)

        regressor_norm = WrapRegressor(learner_prop)
        regressor_norm.calibrate(X_cal, y_cal, de=de)

        intervals = regressor_norm.predict_int(X_test, confidence=confidence)

        return intervals, sigmas_cal

    def _find_bin_thresholds_with_min_size(sigmas_cal_var, min_points, random_seed):
        """
        crepes.extras.binning's min_size parameter requests bins=len(values)//
        min_size equal-frequency bins, but with tied/duplicated difficulty
        scores (common for the ensemble-variance estimator) pd.qcut can still
        leave a handful of bins a few points short of min_size. So, rather than
        trusting the requested bin count outright, verify the actual per-bin
        counts and back off the number of bins until every one of them holds at
        least min_points calibration points.
        """
        number_of_bins = len(sigmas_cal_var) // min_points
        while number_of_bins > 1:
            assigned_bins, bin_thresholds = binning(
                sigmas_cal_var, bins=number_of_bins, seed=random_seed)
            counts = np.bincount(assigned_bins.astype(int))
            if counts.min() >= min_points:
                return bin_thresholds
            number_of_bins -= 1
        return np.array([-np.inf, np.inf])

    learner_prop = base_regressor.learner
    sigmas = {}
    conf_intervals = {}

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    # Standard CP
    print("Computing CI for SCP...")
    base_regressor.calibrate(X_cal, y_cal)
    sigmas["scp"] = np.ones(len(X_cal))
    conf_intervals["scp"] = base_regressor.predict_int(X_test, confidence=confidence)

    # KNN distance
    print("Computing CI for knn_dist NCP...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, scaler=True)
    conf_intervals["knn_dist"], sigmas["knn_dist"] = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN std
    print("Computing CI for knn_std NCP...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
    conf_intervals["knn_std"], sigmas["knn_std"] = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN out-of-bag residuals
    print("Computing CI for knn_res NCP...")
    de_knn_res = DifficultyEstimator()
    de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
    conf_intervals["knn_res"], sigmas["knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, confidence)

    # Random Forest variance
    print("Computing CI for var NCP...")
    de_var = DifficultyEstimator()
    de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
    conf_intervals["var"], sigmas["var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, confidence)

    # Mondrian CP using variance
    print("Computing CI for MCP...")
    min_points = int(1 / (1-confidence) - 1) + 1
    bin_thresholds = _find_bin_thresholds_with_min_size(sigmas["var"], min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas["mondrian"] = np.ones(len(X_cal))
    conf_intervals["mondrian"]= regressor_mond.predict_int(X_test, confidence=confidence)

    # ## 5. Symbolic Regression as an estimator for sigma(x)
    #
    # The idea is that instead of using Symbolic Regression to directly
    # predict confidence bounds, we use it as a difficulty estimator to use
    # in Normalized Conformal Prediction (NCP). Since we need extra data to
    # train the symbolic regressor, we can make use of Random Forest's
    # out-of-bag predictions.

    # ### 5.1. Computing all NPS sigma on OOB predictions
    sigmas_train = {}
    sigmas_cal = {}
    sigmas_test = {}

    # KNN distance
    de_knn_dist_oob = DifficultyEstimator()
    de_knn_dist_oob.fit(X=X_prop_train, scaler=True, oob=True)
    sigmas_train["knn_dist"] = de_knn_dist_oob.apply()
    sigmas_cal["knn_dist"] = de_knn_dist_oob.apply(X_cal)
    sigmas_test["knn_dist"] = de_knn_dist_oob.apply(X_test)

    # KNN std
    de_knn_std_oob = DifficultyEstimator()
    de_knn_std_oob.fit(X=X_prop_train, y=y_prop_train, scaler=True, oob=True)
    sigmas_train["knn_std"] = de_knn_std_oob.apply()
    sigmas_cal["knn_std"] = de_knn_std_oob.apply(X_cal)
    sigmas_test["knn_std"] = de_knn_std_oob.apply(X_test)

    # KNN out-of-bag residuals
    de_knn_res_oob = DifficultyEstimator()
    de_knn_res_oob.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True, oob=True)
    sigmas_train["knn_res"] = de_knn_res_oob.apply()
    sigmas_cal["knn_res"] = de_knn_res_oob.apply(X_cal)
    sigmas_test["knn_res"] = de_knn_res_oob.apply(X_test)

    # Random Forest variance
    de_var_oob = DifficultyEstimator()
    de_var_oob.fit(X=X_prop_train, learner=learner_prop, scaler=True, oob=True)
    sigmas_train["var"] = de_var_oob.apply()
    sigmas_cal["var"] = de_var.apply(X_cal) # For cal and test, use default (no oob) version, otherwise same oob trees are used instead of full model
    sigmas_test["var"] = de_var.apply(X_test)

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

    y_log_abs_residual = np.log(np.abs(residuals_prop_oob))
    y_raw_residual = residuals_prop_oob

    # ### 5.2. Try the candidate fitness functions discussed in NOTES.md
    #
    # Ranked best to worst (2026-08-14 entry): MAE and pinball/quantile loss
    # (both on log(|residual|), robust and coverage-neutral) are preferred
    # over the scale-invariant log-variance loss and Gaussian NLL (both
    # squared-error-based, so more outlier-sensitive; NLL also assumes
    # Gaussian residuals). The last two need custom Julia code since they
    # aren't per-point-decomposable / use a different target.
    logvar_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        delta = dataset.y .- prediction
        return L(sum(delta .^ 2) / dataset.n - (sum(delta) / dataset.n) ^ 2)
    end
    """

    gaussian_nll_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual (not its log)
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        result = 0.0
        for i in 1:dataset.n
            result += dataset.y[i]^2 / (2 * exp(2 * log_sigma[i])) + log_sigma[i]
        end
        return result / dataset.n
    end
    """

    sigma_losses = [
        ("mae", dict(elementwise_loss="L1DistLoss()"), y_log_abs_residual),
        # ("pinball_0.5", dict(elementwise_loss="QuantileLoss(0.5)"), y_log_abs_residual),
        # ("pinball_0.75", dict(elementwise_loss="QuantileLoss(0.75)"), y_log_abs_residual),
        # ("logvar", dict(loss_function=logvar_loss_julia), y_log_abs_residual),
        # ("gaussian_nll", dict(loss_function=gaussian_nll_loss_julia), y_raw_residual),
    ]

    from crepes import ConformalRegressor

    for loss_name, loss_kwargs, y_train_sr in sigma_losses:
        sigma_predictor = PySRRegressor(
            model_selection="best",
            tournament_selection_n=15, # default 15
            populations=31, # default 31
            population_size=30, # must be >= topn:=12 (default 27)
            niterations=100, # default 100
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "log", "exp"],
            temp_equation_file=True, # does not clutter directory with temporary files
            verbosity=1, # can also be set to 0, it should be ok
            random_state=random_seed,
            **loss_kwargs,
        )
        sigma_predictor.fit(X_train_sr, y_train_sr)

        de_sr = DifficultyEstimator()
        de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.predict(X)), scaler=True)

        # WrapRegressor.calibrate()/.predict_int() feed the SAME X to both the
        # wrapped learner (needs the raw data) and de.apply() (needs the
        # sigma-augmented columns) — incompatible here, so calibrate manually via
        # the lower-level ConformalRegressor instead.
        sigmas_cal_sr = de_sr.apply(X_cal_sr)
        sigmas_test_sr = de_sr.apply(X_test_sr)

        cr_sr = ConformalRegressor()
        cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

        cp_key = f"symbolic_regression_{loss_name}"
        conf_intervals[cp_key] = cr_sr.predict_int(
            learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=confidence
        )
        sigmas[cp_key] = sigmas_cal_sr
        print(f"[{loss_name}] chosen SR expression: {sigma_predictor.sympy()}")

    results = {}
    for cp in sigmas.keys():
        results[cp] = {}
        results[cp]["mean"] = np.mean((conf_intervals[cp][:,1] - conf_intervals[cp][:,0]))
        results[cp]["median"] = np.median((conf_intervals[cp][:,1] - conf_intervals[cp][:,0]))
        # this expression below is a bit of a mess, but it's 1 if the measured
        # value falls within the confidence intervals, and 0 otherwise (summed up, divided by n_samples)
        results[cp]["coverage"] = np.sum([1 if (y_test[i] >= conf_intervals[cp][i,0] and
                                y_test[i] <= conf_intervals[cp][i,1]) else 0
                        for i in range(len(y_test))])/len(y_test)

        print(f"{cp.upper()}\n|-- CI mean: {results[cp]["mean"]}\tCI median: {results[cp]["median"]}\tCoverage: {results[cp]["coverage"]}")

    fig, ax = plt.subplots(figsize=(10,8))

    for cp in results.keys():

        # get the information related to coverage
        x = results[cp]["coverage"]

        # get information on median (or mean)
        y = results[cp]["median"]

        ax.scatter(x, y, label=cp)

    # invert x-axis, so that the plot is more readable
    ax.invert_xaxis()

    ax.set_xlabel("coverage on the test set")
    ax.set_ylabel("median amplitude of the confidence intervals")
    ax.legend(loc='best')
    plt.savefig(f"src/pareto_{dataset.name}.png")
    plt.close(fig)
