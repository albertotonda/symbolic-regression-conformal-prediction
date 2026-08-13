# -*- coding: utf-8 -*-
"""
Fitting and calibrating the different conformal predictors compared in the
experiment: the standard conformal regressor, normalized conformal
regressors (one per difficulty-estimation strategy), and Mondrian conformal
regressors.
"""

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, binning

from sklearn.metrics import r2_score

# maps each difficulty-estimation strategy (as used internally, and as a key
# into the sigmas_cal/sigmas_test dicts consumed by symbolic regression) to
# the task_results key its normalized CP intervals are stored under
NORMALIZED_CP_RESULT_KEYS = {
    "knn_dist": "normalized_cp_knn_dist",
    "knn_std": "normalized_cp_knn_std",
    "knn_oob_res": "normalized_cp_knn_res",
    "ensemble_var": "normalized_cp_norm_var",
}


def train_base_regressor(X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test,
                          regressor_class, regressor_params, random_seed):
    """
    Train the base regressor, wrap it as a conformal regressor, and
    calibrate the standard conformal predictor.
    """
    print("Training regressor...")
    base_regressor = regressor_class(**regressor_params)
    if "random_state" in base_regressor.get_params():
         base_regressor.set_params(random_state=random_seed)
    regressor = WrapRegressor(base_regressor)
    regressor.fit(X_prop_train, y_prop_train)

    # get predictions for the test set and calibration set from the learner
    y_cal_pred = regressor.predict(X_cal)
    y_test_pred = regressor.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)

    print("Calibrating conformal regressor...")
    regressor.calibrate(X_cal, y_cal)

    return regressor, y_cal_pred, y_test_pred, r2_test


def fit_difficulty_estimators(X_prop_train, y_prop_train, learner_prop, config):
    """
    Fit one DifficultyEstimator per normalization strategy applicable to the
    current predictor model. KNN-distance and KNN-on-target-std apply to any
    predictor; OOB-residuals and ensemble-variance require an out-of-bag,
    inspectable ensemble, so they're only fit for RandomForestRegressor.

    Returns a dict {sigma_key: DifficultyEstimator}, keyed the same way as
    NORMALIZED_CP_RESULT_KEYS.
    """
    difficulty_estimators = {}

    # distance of KNN in feature space, default k=25
    print("Normalizing confidence intervals using KNN for difficulty estimation...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, k=config.ncp_knn_k, scaler=True)
    difficulty_estimators["knn_dist"] = de_knn_dist

    # standard deviation of KNN in target space
    print("Now normalizing using standard deviations...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, k=config.ncp_knn_k, scaler=True)
    difficulty_estimators["knn_std"] = de_knn_std

    # a third way of normalizing, using absolute residuals; it does not work
    # for XGBoost, because only Random Forest has out-of-bag predictions for
    # each individual learner
    if config.predictor_model == "RandomForestRegressor":
        print("Now normalizing using OOB predictions of each estimator...")
        residuals_prop_oob = y_prop_train - learner_prop.oob_prediction_
        de_knn_res = DifficultyEstimator()
        de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, k=config.ncp_knn_k, scaler=True)
        difficulty_estimators["knn_oob_res"] = de_knn_res

        # a fourth way: using the variance of each element of the ensemble;
        # XGBoostRegressor doesn't expose the individual estimators, so this
        # is also RandomForest-only
        print("Now normalizing using variance of the estimators...")
        de_var = DifficultyEstimator()
        de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
        difficulty_estimators["ensemble_var"] = de_var

    return difficulty_estimators


def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
    """
    Calibrate a normalized conformal regressor using an already-fitted
    DifficultyEstimator. Returns the confidence intervals for the test set,
    together with the difficulty estimates on the calibration and test sets
    (needed later as features for the symbolic regression step).
    """
    sigmas_cal = de.apply(X_cal)

    regressor_norm = WrapRegressor(learner_prop)
    regressor_norm.calibrate(X_cal, y_cal, de=de)

    sigmas_test = de.apply(X_test)
    intervals = regressor_norm.predict_int(X_test, confidence=confidence)

    return intervals, sigmas_cal, sigmas_test


def select_mondrian_source(config, difficulty_estimators, sigmas_cal):
    """
    Pick which already-fitted difficulty estimator (and its calibration-set
    sigmas) the Mondrian conformal regressor should bin on: RandomForest
    ensemble-variance by default, KNN-distance as an experimental
    alternative (config.use_alt_mondrian) for other predictor models, or
    None if neither applies.
    """
    if config.predictor_model == "RandomForestRegressor":
        return difficulty_estimators["ensemble_var"], sigmas_cal["ensemble_var"]
    elif config.use_alt_mondrian:
        return difficulty_estimators["knn_dist"], sigmas_cal["knn_dist"]
    return None


def compute_mondrian_intervals(learner_prop, de_var, sigmas_cal_var, X_cal, y_cal,
                                X_test, confidence, random_seed):
    """
    Calibrate a Mondrian conformal regressor. Bin boundaries are computed
    directly from the calibration set's own ensemble-variance difficulty
    scores (sigmas_cal_var, itself just an inference output of the
    already-fitted learner, so no leakage), using the largest number of
    equal-sized bins for which every bin is guaranteed to hold at least the
    minimum number of calibration points required for a finite conformal
    quantile at this confidence level (crepes.extras.binning's min_size
    parameter). This removes the need to iterate/retry on undersized bins.
    """
    # minimal number of data points per bin is n >= 1/(1-confidence) - 1;
    # +1 as a safety margin, since crepes' own check on the calibration side
    # (base.py: int((1-confidence)*(n+1))-1 >= 0) can trip at the exact
    # boundary count due to floating-point rounding of (1-confidence) for
    # typical confidence levels (e.g. 1-0.9 != 0.1 exactly in binary float)
    min_points = int(1 / (1-confidence) - 1) + 1

    # MondrianCategorizer doesn't expose the "min_size" attribute, so compute manually instead
    _, bin_thresholds = binning(sigmas_cal_var, min_size=min_points, seed=random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    intervals_mond = regressor_mond.predict_int(X_test, confidence=confidence)

    return intervals_mond, number_of_bins
