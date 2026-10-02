from crepes import ConformalRegressor
from crepes.extras import DifficultyEstimator, binning

import numpy as np

def fit_difficulty_estimator(X_prop_train : np.ndarray, 
                            type : str,
                            oob : bool = False,
                            y_prop_train : np.ndarray = None,
                            learner_prop = None,
                            k : int = 25) -> DifficultyEstimator:
    """
    Fit the difficulty estimator depending on type. k is the number of
    neighbors for the KNN-based types.
    """
    de = DifficultyEstimator()
    match type:
        case "knn_dist":
            de.fit(X=X_prop_train, k=k, scaler=True, oob=oob)
        case "knn_std":
            assert y_prop_train is not None
            de.fit(X=X_prop_train, y=y_prop_train, k=k, scaler=True, oob=oob)
        case "knn_res":
            assert y_prop_train is not None
            assert learner_prop is not None
            y_pred_oob = learner_prop.oob_prediction_
            residuals_prop_oob = y_prop_train - y_pred_oob
            de.fit(X=X_prop_train, residuals=residuals_prop_oob, k=k, scaler=True, oob=oob)
        case "var":
            assert learner_prop is not None
            de.fit(X=X_prop_train, learner=learner_prop, scaler=True, oob=oob)
    return de

def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence, X_cal_de=None, X_test_de=None):
    """
    Calibrate a normalized conformal regressor using an already-fitted
    DifficultyEstimator. Returns the confidence intervals for the test set,
    together with the difficulty estimates on the calibration and test sets.
    X_cal_de/X_test_de are the estimator's inputs when they differ from the
    learner's (e.g. augmented with sigma columns); they default to X_cal/X_test.
    """
    sigmas_cal = de.apply(X_cal if X_cal_de is None else X_cal_de)
    sigmas_test = de.apply(X_test if X_test_de is None else X_test_de)

    regressor_norm = ConformalRegressor()
    regressor_norm.fit(residuals=y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal)

    intervals = regressor_norm.predict_int(y_hat=learner_prop.predict(X_test), sigmas=sigmas_test, confidence=confidence)

    return intervals, sigmas_cal, sigmas_test

def mondrian_min_bin_size(confidence):
    """
    Minimum calibration points per Mondrian bin: a finite conformal quantile
    needs n >= 1/(1-confidence) - 1, plus one point of safety margin. The
    ratio is rounded before flooring since 1/(1-0.95) evaluates to
    19.999... in float.
    """
    return int(np.floor(round(1 / (1 - confidence), 9)))

def find_bin_thresholds_with_min_size(sigmas_cal_var, min_points, random_seed):
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