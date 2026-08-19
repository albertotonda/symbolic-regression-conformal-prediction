# -*- coding: utf-8 -*-
"""
Symbolic regression of confidence-interval amplitude: builds a feature
matrix from point predictions, difficulty estimates and the original
features, then fits a PySRRegressor against a custom Julia loss function
that penalizes intervals failing to cover the true value more heavily than
it penalizes wide intervals.
"""

import json
import os
import pickle

import numpy as np

from pysr import PySRRegressor

SR_MODEL_FILENAME = "symbolic_regression_cp.pk"
SR_FEATURES_FILENAME = "sr_features.json"

# unfortunately, using pysr we can define a custom loss function...in Julia.
# since the syntax is different, we can only use a string that is then passed
# to the Julia interpreter internally inside the PySRRegressor object
loss_function_julia_penalize_smaller = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}

    # get predicted values for the current tree
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # 'flag' == false means that evaluating the tree caused an error
    if !flag
        return L(Inf)
    end

    result = 0.0
    coverage = 0.0
    coverage_penalty = 100.0

    # instead of just having a sum of squared means, we penalize more heavily
    # samples for which the predictions are inferior to 'y' (here the difference
    # between the true value and the predicted value)
    for i in 1:length(dataset.y)
        if (prediction[i] < dataset.y[i])
            result += 10 * (prediction[i] - dataset.y[i])^2
        else
            result += (prediction[i] - dataset.y[i])^2
            coverage += 1
        end
    end

    if ((coverage / dataset.n) < 0.95)
        # penalty is equal to the difference between complete coverage and current result * weight
        result += (0.95 - coverage/dataset.n) * dataset.n * coverage_penalty
    end

    return result / dataset.n
end
"""


def run_symbolic_regression(X_cal, X_test, y_cal, y_test, y_cal_pred, y_test_pred,
                             sigmas_cal, sigmas_test,
                             feature_names, task_folder, config, random_seed):
    """
    Train a PySRRegressor to predict the amplitude of a confidence interval
    around the point prediction, using point predictions, difficulty
    estimates, and the original features. Uses a custom Julia loss function
    (loss_function_julia_penalize_smaller) that penalizes intervals that do
    not cover the true value more heavily than it penalizes wide intervals.
    """
    # step 1: prepare data sets with all sigmas and stuff on calibration set
    # and test set; these will be a special version, just for symbolic regression
    X_train_sr = np.zeros((y_cal.shape[0], len(sigmas_cal)+1), dtype=np.float32)
    X_test_sr = np.zeros((y_test.shape[0], len(sigmas_cal)+1), dtype=np.float32)

    # add point predictions
    X_train_sr[:,0] = y_cal_pred
    X_test_sr[:,0] = y_test_pred

    for i, key in enumerate(sigmas_cal.keys()):
        X_train_sr[:,i+1] = sigmas_cal[key]
        X_test_sr[:,i+1] = sigmas_test[key]

    # TODO: information used by the Mondrian conformal predictors is not immediately
    # applicable, unless I use something about the bins? to be explored
    # finally, add the feature information from the original data set
    X_train_sr = np.concatenate((X_train_sr, X_cal), axis=1)
    X_test_sr = np.concatenate((X_test_sr, X_test), axis=1)

    # finally, we need a target (y) for our problem of confidence interval
    # regression; we can obtain that by computing the absolute difference
    # between the y_true and the y_pred for a dataset
    y_train_sr = abs(y_cal - y_cal_pred)

    # now, for the more complex part: we can use a PySRRegressor, but we
    # need to change the fitness function! the fitness function is described
    # as a string (lines of Julia), defined above
    ci_regressor = PySRRegressor(
        tournament_selection_n=config.sr_tournament_selection_n,
        population_size=config.sr_population_size, # must be >= topn (default 12)
        niterations=config.sr_niterations,
        binary_operators=config.sr_binary_operators,
        unary_operators=config.sr_unary_operators,
        loss_function=loss_function_julia_penalize_smaller,
        temp_equation_file=True, # does not clutter directory with temporary files
        verbosity=1, # can also be set to 0, it should be ok
        random_state=random_seed,
        deterministic=True,
        parallelism="serial"
        )

    print("Running symbolic regression...")
    ci_regressor.fit(X_train_sr, y_train_sr)

    print("Now computing confidence intervals for conformal set...")
    ci_amplitude_cal = ci_regressor.predict(X_train_sr)

    print(f"Number of CI with undercoverage in calibration set: {sum(ci_amplitude_cal < y_train_sr)}")

    print("And computing confidence intervals for test set...")
    ci_amplitude_test = ci_regressor.predict(X_test_sr)
    ci_test = np.zeros((y_test.shape[0], 2))
    for i in range(0, y_test.shape[0]):
        ci_test[i,0] = y_test_pred[i] - ci_amplitude_test[i]
        ci_test[i,1] = y_test_pred[i] + ci_amplitude_test[i]

    # save the predictor as a pickle file
    with open(os.path.join(task_folder, SR_MODEL_FILENAME), "wb") as fp:
        pickle.dump(ci_regressor, fp)

    # save the full list of SR feature names for this task (fixed synthetic
    # columns + this dataset's own feature names, in the order used above)
    with open(os.path.join(task_folder, SR_FEATURES_FILENAME), "w") as fp:
        json.dump({"sr_features": list(sigmas_cal.keys()) + feature_names}, fp, indent=2)

    return ci_test
