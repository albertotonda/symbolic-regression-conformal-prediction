# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:17:34 2024

Experiments with Symbolic Regression might take more time and require more
tweaking, so I moved them to a separate file

@author: Alberto
"""

import numpy as np
import openml
import os
import pandas as pd
import seaborn as sns
import sys

from crepes.extras import DifficultyEstimator

from pysr import PySRRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import load_and_preprocess_openml_task, plot_confidence_intervals, plot_pareto, translations

# unfortunately, using pysr we can define a custom loss function...in Julia.
# since the syntax is different, we can only use a string that is then passed
# to the Julia interpreter internally inside the PySRRegressor object
loss_function_julia = """
function my_custom_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    if !completion
        return L(Inf)
    end
    
    elements_covered = 0.0
    fitness_coverage = 0.0
    fitness_amplitude = 0.0
    
    # Julia arrays start indexing at 1! Oh joy!
    for i in 1:length(dataset.y)
        y_true = dataset.y[i]    
        y_pred = dataset.X[1,i] # first column has predicted values
        
        lower_bound = y_pred - prediction[i]
        upper_bound = y_pred + prediction[i]
        
        if (y_true > upper_bound && elements_covered < 0.95 * length(dataset.y))
            fitness_coverage += (y_true - upper_bound)^2
        elseif (y_true < lower_bound && elements_covered < 0.95 * length(dataset.y))
            fitness_coverage += (lower_bound - y_true)^2
        else
            elements_covered += 1
        end
        
        fitness_amplitude += y_pred^2
        
        if (i == 1 && false)
            println("y_true=", y_true)
            println("y_pred=", y_pred)
            println("upper_bound=", upper_bound)
            println("lower_bound=", lower_bound)
            println("fitness_coverage=", fitness_coverage)
            println("fitness_amplitude=", fitness_amplitude)
        end
        
    end
    
    fitness_value = fitness_coverage * 1000 + fitness_amplitude
    #println("fitness_value=", fitness_value)
    
    return fitness_value
    
end
"""

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

def get_confidence_intervals_from_predictors(X, y_pred, predictor_ci) :
    """
    Get confidence intervals from the trained predictor.
    """
    ci = np.zeros((y_pred.shape[0], 2))
    ci_amplitude = predictor_ci.predict(X)
    
    for i in range(0, y_pred.shape[0]) :
        ci[i,0] = y_pred - ci_amplitude[i]
        ci[i,1] = y_pred + ci_amplitude[i]
    
    return ci

if __name__ == "__main__" :
    
    # get all task_ids
    
    # load and pre-process data set
    
    # split
    # normalization
    
    # generation of the features
    # organizing features in a matrix
    # symbolic regression; the fitness function is special
    # - if confidence interval covers true value, amplitude of the confidence interval
    # - if confidence interval does not cover true vale, w * (y - ci)
    random_seed = 42
    results_folder = "results_sr"
    results_csv = "results.csv"
    tasks_too_good = [361236, 361247, 361252, 361254, 361256, 
                      361257, 361268, 361617]
    tasks_too_bad = [361243, 361244, 361261, 361618, 361619]
    
    # filtering warnings is usually bad, but here I am getting lots of annoying
    # FutureWarnings on stuff I cannot modify (it's inside other functions), so
    # I am going to filter them
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # set up plotting
    sns.set_theme(style='darkgrid')
    
    # get task_id for all tasks in the benchmark suite
    suite = openml.study.get_suite(353)
    task_ids = [t for t in suite.tasks]
    
    # remove task_ids that for which we had results that are too good or too bad
    task_ids = [t for t in task_ids if t not in tasks_too_bad and t not in tasks_too_good]
    
    print("After removing data sets with low or high performance, I am left with %d tasks!" % len(task_ids))
    
    # create data structures to store the results
    results_dictionary = {
        'task_id' : [], 'dataset_name' : [], 'r2' : [], 'coverage' : [], 'mean_amplitude' : [], 'median_amplitude' : [],
        }
    
    # prepare directory for the results
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    
    for task_id in task_ids :
        
        # data structure for results related to this task
        task_results = {}
        
        # get the task
        print("Downloading and pre-processing task...")
        df_X, df_y, task = load_and_preprocess_openml_task(task_id)
        
        # get actual numpy values
        X = df_X.values
        y = df_y.values
        
        # get dataset name and create task folder
        dataset = task.get_dataset()
        task_folder = os.path.join(results_folder, dataset.name)
        if not os.path.exists(task_folder) :
            os.makedirs(task_folder)
        
        # training/test split and normalization
        X_prop_train, X_test, y_prop_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                            shuffle=True, random_state=random_seed)
        X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                                    shuffle=True, random_state=random_seed)
        
        # even if normalizing is not really necessary, we do it anyways
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_prop_train = scaler_X.fit_transform(X_prop_train)
        X_cal = scaler_X.transform(X_cal)
        X_test = scaler_X.transform(X_test)
        
        y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1,1)).ravel()
        y_cal = scaler_y.transform(y_cal.reshape(-1,1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1,1)).ravel()
        
        print("Training set: %d samples" % X_prop_train.shape[0])
        print("Calibration set: %d samples" % X_cal.shape[0])
        print("Test set: %d samples" % X_test.shape[0])  
        
        print("Training regressor...")
        regressor = RandomForestRegressor(oob_score=True, random_state=random_seed)
        regressor.fit(X_prop_train, y_prop_train)
        
        # get predictions for the calibration set
        y_cal_pred = regressor.predict(X_cal)
        
        # get predictions for the test set from the learner
        y_test_pred = regressor.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # get all different features in a Pandas dataset?
        print("Preparing necessary features to learn confidence intervals...")
        column_names = ["y_pred", "de_knn", "de_knn_std", "de_knn_oob_res", 
                        "de_cal_ensemble_var"]
        X_train_ci = np.zeros((y_cal.shape[0], len(column_names)), dtype=np.float32)
        X_test_ci = np.zeros((y_test.shape[0], len(column_names)), dtype=np.float32)
        
        # point predictions
        X_train_ci[:,0] = y_cal_pred
        X_test_ci[:,0] = y_test_pred
        
        # difficulty estimation using KNN
        de_knn = DifficultyEstimator()
        de_knn.fit(X=X_prop_train, scaler=True)
        sigmas_cal_knn_dist = de_knn.apply(X_cal)
        sigmas_test_knn_dist = de_knn.apply(X_test)
        
        X_train_ci[:,1] = sigmas_cal_knn_dist
        X_test_ci[:,1] = sigmas_test_knn_dist
        
        # difficulty estimation using standard deviations
        de_knn_std = DifficultyEstimator()
        de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
        sigmas_cal_knn_std = de_knn_std.apply(X_cal)
        sigmas_test_knn_std = de_knn_std.apply(X_test)
        
        X_train_ci[:,2] = sigmas_cal_knn_std
        X_test_ci[:,2] = sigmas_test_knn_std
        
        # difficulty estimation using OOB predictions
        oob_predictions = regressor.oob_prediction_
        residuals_prop_oob = y_prop_train - oob_predictions
        de_knn_res = DifficultyEstimator()
        de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
        sigmas_cal_knn_res = de_knn_res.apply(X_cal)
        sigmas_test_knn_res = de_knn_res.apply(X_test)
        
        X_train_ci[:,3] = sigmas_cal_knn_res
        X_test_ci[:,3] = sigmas_test_knn_res
        
        # difficulty estimation using variance of predictors in ensemble
        de_var = DifficultyEstimator()
        de_var.fit(X=X_prop_train, learner=regressor, scaler=True)
        sigmas_cal_var = de_var.apply(X_cal)
        sigmas_test_var = de_var.apply(X_test)
        
        X_train_ci[:,4] = sigmas_cal_var
        X_test_ci[:,4] = sigmas_test_var
        
        # Mondrian stuff is not applicable, I think; but maybe I could use
        # some binning here and try to associate each y_pred to a different bin?
        # I should think about a way of using the thresholds
        
        # now, I would like to juxtapose the X_cal with X_train_ci and
        # X_test with X_test_ci
        X_train_ci = np.concatenate((X_train_ci, X_cal), axis=1)
        X_test_ci = np.concatenate((X_test_ci, X_test), axis=1)
        
        print("Shape of X_train_ci:", X_train_ci.shape)
        print("Shape of X_test_ci:", X_test_ci.shape)
        
        # finally, we need a target (y) for our problem of confidence interval
        # regression; we can obtain that by computing the absolute difference
        # between the y_true and the y_pred for a dataset
        y_train_ci = abs(y_cal - y_cal_pred)
        
        # now, for the more complex part: we can use a PySRRegressor, but we
        # need to change the fitness function!
        ci_regressor = PySRRegressor(
            #tournament_selection_n=1,
            #populations=1, # TODO this is just for debugging
            #population_size=1, # TODO this is just for debugging
            niterations=1000,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "log", "exp"],
            loss_function=loss_function_julia_penalize_smaller, # defined as a string above
            temp_equation_file=True, # does not clutter directory with temporary files
            verbosity=1,
            random_state=random_seed,
            )
        
        print("Running symbolic regression...")
        ci_regressor.fit(X_train_ci, y_train_ci)
        
        print("Now computing confidence intervals for conformal set...")
        ci_amplitude_cal = ci_regressor.predict(X_train_ci)
        ci_cal = np.zeros((y_cal.shape[0], 2))
        for i in range(0, y_cal.shape[0]) :
            ci_cal[i,0] = y_cal_pred[i] - ci_amplitude_cal[i]
            ci_cal[i,1] = y_cal_pred[i] + ci_amplitude_cal[i]
        
        # in fact, we could use something like
        #ci_cal[:,0] = y_cal_pred - ci_amplitude_cal
        #ci_cal[:,1] = y_cal_pred + ci_amplitude_cal
        
        # and now, compute some stats for the conformal set
        ci_amplitude_mean = np.mean((ci_cal[:,1] - ci_cal[:,0]))
        ci_amplitude_median = np.median((ci_cal[:,1] - ci_cal[:,0]))
        coverage = np.sum([1 if (y_cal[i] >= ci_cal[i,0] and
                               y_cal[i] <= ci_cal[i,1]) else 0
                        for i in range(len(y_cal))])/len(y_cal)
        print("Mean amplitude on conformal set: %.4f" % ci_amplitude_mean)
        print("Median amplitude on conformal set: %.4f" % ci_amplitude_median)
        print("Coverage on conformal set: %.4f" % coverage)
        
        ci_amplitude_test = ci_regressor.predict(X_test_ci)
        ci_test = np.zeros((y_test.shape[0], 2))
        for i in range(0, y_test.shape[0]) :
            ci_test[i,0] = y_test_pred[i] - ci_amplitude_test[i]
            ci_test[i,1] = y_test_pred[i] + ci_amplitude_test[i]
        
        ci_amplitude_mean = np.mean((ci_test[:,1] - ci_test[:,0]))
        ci_amplitude_median = np.median((ci_test[:,1] - ci_test[:,0]))
        coverage = np.sum([1 if (y_test[i] >= ci_test[i,0] and
                               y_test[i] <= ci_test[i,1]) else 0
                        for i in range(len(y_test))])/len(y_test)
        print("Mean amplitude on test set: %.4f" % ci_amplitude_mean)
        print("Median amplitude on test set: %.4f" % ci_amplitude_median)
        print("Coverage on conformal test: %.4f" % coverage)
        
        # save data; first, set of equations (it's already a DataFrame)
        ci_regressor.equations_.to_csv(os.path.join(task_folder, "equations.csv"), index=False)
        
        # then, update the results file
        results_dictionary["task_id"].append(task_id)
        results_dictionary["dataset_name"].append(dataset.name)
        results_dictionary["r2"].append(r2_test)
        results_dictionary["coverage"].append(coverage)
        results_dictionary["mean_amplitude"].append(ci_amplitude_mean)
        results_dictionary["median_amplitude"].append(ci_amplitude_median)
        
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, "results.csv"), index=False)
        
        # TODO remove this, it's only used for debugging
        sys.exit(0)