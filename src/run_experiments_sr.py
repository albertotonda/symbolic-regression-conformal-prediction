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
# since the syntax is different, we can only use a string
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
        
        if (i == 1)
            println("y_true=", y_true)
            println("y_pred=", y_pred)
            println("upper_bound=", upper_bound)
            println("lower_bound=", lower_bound)
            println("fitness_coverage=", fitness_coverage)
            println("fitness_amplitude=", fitness_amplitude)
        end
        
    end
    
    return fitness_coverage * 1000 + fitness_amplitude
    
end
"""


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
        'task_id' : [], 'dataset_name' : [], 'r2' : [],
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
        X_train_ci[:,0] = regressor.predict(X_cal)
        X_test_ci[:,0] = regressor.predict(X_test)
        
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
        
        # now, for the more complex part: we can use a PySRRegressor, but we
        # need to change the fitness function!
        ci_regressor = PySRRegressor(
            niterations=1,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "log", "exp"],
            loss_function=loss_function_julia, # defined as a string above
            temp_equation_file=True, # does not clutter directory with temporary files
            verbosity=1,
            )
        
        print("Running symbolic regression...")
        ci_regressor.fit(X_train_ci, y_cal)
        
        # TODO remove this
        sys.exit(0)