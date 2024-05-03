# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:15:37 2024

@author: Alberto
"""

import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import seaborn as sns
import sys

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

def plot_confidence_intervals(y, y_pred, y_pred_ci) :
    """
    
    """
    # sort y_test values from small to big, along with y_pred_ci
    # using a list is pretty slow, there is probably a smarter way to do this
    # with numpy arrays, but the data set sizes should be small, so who cares
    y_and_ci = []
    for i in range(0, len(y)) :
        y_and_ci.append([y, y_pred, y_pred_ci])
    y_and_ci = sorted(y_and_ci, key=lambda x : x[0])
    
    fig, ax = plt.subplot()
    
    # plot measured values and point predictions for y
    x = range(0, len(y))
    ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
    ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")
    
    # visualize corresponding confidence intervals around point predictions
    ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][0] for x in y_and_ci], color='orange', alpha=0.3)
    
    ax.set_xlabel("Samples sorted by increasing value of target")
    ax.set_xlabel("Value of target y")
    ax.legend(loc='best')
    
    return fig, ax

def load_and_preprocess_openml_task(task_id) :
    """
    Given a task_id, load and pre-process the data related to the task. Pre-processing
    includes converting categorical values to numerical values (e.g. integers),
    and treating missing data, either with imputation or just by ignoring the 
    missing values.

    Parameters
    ----------
    task_id : int
        Id for the target task.

    Returns
    -------
    df_X : pd.DataFrame
        Feature values for the task.
    df_y : pd.DataFrame
        Target values for the task.
    task : openml Task
        Task object, contains a lot of useful information.

    """
    task = openml.tasks.get_task(task_id, download_splits=True)
    
    # the 'task' object above contains a lot of useful information,
    # like the name of the target variable and the id of the dataset
    df_X, df_y = task.get_X_and_y('dataframe')
    
    # check for missing data; if data is missing, operate accordingly
    missing_data = df_X.isnull().sum().sum() + df_y.isnull().sum()
    
    # TODO there should be better ways of taking into account missing data, but for
    # these data sets, all we do is a few special cases where we drop columns
    # that are missing too many data points
    if missing_data > 0 :
        if task_id == 361268 or task_id == 361616 :
            # these two task have several columns with A LOT of missing data,
            # so we are just going to drop them
            df_X.dropna(axis=1, inplace=True)
        else :
            # default solution is dropping rows
            print("Found missing data in data set!")
            df_X.dropna(axis=0, inplace=True)
    
    # check if there are any categorical columns
    df_categorical = df_X.select_dtypes(include=['category', 'object'])
    
    # replace categorical values with integers
    for c in df_categorical.columns :
        df_X[c] = df_X[c].astype('category') # double-check that it is treated as a categorical column
        df_X[c] = df_X[c].cat.codes # replace values with category codes (automatically computed)
       
    return df_X, df_y, task

def evaluate_confidence_intervals(y_true, y_pred, y_ci) :
    """
    Get an evaluation for the confidence intervals. Things that matter:
        - is the true value within the range y_pred +/- y_ci?
        - how large are the ci?
    """
    return

if __name__ == "__main__" :
    
    # for each dataset
    # - split training/calibration/test; 60/20/20 (no cross-validation)
    # - test different conformal predictors
    #   -- regular conformal predictor
    #   -- normalized conformal predictors (N versioni)
    #   -- Mondrian conformal predictors
    # - check for each candidate set of confidence intervals, the tightness and whether the point is inside
    # - Symbolic Regression
    #   -- use as features all statistics computed by normalized and Mondrian
    #   -- plus feature values of the original problem
    #   -- plus (predicted?) value of the target?
    
    # hard-coded variables
    random_seed = 42
    results_folder = "results"
    tasks_too_good = [361236, 361247, 361252, 361254, 361256, 
                      361257, 361268, 361617]
    tasks_too_bad = [361243, 361244, 361261, 361618, 361619]
    
    # get task_id for all tasks in the benchmark suite
    suite = openml.study.get_suite(353)
    task_ids = [t for t in suite.tasks]
    
    # remove task_ids that for which we had results that are too good or too bad
    task_ids = [t for t in task_ids if t not in tasks_too_bad and t not in tasks_too_good]
    
    print("After removing data sets with low or high performance, I am left with %d tasks!" % len(task_ids))
    
    # create data structures to store the results
    results_dictionary = {
        'task_id' : [], 'dataset_name' : [], 'r2' : [], 
        'conformal_predictor_frequency' : [], 'conformal_predictor_mean_amplitude' : [],
        'norm_cp_frequency' : [], 'norm_cp_mean_amplitude' : [],
        'norm_std_frequency' : [], 'norm_std_mean_amplitude' : [],
        'mondrian_cp_frequency' : [], 'mondrian_cp_mean_amplitude' : [],
        'sr_cp_frequency' : [], 'sr_cp_mean_amplitude' : [],
                          }
    
    # prepare directory for the results
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    
    for task_id in task_ids[:1] :
        
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
        regressor = WrapRegressor(XGBRegressor(random_state=random_seed))
        regressor.fit(X_prop_train, y_prop_train)
        
        print("Calibrating conformal regressor...")
        regressor.calibrate(X_cal, y_cal)
        
        print("Getting confidence intervals for the test set from conformal regressor...")
        cp_intervals = regressor.predict_int(X_test, confidence=0.95)
        
        # now we need to access the wrapped learner to re-use it for the other
        # conformal predictors, but it's not difficult
        learner_prop = regressor.learner
        
        print("Normalizing confidence intervals using KNN for difficulty estimation...")
        de_knn = DifficultyEstimator()
        de_knn.fit(X=X_prop_train, scaler=True)
        sigmas_cal_knn_dist = de_knn.apply(X_cal)
        
        regressor_norm_knn_dist = WrapRegressor(learner_prop)
        regressor_norm_knn_dist.calibrate(X_cal, y_cal, sigmas=sigmas_cal_knn_dist)
        
        # now, let's get the confidence intervals for the test set
        sigmas_test_knn_dist = de_knn.apply(X_test)
        intervals_norm_knn_dist = regressor_norm_knn_dist.predict_int(X_test, sigmas=sigmas_test_knn_dist)
        
        # another way of estimating difficulty is by using standard deviations
        print("Now normalizing using standard deviations...")
        de_knn_std = DifficultyEstimator()
        de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
        sigmas_cal_knn_std = de_knn_std.apply(X_cal)
        regressor_norm_knn_std = WrapRegressor(learner_prop)
        regressor_norm_knn_std.calibrate(X_cal, y_cal, sigmas=sigmas_cal_knn_std)
        
        sigmas_test_knn_std = de_knn_std.apply(X_test)
        intervals_norm_knn_std = regressor_norm_knn_std.predict_int(X_test, sigmas=sigmas_test_knn_std)
        
        # a third way of normalizing, using absolute residuals; it does not work
        # for XGBoost, because only Random Forest has out-of-bag predictions for
        # each individual learner...
        if False :
            oob_predictions = regressor.learner.oob_prediction_
            residuals_prop_oob = y_prop_train - oob_predictions
            de_knn_res = DifficultyEstimator()
            de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
            sigmas_cal_knn_res = de_knn_res.apply(X_cal)
            rf_norm_knn_res = WrapRegressor(learner_prop)
            rf_norm_knn_res.calibrate(X_cal, y_cal, sigmas=sigmas_cal_knn_res)
            
            sigmas_test_knn_res = de_knn_res.apply(X_test)
            intervals_norm_knn_res = rf_norm_knn_res.predict_int(X_test, sigmas=sigmas_test_knn_res)
        
        # a fourth way: using the variance of each element of the ensemble (!)

        
    
    