# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:15:37 2024

@author: Alberto
"""

import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import pandas as pd
import seaborn as sns
import sys

from crepes import WrapRegressor
from crepes.extras import binning, DifficultyEstimator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
        y_and_ci.append([y[i], y_pred[i], y_pred_ci[i]])
    y_and_ci = sorted(y_and_ci, key=lambda x : x[0])
    
    fig, ax = plt.subplots(figsize=(10,8))
    
    # plot measured values and point predictions for y
    x = range(0, len(y))
    ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
    ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")
    
    # visualize corresponding confidence intervals around point predictions
    ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][1] for x in y_and_ci], color='orange', alpha=0.3)
    
    ax.set_xlabel("Samples sorted by increasing value of target")
    ax.set_xlabel("Value of target y")
    ax.legend(loc='best')
    
    return fig, ax

def plot_pareto(methods, results_dictionary, translations=None, all_results=False) :
        
    fig, ax = plt.subplots(figsize=(10,8))
    
    for method in methods :
        
        # get the information related to coverage
        key_coverage = method + "_coverage"
        x = results_dictionary[key_coverage]
        
        # get information on median (or mean)
        key_median = method + "_median"
        y = results_dictionary[key_median]
        
        if all_results == False :
            x = x[-1]
            y = y[-1]
        
        if translations is not None :
            method = translations[method]
            
        ax.scatter(x, y, label=method)
    
    # invert x-axis, so that the plot is more readable
    ax.invert_xaxis()
    
    ax.set_xlabel("coverage on the test set")
    ax.set_ylabel("median amplitude of the confidence intervals")
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
    results_csv = "results.csv"
    tasks_too_good = [361236, 361247, 361252, 361254, 361256, 
                      361257, 361268, 361617]
    tasks_too_bad = [361243, 361244, 361261, 361618, 361619]
    
    # this is used to translate internal naming convention to readable strings
    # for the plots
    translations = {
        "conformal_predictor" : "Classic conformal predictor (CP)",
        "normalized_cp_knn_dist" : "CP with intervals normalized using KNN on distance",
        "normalized_cp_knn_std" : "CP with intervals normalized using KNN on standard deviation",
        "normalized_cp_knn_res" : "CP with intervals normalized using KNN on OOB residuals",
        "normalized_cp_norm_var" : "CP with intervals normalized using variance of ensemble predictors",
        "mondrian_cp" : "Mondrian CP",
        }
    
    
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
        #regressor = WrapRegressor(XGBRegressor(random_state=random_seed))
        # the argument 'oob_score=True' is to compute and keep track of the score
        # and performance on samples that are not used to train each predictor
        regressor = WrapRegressor(RandomForestRegressor(oob_score=True, random_state=random_seed))
        regressor.fit(X_prop_train, y_prop_train)
        
        # get predictions for the test set from the learner
        y_test_pred = regressor.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        print("Calibrating conformal regressor...")
        regressor.calibrate(X_cal, y_cal)
        
        print("Getting confidence intervals for the test set from conformal regressor...")
        cp_intervals = regressor.predict_int(X_test, confidence=0.95)
        
        # store confidence intervals
        task_results["conformal_predictor"] = cp_intervals
        
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
        
        task_results["normalized_cp_knn_dist"] = intervals_norm_knn_dist
        
        # another way of estimating difficulty is by using standard deviations
        print("Now normalizing using standard deviations...")
        de_knn_std = DifficultyEstimator()
        de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
        sigmas_cal_knn_std = de_knn_std.apply(X_cal)
        regressor_norm_knn_std = WrapRegressor(learner_prop)
        regressor_norm_knn_std.calibrate(X_cal, y_cal, sigmas=sigmas_cal_knn_std)
        
        sigmas_test_knn_std = de_knn_std.apply(X_test)
        intervals_norm_knn_std = regressor_norm_knn_std.predict_int(X_test, sigmas=sigmas_test_knn_std)
        
        task_results["normalized_cp_knn_std"] = intervals_norm_knn_std
        
        # a third way of normalizing, using absolute residuals; it does not work
        # for XGBoost, because only Random Forest has out-of-bag predictions for
        # each individual learner...but it's a cool idea! maybe I should go back
        # and pick RandomForest as the estimator
        if learner_prop.__class__.__name__ == "RandomForestRegressor" :
            print("Now normalizing using OOB predictions of each estimator...")
            oob_predictions = regressor.learner.oob_prediction_
            residuals_prop_oob = y_prop_train - oob_predictions
            de_knn_res = DifficultyEstimator()
            de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
            sigmas_cal_knn_res = de_knn_res.apply(X_cal)
            rf_norm_knn_res = WrapRegressor(learner_prop)
            rf_norm_knn_res.calibrate(X_cal, y_cal, sigmas=sigmas_cal_knn_res)
            
            sigmas_test_knn_res = de_knn_res.apply(X_test)
            intervals_norm_knn_res = rf_norm_knn_res.predict_int(X_test, sigmas=sigmas_test_knn_res)
            
            task_results["normalized_cp_knn_res"] = intervals_norm_knn_res
        
        # a fourth way: using the variance of each element of the ensemble (!)
        # but we need to check whether XGBoost can actually deal with this;
        # update IT CAN'T, because the XGBoostRegressor object does not have
        # the ._regressor part
        if learner_prop.__class__.__name__ == "RandomForestRegressor" :
            print("Now normalizing using variance of the estimators...")
            de_var = DifficultyEstimator()
            de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
            sigmas_cal_var = de_var.apply(X_cal)
            regressor_norm_var = WrapRegressor(learner_prop)
            regressor_norm_var.calibrate(X_cal, y_cal, sigmas=sigmas_cal_var)
            
            sigmas_test_var = de_var.apply(X_test)
            intervals_norm_var = regressor_norm_var.predict_int(X_test, sigmas=sigmas_test_var)
            
            task_results["normalized_cp_norm_var"] = intervals_norm_var
            
        # Mondrian conformal regressor; in the original version, it is using
        # sigmas_cal_var, but for XGBoost I don't have it... :-D
        # so, in the end we ARE switching back to Random Forest
        if learner_prop.__class__.__name__ == "RandomForestRegressor" :
            print("Now calibrating a Mondrian regressor...")
            bins_cal, bin_thresholds = binning(sigmas_cal_var, bins=20) # why 20 and not 200? magic number
            regressor_mond = WrapRegressor(learner_prop)
            regressor_mond.calibrate(X_cal, y_cal, bins=bins_cal)
            
            bins_test = binning(sigmas_test_var, bins=bin_thresholds)
            intervals_mond = regressor_mond.predict_int(X_test, bins=bins_test)
            
            task_results["mondrian_cp"] = intervals_mond
            
        # TODO: symbolic regression intervals, using all sigmas
        
        # post-processing of the results for the different confidence intervals
        # statistics we are interested in: coverage, mean size, median size
        for method, confidence_intervals in task_results.items() :
            
            ci_amplitude_mean = np.mean((confidence_intervals[:,1] - confidence_intervals[:,0]))
            ci_amplitude_median = np.median((confidence_intervals[:,1] - confidence_intervals[:,0]))
            # this expression below is a bit of a mess, but it's 1 if the measured
            # value falls within the confidence intervals, and 0 otherwise (summed up, divided by n_samples)
            coverage = np.sum([1 if (y_test[i] >= confidence_intervals[i,0] and
                                   y_test[i] <= confidence_intervals[i,1]) else 0
                            for i in range(len(y_test))])/len(y_test)
        
            # add results to global dictionary of results
            key_mean = method + "_mean"
            key_median = method + "_median"
            key_coverage = method + "_coverage"
            
            if key_mean not in results_dictionary :
                results_dictionary[key_mean] = []
                results_dictionary[key_median] = []
                results_dictionary[key_coverage] = []
            
            results_dictionary[key_mean].append(ci_amplitude_mean)
            results_dictionary[key_median].append(ci_amplitude_median)
            results_dictionary[key_coverage].append(coverage)
            
            # plot time! it would be nice to have a classic plot with CI
            # BUT ALSO a plot Pareto-front style, using (for example)
            # median and coverage; just take a few points
            fig, ax = plot_confidence_intervals(y_test[:20], y_test_pred[:20], 
                                                confidence_intervals[:20])
            
            title = "%s on data set \"%s\" (coverage=%.4f, median=%.2f)" % (translations[method], dataset.name, coverage, ci_amplitude_median)
            ax.set_title(title)
            
            plt.savefig(os.path.join(task_folder, method + ".png"), dpi=300)
            plt.close(fig)
            
        # add other necessary details for the row in the results dictionary
        results_dictionary["task_id"].append(task_id)
        results_dictionary["dataset_name"].append(dataset.name)
        results_dictionary["r2"].append(r2_test)
        
        # also plot a Pareto-like scheme
        fig, ax = plot_pareto([k for k in task_results], results_dictionary, translations=translations)
        ax.set_title("Performance of conformal prediction methods on dataset \"%s\"" % dataset.name)
        plt.savefig(os.path.join(task_folder, "pareto.png"), dpi=300)     
        plt.close(fig)
        
        # TODO save local copy of the results?
        
        # save global dictionary of results as DataFrame
        #print(results_dictionary)
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(results_folder, results_csv), index=False)
        
    # and now, a global Pareto front plot
    fig, ax = plot_pareto([k for k in task_results], results_dictionary, translations=translations, all_results=True)
    ax.set_title("Performance of conformal prediction methods on selected CTR-23 datasets")
    plt.savefig(os.path.join(results_folder, "pareto.png"), dpi=300)        

    