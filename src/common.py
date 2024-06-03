# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:18:39 2024

Functions common to all files

@author: Alberto
"""

import matplotlib.pyplot as plt
import openml

# this is used to translate internal naming convention to readable strings
# for the plots
translations = {
    "conformal_predictor" : "Standard conformal predictor",
    "normalized_cp_knn_dist" : "CP with intervals normalized using KNN on distance",
    "normalized_cp_knn_std" : "CP with intervals normalized using KNN on standard deviation",
    "normalized_cp_knn_res" : "CP with intervals normalized using KNN on OOB residuals",
    "normalized_cp_norm_var" : "CP with intervals normalized using variance of ensemble predictors",
    "mondrian_cp" : "Mondrian CP",
    "symbolic_regression_cp" : "Symbolic Regression CP"
    }

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
    
    fig, ax = plt.subplots()#figsize=(10,8))
    
    # plot measured values and point predictions for y
    x = range(0, len(y))
    ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
    ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")
    
    # visualize corresponding confidence intervals around point predictions
    ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][1] for x in y_and_ci], color='orange', alpha=0.3)
    
    ax.set_xlabel("Samples sorted by increasing value of target")
    ax.set_ylabel("Value of target y")
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


if __name__ == "__main__" :
    print("This file is not meant to be run directly, just imported.")