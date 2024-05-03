# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:03:52 2024

@author: Alberto
"""

import openml
import numpy as np
import pandas as pd
import random
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

if __name__ == "__main__" :
    
    # hard-coded variables
    random_seed = 42
    use_predefined_splits = False
    prng = random.Random() # pseudo-random number generator will be useful later
    prng.seed(random_seed)
    
    # load CTR23 regression benchmark suite
    suite = openml.study.get_suite(353)
    
    # get all task_ids, so that if there is a crash we can easily restart
    # by skipping the first results
    task_ids = [t for t in suite.tasks]
    
    # this is for DEBUGGING, let's see if we get a better performance with normalization
    #task_ids = [361244, 361618, 361619, 361269, 361261, 361243]
    
    # prepare data structure to store information
    statistics_dictionary = {'task_id' : [], 'dataset_name' : [], 'target_name': [], 'n_samples' : [],
                             'n_features' : [], 'missing_data' : [], 'categorical_features' : [],
                             'R2' : [], 'MSE' : []}
    
    for task_id in task_ids :
        
        print("Now working on task %d..." % task_id)
        task = openml.tasks.get_task(task_id, download_splits=True)
        
        # the 'task' object above contains a lot of useful information,
        # like the name of the target variable and the id of the dataset
        df_X, df_y = task.get_X_and_y('dataframe')
        
        # check if there is any missing value
        # here below there is a sum().sum() because it is adding up missing values
        # in rows AND THEN in columns
        missing_data = df_X.isnull().sum().sum() + df_y.isnull().sum()
        
        if missing_data > 0 :
            # we actually have to go with a task/dataset-specific correction
            if task_id == 361268 :
                # this task has several columns with A LOT of missing data,
                # so we are just going to drop them
                df_X.dropna(axis=1, inplace=True)
            elif task_id == 361616 :
                # again, a few columns with 800/1200 missing values, get dropped
                df_X.dropna(axis=1, inplace=True)
        
        # check if there are any categorical columns
        df_categorical = df_X.select_dtypes(include=['category', 'object'])
        categorical_features = df_categorical.shape[1]
        
        # convert categorical columns to numerical values
        for c in df_categorical.columns :
            df_X[c] = df_X[c].astype('category') # double-check that it is treated as a categorical column
            df_X[c] = df_X[c].cat.codes # replace values with category codes (automatically computed)
        
        X = df_X.values
        y = df_y.values
        
        # let's also get the name of the dataset
        dataset = task.get_dataset()
        print("Task %d is applied to data set \"%s\" (id=%d)" % (task_id, dataset.name, dataset.dataset_id))
        
        # mean performance of Random Forest 
        rf_r2 = []
        rf_mse = []
        
        for fold in range(0, 10) :
            print("Evaluating RF performance on fold %d..." % fold)
            # initialize random forest regressor
            #rf = RandomForestRegressor(random_state=random_seed)
            rf = XGBRegressor(random_state=random_seed)
            
            if use_predefined_splits :
                # get splits for N-fold cross-validation
                # NOTE: this ignores repetitions, for a few data sets the evaluation
                # is something like 10x(10-fold cross-validation), and here we are only
                # performing one
                train_index, test_index = task.get_train_test_split_indices(fold=fold)
            else :
                # otherwise, we go for a nice 50/50 split, just like the funny
                # guys that use conformal predictors; we need to instantiate
                # the object managing the cross-validation with a different random
                # seed at each iteration, to avoid issues
                cv_random_seed = random.randint(0, 10000)
                kf = KFold(n_splits=2, shuffle=True, random_state=cv_random_seed)
                folds = [(train_index, test_index) for train_index, test_index in kf.split(X, y)]
                train_index, test_index = folds[0]
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # normalization (it should not impact performance at all, let's see)
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1,1)).ravel()
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            rf_r2.append(r2_score(y_test, y_pred))
            rf_mse.append(mean_squared_error(y_test, y_pred))
            
        # what I am interested in knowing:
        # number of samples
        # number of features
        # name of the target
        # missing data?
        # categorical variables?
        # mean performance of random forest?
        statistics_dictionary['task_id'].append(task_id)
        statistics_dictionary['dataset_name'].append(dataset.name)
        statistics_dictionary['target_name'].append(task.target_name)
        statistics_dictionary['n_samples'].append('{:,}'.format(df_X.shape[0]))
        statistics_dictionary['n_features'].append('{:,}'.format(df_X.shape[1]))
        statistics_dictionary['missing_data'].append('{:,}'.format(missing_data))
        statistics_dictionary['categorical_features'].append(categorical_features)
        
        statistics_dictionary['R2'].append("%.2f +/- %.2f" % (np.mean(rf_r2), np.std(rf_r2)))
        statistics_dictionary['MSE'].append("%.2f +/- %.2f" % (np.mean(rf_mse), np.std(rf_mse)))
        
        df_statistics = pd.DataFrame.from_dict(statistics_dictionary)
        df_statistics.to_csv("OpenML-CTR23-statistics.csv", index=False)
        
        #sys.exit(0)