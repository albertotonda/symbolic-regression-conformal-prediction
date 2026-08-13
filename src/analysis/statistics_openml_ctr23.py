# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:03:52 2024

@author: Alberto
"""

import numpy as np
import openml
import pandas as pd
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# run as `python -m analysis.statistics_openml_ctr23` from src/, so src/ (this
# module's parent) is on sys.path and sibling modules import flatly
from data import load_and_preprocess_openml_task

if __name__ == "__main__" :

    # hard-coded variables
    random_seed = 42
    use_predefined_splits = False
    prng = random.Random() # pseudo-random number generator will be useful later
    prng.seed(random_seed)
    regressor_classes = [RandomForestRegressor, XGBRegressor]

    # load CTR23 regression benchmark suite
    suite = openml.study.get_suite(353)

    # get all task_ids, so that if there is a crash we can easily restart
    # by skipping the first results
    task_ids = [t for t in suite.tasks]

    # prepare data structure to store information
    statistics_dictionary = {'task_id' : [], 'dataset_name' : [], 'target_name': [], 'n_samples' : [],
                             'n_features' : [], 'missing_data' : [], 'categorical_features' : [],}
    
    for regressor_class in regressor_classes :
        statistics_dictionary['R2_' + regressor_class.__name__] = []
        statistics_dictionary['MSE_' + regressor_class.__name__] = []
    
    for task_id in task_ids :

        print("Now working on task %d..." % task_id)

        # descriptive stats (missing data, categorical columns) are computed
        # from the raw data, since load_and_preprocess_openml_task returns
        # data that's already been cleaned/encoded
        raw_task = openml.tasks.get_task(task_id, download_splits=True)
        df_X_raw, df_y_raw = raw_task.get_X_and_y('dataframe')
        # here below there is a sum().sum() because it is adding up missing values
        # in rows AND THEN in columns
        missing_data = df_X_raw.isnull().sum().sum() + df_y_raw.isnull().sum()
        categorical_features = df_X_raw.select_dtypes(include=['category', 'object']).shape[1]

        df_X, df_y, task = load_and_preprocess_openml_task(task_id)
        X = df_X.values
        y = df_y.values

        # let's also get the name of the dataset
        dataset = task.get_dataset()
        print("Task %d is applied to data set \"%s\" (id=%d)" % (task_id, dataset.name, dataset.dataset_id))
        
        for regressor_class in regressor_classes :
            # mean performance of regressor 
            regressor_name = regressor_class.__name__
            regressor_r2 = []
            regressor_mse = []
            
            for fold in range(0, 10) :
                print("Evaluating \"%s\" performance on fold %d..." % (regressor_name, fold))
                # initialize regressor
                regressor = regressor_class(n_estimators=500, random_state=random_seed, n_jobs=-1)
                
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
                
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                
                regressor_r2.append(r2_score(y_test, y_pred))
                regressor_mse.append(mean_squared_error(y_test, y_pred))
                
            statistics_dictionary['R2_' + regressor_name].append("%.2f +/- %.2f" % (np.mean(regressor_r2), np.std(regressor_r2)))
            statistics_dictionary['MSE_' + regressor_name].append("%.2f +/- %.2f" % (np.mean(regressor_mse), np.std(regressor_mse)))
            
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
        
        
        df_statistics = pd.DataFrame.from_dict(statistics_dictionary)
        df_statistics.to_csv("OpenML-CTR23-statistics.csv", index=False)