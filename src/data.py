# -*- coding: utf-8 -*-
"""
Loading and pre-processing OpenML-CTR23 tasks: fetching the benchmark
suite's task ids, downloading/cleaning a single task's data, and splitting
it into train/calibration/test sets.
"""

import os

import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train/calibration/test split proportions: 50% train, then the remaining
# 50% split evenly again into calibration and test (25%/25% overall)
_SPLIT_TEST_SIZE = 0.5


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


def get_benchmark_task_ids(suite_id, tasks_too_good, tasks_too_bad):
    """
    Get the task_ids for all tasks in the benchmark suite, removing the ones
    for which we already know performance is too good or too bad.
    """
    suite = openml.study.get_suite(suite_id)
    task_ids = [t for t in suite.tasks]

    # remove task_ids that for which we had results that are too good or too bad
    task_ids = [t for t in task_ids if t not in tasks_too_bad and t not in tasks_too_good]

    print("After removing data sets with low or high performance, I am left with %d tasks!" % len(task_ids))

    return task_ids


def prepare_task_data(task_id, results_folder, random_seed):
    """
    Download and pre-process a task, split it into training/calibration/test
    sets, and normalize features and target.
    """
    print("Downloading and pre-processing task %d..." % (task_id))
    df_X, df_y, task = load_and_preprocess_openml_task(task_id)

    # get names for features and target
    feature_names = [c for c in df_X.columns]

    # get actual numpy values
    X = df_X.values
    y = df_y.values

    # get dataset name and create task folder
    dataset = task.get_dataset()
    task_folder = os.path.join(results_folder, dataset.name)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    print("Starting work on dataset \"%s\" for task %d..." % (dataset.name, task_id))

    # training/test split and normalization; 50/25/25 split
    X_prop_train, X_test, y_prop_train, y_test = train_test_split(X, y, test_size=_SPLIT_TEST_SIZE,
                                                        shuffle=True, random_state=random_seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=_SPLIT_TEST_SIZE,
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

    return (X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test,
            feature_names, dataset, task_folder)
