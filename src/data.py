# -*- coding: utf-8 -*-
"""
Loading and pre-processing OpenML-CTR23 tasks: fetching the benchmark
suite's task ids, downloading/cleaning a single task's data, and splitting
it into train/calibration/test sets.
"""

import os

import openml

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass


# train/calibration/test split proportions: 50% train, then the remaining
# 50% split evenly again into calibration and test (25%/25% overall)
_SPLIT_TEST_SIZE = 0.5

@dataclass
class Dataset:
    """ Custom structure to hold generic dataset information """
    name : str
    n_samples: int
    n_features: int
    missing_data: bool
    categorical: list[bool]
    df_X : pd.DataFrame
    df_y : pd.Series
    id : int = None

    def __str__(self):
        return f"""
ID: {self.id}
Name: {self.name}
n_samples: {self.n_samples}
n_features: {self.n_features}
missing_data: {self.missing_data}
categorical: {any(self.categorical)}
"""


def load_and_preprocess_openml_task(task_id) :
    """
    Given a task_id, load and pre-process the data related to the task. Pre-processing
    includes converting categorical values to numerical values (e.g. integers),
    and treating missing data by imputation.

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
    task = openml.tasks.get_task(task_id)
    openml_dataset = task.get_dataset()
    X_raw, y_raw, categorical, features = openml_dataset.get_data(target=openml_dataset.default_target_attribute)

    X = X_raw.loc[y_raw.notna()]
    y = y_raw.dropna()

    for i, c in enumerate(X.columns):
        if categorical[i]:
            X[c] = X[c].cat.codes

    # Default solution is to drop columns with missing data
    X.dropna(axis=1, how="any", inplace=True)

    return Dataset(
        id=task_id, 
        name=openml_dataset.name, 
        n_samples=X.shape[0], 
        n_features=X.shape[1], 
        missing_data=bool(X_raw.isna().any().any() or y_raw.isna().any()),
        categorical=categorical,
        df_X=X,
        df_y=y
    )


def split_and_normalize_data(df_X, df_y, random_seed):
    """
    Split data in training, calibration, test sets and normalize them
    """
    X = df_X.values
    y = df_y.values

    X_prop_train, X_test, y_prop_train, y_test = train_test_split(
        X, y, test_size=_SPLIT_TEST_SIZE, shuffle=True, random_state=random_seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test, y_test, test_size=_SPLIT_TEST_SIZE, shuffle=True, random_state=random_seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_prop_train = scaler_X.fit_transform(X_prop_train)
    X_cal        = scaler_X.transform(X_cal)
    X_test       = scaler_X.transform(X_test)

    y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1, 1)).ravel()
    y_cal        = scaler_y.transform(y_cal.reshape(-1, 1)).ravel()
    y_test       = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test
