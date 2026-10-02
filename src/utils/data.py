# -*- coding: utf-8 -*-
"""
Loading and pre-processing OpenML-CTR23 tasks: fetching the benchmark
suite's task ids, downloading/cleaning a single task's data, and splitting
it into train/calibration/test sets.
"""

import openml
import numpy as np

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

    cols = X.select_dtypes(exclude=["number"]).columns
    for i, c in enumerate(X.columns):
        if (c in cols) and (not categorical[i]) and (X[c].nunique() < X.shape[0] // 4): # heuristic to detect categories
            X[c] = X[c].astype("category")
            categorical[i] = True
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


SPLIT_NAMES = ("train", "sr_train", "cal", "test")

def split_and_normalize_data_with_sr(df_X, df_y, split, random_seed):
    """
    Split data in training, SR-training, calibration and test sets using the
    percentages in `split` (keys SPLIT_NAMES, summing to 100), and normalize
    them with scalers fitted on the training set. Returns a dict mapping each
    split name to (X, y); a split with 0% maps to (None, None).
    """
    if sum(split[name] for name in SPLIT_NAMES) != 100:
        raise ValueError(f"split percentages must sum to 100, got {dict(split)}")

    X = df_X.values
    y = df_y.values

    # boundaries of each split in a shuffled index array
    perm = np.random.default_rng(random_seed).permutation(len(X))
    cumulative = np.cumsum([split[name] for name in SPLIT_NAMES])
    bounds = np.concatenate(([0], np.round(cumulative / 100 * len(X)).astype(int)))

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    train_idx = perm[bounds[0]:bounds[1]]
    scaler_X.fit(X[train_idx])
    scaler_y.fit(y[train_idx].reshape(-1, 1))

    splits = {}
    for i, name in enumerate(SPLIT_NAMES):
        idx = perm[bounds[i]:bounds[i + 1]]
        if split[name] == 0:
            splits[name] = (None, None)
            continue
        splits[name] = (
            scaler_X.transform(X[idx]),
            scaler_y.transform(y[idx].reshape(-1, 1)).ravel(),
        )
    return splits
