# -*- coding: utf-8 -*-
"""
Extra functions
"""

import math
import os
from datetime import datetime

import numpy as np

def setup_results_folder(prefix, random_seed):
    """
    Create (and return the path to) a fresh timestamped results folder,
    named results-<prefix>-<random_seed>_<timestamp>, in the current
    working directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = "results-%s-%d_%s" % (prefix, random_seed, timestamp)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


def log_equations(sr_model, task_folder, label):
    """
    Print the full set of candidate equations found by a fitted
    PySRRegressor (its Pareto front of complexity vs. loss), marking the one
    actually selected per sr_model.model_selection, and save the same table
    as a CSV file (<label>_equations.csv) in task_folder.
    """
    equations = sr_model.equations_.copy()
    equations["chosen"] = (equations.index == sr_model.get_best().name)

    print("\n%s: candidate equations (model_selection=%r)" % (label, sr_model.model_selection))
    print(equations[["complexity", "loss", "score", "equation", "chosen"]].to_string(index=False))

    equations.to_csv(os.path.join(task_folder, "%s_equations.csv" % label), index=False)


def fit_with_early_stopping(sr_model, X, y, chunk_size=1, patience=3, min_relative_improvement=1e-3):
    """
    Fit a PySRRegressor in blocks of `chunk_size` iterations (via warm_start),
    stopping once the best loss on the Pareto front has improved by less than
    `min_relative_improvement` for `patience` consecutive blocks in a row.
    Runs the full sr_model.niterations if the plateau is never reached.

    Mutates and returns sr_model, mirroring sr_model.fit's own in-place API.
    """
    total_iterations = sr_model.niterations
    n_blocks = math.ceil(total_iterations / chunk_size)
    sr_model.niterations = chunk_size

    best_loss = np.inf
    stall_count = 0

    for block in range(n_blocks):
        sr_model.warm_start = block > 0
        sr_model.fit(X, y)

        current_best_loss = sr_model.equations_["loss"].min()
        if not np.isfinite(best_loss):
            relative_improvement = np.inf
        elif best_loss == 0:
            relative_improvement = 0.0  # already a perfect fit, no room left to improve
        else:
            relative_improvement = (best_loss - current_best_loss) / best_loss
        best_loss = min(best_loss, current_best_loss)

        if relative_improvement < min_relative_improvement:
            stall_count += 1
            if stall_count >= patience:
                print(
                    "Early stopping: best loss plateaued for %d blocks (ran %d/%d iterations)"
                    % (patience, (block + 1) * chunk_size, total_iterations)
                )
                break
        else:
            stall_count = 0

    sr_model.niterations = total_iterations
    return sr_model


def read_tensorboard_scalar(log_dir, tag):
    """
    Read back one scalar time series logged during a PySRRegressor fit via
    `logger_spec=TensorBoardLoggerSpec(log_dir=log_dir, ...)`.

    Returns (steps, values) as parallel lists. `tag="search/data/summaries/min_loss"`
    is the best loss on the Pareto front, logged throughout the search itself
    (single continuous Julia run, no Python-level restarts).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()
    events = accumulator.Scalars(tag)

    return [e.step for e in events], [e.value for e in events]