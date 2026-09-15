# -*- coding: utf-8 -*-
"""
Extra functions
"""

import os
from datetime import datetime

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