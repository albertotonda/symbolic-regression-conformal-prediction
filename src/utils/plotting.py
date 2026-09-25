# -*- coding: utf-8 -*-
"""
Every matplotlib/seaborn figure produced by the experiment scripts lives
here, so plot appearance can be iterated on without touching the statistics
computations (evaluate.py) or the experiment orchestration (run_*.py,
experiments/*.py).
"""

import matplotlib
matplotlib.use("Agg") # headless batch scripts, only ever save figures to file
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style='darkgrid')


# translates internal naming convention to readable strings for plot
# legends/titles; methods with no entry here just fall back to their raw key
translations = {
    "standard_cp" : "Standard conformal predictor",
    "knn_dist" : "CP normalized using KNN on distance",
    "knn_std" : "CP normalized using KNN on standard deviation",
    "knn_res" : "CP normalized using KNN on OOB residuals",
    "var" : "CP normalized using variance of ensemble predictors",
    "mondrian_cp" : "Mondrian CP",
    "sr_bin_crossfit" : "Symbolic Regression CP (bin-crossfit loss)",
    "sr_pinball" : "Symbolic Regression CP (pinball loss)",
    }

# a fixed, colorblind-checked categorical palette (8 hues, order matters --
# it's the CVD-safety mechanism, not cosmetic; see dataviz skill's
# references/palette.md), assigned by method identity rather than by
# plot-order/index -- so a method keeps the same color across every figure
# it appears in, even when a run only computes a subset of methods (e.g.
# non-RandomForest predictors in run_interval_sr.py skip several of them).
_CATEGORICAL_PALETTE = [
    "#2a78d6", # blue
    "#eb6834", # orange
    "#eda100", # yellow
    "#e87ba4", # magenta
    "#008300", # green
    "#4a3aa7", # violet
    "#e34948", # red
]
_METHOD_ORDER = [
    "standard_cp",
    "knn_dist",
    "knn_std",
    "knn_res",
    "var",
    "mondrian_cp",
    "sr_bin_crossfit",
    "sr_pinball",
]
METHOD_COLORS = dict(zip(_METHOD_ORDER, _CATEGORICAL_PALETTE))
_FALLBACK_COLOR = "#898781" # muted ink, for any method key not listed above


def plot_confidence_intervals(method, y_test, y_test_pred, confidence_intervals,
                               dataset_name, coverage, ci_amplitude_median, save_path):
    """
    Plot measured values, point predictions, and their confidence
    intervals, for a handful of test samples sorted by increasing target
    value, and save the figure to save_path.
    """
    print(f"Plotting confidence intervals for {method}")
    y = y_test[:20]
    y_pred = y_test_pred[:20]
    y_pred_ci = confidence_intervals[:20]

    # sort (y, y_pred, ci) tuples by y ascending; list-based since these are small samples
    y_and_ci = []
    for i in range(0, len(y)):
        y_and_ci.append([y[i], y_pred[i], y_pred_ci[i]])
    y_and_ci = sorted(y_and_ci, key=lambda x: x[0])

    fig, ax = plt.subplots()

    x = range(0, len(y))
    ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
    ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")

    ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][1] for x in y_and_ci], color='orange', alpha=0.3)

    ax.set_xlabel("Samples sorted by increasing value of target")
    ax.set_ylabel("Value of target y")
    ax.legend(loc='best')

    title = "%s on data set \"%s\" (coverage=%.4f, median=%.2f)" % (
        translations.get(method, method), dataset_name, coverage, ci_amplitude_median)
    ax.set_title(title)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pareto(methods, medians, coverages, title, save_path, translations=translations):
    """
    Scatter one point per method: (coverage, median CI amplitude), taken
    directly from the medians/coverages dicts (one scalar per method), and
    save the figure to save_path.
    """
    print("Plotting pareto for method comparison")
    labels = [translations.get(m, m) if translations is not None else m for m in methods]
    palette = {label: METHOD_COLORS.get(m, _FALLBACK_COLOR) for m, label in zip(methods, labels)}

    df = pd.DataFrame({
        "coverage": [coverages[m] for m in methods],
        "median": [medians[m] for m in methods],
        "method": labels,
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=df, x="coverage", y="median", hue="method", hue_order=labels,
        palette=palette, s=90, edgecolor="white", linewidth=0.8, ax=ax,
    )

    ax.invert_xaxis()

    ax.set_xlabel("coverage on the test set")
    ax.set_ylabel("median amplitude of the confidence intervals")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              title=None, fontsize='small', markerscale=0.7, framealpha=0.85)

    ax.set_title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curve(steps, losses, label, save_path):
    """
    Best loss on the Pareto front over the course of one PySR search
    (logged throughout the run via TensorBoardLoggerSpec), log-scale since SR
    losses commonly span orders of magnitude as the search converges.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, losses, color=_CATEGORICAL_PALETTE[0], linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Search step')
    ax.set_ylabel('Best loss (Pareto front)')
    ax.set_title(f"{label}: loss vs. search step")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
