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
    "conformal_predictor" : "Standard conformal predictor",
    "normalized_cp_knn_dist" : "CP normalized using KNN on distance",
    "normalized_cp_knn_std" : "CP normalized using KNN on standard deviation",
    "normalized_cp_knn_res" : "CP normalized using KNN on OOB residuals",
    "normalized_cp_norm_var" : "CP normalized using variance of ensemble predictors",
    "mondrian_cp" : "Mondrian CP",
    "symbolic_regression_cp" : "Symbolic Regression CP",
    "symbolic_regression_bin_crossfit" : "Symbolic Regression CP (bin-crossfit loss)",
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
    "#1baf7a", # aqua
    "#eda100", # yellow
    "#e87ba4", # magenta
    "#008300", # green
    "#4a3aa7", # violet
    "#e34948", # red
]
_METHOD_ORDER = [
    "conformal_predictor",
    "normalized_cp_knn_dist",
    "normalized_cp_knn_std",
    "normalized_cp_knn_res",
    "normalized_cp_norm_var",
    "mondrian_cp",
    "symbolic_regression_cp",
    "symbolic_regression_bin_crossfit",
]
METHOD_COLORS = dict(zip(_METHOD_ORDER, _CATEGORICAL_PALETTE))
_FALLBACK_COLOR = "#898781" # muted ink, for any method key not listed above


def plot_confidence_intervals(y, y_pred, y_pred_ci):
    """
    Plot measured values, point predictions, and their confidence
    intervals, for a handful of samples sorted by increasing target value.
    """
    # sort y_test values from small to big, along with y_pred_ci
    # using a list is pretty slow, there is probably a smarter way to do this
    # with numpy arrays, but the data set sizes should be small, so who cares
    y_and_ci = []
    for i in range(0, len(y)):
        y_and_ci.append([y[i], y_pred[i], y_pred_ci[i]])
    y_and_ci = sorted(y_and_ci, key=lambda x: x[0])

    fig, ax = plt.subplots()

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


def save_confidence_interval_plot(method, y_test, y_test_pred, confidence_intervals,
                                   dataset_name, coverage, ci_amplitude_median, save_path):
    """
    Build the confidence-interval plot (plot_confidence_intervals) for a
    handful of test samples and save it to save_path.
    """
    fig, ax = plot_confidence_intervals(y_test[:20], y_test_pred[:20], confidence_intervals[:20])

    title = "%s on data set \"%s\" (coverage=%.4f, median=%.2f)" % (
        translations.get(method, method), dataset_name, coverage, ci_amplitude_median)
    ax.set_title(title)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_method_pareto(methods, medians, coverages, translations=translations):
    """
    Scatter one point per method: (coverage, median CI amplitude), taken
    directly from the medians/coverages dicts (one scalar per method).
    """
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

    # invert x-axis, so that the plot is more readable
    ax.invert_xaxis()

    ax.set_xlabel("coverage on the test set")
    ax.set_ylabel("median amplitude of the confidence intervals")
    # compact legend below the axes, wrapped into columns -- stays out of
    # the way of the data (unlike loc='best', which can land on top of
    # points) without ballooning the figure width the way a right-hand
    # outside-axes legend would
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              title=None, fontsize='small', markerscale=0.7, framealpha=0.85)

    return fig, ax


def save_method_pareto_plot(methods, medians, coverages, title, save_path):
    """
    Build the Pareto scatter plot (plot_pareto) and save it to save_path.
    """
    fig, ax = plot_method_pareto(methods, medians, coverages)
    ax.set_title(title)
    # bbox_inches='tight' recomputes the layout at save time, so the
    # legend placed outside the axes (see plot_pareto) doesn't get clipped
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_target_distribution(y_values, dataset_name, save_path):
    """
    Histogram of the raw target values for one dataset.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(y_values, bins=60, ax=ax, color=METHOD_COLORS["conformal_predictor"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Count')
    ax.set_title(f"Distribution of target in '{dataset_name}'")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_pareto_fronts(fronts, path, title="Pareto fronts (final population)"):
    """Plot successive Pareto fronts (complexity vs. loss, log scale) and save to `path`.

    `fronts` is the list of DataFrames returned by
    `recompute_pareto_fronts_from_population` for a single output (each needs
    `complexity` and `loss` columns).
    """
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for i, front_df in enumerate(fronts):
        color = _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
        ax.plot(
            front_df["complexity"],
            front_df["loss"],
            marker="o",
            markersize=6,
            linewidth=1.5,
            color=color,
            label=f"Front {i + 1}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Complexity", color="#0b0b0b")
    ax.set_ylabel("Loss (log scale)", color="#0b0b0b")
    ax.set_title(title, color="#0b0b0b")

    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    if len(fronts) >= 2:
        ax.legend(frameon=False, labelcolor="#52514e")

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)