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
import numpy as np
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


def plot_target_distribution(y_values, dataset_name, save_path):
    """
    Histogram of the raw target values for one dataset.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(y_values, bins=60, ax=ax, color=METHOD_COLORS["standard_cp"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Count')
    ax.set_title(f"Distribution of target in '{dataset_name}'")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_binned_sigma_metric(binned_stats, metric, save_path, highlighted_keys=None,
                            use_complexity=False, cmap="plasma"):
    """
    Coverage or median width vs. binned difficulty score, for a set of
    series sharing the same bin axis.

    `binned_stats` is {key: DataFrame} (needs `bin` and `metric` columns).
    """
    print(f"Plotting {metric.replace("_", " ")} vs binned sigmas for {"equations" if use_complexity else "methods"} comparison")
    highlighted_keys = highlighted_keys or []

    fig, ax = plt.subplots(figsize=(8, 5.5))

    sm = None
    if use_complexity:
        vals = list(binned_stats.keys())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(vals), vmax=max(vals)))

    all_bins = set()
    handles, legend_labels = [], []
    for key, df_bins in binned_stats.items():
        all_bins.update(df_bins["bin"])
        is_highlighted = key in highlighted_keys
        line_color = sm.to_rgba(key) if sm is not None else None

        line, = ax.plot(
            df_bins["bin"], df_bins[metric], marker="o", color=line_color,
            markersize=6 if is_highlighted else 3,
            linewidth=2.5 if is_highlighted else 1,
            alpha=1.0 if is_highlighted else (0.6 if sm is not None else 0.85),
            zorder=3 if is_highlighted else 1,
            markeredgewidth=1.2 if is_highlighted else 0,
        )
        # equation mode: only the highlighted (chosen) equation is labeled
        # method mode (no `complexity`): every series gets a legend entry
        if use_complexity:
            if is_highlighted:
                handles.append(line)
                legend_labels.append("Chosen")
        else:
            handles.append(line)
            legend_labels.append(key)

    ax.set_ylabel(metric)
    ax.set_xlabel("Difficulty quantile bin in increasing order")
    ax.set_xticks(sorted(all_bins))
    if handles:
        ax.legend(handles=handles, labels=legend_labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize='small', framealpha=0.85)
    if sm is not None:
        fig.colorbar(sm, ax=ax, label="Complexity")
    ax.set_title(f"{metric.replace("_", " ").title()} vs binned difficulty scores")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sigma_distributions(sigmas_by_key, title, save_path):
    """
    Violin plot comparing the marginal distribution of each key's
    test-set difficulty score.
    
    `sigmas_by_key` : dict(key: sigmas)
    """
    print("Plotting sigmas distributions")
    rows = []
    for key, sigmas in sigmas_by_key.items():
        sigmas = pd.Series(sigmas, dtype=float)
        rows.extend({"key": key, "sigma": v} for v in sigmas)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(_METHOD_ORDER)), 6))
    sns.violinplot(data=df, x="key", y="sigma", hue="key", order=_METHOD_ORDER,
                   hue_order=_METHOD_ORDER, palette=_CATEGORICAL_PALETTE, legend=False, ax=ax, cut=0)
    ax.set_yscale("log")
    ax.set_xlabel(None)
    ax.set_ylabel("Difficulty score")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_equation_performance_vs_complexity(df_hof, loss_name, dataset_name, save_path):
    """
    Two side-by-side panels (mean, median interval width) scattering every
    Hall-of-Fame equation by (coverage, width), colored by complexity.

    `df_hof` needs `coverage`, `ci_mean`, `ci_median`, `Chosen` columns and
    an index of equation complexity.
    """
    print("Plotting equation performance vs complexity")
    complexity = df_hof.index.values.astype(float)
    vmin, vmax = complexity.min(), complexity.max()
    chosen_mask = df_hof["Chosen"].values.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [("ci_mean", "Interval mean", axes[0]), ("ci_median", "Interval median", axes[1])]

    for column, ylabel, ax in panels:
        sc = ax.scatter(
            df_hof["coverage"], df_hof[column],
            c=complexity, cmap="plasma", vmin=vmin, vmax=vmax,
            s=50, alpha=0.8, edgecolor="white", linewidth=0.3,
        )
        if chosen_mask.any():
            ax.scatter(
                df_hof["coverage"][chosen_mask], df_hof[column][chosen_mask],
                c=complexity[chosen_mask], cmap="plasma", vmin=vmin, vmax=vmax,
                s=200, edgecolor="black", linewidth=1.2, label="Chosen equation",
            )
            ax.legend(loc='best', fontsize='small')
        ax.invert_xaxis()
        ax.set_xlabel("Coverage")
        ax.set_ylabel(ylabel)
        fig.colorbar(sc, ax=ax, label="Complexity")

    fig.suptitle(f"Symbolic regression equations ({loss_name}) on \"{dataset_name}\"")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sigma_vs_residuals(df_hof, abs_res, loss_name, dataset_name, save_path,
                                     n_bins=15, cmap="plasma"):
    """
    For every Hall-of-Fame equation, its test-set difficulty score plotted
    against the base learner's absolute residual, both aggregated (median)
    over `n_bins` equal-frequency bins of residual rank, since raw
    per-point sigma is too noisy to compare across equations. All equations
    share the same residual bins, so curves stay directly comparable
    bin-for-bin.

    `df_hof` needs a `Chosen` column, a `sigmas` column (Python list of
    per-test-point sigma, one entry per row), and an index of
    equation complexity. `abs_res` is the base learner's absolute residual
    on the test set, same length and ordering as each row's `sigmas`.
    """
    print("Plotting sigmas vs binned residuals")
    abs_res = np.asarray(abs_res)
    bins = np.array_split(np.argsort(abs_res), n_bins)
    bin_res_medians = [np.median(abs_res[b]) for b in bins]

    complexity = df_hof.index.values.astype(float)
    vmin, vmax = complexity.min(), complexity.max()
    chosen_mask = df_hof["Chosen"].values.astype(bool)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    fig, ax = plt.subplots()

    for row_complexity, row in df_hof.iterrows():
        sigmas = np.asarray(row["sigmas"])
        is_chosen = bool(row["Chosen"])
        bin_sigma_medians = [np.median(sigmas[b]) for b in bins]
        ax.plot(
            bin_res_medians, bin_sigma_medians, marker="o", markersize=4,
            color=sm.to_rgba(row_complexity),
            alpha=1.0 if is_chosen else 0.6,
            linewidth=3 if is_chosen else 1,
            label="Chosen equation" if is_chosen else None,
        )

    fig.colorbar(sm, ax=ax, label="Complexity")

    ax.set_xlabel("Absolute residuals (binned, median per bin)")
    ax.set_yscale("log")
    ax.set_ylabel("Equation's predicted sigma on test set (median per bin)")
    ax.set_title(f"Sigma test by equation ({loss_name}) on \"{dataset_name}\"")
    if chosen_mask.any():
        ax.legend(loc='best', fontsize='small')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_pareto_fronts(fronts, path, title="Pareto fronts (final population)"):
    """Plot successive Pareto fronts (complexity vs. loss, log scale) and save to `path`.

    `fronts` is the list of DataFrames returned by
    `recompute_pareto_fronts_from_population` for a single output (each needs
    `complexity` and `loss` columns).
    """
    print("Plotting final evolution pareto fronts")
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