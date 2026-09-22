# -*- coding: utf-8 -*-
"""Heatmap: rows = datasets, columns = residual-rank deciles -- coverage or
median width within each (dataset, decile) cell.

Binned on the base regressor's *absolute residual* rank, not each method's
own sigma rank: sigma is a per-method estimate, so a method's own sigma
deciles order test points differently from another method's -- binning by
sigma would make column "D5" mean a different set of points in every
method's panel, silently comparing different points side by side. Binning
by residual rank instead means every method's decile columns condition on
the exact same points (the ones actually hardest to predict), which is
what makes the Grid view's method-to-method comparison meaningful. This
also means a method with a constant sigma (standard_cp, mondrian_cp) is no
longer excluded -- unlike a sigma-rank bin, a residual-rank bin is always
well-defined regardless of what the method's own difficulty estimate looks
like.

A grid view shows every method side by side (same dataset row order and
color scale in every panel); a detail view shows one method at a time,
larger.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

N_DECILES = 10
# Diverging around target coverage: white = perfect, red = over-covered,
# blue = under-covered. Reuses the dashboard's own blue/rust hues (see
# data.py's _CATEGORICAL_PALETTE) instead of a stock RdBu scale, so the
# under/over direction is unambiguous regardless of a colorscale's default
# orientation.
COVERAGE_COLORSCALE = [
    [0.0, "#2b6bab"],
    [0.5, "#ffffff"],
    [1.0, "#ab4b2b"],
]

st.set_page_config(page_title="Difficulty Heatmap", layout="wide")
st.title("Coverage / width by residual-rank decile")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

all_datasets = sorted(data.load_results(run_path)["dataset_name"].unique())
per_point_by_dataset = {
    ds: data.load_per_point(run_path, ds) for ds in all_datasets if data.has_per_point_data(run_path, ds)
}

if not per_point_by_dataset:
    st.info(
        "No per-point data in this run — it predates `src/run_sigma_sr.py` "
        "writing `testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` "
        "(or the older `per_point.csv`). Select a newer run, or re-run the "
        "experiment, to see this plot."
    )
    st.stop()

all_methods = sorted({m for df in per_point_by_dataset.values() for m in data.per_point_methods(df)})
target_coverage = data.target_coverage(run_path)

# one decile assignment per dataset, shared by every method (see module
# docstring) -- ranked first so tied residuals (rare, but possible) still
# always split into exactly N_DECILES equal-size bins via qcut.
deciles_by_dataset = {
    ds: pd.qcut(df["abs_residual"].rank(method="first"), N_DECILES, labels=False)
    for ds, df in per_point_by_dataset.items()
}

metric = st.selectbox("Metric", options=["Coverage", "Median width"], key="heatmap_metric")
if metric == "Coverage":
    st.caption(f"White = target coverage ({target_coverage:.2f}); red = over-covered, blue = under-covered.")
elif metric == "Median width":
    st.caption("Color is log-scaled (a linear scale gets washed out by a single wide outlier cell); hover shows the actual width.")


def color_z(matrix, metric):
    """The matrix actually driving cell *color* -- log10 for Median width,
    so one outlier cell doesn't stretch the whole scale and wash out every
    other cell's color (interval width is strictly positive and often
    right-skewed, same reason sigma itself is log-scaled elsewhere in this
    dashboard). Coverage is already a bounded [0, 1] rate, no such skew, so
    it's used as-is. Hover text always shows the true (linear) value via
    `customdata`, never this transformed matrix, so it stays directly
    readable regardless of how color is computed.
    """
    if metric == "Median width":
        return np.log10(matrix.clip(lower=1e-12))
    return matrix


def build_matrix(method, metric):
    """Full-`all_datasets`-shaped decile matrix for one method (NaN rows for
    datasets excluded because they have no per-point data for it), plus the
    exclusion count."""
    value_col = f"covered_{method}" if metric == "Coverage" else f"width_{method}"

    rows = {}
    excluded = 0
    for ds in all_datasets:
        df = per_point_by_dataset.get(ds)
        if df is None or value_col not in df.columns:
            excluded += 1
            continue
        agg = df.groupby(deciles_by_dataset[ds])[value_col].agg("mean" if metric == "Coverage" else "median")
        rows[ds] = {int(d): agg.loc[d] for d in agg.index}

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix = matrix.reindex(all_datasets, axis=0)
    matrix = matrix.reindex(range(N_DECILES), axis=1)
    included = len(all_datasets) - excluded
    return matrix, included, excluded


tab_grid, tab_detail = st.tabs(["Grid (all methods)", "Single method (detail)"])

with tab_grid:
    cols = min(3, len(all_methods))
    rows_n = math.ceil(len(all_methods) / cols)
    fig = make_subplots(
        rows=rows_n, cols=cols,
        subplot_titles=[data.method_label(m) for m in all_methods],
    )
    zmin = zmax = None
    matrices = {}
    for m in all_methods:
        matrix, *_ = build_matrix(m, metric)
        matrices[m] = matrix
        finite = color_z(matrix, metric).values
        finite = finite[~pd.isna(finite)]
        if finite.size:
            zmin = finite.min() if zmin is None else min(zmin, finite.min())
            zmax = finite.max() if zmax is None else max(zmax, finite.max())

    for i, m in enumerate(all_methods):
        row, col = i // cols + 1, i % cols + 1
        matrix = matrices[m]
        fig.add_trace(
            go.Heatmap(
                z=color_z(matrix, metric).values,
                customdata=matrix.values,
                x=[f"D{c + 1}" for c in matrix.columns],
                y=matrix.index,
                coloraxis="coloraxis",
                hovertemplate=(
                    f"<b>{data.method_label(m)}</b><br>dataset=%{{y}}<br>residual decile=%{{x}}<br>"
                    + metric.lower() + "=%{customdata:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    panel_height = max(220, 22 * len(all_datasets) + 90)
    if metric == "Coverage":
        # cmid needs cauto (the Plotly default), so cmin/cmax are left unset
        # here -- passing them alongside cmid would pin the range and stop
        # it from staying centered on target_coverage.
        coloraxis = dict(colorscale=COVERAGE_COLORSCALE, cmid=target_coverage, colorbar=dict(title=metric))
    else:
        coloraxis = dict(colorscale="Blues", cmin=zmin, cmax=zmax, colorbar=dict(title="log10(Median width)"))
    fig.update_layout(
        width=cols * 480, height=rows_n * panel_height,
        coloraxis=coloraxis,
        title=f"{metric} by residual-rank decile — all methods",
    )
    st.plotly_chart(fig, width="content")

with tab_detail:
    method = st.selectbox(
        "Method", options=all_methods, format_func=data.method_label, key="heatmap_method",
    )
    matrix, included, excluded = build_matrix(method, metric)
    matrix_present = matrix.dropna(how="all")

    if matrix_present.empty:
        st.info(
            f"No dataset in this run has per-point data for **{data.method_label(method)}**. "
            "Pick a different method, or a newer run."
        )
        st.stop()

    if excluded:
        st.caption(f"{included} dataset(s) included, {excluded} excluded (no per-point data).")

    if metric == "Coverage":
        color_kwargs = dict(colorscale=COVERAGE_COLORSCALE, zmid=target_coverage, colorbar=dict(title=metric))
    else:
        color_kwargs = dict(colorscale="Blues", colorbar=dict(title="log10(Median width)"))
    fig = go.Figure(
        data=go.Heatmap(
            z=color_z(matrix_present, metric).values,
            customdata=matrix_present.values,
            x=[f"D{c + 1}" for c in matrix_present.columns],
            y=matrix_present.index,
            hovertemplate=(
                "dataset=%{y}<br>residual decile=%{x}<br>"
                + metric.lower() + "=%{customdata:.3f}<extra></extra>"
            ),
            **color_kwargs,
        )
    )
    fig.update_layout(
        width=data.DETAIL_SIZE,
        height=max(data.DETAIL_SIZE * 0.5, 40 * len(matrix_present.index) + 150),
        xaxis_title="Residual-rank decile (increasing |residual|)",
        yaxis_title=None,
        title=f"{metric} by residual-rank decile — {data.method_label(method)}",
    )
    st.plotly_chart(fig, width="content")
