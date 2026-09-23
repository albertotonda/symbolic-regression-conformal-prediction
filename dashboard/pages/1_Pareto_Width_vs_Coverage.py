# -*- coding: utf-8 -*-
"""Pareto plot: median CI width vs. coverage, one point per method.

A grid view shows every dataset's Pareto front side by side for easy
cross-dataset comparison; a detail view shows one dataset at a time, larger.
"""

import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

CELL_SIZE, LEGEND_MARGIN, DETAIL_SIZE = data.CELL_SIZE, data.LEGEND_MARGIN, data.DETAIL_SIZE

st.title("Pareto: median CI width vs. coverage")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

df = data.load_results(run_path)
all_methods = data.discover_methods(df)
all_datasets = sorted(df["dataset_name"].unique())
target_coverage = data.target_coverage(run_path)

col1, col2 = st.columns([3, 1])
with col1:
    methods = st.multiselect(
        "Methods", options=all_methods, default=all_methods,
        format_func=data.method_label,
        key="pareto_methods",
    )
with col2:
    st.metric("Target coverage", f"{target_coverage:.2f}")

if not methods:
    st.warning("Select at least one method.")
    st.stop()


def _add_method_traces(fig, row_data, methods, seen_methods, row=None, col=None, marker_size=10):
    for m in methods:
        cov = row_data.get(f"{m}_coverage")
        med = row_data.get(f"{m}_median")
        if cov is None or med is None or math.isnan(cov) or math.isnan(med):
            continue
        first_time = m not in seen_methods
        seen_methods.add(m)
        fig.add_trace(
            go.Scatter(
                x=[cov], y=[med], mode="markers",
                marker=dict(size=marker_size, color=data.method_color(m),
                            line=dict(width=1, color="white")),
                name=data.method_label(m), legendgroup=m, showlegend=first_time,
                hovertemplate=(
                    f"<b>{data.method_label(m)}</b><br>"
                    "coverage=%{x:.3f}<br>median width=%{y:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )


def _add_target_line(fig, row=None, col=None):
    fig.add_vline(x=target_coverage, line_dash="dash", line_color="gray",
                   opacity=0.5, row=row, col=col)


tab_grid, tab_detail = st.tabs(["Grid (all datasets)", "Single dataset (detail)"])

with tab_grid:
    datasets = st.multiselect(
        "Datasets", options=all_datasets, default=all_datasets, key="pareto_grid_datasets",
    )
    if not datasets:
        st.warning("Select at least one dataset.")
    else:
        cols = min(4, len(datasets))
        rows = math.ceil(len(datasets) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=datasets)
        seen_methods = set()
        for i, ds in enumerate(datasets):
            row, col = i // cols + 1, i % cols + 1
            row_data = df[df["dataset_name"] == ds].iloc[0]
            _add_method_traces(fig, row_data, methods, seen_methods, row=row, col=col)
            _add_target_line(fig, row=row, col=col)
            fig.update_xaxes(autorange="reversed", row=row, col=col)
        fig.update_layout(
            width=cols * CELL_SIZE, height=rows * CELL_SIZE + LEGEND_MARGIN,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0 + 90 / (rows * CELL_SIZE), x=0),
            margin=dict(t=LEGEND_MARGIN),
        )
        st.plotly_chart(fig, width="content")

with tab_detail:
    dataset = st.selectbox("Dataset", options=all_datasets, key="pareto_detail_dataset")
    row_data = df[df["dataset_name"] == dataset].iloc[0]
    fig = go.Figure()
    _add_method_traces(fig, row_data, methods, set(), marker_size=16)
    _add_target_line(fig)
    fig.update_xaxes(autorange="reversed", title="Coverage on the test set")
    fig.update_yaxes(title="Median amplitude of the confidence intervals")
    fig.update_layout(width=DETAIL_SIZE, height=DETAIL_SIZE, title=f'"{dataset}"')
    st.plotly_chart(fig, width="content")
