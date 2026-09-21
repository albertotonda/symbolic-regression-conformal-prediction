# -*- coding: utf-8 -*-
"""Interval width vs. residual rank: test points binned into equal-count
bins by ascending absolute residual, median width per bin, one line per
method — shows whether a method's intervals actually widen where the base
regressor is more wrong."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

N_BINS = 15

st.set_page_config(page_title="Width by Residual Rank", layout="wide")
st.title("Interval width by residual rank")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

results_df = data.load_results(run_path)
all_datasets = sorted(results_df["dataset_name"].unique())
datasets_with_data = [ds for ds in all_datasets if data.has_per_point_data(run_path, ds)]

if not datasets_with_data:
    st.info(
        "This run has no per-point data — it predates `src/run_sigma_sr.py` "
        "writing `testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` "
        "(or the older `per_point.csv`). Pick a newer run, or re-run the experiment."
    )
    st.stop()

dataset = st.selectbox("Dataset", options=datasets_with_data)
df = data.load_per_point(run_path, dataset)
all_methods = data.per_point_methods(df)

methods = st.multiselect(
    "Methods", options=all_methods, default=all_methods,
    format_func=data.method_label,
)
if not methods:
    st.warning("Select at least one method.")
    st.stop()

abs_residual = df["abs_residual"].to_numpy()
bins = np.array_split(np.argsort(abs_residual), N_BINS)
bin_indices = np.arange(len(bins))

fig = go.Figure()
for m in methods:
    widths = df[f"width_{m}"].to_numpy()
    bin_medians = [np.median(widths[b]) for b in bins]
    fig.add_trace(go.Scatter(
        x=bin_indices, y=bin_medians, mode="lines+markers",
        line=dict(color=data.method_color(m), width=2.5),
        marker=dict(size=7),
        name=data.method_label(m),
        hovertemplate=f"<b>{data.method_label(m)}</b><br>bin=%{{x}}<br>median width=%{{y:.3f}}<extra></extra>",
    ))

fig.update_xaxes(title="Absolute residual bin (equal count per bin, increasing order)", tickmode="array", tickvals=bin_indices)
fig.update_yaxes(title="Median interval width")
fig.update_layout(
    width=data.DETAIL_SIZE * 1.1, height=data.DETAIL_SIZE * 0.7,
    title=f'"{dataset}"',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="content")
