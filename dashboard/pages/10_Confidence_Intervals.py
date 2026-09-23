# -*- coding: utf-8 -*-
"""Confidence interval sanity check: actual target, point prediction, and
interval band for a handful of test points sorted by increasing target
value -- an intuitive "does this look reasonable" view, one method at a
time, that the PNG-to-CSV refactor of run_sigma_sr.py dropped from the
pipeline's own output."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.title("Confidence intervals — sanity check")

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
datasets_with_data = [ds for ds in all_datasets if data.has_interval_detail_data(run_path, ds)]

if not datasets_with_data:
    st.info(
        "No dataset in this run has both `testing_data.csv` and "
        "`methods_intervals.csv` — this view needs the raw prediction/bounds, "
        "which the older `per_point.csv` format didn't keep."
    )
    st.stop()

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    dataset = st.selectbox("Dataset", options=datasets_with_data, key="ci_dataset")

testing = data.load_testing_data(run_path, dataset)
intervals = data.load_intervals(run_path, dataset)
all_methods = data.discover_methods(results_df)
available_methods = [m for m in all_methods if m in intervals["method"].unique()]

with col2:
    method = st.selectbox("Method", options=available_methods, format_func=data.method_label,
                           key="ci_method")
with col3:
    n_points = st.slider("Points shown", min_value=10, max_value=100, value=30, step=5, key="ci_n_points")

method_intervals = intervals[intervals["method"] == method].set_index("index")
df = testing.join(method_intervals[["lower_bound", "upper_bound"]])
df = df.sort_values("y").reset_index(drop=True)

if len(df) > n_points:
    sample_idx = sorted(range(0, len(df), max(len(df) // n_points, 1)))[:n_points]
    df = df.loc[sample_idx].reset_index(drop=True)

x = list(range(len(df)))
color = data.method_color(method)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x + x[::-1], y=list(df["upper_bound"]) + list(df["lower_bound"][::-1]),
    fill="toself", fillcolor=color, opacity=0.25,
    line=dict(width=0), hoverinfo="skip", showlegend=True, name="Interval",
))
fig.add_trace(go.Scatter(
    x=x, y=df["y"], mode="markers", name="Measured value",
    marker=dict(symbol="circle", size=8, color="#2b6bab"),
))
fig.add_trace(go.Scatter(
    x=x, y=df["y_pred"], mode="markers", name="Prediction",
    marker=dict(symbol="x", size=8, color="#ab4b2b"),
))

covered = ((df["y"] >= df["lower_bound"]) & (df["y"] <= df["upper_bound"])).mean()
fig.update_xaxes(title="Test samples, sorted by increasing measured value")
fig.update_yaxes(title="Target value")
fig.update_layout(
    width=data.DETAIL_SIZE * 1.3, height=data.DETAIL_SIZE * 0.75,
    title=f'"{dataset}" — {data.method_label(method)} (coverage on shown points: {covered:.1%})',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="content")
