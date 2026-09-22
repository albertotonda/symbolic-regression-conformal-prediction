# -*- coding: utf-8 -*-
"""Method head-to-head: one point per test sample, interval width from
method A vs. method B, colored by the base regressor's absolute residual --
shows *where* two methods disagree on a point-by-point basis (not just in
aggregate median/coverage), and whether the disagreement tracks how hard
that particular point is."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.set_page_config(page_title="Method Head-to-Head", layout="wide")
st.title("Method head-to-head: per-point interval width")

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
        "No per-test-point data available for this run — it predates "
        "`src/run_sigma_sr.py` writing `testing_data.csv`/`methods_sigmas_test.csv`/"
        "`methods_intervals.csv` (or the older `per_point.csv`)."
    )
    st.stop()

dataset = st.selectbox("Dataset", options=datasets_with_data, key="h2h_dataset")
df = data.load_per_point(run_path, dataset)
all_methods = data.per_point_methods(df)
if len(all_methods) < 2:
    st.warning("Need at least two methods with per-point data.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    default_a = next((m for m in all_methods if m.startswith("sr_")), all_methods[0])
    method_a = st.selectbox("Method A", options=all_methods, index=all_methods.index(default_a),
                             format_func=data.method_label, key="h2h_method_a")
with col2:
    remaining = [m for m in all_methods if m != method_a]
    default_b = "standard_cp" if "standard_cp" in remaining else remaining[0]
    method_b = st.selectbox("Method B", options=remaining, index=remaining.index(default_b),
                             format_func=data.method_label, key="h2h_method_b")

width_a = df[f"width_{method_a}"]
width_b = df[f"width_{method_b}"]
lo = min(width_a.min(), width_b.min())
hi = max(width_a.max(), width_b.max())

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=width_b, y=width_a, mode="markers",
    marker=dict(
        size=6, color=df["abs_residual"], colorscale="Viridis", opacity=0.75,
        colorbar=dict(title="|residual|"),
    ),
    hovertemplate=(
        f"{data.method_label(method_b)}" + "=%{x:.3f}<br>"
        f"{data.method_label(method_a)}" + "=%{y:.3f}<br>"
        "|residual|=%{marker.color:.3f}<extra></extra>"
    ),
))
fig.add_trace(go.Scatter(
    x=[lo, hi], y=[lo, hi], mode="lines",
    line=dict(color="gray", dash="dash", width=1.5),
    showlegend=False, hoverinfo="skip",
))

fig.update_xaxes(title=f"{data.method_label(method_b)} — interval width", type="log")
fig.update_yaxes(title=f"{data.method_label(method_a)} — interval width", type="log")
fig.update_layout(
    width=data.DETAIL_SIZE, height=data.DETAIL_SIZE,
    title=f'"{dataset}"',
)
st.plotly_chart(fig, width="content")

narrower_a = (width_a < width_b).mean()
st.caption(
    f"{data.method_label(method_a)} is narrower on {narrower_a:.1%} of test points "
    f"(points below the diagonal); {data.method_label(method_b)} is narrower on "
    f"{1 - narrower_a:.1%}."
)
