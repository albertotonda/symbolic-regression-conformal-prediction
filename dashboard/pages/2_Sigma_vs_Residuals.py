# -*- coding: utf-8 -*-
"""Sigma vs. residuals: X = log10(absolute residual), Y = raw difficulty
score (sigma), one point per test sample, one subplot per method.

Each method's sigma lives on its own scale (a KNN-distance sigma and an
ensemble-variance sigma can differ by orders of magnitude), so each
subplot keeps its own independent y-axis rather than sharing one.
"""

import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.set_page_config(page_title="Sigma vs Residuals", layout="wide")
st.title("Sigma vs. residuals")

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
        "`methods_intervals.csv` (or the older `per_point.csv`). Re-run the "
        "experiment to get per-point sigma/residual data, or pick a newer run "
        "from the Home page."
    )
    st.stop()

dataset = st.selectbox("Dataset", options=datasets_with_data, key="sigma_vs_residuals_dataset")

df = data.load_per_point(run_path, dataset)
all_methods = data.per_point_methods(df)

methods = st.multiselect(
    "Methods", options=all_methods, default=all_methods,
    format_func=data.method_label,
    key="sigma_vs_residuals_methods",
)

if not methods:
    st.warning("Select at least one method.")
    st.stop()

log_resid = np.log10(df["abs_residual"].clip(lower=1e-12))

cols = min(4, len(methods))
rows = math.ceil(len(methods) / cols)
fig = make_subplots(rows=rows, cols=cols, subplot_titles=[data.method_label(m) for m in methods])

for i, m in enumerate(methods):
    row, col = i // cols + 1, i % cols + 1
    fig.add_trace(
        go.Scattergl(
            x=log_resid, y=df[f"sigma_{m}"], mode="markers",
            marker=dict(size=5, color=data.method_color(m), opacity=0.55),
            showlegend=False,
            hovertemplate=(
                f"<b>{data.method_label(m)}</b><br>"
                "log10(abs residual)=%{x:.3f}<br>sigma=%{y:.4g}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )
    fig.update_xaxes(title="log10(abs residual)", row=row, col=col)
    fig.update_yaxes(title="sigma", row=row, col=col)

fig.update_layout(
    width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
    showlegend=False,
    margin=dict(t=60),
    title=f'"{dataset}"',
)
st.plotly_chart(fig, width="content")
