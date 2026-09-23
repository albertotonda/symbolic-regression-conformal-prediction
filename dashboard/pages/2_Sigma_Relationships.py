# -*- coding: utf-8 -*-
"""Sigma vs. residuals/width: X = log10(target), Y = raw difficulty score
(sigma), one point per sample, one subplot per method -- the direct,
unbinned relationship between a method's own difficulty estimate and an
actual outcome. Calibration-side sigma has no notion of interval width or
coverage (split conformal never scores the calibration set itself), so
"Width" is only offered for the test set.

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

st.title("Sigma vs. residuals / width")

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
datasets_with_test = [ds for ds in all_datasets if data.has_per_point_data(run_path, ds)]
datasets_with_cal = [ds for ds in all_datasets if data.has_calibration_per_point_data(run_path, ds)]
datasets_with_data = sorted(set(datasets_with_test) | set(datasets_with_cal))

if not datasets_with_data:
    st.info(
        "No per-point data available for this run — it predates "
        "`src/run_sigma_sr.py` writing `testing_data.csv`/`calibration_data.csv`/"
        "`methods_sigmas_*.csv` (or the older test-only `per_point.csv`). Re-run "
        "the experiment, or pick a newer run from the Home page."
    )
    st.stop()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    dataset = st.selectbox("Dataset", options=datasets_with_data, key="sigma_rel_dataset")
with col2:
    source_options = [s for s, avail in [("Test", dataset in datasets_with_test),
                                          ("Calibration", dataset in datasets_with_cal)] if avail]
    source = st.selectbox("Data", options=source_options, key="sigma_rel_source")
with col3:
    target_options = ["Residual", "Width"] if source == "Test" else ["Residual"]
    target = st.selectbox("vs.", options=target_options, key="sigma_rel_target")

df = data.load_per_point(run_path, dataset) if source == "Test" else data.load_per_point_calibration(run_path, dataset)
all_methods = data.per_point_methods(df)

methods = st.multiselect(
    "Methods", options=all_methods, default=all_methods,
    format_func=data.method_label,
    key="sigma_rel_methods",
)

if not methods:
    st.warning("Select at least one method.")
    st.stop()

target_col_by_method = (
    {m: f"width_{m}" for m in methods} if target == "Width" else {m: "abs_residual" for m in methods}
)
target_label = "abs residual" if target == "Residual" else "width"

cols = min(4, len(methods))
rows = math.ceil(len(methods) / cols)
fig = make_subplots(rows=rows, cols=cols, subplot_titles=[data.method_label(m) for m in methods])

for i, m in enumerate(methods):
    row, col = i // cols + 1, i % cols + 1
    log_target = np.log10(df[target_col_by_method[m]].clip(lower=1e-12))
    fig.add_trace(
        go.Scattergl(
            x=log_target, y=df[f"sigma_{m}"], mode="markers",
            marker=dict(size=5, color=data.method_color(m), opacity=0.55),
            showlegend=False,
            hovertemplate=(
                f"<b>{data.method_label(m)}</b><br>"
                f"log10({target_label})=" + "%{x:.3f}<br>sigma=%{y:.4g}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )
    fig.update_xaxes(title=f"log10({target_label})", row=row, col=col)
    fig.update_yaxes(title="sigma", row=row, col=col)

fig.update_layout(
    width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
    showlegend=False,
    margin=dict(t=60),
    title=f'"{dataset}" — {source} sigma vs. {target_label}',
)
st.plotly_chart(fig, width="content")
