# -*- coding: utf-8 -*-
"""Sigma vs. coverage: for each method, test points sorted by that method's
own sigma, then a sliding-window empirical coverage curve over that sorted
order -- the direct, unbinned way to see whether a method's own difficulty
estimate actually tracks coverage failures.

Per-point coverage is a 0/1 indicator, too noisy to read raw against a
continuous sigma axis, so it needs *some* smoothing -- a sliding window
over each method's own sorted sigma, rather than discrete quantile bins
(the Difficulty Heatmap's old approach), so there's no arbitrary bin
boundary. Deliberately kept to independent small-multiples (one sigma axis
per method) rather than a shared axis across methods: methods don't rank
test points in the same order by their own sigma, so forcing them onto one
shared axis (like the Heatmap used to) would silently compare different
points across methods -- see 4_Difficulty_Heatmap.py's module docstring for
the same issue, fixed there by binning on residual rank instead.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.title("Sigma vs. coverage (sliding window)")

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

dataset = st.selectbox("Dataset", options=datasets_with_data, key="sigma_cov_dataset")
df = data.load_per_point(run_path, dataset)
methods_all = data.per_point_methods(df)
target_coverage = data.target_coverage(run_path)

# a constant sigma (e.g. standard_cp/mondrian_cp) has no ordering to slide a
# window over -- same reason the Sigma Ridgeline page excludes them.
all_methods = [m for m in methods_all if df[f"sigma_{m}"].nunique() > 1]
constant_methods = [m for m in methods_all if m not in all_methods]

if not all_methods:
    st.info("No method has a non-constant difficulty score for this dataset.")
    st.stop()
if constant_methods:
    st.caption(
        "Not shown (constant sigma, no ordering to slide a window over): "
        + ", ".join(data.method_label(m) for m in constant_methods)
    )

col1, col2 = st.columns([3, 1])
with col1:
    methods = st.multiselect(
        "Methods", options=all_methods, default=all_methods,
        format_func=data.method_label,
        key="sigma_cov_methods",
    )
with col2:
    window_pct = st.slider("Window size (% of points)", min_value=5, max_value=50, value=10, step=5,
                            key="sigma_cov_window")

if not methods:
    st.warning("Select at least one method.")
    st.stop()

n = len(df)
window = max(3, round(n * window_pct / 100))

cols = min(4, len(methods))
rows = math.ceil(len(methods) / cols)
fig = make_subplots(rows=rows, cols=cols, subplot_titles=[data.method_label(m) for m in methods])

for i, m in enumerate(methods):
    row, col = i // cols + 1, i % cols + 1
    order = df[f"sigma_{m}"].to_numpy().argsort()
    sigma_sorted = df[f"sigma_{m}"].to_numpy()[order]
    covered_sorted = df[f"covered_{m}"].to_numpy()[order].astype(float)
    smoothed = pd.Series(covered_sorted).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    fig.add_trace(
        go.Scatter(
            x=sigma_sorted, y=smoothed, mode="lines",
            line=dict(color=data.method_color(m), width=2.5),
            showlegend=False,
            hovertemplate=(
                f"<b>{data.method_label(m)}</b><br>sigma=%{{x:.4g}}<br>"
                "coverage (windowed)=%{y:.3f}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )
    fig.add_hline(y=target_coverage, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    fig.update_xaxes(title="sigma", row=row, col=col)
    fig.update_yaxes(title="Coverage (windowed)", range=[0, 1.05], row=row, col=col)

fig.update_layout(
    width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
    margin=dict(t=60),
    title=f'"{dataset}" — window = {window} of {n} points ({window_pct}%)',
)
st.plotly_chart(fig, width="content")
