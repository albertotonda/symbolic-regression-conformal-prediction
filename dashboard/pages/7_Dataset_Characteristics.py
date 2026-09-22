# -*- coding: utf-8 -*-
"""Dataset characteristics vs. method performance: joins each dataset's
OpenML-CTR23 metadata (size, dimensionality, this run's own base-regressor
R2) onto its median-width ratio between two chosen methods, to see whether
a method's relative performance is predictable from what the dataset looks
like (e.g. "SR only wins on datasets above N samples")."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.set_page_config(page_title="Dataset Characteristics", layout="wide")
st.title("Dataset characteristics vs. method performance")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

results_df = data.load_results(run_path)
all_methods = data.discover_methods(results_df)
if len(all_methods) < 2:
    st.warning("Need at least two methods with results to compare.")
    st.stop()

characteristics = data.load_dataset_characteristics()
merged = results_df.merge(characteristics, on="dataset_name", how="left", suffixes=("", "_meta"))
n_matched = merged["n_samples"].notna().sum()
if n_matched < len(merged):
    st.caption(
        f"{n_matched}/{len(merged)} datasets matched to OpenML-CTR23 metadata by name "
        f"(the rest aren't in `results/OpenML-CTR23-statistics-500-estimators-10-fold-cv.csv`)."
    )
merged = merged[merged["n_samples"].notna()]
if merged.empty:
    st.warning("No dataset in this run matched the OpenML-CTR23 metadata table.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    default_a = next((m for m in all_methods if m.startswith("sr_")), all_methods[0])
    method_a = st.selectbox("Method A (numerator)", options=all_methods,
                             index=all_methods.index(default_a), format_func=data.method_label,
                             key="charact_method_a")
with col2:
    default_b = "standard_cp" if "standard_cp" in all_methods else all_methods[0]
    method_b = st.selectbox("Method B (denominator)", options=all_methods,
                             index=all_methods.index(default_b), format_func=data.method_label,
                             key="charact_method_b")

merged["width_ratio"] = merged[f"{method_a}_median"] / merged[f"{method_b}_median"]
merged = merged.dropna(subset=["width_ratio"])

CHARACTERISTICS = {
    "n_samples (log10)": ("n_samples", True),
    "n_features": ("n_features", False),
    "missing_data": ("missing_data", False),
    "categorical_features": ("categorical_features", False),
    "r2 (this run's base regressor)": ("r2", False),
}
x_label = st.selectbox("Dataset characteristic (x-axis)", options=list(CHARACTERISTICS.keys()),
                        key="charact_x")
x_col, log_x = CHARACTERISTICS[x_label]
x_values = np.log10(merged[x_col]) if log_x else merged[x_col]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_values, y=merged["width_ratio"], mode="markers+text",
    text=merged["dataset_name"], textposition="top center",
    textfont=dict(size=9),
    marker=dict(size=11, color=data.method_color(method_a), line=dict(width=1, color="white")),
    hovertemplate=(
        "<b>%{text}</b><br>" + x_label + "=%{x:.3g}<br>"
        f"{data.method_label(method_a)} / {data.method_label(method_b)}" + "=%{y:.3f}<extra></extra>"
    ),
))
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6,
              annotation_text="equal width", annotation_position="bottom right")

fig.update_xaxes(title=x_label)
fig.update_yaxes(title=f"Median width ratio: {data.method_label(method_a)} / {data.method_label(method_b)}")
fig.update_layout(
    width=data.DETAIL_SIZE * 1.2, height=data.DETAIL_SIZE,
    title="Width ratio vs. dataset characteristics",
)
st.plotly_chart(fig, width="content")

r = np.corrcoef(x_values, merged["width_ratio"])[0, 1] if len(merged) > 1 else float("nan")
st.caption(f"Pearson correlation (x, ratio): {r:.3f} over {len(merged)} dataset(s).")
