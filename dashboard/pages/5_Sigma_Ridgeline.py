# -*- coding: utf-8 -*-
"""Ridgeline plot of each method's sigma distribution, for one dataset.

All lanes share one linear sigma x-axis. Lanes are ordered top-to-bottom in
the canonical method order -- a density curve is order-invariant over the
underlying points, so "sorted by increasing absolute residual" isn't
meaningful for placing points within a lane; it only matters for the
scatter/line plots (pages 2 and 3).
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

LANE_HEIGHT = 0.85
N_GRID = 300


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


st.title("Sigma distributions (ridgeline)")

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
        "No dataset in this run has per-point data — it predates `src/run_sigma_sr.py` "
        "writing `testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` "
        "(or the older `per_point.csv`). Re-run the experiment to generate it, or "
        "pick a newer run on the Home page."
    )
    st.stop()

dataset = st.selectbox("Dataset", options=datasets_with_data, key="ridgeline_dataset")
df = data.load_per_point(run_path, dataset)
methods_all = data.per_point_methods(df)

# a constant sigma (e.g. standard_cp/mondrian_cp, which don't have a real
# per-point difficulty score) has no distribution to draw, and gaussian_kde
# raises on zero-variance input -- so those methods aren't selectable here.
all_methods = [m for m in methods_all if df[f"sigma_{m}"].nunique() > 1]
constant_methods = [m for m in methods_all if m not in all_methods]

if not all_methods:
    st.info("No selected method has a non-constant difficulty score for this dataset.")
    st.stop()
if constant_methods:
    st.caption(
        "Not shown (constant sigma, no distribution to draw): "
        + ", ".join(data.method_label(m) for m in constant_methods)
    )

methods = st.multiselect(
    "Methods", options=all_methods, default=all_methods,
    format_func=data.method_label,
    key="ridgeline_methods",
)
if not methods:
    st.warning("Select at least one method.")
    st.stop()

# keep the canonical order (all_methods) among the selected subset, so
# lane order stays stable regardless of multiselect click order
ordered_methods = [m for m in all_methods if m in methods]
n = len(ordered_methods)

sigmas = {m: df[f"sigma_{m}"].to_numpy() for m in ordered_methods}
x_min = min(v.min() for v in sigmas.values())
x_max = max(v.max() for v in sigmas.values())
pad = 0.05 * (x_max - x_min or 1.0)
x_grid = np.linspace(x_min - pad, x_max + pad, N_GRID)

fig = go.Figure()
for i, m in enumerate(ordered_methods):
    offset = n - 1 - i  # canonical-order method 0 drawn at the top
    kde = gaussian_kde(sigmas[m])
    density = kde(x_grid)
    peak = density.max()
    scaled = (density / peak * LANE_HEIGHT) if peak > 0 else density

    fig.add_trace(go.Scatter(
        x=x_grid, y=np.full(N_GRID, offset), mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=offset + scaled, mode="lines", fill="tonexty",
        line=dict(width=1.5, color=data.method_color(m)),
        fillcolor=_hex_to_rgba(data.method_color(m), 0.6),
        showlegend=False,
        hovertemplate=f"<b>{data.method_label(m)}</b><br>sigma=%{{x:.3g}}<extra></extra>",
    ))
    fig.add_annotation(
        x=x_min - pad, y=offset + 0.05, text=data.method_label(m),
        showarrow=False, xanchor="left", font=dict(size=12),
    )

fig.update_xaxes(title="sigma")
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[-0.2, n - 1 + LANE_HEIGHT + 0.3])
fig.update_layout(
    width=int(data.DETAIL_SIZE * 1.1),
    height=max(int(data.DETAIL_SIZE * 0.6), 90 * n + 150),
    title=f'"{dataset}"',
    plot_bgcolor="white",
)
st.plotly_chart(fig, width="content")
