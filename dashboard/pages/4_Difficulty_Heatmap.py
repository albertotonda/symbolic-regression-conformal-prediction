# -*- coding: utf-8 -*-
"""Heatmap: rows = datasets, columns = difficulty (sigma) deciles, for one
method at a time -- coverage or median width within each (dataset, decile)
cell, binned independently per dataset since sigma scale varies by dataset."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

N_DECILES = 10

st.set_page_config(page_title="Difficulty Heatmap", layout="wide")
st.title("Coverage / width by difficulty decile")

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

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Method", options=all_methods, format_func=data.method_label, key="heatmap_method",
    )
with col2:
    metric = st.selectbox(
        "Metric", options=["Coverage", "Median width"], key="heatmap_metric",
    )

sigma_col = f"sigma_{method}"
value_col = f"covered_{method}" if metric == "Coverage" else f"width_{method}"

rows = []
included, excluded, constant_sigma = 0, 0, 0
for ds in all_datasets:
    df = per_point_by_dataset.get(ds)
    if df is None or sigma_col not in df.columns:
        excluded += 1
        continue
    if df[sigma_col].nunique() <= 1:
        # a constant sigma (e.g. standard_cp/mondrian_cp, which don't have a
        # real per-point difficulty score) can't be split into deciles --
        # pd.qcut quietly returns all-NaN bins rather than raising, so this
        # would otherwise render a blank row instead of a clear message.
        constant_sigma += 1
        continue
    included += 1
    # rank first, then qcut the ranks: with plain qcut, tied sigma values
    # (common for e.g. the ensemble-variance estimator) make duplicates="drop"
    # silently collapse to fewer than N_DECILES bins -- and a *different*
    # bin count per dataset means column "D9" isn't the same difficulty
    # percentile in every row, which breaks the whole point of this heatmap
    # (comparing datasets side by side). Ranking first guarantees unique
    # values, so every dataset always gets exactly N_DECILES equal-size bins.
    deciles = pd.qcut(df[sigma_col].rank(method="first"), N_DECILES, labels=False)
    agg = df.groupby(deciles)[value_col].agg("mean" if metric == "Coverage" else "median")
    row = {int(d): agg.loc[d] for d in agg.index}
    row["dataset_name"] = ds
    rows.append(row)

if not rows:
    if constant_sigma:
        st.info(
            f"**{data.method_label(method)}** doesn't have a real per-point difficulty "
            "score (its sigma is constant), so it can't be split into difficulty deciles. "
            "Pick a different method."
        )
    else:
        st.info(
            f"No dataset in this run has per-point data for **{data.method_label(method)}**. "
            "Pick a different method, or a newer run."
        )
    st.stop()

matrix = pd.DataFrame(rows).set_index("dataset_name")
matrix = matrix.reindex(sorted(matrix.columns), axis=1)

if excluded or constant_sigma:
    st.caption(
        f"{included} dataset(s) included, {excluded} excluded (no per-point data), "
        f"{constant_sigma} excluded (constant sigma for this method)."
    )

fig = go.Figure(
    data=go.Heatmap(
        z=matrix.values,
        x=[f"D{c + 1}" for c in matrix.columns],
        y=matrix.index,
        colorscale="Blues",
        colorbar=dict(title=metric),
        hovertemplate="dataset=%{y}<br>decile=%{x}<br>" + metric.lower() + "=%{z:.3f}<extra></extra>",
    )
)
fig.update_layout(
    width=data.DETAIL_SIZE,
    height=max(data.DETAIL_SIZE * 0.5, 40 * len(matrix.index) + 150),
    xaxis_title="Difficulty decile (increasing sigma)",
    yaxis_title=None,
    title=f"{metric} by difficulty decile — {data.method_label(method)}",
)
st.plotly_chart(fig, width="content")
