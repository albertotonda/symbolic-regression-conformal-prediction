# -*- coding: utf-8 -*-
"""Dataset analysis: distribution of the target/predictions/residuals on
the calibration and test splits, one dataset at a time -- the only two
splits with raw y saved to disk (there's no per-run training-set y file).
y is normalized (z-scored) per dataset by split_and_normalize_data
(src/utils/data.py), not in the dataset's original units.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.title("Dataset analysis")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

all_datasets = sorted(data.load_results(run_path)["dataset_name"].unique())
dataset = st.selectbox("Dataset", options=all_datasets, key="dataset_analysis_dataset")

test_df = data.load_testing_data(run_path, dataset)
cal_df = data.load_calibration_data(run_path, dataset)
if test_df is None and cal_df is None:
    st.info(
        f'No raw per-point data for "{dataset}" — needs `testing_data.csv` '
        "and/or `calibration_data.csv`, written by `src/run_sigma_sr.py`."
    )
    st.stop()

variable = st.selectbox("Variable", options=["y", "y_pred", "residuals"], key="dataset_analysis_variable")

col_plot, col_desc = st.columns([1.4, 1])

with col_plot:
    fig = go.Figure()
    for split_label, df_split, color in [("Test", test_df, "#2b6bab"), ("Calibration", cal_df, "#ab4b2b")]:
        if df_split is None:
            continue
        fig.add_trace(go.Histogram(
            x=df_split[variable], name=f"{split_label} (n={len(df_split)})",
            marker=dict(color=color), opacity=0.6, histnorm="probability density",
            hovertemplate=f"{split_label}<br>{variable}=" + "%{x:.3g}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        width=data.DETAIL_SIZE * 0.9, height=data.DETAIL_SIZE * 0.65,
        title=f'"{dataset}" — {variable} distribution',
        xaxis_title=variable,
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="content")

with col_desc:
    st.markdown("**Dataset description** (from OpenML)")
    info = data.load_dataset_description(dataset)
    if info is None:
        st.caption(
            "Not available -- either this dataset isn't in the OpenML-CTR23 "
            "metadata table (`results/OpenML-CTR23-statistics-...csv`), or the "
            "live OpenML API call failed (e.g. no network access)."
        )
    else:
        with st.container(height=int(data.DETAIL_SIZE * 0.55), border=True):
            st.markdown(info["description"] or "*No description text provided by OpenML.*")
        st.caption(f"[OpenML dataset page ↗](https://www.openml.org/d/{info['dataset_id']})")

stats_rows = []
for split_label, df_split in [("Test", test_df), ("Calibration", cal_df)]:
    if df_split is None:
        continue
    s = df_split[variable]
    stats_rows.append({
        "Split": split_label, "n": len(s), "mean": s.mean(), "std": s.std(),
        "min": s.min(), "median": s.median(), "max": s.max(),
    })
st.dataframe(pd.DataFrame(stats_rows).set_index("Split"), width="stretch")

if variable == "y":
    st.caption("y is normalized (z-scored) per dataset during preprocessing, not in original units.")
