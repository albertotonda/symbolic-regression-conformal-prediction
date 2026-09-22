# -*- coding: utf-8 -*-
"""Hall-of-Fame trade-off: every equation on the SR search's final
complexity/loss Pareto front, scattered by (coverage, interval width) and
colored by complexity -- shows whether the chosen equation actually sits at
a good point of that trade-off, or whether a simpler/more complex sibling
would have done as well or better."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.set_page_config(page_title="Hall of Fame Trade-off", layout="wide")
st.title("Hall-of-Fame equation trade-off")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

all_datasets = sorted(data.load_results(run_path)["dataset_name"].unique())
datasets_with_hof = [ds for ds in all_datasets if data.discover_losses(run_path, ds)]

if not datasets_with_hof:
    st.info(
        "No dataset in this run has a Hall-of-Fame file (`hof_<loss>.csv`) — "
        "either the run predates `src/run_sigma_sr.py` saving it, or the SR "
        "step hasn't completed for any dataset yet."
    )
    st.stop()

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    dataset = st.selectbox("Dataset", options=datasets_with_hof, key="hof_dataset")
with col2:
    losses = data.discover_losses(run_path, dataset)
    loss_name = st.selectbox("SR loss", options=losses, key="hof_loss")
with col3:
    metric = st.radio("Width", options=["ci_median", "ci_mean"], key="hof_metric")

df_hof = data.load_hof(run_path, dataset, loss_name)
target_coverage = data.target_coverage(run_path)

chosen_mask = df_hof["Chosen"].astype(bool)
complexity = df_hof.index.to_series(name="Complexity")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_hof["coverage"], y=df_hof[metric], mode="markers",
    marker=dict(
        size=12, color=complexity, colorscale="Plasma",
        colorbar=dict(title="Complexity"),
        line=dict(width=1, color="white"),
    ),
    customdata=list(zip(complexity, df_hof["Loss"], df_hof["Equation"])),
    hovertemplate=(
        "complexity=%{customdata[0]}<br>loss=%{customdata[1]:.4g}<br>"
        "coverage=%{x:.3f}<br>" + metric + "=%{y:.3f}<br>"
        "%{customdata[2]}<extra></extra>"
    ),
    name="Equations",
))
if chosen_mask.any():
    fig.add_trace(go.Scatter(
        x=df_hof.loc[chosen_mask, "coverage"], y=df_hof.loc[chosen_mask, metric],
        mode="markers",
        marker=dict(size=20, color=complexity[chosen_mask], colorscale="Plasma",
                    cmin=complexity.min(), cmax=complexity.max(),
                    line=dict(width=2.5, color="black")),
        customdata=list(zip(complexity[chosen_mask], df_hof.loc[chosen_mask, "Loss"],
                             df_hof.loc[chosen_mask, "Equation"])),
        hovertemplate=(
            "<b>Chosen</b><br>complexity=%{customdata[0]}<br>loss=%{customdata[1]:.4g}<br>"
            "coverage=%{x:.3f}<br>" + metric + "=%{y:.3f}<br>"
            "%{customdata[2]}<extra></extra>"
        ),
        name="Chosen equation",
        showlegend=True,
    ))

fig.add_vline(x=target_coverage, line_dash="dash", line_color="gray", opacity=0.5)
fig.update_xaxes(autorange="reversed", title="Coverage on the test set")
fig.update_yaxes(title=f"Interval {metric.split('_')[1]} amplitude")
fig.update_layout(
    width=data.DETAIL_SIZE, height=data.DETAIL_SIZE,
    title=f'"{dataset}" — {data.method_label(f"sr_{loss_name}")}',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="content")

with st.expander("Full Hall of Fame table", expanded=False):
    st.dataframe(
        df_hof[["Loss", "Equation", "Chosen", "coverage", "ci_median", "ci_mean"]]
        .sort_index(),
        width="stretch",
    )
