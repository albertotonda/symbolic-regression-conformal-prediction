# -*- coding: utf-8 -*-
"""Training dynamics: the SR search's loss-vs-iteration curve, read
directly from TensorBoard logs, plus a convergence-speed comparison across
datasets -- useful even without early stopping turned on, to see whether
some datasets need much longer searches than others."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.set_page_config(page_title="Training Dynamics", layout="wide")
st.title("SR search: loss vs. iteration")

run_path = st.session_state.get("run_path")
if run_path is None:
    runs = data.discover_runs()
    if not runs:
        st.error("No results-* run folders found next to the repo root.")
        st.stop()
    run_path = runs[0]["path"]
    st.warning(f"No run selected on the Home page — defaulting to `{runs[0]['name']}`.")

all_datasets = sorted(data.load_results(run_path)["dataset_name"].unique())
losses_by_dataset = {ds: data.discover_losses(run_path, ds) for ds in all_datasets}
all_losses = sorted({loss for losses in losses_by_dataset.values() for loss in losses})

if not all_losses:
    st.info(
        "No dataset in this run has a Hall-of-Fame file, so there's no SR "
        "loss to show a training curve for."
    )
    st.stop()

loss_name = st.selectbox("SR loss", options=all_losses, key="training_loss")
datasets_with_tb = [
    ds for ds in all_datasets
    if loss_name in losses_by_dataset[ds] and data.tb_log_dir(run_path, ds, loss_name).exists()
]
if not datasets_with_tb:
    st.info(f"No TensorBoard log found for loss `{loss_name}` in this run.")
    st.stop()

config_tolerance = data.early_stop_params(run_path)["min_relative_improvement"]
tolerance_options = sorted({1e-4, 1e-3, 1e-2, 5e-2, config_tolerance})
tolerance = st.select_slider(
    "Convergence tolerance (fraction of total improvement still missing)",
    options=tolerance_options, value=config_tolerance,
    format_func=lambda v: f"{v:.4g}" + ("  (this run's early_stop_min_improvement)" if v == config_tolerance else ""),
    key="training_tolerance",
)

curves = {}
for ds in datasets_with_tb:
    steps, losses = data.load_loss_curve(run_path, ds, loss_name)
    if steps:
        curves[ds] = (steps, losses)

convergence = {
    ds: data.compute_convergence_step(steps, losses, tolerance=tolerance)
    for ds, (steps, losses) in curves.items()
}
total_steps = {ds: steps[-1] for ds, (steps, losses) in curves.items()}
final_loss = {ds: losses[-1] for ds, (steps, losses) in curves.items()}

tab_convergence, tab_curves = st.tabs(["Convergence comparison", "Loss curves"])

with tab_convergence:
    ordered = sorted(curves, key=lambda ds: convergence[ds] / total_steps[ds])
    fractions = [convergence[ds] / total_steps[ds] for ds in ordered]

    fig = go.Figure(go.Bar(
        x=ordered, y=fractions,
        marker=dict(color=data.METHOD_COLORS.get("sr_bin_crossfit", "#4d1fd6")),
        customdata=[[convergence[ds], total_steps[ds], final_loss[ds]] for ds in ordered],
        hovertemplate=(
            "<b>%{x}</b><br>converged at step %{customdata[0]} / %{customdata[1]}<br>"
            "final best loss=%{customdata[2]:.4g}<extra></extra>"
        ),
    ))
    fig.update_yaxes(title="Fraction of logged steps until convergence", range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(title=None, tickangle=30)
    fig.update_layout(
        width=max(data.DETAIL_SIZE, 60 * len(ordered)), height=data.DETAIL_SIZE * 0.7,
        title=f"When does the search converge? — {data.method_label(f'sr_{loss_name}')}",
    )
    st.plotly_chart(fig, width="content")

with tab_curves:
    datasets = st.multiselect("Datasets", options=datasets_with_tb, default=datasets_with_tb,
                               key="training_curve_datasets")
    fig = go.Figure()
    for ds in datasets:
        if ds not in curves:
            continue
        steps, losses = curves[ds]
        fig.add_trace(go.Scatter(
            x=steps, y=losses, mode="lines", name=ds,
            hovertemplate=f"<b>{ds}</b><br>step=%{{x}}<br>loss=%{{y:.4g}}<extra></extra>",
        ))
        conv_step = convergence[ds]
        match_idx = np.flatnonzero(np.asarray(steps) == conv_step)
        conv_loss = losses[match_idx[0]] if match_idx.size else final_loss[ds]
        fig.add_trace(go.Scatter(
            x=[conv_step], y=[conv_loss], mode="markers",
            marker=dict(symbol="star", size=12, color="black"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_yaxes(title="Best loss (Pareto front)", type="log")
    fig.update_xaxes(title="Search step")
    fig.update_layout(
        width=data.DETAIL_SIZE * 1.3, height=data.DETAIL_SIZE,
        title=f"Loss vs. search step (★ = convergence) — {data.method_label(f'sr_{loss_name}')}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="content")
