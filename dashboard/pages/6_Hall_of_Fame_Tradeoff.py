# -*- coding: utf-8 -*-
"""Hall-of-Fame analysis: everything the SR search's final complexity/loss
Pareto front looked like for one dataset/loss, across five tabs --

- Trade-off: every equation scattered by (coverage, interval width),
  colored by complexity, chosen one highlighted -- does the chosen
  equation actually sit at a good point of the trade-off?
- Sigma vs Outcome: each equation's own per-point sigma against residual or
  width -- raw scatter plus a binned (equal-count bins on the target,
  median sigma per bin) trend line on top, one subplot per equation (own
  axis, not shared -- see 4_Difficulty_Heatmap.py's docstring for why a
  shared axis across different sigma estimators would misalign) -- does a
  more complex equation's difficulty estimate actually track the outcome
  tighter, or is it noise?
- Interval Width: per-equation small multiples, width vs. y_pred or
  right-sizing (|residual| quantile vs. half-width, grouped by the
  equation's own width); see interval_width_view.py -- where does each
  equation spend its width, and is it the right size?
- Conditional Coverage: same per-equation small multiples, sliding-window
  empirical coverage along a selectable x-axis (y_pred or own sigma),
  then mean width vs. worst-group coverage per equation (y_pred bins,
  own-sigma bins, or worst slab); see
  conditional_coverage_view.py -- does added complexity buy real
  conditional calibration?
- Complexity x Decile: coverage/width heatmap, rows = complexity, columns
  = residual-rank decile (shared across every equation, so columns are
  directly comparable row to row) -- where along the complexity path does
  conditional coverage actually improve?
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_coverage_view  # noqa: E402
import interval_width_view  # noqa: E402
import data  # noqa: E402

N_DECILES = 10
N_BINS = 15
COVERAGE_COLORSCALE = [
    [0.0, "#2b6bab"],
    [0.5, "#ffffff"],
    [1.0, "#ab4b2b"],
]

st.title("Hall-of-Fame equations")

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

col1, col2 = st.columns([2, 2])
with col1:
    dataset = st.selectbox("Dataset", options=datasets_with_hof, key="hof_dataset")
with col2:
    losses = data.discover_losses(run_path, dataset)
    loss_name = st.selectbox("SR loss", options=losses, key="hof_loss")

df_hof = data.load_hof(run_path, dataset, loss_name)
target_coverage = data.target_coverage(run_path)
chosen_mask = df_hof["Chosen"].astype(bool)
complexities = sorted(df_hof.index)
chosen_complexity = int(df_hof.index[chosen_mask][0]) if chosen_mask.any() else None


def complexity_color(c):
    lo, hi = complexities[0], complexities[-1]
    frac = (c - lo) / (hi - lo) if hi > lo else 0.5
    return plotly.colors.sample_colorscale("Plasma", [frac])[0]


def panel_title(c):
    return f"C{c} ★" if c == chosen_complexity else f"C{c}"


tab_tradeoff, tab_outcome, tab_width_resid, tab_coverage, tab_heatmap = st.tabs(
    ["Trade-off", "Sigma vs Outcome", "Interval Width", "Conditional Coverage", "Complexity x Decile"]
)

with tab_tradeoff:
    metric = st.radio("Width", options=["ci_median", "ci_mean"], key="hof_metric")
    complexity_series = df_hof.index.to_series(name="Complexity")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hof["coverage"], y=df_hof[metric], mode="markers",
        marker=dict(
            size=12, color=complexity_series, colorscale="Plasma",
            colorbar=dict(title="Complexity"),
            line=dict(width=1, color="white"),
        ),
        customdata=list(zip(complexity_series, df_hof["Loss"], df_hof["Equation"])),
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
            marker=dict(size=20, color=complexity_series[chosen_mask], colorscale="Plasma",
                        cmin=complexity_series.min(), cmax=complexity_series.max(),
                        line=dict(width=2.5, color="black")),
            customdata=list(zip(complexity_series[chosen_mask], df_hof.loc[chosen_mask, "Loss"],
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
            df_hof[["Loss", "Equation", "Chosen", "coverage", "ci_median", "ci_mean"]].sort_index(),
            width="stretch",
        )

has_per_point = data.has_hof_per_point_data(run_path, dataset, loss_name)
if not has_per_point:
    no_data_message = (
        "No per-equation per-point data for this dataset/loss — needs "
        "`testing_data.csv` + `hof_sigmas_test_<loss>.csv` + "
        "`hof_intervals_<loss>.csv`."
    )
else:
    df_pp = data.load_hof_per_point(run_path, dataset, loss_name)

with tab_outcome:
    if not has_per_point:
        st.info(no_data_message)
    else:
        target = st.radio("vs.", options=["Residual", "Width"], key="hof_outcome_target")
        target_label = "abs residual" if target == "Residual" else "width"

        cols = min(4, len(complexities))
        rows = math.ceil(len(complexities) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[panel_title(c) for c in complexities])
        for i, c in enumerate(complexities):
            row, col = i // cols + 1, i % cols + 1
            col_name = "abs_residual" if target == "Residual" else f"width_{c}"
            target_vals = df_pp[col_name].clip(lower=1e-12).to_numpy()
            sigma_vals = df_pp[f"sigma_{c}"].to_numpy()
            log_target = np.log10(target_vals)
            fig.add_trace(
                go.Scattergl(
                    x=log_target, y=sigma_vals, mode="markers",
                    marker=dict(size=4, color=complexity_color(c), opacity=0.22),
                    showlegend=False,
                    hovertemplate=f"C{c}<br>log10({target_label})=%{{x:.3f}}<br>sigma=%{{y:.4g}}<extra></extra>",
                ),
                row=row, col=col,
            )

            bins = np.array_split(np.argsort(target_vals), N_BINS)
            bin_log_target = [np.log10(np.median(target_vals[b])) for b in bins]
            bin_sigma = [np.median(sigma_vals[b]) for b in bins]
            # white halo drawn under the trend line so it stays legible on
            # top of a dense scatter, same treatment as method_comparison.py's
            # Sigma Relationships tab.
            fig.add_trace(
                go.Scatter(
                    x=bin_log_target, y=bin_sigma, mode="lines",
                    line=dict(color="white", width=6),
                    opacity=0.85,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=bin_log_target, y=bin_sigma, mode="lines+markers",
                    line=dict(color=complexity_color(c), width=3.5),
                    marker=dict(size=9, color=complexity_color(c), line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=(
                        f"C{c} (binned)<br>log10(median {target_label})="
                        + "%{x:.3f}<br>median sigma=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            fig.update_xaxes(title=f"log10({target_label})", row=row, col=col)
            fig.update_yaxes(title="sigma", row=row, col=col)
        fig.update_layout(
            width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
            margin=dict(t=60),
            title=f'"{dataset}" — {data.method_label(f"sr_{loss_name}")} equations, sigma vs. {target_label}',
        )
        st.plotly_chart(fig, width="content")

with tab_width_resid:
    if not has_per_point:
        st.info(no_data_message)
    else:
        interval_width_view.render(
            keys=complexities,
            width_by_key={c: df_pp[f"width_{c}"].to_numpy() for c in complexities},
            label=panel_title, color=complexity_color,
            testing=data.load_testing_data(run_path, dataset).reset_index(drop=True),
            target_coverage=target_coverage,
            title=f'"{dataset}" — {data.method_label(f"sr_{loss_name}")} equations',
            key_prefix="hof_width",
        )

with tab_coverage:
    if not has_per_point:
        st.info(no_data_message)
    else:
        features = data.load_testing_features(run_path, dataset)
        conditional_coverage_view.render(
            keys=complexities,
            covered_by_key={c: df_pp[f"covered_{c}"].to_numpy() for c in complexities},
            sigma_by_key={c: df_pp[f"sigma_{c}"].to_numpy() for c in complexities},
            width_by_key={c: df_pp[f"width_{c}"].to_numpy() for c in complexities},
            label=panel_title, color=complexity_color,
            testing=data.load_testing_data(run_path, dataset).reset_index(drop=True),
            features=None if features is None else features.reset_index(drop=True),
            target_coverage=target_coverage,
            title=f'"{dataset}" — {data.method_label(f"sr_{loss_name}")} equations',
            key_prefix="hof_cond_cov",
            highlight=chosen_complexity,
        )

with tab_heatmap:
    if not has_per_point:
        st.info(no_data_message)
    else:
        heatmap_metric = st.selectbox("Metric", options=["Coverage", "Median width"], key="hof_heatmap_metric")
        if heatmap_metric == "Coverage":
            st.caption(f"White = target coverage ({target_coverage:.2f}); red = over-covered, blue = under-covered.")

        deciles = pd.qcut(df_pp["abs_residual"].rank(method="first"), N_DECILES, labels=False)
        rows_dict = {}
        for c in complexities:
            value_col = f"covered_{c}" if heatmap_metric == "Coverage" else f"width_{c}"
            agg = df_pp.groupby(deciles)[value_col].agg("mean" if heatmap_metric == "Coverage" else "median")
            rows_dict[panel_title(c)] = {int(d): agg.loc[d] for d in agg.index}

        matrix = pd.DataFrame.from_dict(rows_dict, orient="index")
        matrix = matrix.reindex(range(N_DECILES), axis=1)

        if heatmap_metric == "Coverage":
            color_kwargs = dict(colorscale=COVERAGE_COLORSCALE, zmid=target_coverage)
        else:
            color_kwargs = dict(colorscale="Blues")
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=[f"D{c + 1}" for c in matrix.columns],
                y=matrix.index,
                colorbar=dict(title=heatmap_metric),
                hovertemplate=(
                    "%{y}<br>residual decile=%{x}<br>" + heatmap_metric.lower() + "=%{z:.3f}<extra></extra>"
                ),
                **color_kwargs,
            )
        )
        fig.update_layout(
            width=data.DETAIL_SIZE,
            height=max(data.DETAIL_SIZE * 0.5, 40 * len(matrix.index) + 150),
            xaxis_title="Residual-rank decile (increasing |residual|)",
            yaxis_title="Complexity",
            title=(
                f"{heatmap_metric} by residual-rank decile, across complexity — "
                f'"{dataset}" — {data.method_label(f"sr_{loss_name}")}'
            ),
        )
        st.plotly_chart(fig, width="content")
