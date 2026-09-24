# -*- coding: utf-8 -*-
"""
Interval Width tab, shared by the Method Comparison page (one panel per CP
method) and the Hall of Fame page (one panel per SR equation), with a
selectable x-axis:
- y_pred: per-point width against the base prediction plus a sliding-window
  median, same x for every panel -- where does each method spend its width?
- Own sigma: right-sizing -- points grouped by their own interval width
  (the same order as their own sigma for normalized CP; also covers
  Mondrian's per-category widths), then each group's target-level quantile
  of |residual| against its median half-width. On the diagonal, intervals
  are exactly as wide as the errors they get.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import conditional_coverage as cc
import data

X_AXES = ["y_pred", "Own sigma"]
X_AXIS_QUESTIONS = {
    "y_pred": (
        "Where does each method spend its width? Same points in every panel. Read it next to the "
        "Conditional Coverage y_pred view: narrow width with a coverage dip means over-confidence, "
        "wide width with over-coverage means wasted width."
    ),
    "Own sigma": (
        "Is each interval the right size for the errors it gets? Points are grouped by their own "
        "width; each marker is one group's target-level quantile of |residual| against its median "
        "half-width. Above the diagonal: too narrow there. Below: too wide."
    ),
}
N_BINS = 15


def render(keys, width_by_key, label, color, testing, target_coverage, title, key_prefix):
    """Draw the tab body.

    keys: panels to show, in order. width_by_key: key -> per-point interval
    width, aligned with `testing` rows. label/color: key -> str. testing:
    frame with y_pred and residuals.
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        x_axis = st.radio("x-axis", options=X_AXES, horizontal=True, key=f"{key_prefix}_x_axis")
        st.caption(f"**Question:** {X_AXIS_QUESTIONS[x_axis]}")
    with col2:
        if x_axis == "y_pred":
            window_pct = st.slider("Window size (% of points)", min_value=5, max_value=50, value=10, step=5,
                                    key=f"{key_prefix}_window")

    n = len(testing)
    y_pred = testing["y_pred"].to_numpy()
    abs_residual = testing["residuals"].abs().to_numpy()

    cols = min(4, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[label(k) for k in keys])

    for i, k in enumerate(keys):
        row, col = i // cols + 1, i % cols + 1
        width = np.asarray(width_by_key[k], dtype=float)
        if x_axis == "y_pred":
            _add_width_vs_y_pred(fig, row, col, y_pred, width, label(k), color(k), window_pct, n)
        else:
            _add_right_sizing(fig, row, col, width, abs_residual, target_coverage, label(k), color(k))

    if x_axis == "y_pred":
        window = max(3, round(n * window_pct / 100))
        full_title = f"{title}, window = {window} of {n} points ({window_pct}%)"
    else:
        full_title = f"{title}, {target_coverage:.0%} quantile of |residual| per width group"
    fig.update_layout(
        width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
        margin=dict(t=60), showlegend=False, title=full_title,
    )
    st.plotly_chart(fig, width="content")


def _add_width_vs_y_pred(fig, row, col, y_pred, width, name, color, window_pct, n):
    window = max(3, round(n * window_pct / 100))
    order = np.argsort(y_pred, kind="stable")
    x_sorted, width_sorted = y_pred[order], width[order]
    smoothed = pd.Series(width_sorted).rolling(window=window, center=True, min_periods=1).median().to_numpy()

    fig.add_trace(go.Scattergl(
        x=x_sorted, y=width_sorted, mode="markers",
        marker=dict(size=4, color=color, opacity=0.22),
        hovertemplate=f"<b>{name}</b><br>y_pred=%{{x:.4g}}<br>width=%{{y:.3f}}<extra></extra>",
    ), row=row, col=col)
    # white halo under the trend line keeps it legible over a dense scatter
    fig.add_trace(go.Scatter(
        x=x_sorted, y=smoothed, mode="lines", line=dict(color="white", width=6),
        opacity=0.85, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=smoothed, mode="lines", line=dict(color=color, width=3.5),
        hovertemplate=f"<b>{name}</b> (windowed median)<br>y_pred=%{{x:.4g}}<br>width=%{{y:.3f}}<extra></extra>",
    ), row=row, col=col)
    fig.update_xaxes(title="Prediction y_pred", row=row, col=col)
    fig.update_yaxes(title="Interval width", row=row, col=col)


def _add_right_sizing(fig, row, col, width, abs_residual, level, name, color):
    half, quantile, sizes = cc.right_sizing_bins(width, abs_residual, level, n_bins=N_BINS)
    lo = min(half.min(), quantile.min())
    hi = max(half.max(), quantile.max())

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash", width=1.5),
        hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=half, y=quantile, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
        customdata=sizes,
        hovertemplate=(
            f"<b>{name}</b><br>median half-width=%{{x:.3f}}<br>"
            f"{level:.0%} quantile of |residual|=%{{y:.3f}}<br>points=%{{customdata}}<extra></extra>"
        ),
    ), row=row, col=col)
    fig.update_xaxes(title="Median half-width", row=row, col=col)
    fig.update_yaxes(title=f"{level:.0%} quantile of |residual|", row=row, col=col)
