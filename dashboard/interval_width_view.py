# -*- coding: utf-8 -*-
"""
Interval Width tab, shared by the Method Comparison page (one panel per CP
method) and the Hall of Fame page (one panel per SR equation): per-point
width against the base prediction y_pred plus a sliding-window median,
same x for every panel -- where does each method spend its width?
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data

QUESTION = (
    "Where does each method spend its width? Same points in every panel. Read it next to the "
    "Conditional Coverage y_pred view: narrow width with a coverage dip means over-confidence, "
    "wide width with over-coverage means wasted width."
)


def render(keys, width_by_key, label, color, testing, title, key_prefix):
    """Draw the tab body.

    keys: panels to show, in order. width_by_key: key -> per-point interval
    width, aligned with `testing` rows. label/color: key -> str. testing:
    frame with y_pred.
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"**Question:** {QUESTION}")
    with col2:
        window_pct = st.slider("Window size (% of points)", min_value=5, max_value=50, value=10, step=5,
                                key=f"{key_prefix}_window")

    n = len(testing)
    window = max(3, round(n * window_pct / 100))
    y_pred = testing["y_pred"].to_numpy()
    order = np.argsort(y_pred, kind="stable")
    x_sorted = y_pred[order]

    cols = min(4, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[label(k) for k in keys])

    for i, k in enumerate(keys):
        row, col = i // cols + 1, i % cols + 1
        width_sorted = np.asarray(width_by_key[k], dtype=float)[order]
        smoothed = pd.Series(width_sorted).rolling(window=window, center=True, min_periods=1).median().to_numpy()

        fig.add_trace(go.Scattergl(
            x=x_sorted, y=width_sorted, mode="markers",
            marker=dict(size=4, color=color(k), opacity=0.22),
            hovertemplate=f"<b>{label(k)}</b><br>y_pred=%{{x:.4g}}<br>width=%{{y:.3f}}<extra></extra>",
        ), row=row, col=col)
        # white halo under the trend line keeps it legible over a dense scatter
        fig.add_trace(go.Scatter(
            x=x_sorted, y=smoothed, mode="lines", line=dict(color="white", width=6),
            opacity=0.85, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x_sorted, y=smoothed, mode="lines", line=dict(color=color(k), width=3.5),
            hovertemplate=(
                f"<b>{label(k)}</b> (windowed median)<br>y_pred=%{{x:.4g}}<br>width=%{{y:.3f}}<extra></extra>"
            ),
        ), row=row, col=col)
        fig.update_xaxes(title="Prediction y_pred", row=row, col=col)
        fig.update_yaxes(title="Interval width", row=row, col=col)

    fig.update_layout(
        width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
        margin=dict(t=60), showlegend=False,
        title=f"{title}, window = {window} of {n} points ({window_pct}%)",
    )
    st.plotly_chart(fig, width="content")
