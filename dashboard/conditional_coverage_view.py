# -*- coding: utf-8 -*-
"""
Conditional Coverage tab, shared by the Method Comparison page (one panel
per CP method) and the Hall of Fame page (one panel per SR equation).

1. One sliding-window coverage plot, with a selectable x-axis. Every axis
   is known before seeing y (|residual| and y are not offered: they
   condition on the outcome, so their top bins are under-covered for any
   method, even an oracle sigma):
   - y_pred: the base prediction, shared by every panel.
   - Own sigma: each panel's own difficulty score, as a quantile (0-1) so
     panels share an axis. Checks whether a method is calibrated with
     respect to its own difficulty claims; panels hold different points.
2. Mean width vs. worst-group coverage, one point per panel, with groups
   defined from information known before seeing y: y_pred bins, own-sigma
   bins, or the worst slab of the test features (see
   conditional_coverage.py; each panel gets its own adversarial search).
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
# question each x-axis answers, shown under the selector
X_AXIS_QUESTIONS = {
    "y_pred": (
        "For a given predicted value, is coverage on target? Same points in every panel, "
        "so panels compare directly."
    ),
    "Own sigma": (
        "When the method says a point is easy or hard, is it right? A dip at high sigma means hard "
        "points are underestimated; over-coverage at low sigma means width is wasted on easy points. "
        "Panels hold different points."
    ),
}
GROUPINGS = ["Worst slab", "y_pred bins", "Own sigma bins"]
_SEED = 0
_NO_FEATURES = (
    "This run has no `testing_features.csv`. Rebuild it with "
    "`uv run src/analysis/backfill_testing_features.py <run folder>`."
)


@st.cache_data(show_spinner="Searching worst slabs…")
def _cached_worst_slabs(X, covered_by_key, delta):
    return cc.worst_slabs(X, covered_by_key, delta=delta, seed=_SEED)


def _is_constant(values):
    return np.unique(values).size <= 1


def _quantile(values):
    """Rank of each value as a fraction in [0, 1]."""
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def render(keys, covered_by_key, sigma_by_key, width_by_key, label, color, testing, features,
           target_coverage, title, key_prefix, highlight=None):
    """Draw the tab body.

    keys: panels to show, in order. covered_by_key / sigma_by_key /
    width_by_key: key -> per-point array, aligned with `testing` rows.
    label/color: key -> str. testing: frame with y, y_pred, residuals.
    features: normalized test features aligned with `testing`, or None if
    the run lacks them. highlight: key drawn with an outline in the
    width-vs-coverage chart (e.g. the chosen equation).
    """
    covered_by_key = {k: np.asarray(covered_by_key[k], dtype=bool) for k in keys}
    X = None if features is None else features.to_numpy(dtype=float)

    _render_coverage_plot(keys, covered_by_key, sigma_by_key, label, color, testing,
                          target_coverage, title, key_prefix)
    st.divider()
    _render_width_vs_worst_group(keys, covered_by_key, sigma_by_key, width_by_key, label, color,
                                 testing, X, target_coverage, title, key_prefix, highlight)


def _render_coverage_plot(keys, covered_by_key, sigma_by_key, label, color, testing,
                          target_coverage, title, key_prefix):
    col1, col2 = st.columns([2, 1])
    with col1:
        x_axis = st.radio("x-axis", options=X_AXES, horizontal=True, key=f"{key_prefix}_x_axis")
        st.caption(f"**Question:** {X_AXIS_QUESTIONS[x_axis]}")
    with col2:
        window_pct = st.slider("Window size (% of points)", min_value=5, max_value=50, value=10, step=5,
                                key=f"{key_prefix}_window")

    n = len(testing)
    window = max(3, round(n * window_pct / 100))
    panel_keys = keys

    if x_axis == "Own sigma":
        constant = [k for k in keys if _is_constant(sigma_by_key[k])]
        panel_keys = [k for k in keys if k not in constant]
        if constant:
            st.caption("Not shown (constant sigma, nothing to order by): " + ", ".join(label(k) for k in constant))
        if not panel_keys:
            st.info("No selected panel has a non-constant sigma.")
            return

    def _order_values(k):
        if x_axis == "y_pred":
            return testing["y_pred"].to_numpy()
        return _quantile(sigma_by_key[k])

    cols = min(4, len(panel_keys))
    rows = math.ceil(len(panel_keys) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[label(k) for k in panel_keys])
    x_title = {"y_pred": "Prediction y_pred", "Own sigma": "Own sigma quantile"}[x_axis]

    for i, k in enumerate(panel_keys):
        row, col = i // cols + 1, i % cols + 1
        x_sorted, smoothed = cc.sliding_coverage(_order_values(k), covered_by_key[k], window)
        fig.add_trace(
            go.Scattergl(
                x=x_sorted, y=smoothed, mode="markers",
                marker=dict(size=5, color=color(k), opacity=0.25),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label(k)}</b><br>{x_axis}=%{{x:.4g}}<br>coverage (windowed)=%{{y:.3f}}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
        fig.add_hline(y=target_coverage, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
        fig.update_xaxes(title=x_title, row=row, col=col)
        fig.update_yaxes(title="Coverage (windowed)", range=[0, 1.05], row=row, col=col)

    fig.update_layout(
        width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
        margin=dict(t=60),
        title=f"{title}, window = {window} of {n} points ({window_pct}%)",
    )
    st.plotly_chart(fig, width="content")


def _render_width_vs_worst_group(keys, covered_by_key, sigma_by_key, width_by_key, label, color,
                                 testing, X, target_coverage, title, key_prefix, highlight):
    st.subheader("Mean width vs. worst-group coverage")
    col1, col2 = st.columns([2, 1])
    with col1:
        grouping = st.radio("Groups", options=GROUPINGS, horizontal=True, key=f"{key_prefix}_wg_grouping")
    with col2:
        min_pct = st.slider("Min group size (% of points)", min_value=10, max_value=50, value=20, step=5,
                             key=f"{key_prefix}_wg_min")
    st.caption(
        "Groups are defined before seeing y. The worst group is picked on half of the test points and "
        "its coverage measured on the other half. Best methods sit at small width with worst-group "
        "coverage near the target. Own-sigma bins are undefined for a constant sigma."
    )

    heldout_group_size = round(len(testing) / 2 * min_pct / 100)
    if heldout_group_size < 100:
        st.warning(
            f"Only about {heldout_group_size} held-out points per group: worst-group coverage is noisy "
            f"(±{1.96 * math.sqrt(target_coverage * (1 - target_coverage) / max(heldout_group_size, 1)):.2f} "
            "at the target). Compare methods cautiously on this dataset."
        )

    if grouping == "Worst slab":
        if X is None:
            st.info(_NO_FEATURES)
            return
        slabs = _cached_worst_slabs(X, covered_by_key, min_pct / 100)
        worst = {k: slabs[k].heldout_coverage for k in keys}
    elif grouping == "y_pred bins":
        y_pred = testing["y_pred"].to_numpy()
        worst = {k: cc.worst_bin(y_pred, covered_by_key[k], min_pct / 100, seed=_SEED)[0] for k in keys}
    else:
        worst = {k: cc.worst_bin(sigma_by_key[k], covered_by_key[k], min_pct / 100, seed=_SEED)[0] for k in keys}

    summary = pd.DataFrame({
        "Mean width": [float(np.mean(width_by_key[k])) for k in keys],
        "Worst-group coverage": [worst[k] for k in keys],
        "Marginal coverage": [float(covered_by_key[k].mean()) for k in keys],
    }, index=keys)
    shown = summary.dropna(subset=["Worst-group coverage"])
    if shown.empty:
        st.info("No panel has a defined worst-group coverage for this grouping.")
        return

    fig = go.Figure()
    for k, row in shown.iterrows():
        is_highlight = k == highlight
        fig.add_trace(go.Scatter(
            x=[row["Worst-group coverage"]], y=[row["Mean width"]], mode="markers",
            marker=dict(size=18 if is_highlight else 13, color=color(k),
                        line=dict(width=2.5 if is_highlight else 1, color="black" if is_highlight else "white")),
            name=label(k),
            hovertemplate=(
                f"<b>{label(k)}</b><br>worst-group coverage=%{{x:.3f}}<br>mean width=%{{y:.3f}}<br>"
                f"marginal coverage={row['Marginal coverage']:.3f}<extra></extra>"
            ),
        ))
    fig.add_vline(x=target_coverage, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(title=f"Worst-group coverage ({grouping}, held-out)")
    fig.update_yaxes(title="Mean interval width")
    fig.update_layout(
        width=data.DETAIL_SIZE, height=data.DETAIL_SIZE * 0.8,
        title=title,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    st.plotly_chart(fig, width="content")

    table = summary.copy()
    table.index = [label(k) for k in keys]
    st.dataframe(table.style.format("{:.3f}", na_rep="—"), width="stretch")
