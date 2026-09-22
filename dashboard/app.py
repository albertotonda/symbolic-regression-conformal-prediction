# -*- coding: utf-8 -*-
"""Landing page: pick which results run to explore; other pages (in
pages/) read the selection back from st.session_state["run_path"]."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

st.set_page_config(page_title="CP Methods Comparison", layout="wide")
st.title("Conformal Prediction Methods — Comparison Dashboard")

runs = data.discover_runs()
CUSTOM = "Other folder..."
options = [r["name"] for r in runs] + [CUSTOM]
selected = st.selectbox(
    "Results run",
    options=options,
    format_func=lambda name: name if name == CUSTOM else (
        f"{name}  ({next(r['n_datasets'] for r in runs if r['name'] == name)} datasets)"
    ),
    # explicit key so the choice survives navigating away to another page
    # and back -- without it, Streamlit re-renders this widget at its
    # default (the newest run) on every visit to this page, silently
    # overwriting st.session_state["run_path"] below
    key="run_select",
)

if selected == CUSTOM:
    run_path = st.text_input(
        "Folder path (absolute, or relative to the repo root)",
        value=st.session_state.get("run_path", ""),
        key="run_select_custom_path",
    )
    if run_path and not Path(run_path).is_absolute():
        run_path = str(data.REPO_ROOT / run_path)
else:
    run_path = next(r["path"] for r in runs if r["name"] == selected)

if not run_path:
    st.stop()
if not (Path(run_path) / "results.csv").exists():
    st.error(f"No `results.csv` found in `{run_path}`.")
    st.stop()

st.session_state["run_path"] = run_path
st.caption(f"Reading from `{run_path}`")
st.info("Pick a run above, then open a plot from the sidebar.")

df = data.load_results(run_path)

methods = data.discover_methods(df)
if methods:
    st.subheader("Method leaderboard")
    st.caption(
        "How often each method is Pareto-non-dominated on (coverage, median width) "
        "across this run's datasets — same dominance rule as "
        "`src/analysis/check_pareto_optimality.py`'s `results-statistics.csv`, computed "
        "live here instead of requiring that script to have been run first."
    )
    dominance = data.compute_pareto_dominance(df, methods)
    dominance = dominance.reindex(methods).sort_values("non_dominated", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[data.method_label(m) for m in dominance.index], x=dominance["non_dominated"],
        orientation="h", name="Non-dominated",
        marker=dict(color=[data.method_color(m) for m in dominance.index]),
        customdata=dominance[["dominated", "alone"]].values,
        hovertemplate="non-dominated on %{x} dataset(s)<br>alone: %{customdata[1]}<extra></extra>",
    ))
    fig.update_xaxes(title="Number of datasets where non-dominated")
    fig.update_yaxes(title=None)
    fig.update_layout(
        width=data.DETAIL_SIZE, height=max(220, 40 * len(methods) + 80),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig, width="content")

st.dataframe(df, width="stretch")
