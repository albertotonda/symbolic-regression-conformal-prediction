# -*- coding: utf-8 -*-
"""Method comparison: every cross-method view, as tabs of one page (mirrors
Hall of Fame's one-page/many-tabs layout instead of one page per view).

A single **Dataset** selector above the tabs drives every per-dataset tab
(Sigma Relationships, Width by Residual Rank, Sigma Ridgeline, Method
Head-to-Head, Confidence Intervals, Sigma vs Coverage, and Pareto's Detail
view) -- pick it once instead of on each tab separately. Pareto's Grid view,
Difficulty Heatmap, and Dataset Characteristics are inherently cross-dataset
(every dataset at once) and ignore it.

Streamlit tabs can't nest, so Pareto's and Difficulty Heatmap's own
Grid-vs-Detail split (previously a second level of `st.tabs`) is a radio
button instead, inside their own outer tab.

Each tab's body is wrapped in a small function called immediately below its
definition, so an internal `st.stop()` -- kept from each view's original
"no data for this" guard clauses -- only skips that tab's own render instead
of halting the whole page (a bare `st.stop()` stops the *entire* script,
which would blank out every tab after it).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402

st.title("Method comparison")

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
target_coverage = data.target_coverage(run_path)

dataset = st.selectbox(
    "Dataset",
    options=all_datasets,
    key="mc_dataset",
    help=(
        "Used by every per-dataset tab (Sigma Relationships, Width by Residual "
        "Rank, Sigma Ridgeline, Method Head-to-Head, Confidence Intervals, Sigma "
        "vs Coverage, and Pareto's Detail view). Pareto's Grid view, Difficulty "
        "Heatmap, and Dataset Characteristics always show every dataset "
        "regardless of this."
    ),
)

(
    tab_pareto, tab_sigma_rel, tab_width_rank, tab_heatmap, tab_ridgeline,
    tab_characteristics, tab_head_to_head, tab_ci, tab_sigma_cov,
) = st.tabs([
    "Pareto", "Sigma Relationships", "Width by Residual Rank", "Difficulty Heatmap",
    "Sigma Ridgeline", "Dataset Characteristics", "Method Head-to-Head",
    "Confidence Intervals", "Sigma vs Coverage",
])

# ---------------------------------------------------------------------------
# Pareto: median CI width vs. coverage, one point per method.
# ---------------------------------------------------------------------------
with tab_pareto:
    def _render_pareto():
        all_methods = data.discover_methods(results_df)

        def _add_method_traces(fig, row_data, methods, seen_methods, row=None, col=None, marker_size=10):
            for m in methods:
                cov = row_data.get(f"{m}_coverage")
                med = row_data.get(f"{m}_median")
                if cov is None or med is None or math.isnan(cov) or math.isnan(med):
                    continue
                first_time = m not in seen_methods
                seen_methods.add(m)
                fig.add_trace(
                    go.Scatter(
                        x=[cov], y=[med], mode="markers",
                        marker=dict(size=marker_size, color=data.method_color(m),
                                    line=dict(width=1, color="white")),
                        name=data.method_label(m), legendgroup=m, showlegend=first_time,
                        hovertemplate=(
                            f"<b>{data.method_label(m)}</b><br>"
                            "coverage=%{x:.3f}<br>median width=%{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )

        def _add_target_line(fig, row=None, col=None):
            fig.add_vline(x=target_coverage, line_dash="dash", line_color="gray",
                           opacity=0.5, row=row, col=col)

        col1, col2 = st.columns([3, 1])
        with col1:
            methods = st.multiselect(
                "Methods", options=all_methods, default=all_methods,
                format_func=data.method_label,
                key="pareto_methods",
            )
        with col2:
            st.metric("Target coverage", f"{target_coverage:.2f}")

        if not methods:
            st.warning("Select at least one method.")
            return

        view = st.radio("View", options=["Grid (all datasets)", "Detail (single dataset)"],
                         key="pareto_view", horizontal=True)

        if view == "Grid (all datasets)":
            datasets = st.multiselect(
                "Datasets", options=all_datasets, default=all_datasets, key="pareto_grid_datasets",
            )
            if not datasets:
                st.warning("Select at least one dataset.")
                return
            cols = min(4, len(datasets))
            rows = math.ceil(len(datasets) / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=datasets)
            seen_methods = set()
            for i, ds in enumerate(datasets):
                row, col = i // cols + 1, i % cols + 1
                row_data = results_df[results_df["dataset_name"] == ds].iloc[0]
                _add_method_traces(fig, row_data, methods, seen_methods, row=row, col=col)
                _add_target_line(fig, row=row, col=col)
                fig.update_xaxes(autorange="reversed", row=row, col=col)
            fig.update_layout(
                width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + data.LEGEND_MARGIN,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.0 + 90 / (rows * data.CELL_SIZE), x=0),
                margin=dict(t=data.LEGEND_MARGIN),
            )
            st.plotly_chart(fig, width="content")
        else:
            row_data = results_df[results_df["dataset_name"] == dataset].iloc[0]
            fig = go.Figure()
            _add_method_traces(fig, row_data, methods, set(), marker_size=16)
            _add_target_line(fig)
            fig.update_xaxes(autorange="reversed", title="Coverage on the test set")
            fig.update_yaxes(title="Median amplitude of the confidence intervals")
            fig.update_layout(width=data.DETAIL_SIZE, height=data.DETAIL_SIZE, title=f'"{dataset}"')
            st.plotly_chart(fig, width="content")

    _render_pareto()

# ---------------------------------------------------------------------------
# Sigma vs. residuals/width, unbinned, one subplot per method.
# ---------------------------------------------------------------------------
with tab_sigma_rel:
    def _render_sigma_relationships():
        datasets_with_test = [ds for ds in all_datasets if data.has_per_point_data(run_path, ds)]
        datasets_with_cal = [ds for ds in all_datasets if data.has_calibration_per_point_data(run_path, ds)]

        if dataset not in datasets_with_test and dataset not in datasets_with_cal:
            st.info(
                f'No per-point data for "{dataset}" — it predates `src/run_sigma_sr.py` writing '
                "`testing_data.csv`/`calibration_data.csv`/`methods_sigmas_*.csv` (or the older "
                "test-only `per_point.csv`). Pick a different dataset above, or a newer run."
            )
            return

        col1, col2 = st.columns(2)
        with col1:
            source_options = [s for s, avail in [("Test", dataset in datasets_with_test),
                                                  ("Calibration", dataset in datasets_with_cal)] if avail]
            source = st.selectbox("Data", options=source_options, key="sigma_rel_source")
        with col2:
            target_options = ["Residual", "Width"] if source == "Test" else ["Residual"]
            target = st.selectbox("vs.", options=target_options, key="sigma_rel_target")

        df = data.load_per_point(run_path, dataset) if source == "Test" else data.load_per_point_calibration(run_path, dataset)
        all_methods = data.per_point_methods(df)

        methods = st.multiselect(
            "Methods", options=all_methods, default=all_methods,
            format_func=data.method_label,
            key="sigma_rel_methods",
        )
        if not methods:
            st.warning("Select at least one method.")
            return

        target_col_by_method = (
            {m: f"width_{m}" for m in methods} if target == "Width" else {m: "abs_residual" for m in methods}
        )
        target_label = "abs residual" if target == "Residual" else "width"

        cols = min(4, len(methods))
        rows = math.ceil(len(methods) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[data.method_label(m) for m in methods])

        for i, m in enumerate(methods):
            row, col = i // cols + 1, i % cols + 1
            log_target = np.log10(df[target_col_by_method[m]].clip(lower=1e-12))
            fig.add_trace(
                go.Scattergl(
                    x=log_target, y=df[f"sigma_{m}"], mode="markers",
                    marker=dict(size=5, color=data.method_color(m), opacity=0.55),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{data.method_label(m)}</b><br>"
                        f"log10({target_label})=" + "%{x:.3f}<br>sigma=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            fig.update_xaxes(title=f"log10({target_label})", row=row, col=col)
            fig.update_yaxes(title="sigma", row=row, col=col)

        fig.update_layout(
            width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
            showlegend=False,
            margin=dict(t=60),
            title=f'"{dataset}" — {source} sigma vs. {target_label}',
        )
        st.plotly_chart(fig, width="content")

    _render_sigma_relationships()

# ---------------------------------------------------------------------------
# Interval width vs. residual rank, one line per method.
# ---------------------------------------------------------------------------
with tab_width_rank:
    def _render_width_by_rank():
        N_BINS = 15
        if not data.has_per_point_data(run_path, dataset):
            st.info(
                f'No per-point data for "{dataset}" — it predates `src/run_sigma_sr.py` writing '
                "`testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` (or the older "
                "`per_point.csv`). Pick a different dataset above, or a newer run."
            )
            return

        df = data.load_per_point(run_path, dataset)
        all_methods = data.per_point_methods(df)

        methods = st.multiselect(
            "Methods", options=all_methods, default=all_methods,
            format_func=data.method_label,
            key="width_by_rank_methods",
        )
        if not methods:
            st.warning("Select at least one method.")
            return

        abs_residual = df["abs_residual"].to_numpy()
        bins = np.array_split(np.argsort(abs_residual), N_BINS)
        bin_indices = np.arange(len(bins))

        fig = go.Figure()
        for m in methods:
            widths = df[f"width_{m}"].to_numpy()
            bin_medians = [np.median(widths[b]) for b in bins]
            fig.add_trace(go.Scatter(
                x=bin_indices, y=bin_medians, mode="lines+markers",
                line=dict(color=data.method_color(m), width=2.5),
                marker=dict(size=7),
                name=data.method_label(m),
                hovertemplate=f"<b>{data.method_label(m)}</b><br>bin=%{{x}}<br>median width=%{{y:.3f}}<extra></extra>",
            ))

        fig.update_xaxes(title="Absolute residual bin (equal count per bin, increasing order)",
                          tickmode="array", tickvals=bin_indices)
        fig.update_yaxes(title="Median interval width")
        fig.update_layout(
            width=data.DETAIL_SIZE * 1.1, height=data.DETAIL_SIZE * 0.7,
            title=f'"{dataset}"',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, width="content")

    _render_width_by_rank()

# ---------------------------------------------------------------------------
# Heatmap: rows = datasets, columns = residual-rank deciles.
# ---------------------------------------------------------------------------
with tab_heatmap:
    def _render_heatmap():
        N_DECILES = 10
        # Diverging around target coverage: white = perfect, red = over-covered,
        # blue = under-covered. Reuses the dashboard's own blue/rust hues (see
        # data.py's _CATEGORICAL_PALETTE) instead of a stock RdBu scale, so the
        # under/over direction is unambiguous regardless of a colorscale's
        # default orientation.
        COVERAGE_COLORSCALE = [[0.0, "#2b6bab"], [0.5, "#ffffff"], [1.0, "#ab4b2b"]]

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
            return

        all_methods = sorted({m for df in per_point_by_dataset.values() for m in data.per_point_methods(df)})

        # one decile assignment per dataset, shared by every method -- binned
        # on residual rank (method-independent), not each method's own sigma
        # rank, so columns condition on the same points in every method's
        # panel; see the module docstring's mismatch note for why.
        deciles_by_dataset = {
            ds: pd.qcut(df["abs_residual"].rank(method="first"), N_DECILES, labels=False)
            for ds, df in per_point_by_dataset.items()
        }

        metric = st.selectbox("Metric", options=["Coverage", "Median width"], key="heatmap_metric")
        if metric == "Coverage":
            st.caption(f"White = target coverage ({target_coverage:.2f}); red = over-covered, blue = under-covered.")

        def build_matrix(method, metric):
            value_col = f"covered_{method}" if metric == "Coverage" else f"width_{method}"
            rows = {}
            excluded = 0
            for ds in all_datasets:
                df = per_point_by_dataset.get(ds)
                if df is None or value_col not in df.columns:
                    excluded += 1
                    continue
                agg = df.groupby(deciles_by_dataset[ds])[value_col].agg("mean" if metric == "Coverage" else "median")
                rows[ds] = {int(d): agg.loc[d] for d in agg.index}
            matrix = pd.DataFrame.from_dict(rows, orient="index")
            matrix = matrix.reindex(all_datasets, axis=0)
            matrix = matrix.reindex(range(N_DECILES), axis=1)
            included = len(all_datasets) - excluded
            return matrix, included, excluded

        view = st.radio("View", options=["Grid (all methods)", "Detail (single method)"],
                         key="heatmap_view", horizontal=True)

        if view == "Grid (all methods)":
            cols = min(3, len(all_methods))
            rows_n = math.ceil(len(all_methods) / cols)
            fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=[data.method_label(m) for m in all_methods])
            zmin = zmax = None
            matrices = {}
            for m in all_methods:
                matrix, *_ = build_matrix(m, metric)
                matrices[m] = matrix
                finite = matrix.values[~pd.isna(matrix.values)]
                if finite.size:
                    zmin = finite.min() if zmin is None else min(zmin, finite.min())
                    zmax = finite.max() if zmax is None else max(zmax, finite.max())

            for i, m in enumerate(all_methods):
                row, col = i // cols + 1, i % cols + 1
                matrix = matrices[m]
                fig.add_trace(
                    go.Heatmap(
                        z=matrix.values,
                        x=[f"D{c + 1}" for c in matrix.columns],
                        y=matrix.index,
                        coloraxis="coloraxis",
                        hovertemplate=(
                            f"<b>{data.method_label(m)}</b><br>dataset=%{{y}}<br>residual decile=%{{x}}<br>"
                            + metric.lower() + "=%{z:.3f}<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )

            panel_height = max(220, 22 * len(all_datasets) + 90)
            if metric == "Coverage":
                coloraxis = dict(colorscale=COVERAGE_COLORSCALE, cmid=target_coverage, colorbar=dict(title=metric))
            else:
                coloraxis = dict(colorscale="Blues", cmin=zmin, cmax=zmax, colorbar=dict(title=metric))
            fig.update_layout(
                width=cols * 480, height=rows_n * panel_height,
                coloraxis=coloraxis,
                title=f"{metric} by residual-rank decile — all methods",
            )
            st.plotly_chart(fig, width="content")
        else:
            method = st.selectbox("Method", options=all_methods, format_func=data.method_label, key="heatmap_method")
            matrix, included, excluded = build_matrix(method, metric)
            matrix_present = matrix.dropna(how="all")

            if matrix_present.empty:
                st.info(
                    f"No dataset in this run has per-point data for **{data.method_label(method)}**. "
                    "Pick a different method, or a newer run."
                )
                return

            if excluded:
                st.caption(f"{included} dataset(s) included, {excluded} excluded (no per-point data).")

            if metric == "Coverage":
                color_kwargs = dict(colorscale=COVERAGE_COLORSCALE, zmid=target_coverage)
            else:
                color_kwargs = dict(colorscale="Blues")
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix_present.values,
                    x=[f"D{c + 1}" for c in matrix_present.columns],
                    y=matrix_present.index,
                    colorbar=dict(title=metric),
                    hovertemplate="dataset=%{y}<br>residual decile=%{x}<br>" + metric.lower() + "=%{z:.3f}<extra></extra>",
                    **color_kwargs,
                )
            )
            fig.update_layout(
                width=data.DETAIL_SIZE,
                height=max(data.DETAIL_SIZE * 0.5, 40 * len(matrix_present.index) + 150),
                xaxis_title="Residual-rank decile (increasing |residual|)",
                yaxis_title=None,
                title=f"{metric} by residual-rank decile — {data.method_label(method)}",
            )
            st.plotly_chart(fig, width="content")

    _render_heatmap()

# ---------------------------------------------------------------------------
# Ridgeline plot of each method's sigma distribution, for one dataset.
# ---------------------------------------------------------------------------
with tab_ridgeline:
    def _render_ridgeline():
        LANE_HEIGHT = 0.85
        N_GRID = 300

        def _hex_to_rgba(hex_color, alpha):
            hex_color = hex_color.lstrip("#")
            r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"

        if not data.has_per_point_data(run_path, dataset):
            st.info(
                f'No per-point data for "{dataset}" — it predates `src/run_sigma_sr.py` writing '
                "`testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` (or the older "
                "`per_point.csv`). Pick a different dataset above, or a newer run."
            )
            return

        df = data.load_per_point(run_path, dataset)
        methods_all = data.per_point_methods(df)

        # a constant sigma (e.g. standard_cp/mondrian_cp) has no distribution to
        # draw, and gaussian_kde raises on zero-variance input.
        all_methods = [m for m in methods_all if df[f"sigma_{m}"].nunique() > 1]
        constant_methods = [m for m in methods_all if m not in all_methods]

        if not all_methods:
            st.info("No method has a non-constant difficulty score for this dataset.")
            return
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
            return

        ordered_methods = [m for m in all_methods if m in methods]
        n = len(ordered_methods)

        sigmas = {m: df[f"sigma_{m}"].to_numpy() for m in ordered_methods}
        x_min = min(v.min() for v in sigmas.values())
        x_max = max(v.max() for v in sigmas.values())
        pad = 0.05 * (x_max - x_min or 1.0)
        x_grid = np.linspace(x_min - pad, x_max + pad, N_GRID)

        fig = go.Figure()
        for i, m in enumerate(ordered_methods):
            offset = n - 1 - i
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
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                          range=[-0.2, n - 1 + LANE_HEIGHT + 0.3])
        fig.update_layout(
            width=int(data.DETAIL_SIZE * 1.1),
            height=max(int(data.DETAIL_SIZE * 0.6), 90 * n + 150),
            title=f'"{dataset}"',
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="content")

    _render_ridgeline()

# ---------------------------------------------------------------------------
# Dataset characteristics vs. method performance (always cross-dataset).
# ---------------------------------------------------------------------------
with tab_characteristics:
    def _render_characteristics():
        all_methods = data.discover_methods(results_df)
        if len(all_methods) < 2:
            st.warning("Need at least two methods with results to compare.")
            return

        characteristics = data.load_dataset_characteristics()
        merged = results_df.merge(characteristics, on="dataset_name", how="left", suffixes=("", "_meta"))
        n_matched = merged["n_samples"].notna().sum()
        if n_matched < len(merged):
            st.caption(
                f"{n_matched}/{len(merged)} datasets matched to OpenML-CTR23 metadata by name "
                f"(the rest aren't in `results/OpenML-CTR23-statistics-500-estimators-10-fold-cv.csv`)."
            )
        merged = merged[merged["n_samples"].notna()]
        if merged.empty:
            st.warning("No dataset in this run matched the OpenML-CTR23 metadata table.")
            return

        col1, col2 = st.columns(2)
        with col1:
            default_a = next((m for m in all_methods if m.startswith("sr_")), all_methods[0])
            method_a = st.selectbox("Method A (numerator)", options=all_methods,
                                     index=all_methods.index(default_a), format_func=data.method_label,
                                     key="charact_method_a")
        with col2:
            default_b = "standard_cp" if "standard_cp" in all_methods else all_methods[0]
            method_b = st.selectbox("Method B (denominator)", options=all_methods,
                                     index=all_methods.index(default_b), format_func=data.method_label,
                                     key="charact_method_b")

        merged["width_ratio"] = merged[f"{method_a}_median"] / merged[f"{method_b}_median"]
        merged = merged.dropna(subset=["width_ratio"])

        CHARACTERISTICS = {
            "n_samples (log10)": ("n_samples", True),
            "n_features": ("n_features", False),
            "missing_data": ("missing_data", False),
            "categorical_features": ("categorical_features", False),
            "r2 (this run's base regressor)": ("r2", False),
        }
        x_label = st.selectbox("Dataset characteristic (x-axis)", options=list(CHARACTERISTICS.keys()),
                                key="charact_x")
        x_col, log_x = CHARACTERISTICS[x_label]
        x_values = np.log10(merged[x_col]) if log_x else merged[x_col]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, y=merged["width_ratio"], mode="markers+text",
            text=merged["dataset_name"], textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=11, color=data.method_color(method_a), line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>%{text}</b><br>" + x_label + "=%{x:.3g}<br>"
                f"{data.method_label(method_a)} / {data.method_label(method_b)}" + "=%{y:.3f}<extra></extra>"
            ),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6,
                      annotation_text="equal width", annotation_position="bottom right")

        fig.update_xaxes(title=x_label)
        fig.update_yaxes(title=f"Median width ratio: {data.method_label(method_a)} / {data.method_label(method_b)}")
        fig.update_layout(
            width=data.DETAIL_SIZE * 1.2, height=data.DETAIL_SIZE,
            title="Width ratio vs. dataset characteristics",
        )
        st.plotly_chart(fig, width="content")

        r = np.corrcoef(x_values, merged["width_ratio"])[0, 1] if len(merged) > 1 else float("nan")
        st.caption(f"Pearson correlation (x, ratio): {r:.3f} over {len(merged)} dataset(s).")

    _render_characteristics()

# ---------------------------------------------------------------------------
# Method head-to-head: per-point interval width, two methods.
# ---------------------------------------------------------------------------
with tab_head_to_head:
    def _render_head_to_head():
        if not data.has_per_point_data(run_path, dataset):
            st.info(
                f'No per-point data for "{dataset}" — it predates `src/run_sigma_sr.py` writing '
                "`testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` (or the older "
                "`per_point.csv`). Pick a different dataset above, or a newer run."
            )
            return

        df = data.load_per_point(run_path, dataset)
        all_methods = data.per_point_methods(df)
        if len(all_methods) < 2:
            st.warning("Need at least two methods with per-point data.")
            return

        col1, col2 = st.columns(2)
        with col1:
            default_a = next((m for m in all_methods if m.startswith("sr_")), all_methods[0])
            method_a = st.selectbox("Method A", options=all_methods, index=all_methods.index(default_a),
                                     format_func=data.method_label, key="h2h_method_a")
        with col2:
            remaining = [m for m in all_methods if m != method_a]
            default_b = "standard_cp" if "standard_cp" in remaining else remaining[0]
            method_b = st.selectbox("Method B", options=remaining, index=remaining.index(default_b),
                                     format_func=data.method_label, key="h2h_method_b")

        width_a = df[f"width_{method_a}"]
        width_b = df[f"width_{method_b}"]
        lo = min(width_a.min(), width_b.min())
        hi = max(width_a.max(), width_b.max())

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=width_b, y=width_a, mode="markers",
            marker=dict(
                size=6, color=df["abs_residual"], colorscale="Viridis", opacity=0.75,
                colorbar=dict(title="|residual|"),
            ),
            hovertemplate=(
                f"{data.method_label(method_b)}" + "=%{x:.3f}<br>"
                f"{data.method_label(method_a)}" + "=%{y:.3f}<br>"
                "|residual|=%{marker.color:.3f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="gray", dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

        fig.update_xaxes(title=f"{data.method_label(method_b)} — interval width", type="log")
        fig.update_yaxes(title=f"{data.method_label(method_a)} — interval width", type="log")
        fig.update_layout(
            width=data.DETAIL_SIZE, height=data.DETAIL_SIZE,
            title=f'"{dataset}"',
        )
        st.plotly_chart(fig, width="content")

        narrower_a = (width_a < width_b).mean()
        st.caption(
            f"{data.method_label(method_a)} is narrower on {narrower_a:.1%} of test points "
            f"(points below the diagonal); {data.method_label(method_b)} is narrower on "
            f"{1 - narrower_a:.1%}."
        )

    _render_head_to_head()

# ---------------------------------------------------------------------------
# Confidence interval sanity check: actual/predicted/interval, one method.
# ---------------------------------------------------------------------------
with tab_ci:
    def _render_confidence_intervals():
        if not data.has_interval_detail_data(run_path, dataset):
            st.info(
                f'"{dataset}" has no `testing_data.csv` + `methods_intervals.csv` — this view needs '
                "the raw prediction/bounds, which the older `per_point.csv` format didn't keep. Pick "
                "a different dataset above, or a newer run."
            )
            return

        testing = data.load_testing_data(run_path, dataset)
        intervals = data.load_intervals(run_path, dataset)
        all_methods = data.discover_methods(results_df)
        available_methods = [m for m in all_methods if m in intervals["method"].unique()]

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Method", options=available_methods, format_func=data.method_label,
                                   key="ci_method")
        with col2:
            n_points = st.slider("Points shown", min_value=10, max_value=100, value=30, step=5, key="ci_n_points")

        method_intervals = intervals[intervals["method"] == method].set_index("index")
        df = testing.join(method_intervals[["lower_bound", "upper_bound"]])
        df = df.sort_values("y").reset_index(drop=True)

        if len(df) > n_points:
            sample_idx = sorted(range(0, len(df), max(len(df) // n_points, 1)))[:n_points]
            df = df.loc[sample_idx].reset_index(drop=True)

        x = list(range(len(df)))
        color = data.method_color(method)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=list(df["upper_bound"]) + list(df["lower_bound"][::-1]),
            fill="toself", fillcolor=color, opacity=0.25,
            line=dict(width=0), hoverinfo="skip", showlegend=True, name="Interval",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df["y"], mode="markers", name="Measured value",
            marker=dict(symbol="circle", size=8, color="#2b6bab"),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df["y_pred"], mode="markers", name="Prediction",
            marker=dict(symbol="x", size=8, color="#ab4b2b"),
        ))

        covered = ((df["y"] >= df["lower_bound"]) & (df["y"] <= df["upper_bound"])).mean()
        fig.update_xaxes(title="Test samples, sorted by increasing measured value")
        fig.update_yaxes(title="Target value")
        fig.update_layout(
            width=data.DETAIL_SIZE * 1.3, height=data.DETAIL_SIZE * 0.75,
            title=f'"{dataset}" — {data.method_label(method)} (coverage on shown points: {covered:.1%})',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, width="content")

    _render_confidence_intervals()

# ---------------------------------------------------------------------------
# Sigma vs. coverage: sliding-window empirical coverage, one method's own
# sorted sigma at a time.
# ---------------------------------------------------------------------------
with tab_sigma_cov:
    def _render_sigma_vs_coverage():
        if not data.has_per_point_data(run_path, dataset):
            st.info(
                f'No per-point data for "{dataset}" — it predates `src/run_sigma_sr.py` writing '
                "`testing_data.csv`/`methods_sigmas_test.csv`/`methods_intervals.csv` (or the older "
                "`per_point.csv`). Pick a different dataset above, or a newer run."
            )
            return

        df = data.load_per_point(run_path, dataset)
        methods_all = data.per_point_methods(df)

        # a constant sigma (e.g. standard_cp/mondrian_cp) has no ordering to
        # slide a window over -- same reason Sigma Ridgeline excludes them.
        all_methods = [m for m in methods_all if df[f"sigma_{m}"].nunique() > 1]
        constant_methods = [m for m in methods_all if m not in all_methods]

        if not all_methods:
            st.info("No method has a non-constant difficulty score for this dataset.")
            return
        if constant_methods:
            st.caption(
                "Not shown (constant sigma, no ordering to slide a window over): "
                + ", ".join(data.method_label(m) for m in constant_methods)
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            methods = st.multiselect(
                "Methods", options=all_methods, default=all_methods,
                format_func=data.method_label,
                key="sigma_cov_methods",
            )
        with col2:
            window_pct = st.slider("Window size (% of points)", min_value=5, max_value=50, value=10, step=5,
                                    key="sigma_cov_window")

        if not methods:
            st.warning("Select at least one method.")
            return

        n = len(df)
        window = max(3, round(n * window_pct / 100))

        cols = min(4, len(methods))
        rows = math.ceil(len(methods) / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[data.method_label(m) for m in methods])

        for i, m in enumerate(methods):
            row, col = i // cols + 1, i % cols + 1
            order = df[f"sigma_{m}"].to_numpy().argsort()
            sigma_sorted = df[f"sigma_{m}"].to_numpy()[order]
            covered_sorted = df[f"covered_{m}"].to_numpy()[order].astype(float)
            smoothed = pd.Series(covered_sorted).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=sigma_sorted, y=smoothed, mode="lines",
                    line=dict(color=data.method_color(m), width=2.5),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{data.method_label(m)}</b><br>sigma=%{{x:.4g}}<br>"
                        "coverage (windowed)=%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            fig.add_hline(y=target_coverage, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
            fig.update_xaxes(title="sigma", row=row, col=col)
            fig.update_yaxes(title="Coverage (windowed)", range=[0, 1.05], row=row, col=col)

        fig.update_layout(
            width=cols * data.CELL_SIZE, height=rows * data.CELL_SIZE + 60,
            margin=dict(t=60),
            title=f'"{dataset}" — window = {window} of {n} points ({window_pct}%)',
        )
        st.plotly_chart(fig, width="content")

    _render_sigma_vs_coverage()
