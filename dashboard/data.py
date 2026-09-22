# -*- coding: utf-8 -*-
"""
Shared data-loading layer for the dashboard: discovering results-* run
folders, loading their results.csv, and deriving method lists/labels/colors.
Method-naming conventions differ between runs (older runs use prefixed
names like `normalized_cp_knn_dist`, newer ones use short names like
`knn_dist`), so methods are always read from a run's own columns rather
than assumed from a fixed schema.
"""

import colorsys
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import translations  # noqa: E402

# A categorical palette independent of src/utils/plotting.py's, sized for
# this dashboard's own use (a scatter/small-multiples Pareto plot, which
# needs all-pairs color separation, not just adjacent-pair/legend order).
# Found by maximizing the worst-pair CVD/normal-vision distance over an
# OKLCH-sampled candidate pool, then confirmed with the dataviz skill's
# validator (scripts/validate_palette.py): all 6 checks pass in light mode
# for both the adjacent and --pairs all gates (WARN-only on the 6-8 CVD
# band and sub-3:1 contrast, both mitigated here by the always-on legend
# and per-point hover labels).
_CATEGORICAL_PALETTE = [
    "#2b6bab",  # blue
    "#c49f31",  # gold
    "#ab4b2b",  # rust
    "#47d18c",  # mint
    "#d61f7a",  # magenta
    "#4d1fd6",  # indigo
    "#af47d1",  # purple
    "#1fa8d6",  # sky blue
]
_KNOWN_METHOD_ORDER = [
    "standard_cp",
    "knn_dist",
    "knn_std",
    "knn_res",
    "var",
    "mondrian_cp",
    "sr_bin_crossfit",
]
METHOD_COLORS = dict(zip(_KNOWN_METHOD_ORDER, _CATEGORICAL_PALETTE))

# Shared figure sizing, so every page's plots stay consistent and there's
# one place to bump when "make the plots bigger" comes up again.
CELL_SIZE = 440  # px per subplot, kept equal on both axes so each cell is square
LEGEND_MARGIN = 180  # vertical space above a grid for the shared legend, clear of subplot titles
DETAIL_SIZE = 860  # px, single-item detail/full-size view


def discover_runs() -> list[dict]:
    """List results-* run folders that contain a results.csv, newest first."""
    runs = []
    for path in REPO_ROOT.glob("results-*"):
        results_csv = path / "results.csv"
        if not results_csv.exists():
            continue
        n_datasets = max(sum(1 for _ in open(results_csv)) - 1, 0)
        runs.append({
            "name": path.name,
            "path": str(path),
            "n_datasets": n_datasets,
            "mtime": path.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


@st.cache_data
def load_results(run_path: str) -> pd.DataFrame:
    """Load a run's results.csv (one row per dataset)."""
    return pd.read_csv(Path(run_path) / "results.csv")


def discover_methods(df: pd.DataFrame) -> list[str]:
    """Method keys present in a results.csv, known methods first (in the
    repo's canonical order) then any unrecognized ones alphabetically."""
    present = {c[: -len("_median")] for c in df.columns if c.endswith("_median")}
    ordered = [m for m in _KNOWN_METHOD_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def method_label(method: str) -> str:
    return translations.get(method, method.replace("_", " ").title())


def target_coverage(run_path: str) -> float:
    """The run's configured target confidence level, e.g. 0.95, read from
    its saved config.yaml snapshot; defaults to 0.95 if not found."""
    config_path = Path(run_path) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if isinstance(config, dict) and "confidence" in config:
            return float(config["confidence"])
    return 0.95


def _per_point_source_files(run_path: str, dataset_name: str) -> dict[str, Path]:
    dataset_dir = Path(run_path) / dataset_name
    return {
        "legacy": dataset_dir / "per_point.csv",
        "testing": dataset_dir / "testing_data.csv",
        "sigmas": dataset_dir / "methods_sigmas_test.csv",
        "intervals": dataset_dir / "methods_intervals.csv",
    }


def has_per_point_data(run_path: str, dataset_name: str) -> bool:
    paths = _per_point_source_files(run_path, dataset_name)
    if paths["legacy"].exists():
        return True
    return paths["testing"].exists() and paths["sigmas"].exists() and paths["intervals"].exists()


def has_interval_detail_data(run_path: str, dataset_name: str) -> bool:
    """Whether raw per-point (y, y_pred, lower_bound, upper_bound) can be
    reconstructed for this dataset -- needs the current `testing_data.csv` +
    `methods_intervals.csv` format; unlike `has_per_point_data`, the older
    `per_point.csv` format doesn't qualify since it only ever stored derived
    width/coverage, not the raw prediction or interval bounds."""
    paths = _per_point_source_files(run_path, dataset_name)
    return paths["testing"].exists() and paths["intervals"].exists()


@st.cache_data
def load_testing_data(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Raw test-set (y, y_pred, residuals), indexed the same 0-based way as
    `methods_intervals.csv`."""
    path = _per_point_source_files(run_path, dataset_name)["testing"]
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="index")


@st.cache_data
def load_intervals(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Long-format (method, index, lower_bound, upper_bound) for every
    method on this dataset's test set."""
    path = _per_point_source_files(run_path, dataset_name)["intervals"]
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_per_point(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Per-test-point sigma/width/coverage per method + the base
    regressor's absolute residual, for one dataset.

    Two source formats, tried in order:
    - `per_point.csv`, written directly by src/run_sigma_sr.py between
      2026-09-18 and its 2026-09-21 refactor (commit d5d2d40, "save csv
      instead of plots") -- kept for old runs that still have it.
    - `testing_data.csv` + `methods_sigmas_test.csv` +
      `methods_intervals.csv`, the format written since that refactor --
      reassembled here into the same per-point shape (same three
      `{abs_residual, sigma_<method>, width_<method>, covered_<method>}`
      columns) so every dashboard page keeps working unchanged. All three
      share the same 0-based test-set row order, so they join on that
      shared `index` column.

    Returns None if neither format is present, so callers can show a
    clear "not available" state instead of crashing.
    """
    paths = _per_point_source_files(run_path, dataset_name)

    if paths["legacy"].exists():
        return pd.read_csv(paths["legacy"])

    if not (paths["testing"].exists() and paths["sigmas"].exists() and paths["intervals"].exists()):
        return None

    testing = pd.read_csv(paths["testing"], index_col="index")
    sigmas = pd.read_csv(paths["sigmas"], index_col="index")
    intervals = pd.read_csv(paths["intervals"])

    df = pd.DataFrame(index=testing.index)
    df["abs_residual"] = testing["residuals"].abs()

    for method in sigmas.columns:
        df[f"sigma_{method}"] = sigmas[method]
        method_intervals = intervals.loc[intervals["method"] == method].set_index("index")
        df[f"width_{method}"] = method_intervals["upper_bound"] - method_intervals["lower_bound"]
        df[f"covered_{method}"] = (
            (testing["y"] >= method_intervals["lower_bound"])
            & (testing["y"] <= method_intervals["upper_bound"])
        )

    return df.reset_index(drop=True)


def per_point_methods(df: pd.DataFrame) -> list[str]:
    """Method keys with per-point data in a per_point.csv, same ordering
    convention as discover_methods()."""
    present = {c[len("sigma_"):] for c in df.columns if c.startswith("sigma_")}
    ordered = [m for m in _KNOWN_METHOD_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def method_color(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # deterministic fallback color for methods outside the fixed palette
    # (e.g. older runs' naming), so a given method key is stable across
    # reruns/pages even though it wasn't assigned a curated color.
    digest = hashlib.sha256(method.encode()).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.55)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def discover_losses(run_path: str, dataset_name: str) -> list[str]:
    """SR loss names with a `hof_<loss>.csv` in a dataset's folder (as
    opposed to the per-loss `hof_intervals_<loss>.csv`/
    `hof_sigmas_cal_<loss>.csv`/`hof_sigmas_test_<loss>.csv` siblings, which
    share the `hof_` prefix but aren't the main table)."""
    dataset_dir = Path(run_path) / dataset_name
    losses = []
    for path in dataset_dir.glob("hof_*.csv"):
        stem = path.stem[len("hof_"):]
        if stem.startswith("intervals_") or stem.startswith("sigmas_cal_") or stem.startswith("sigmas_test_"):
            continue
        losses.append(stem)
    return sorted(losses)


@st.cache_data
def load_hof(run_path: str, dataset_name: str, loss_name: str) -> pd.DataFrame | None:
    """One SR loss's Hall of Fame for one dataset: one row per equation on
    the complexity/loss Pareto front, indexed by `Complexity`, with columns
    `Loss`, `Equation`, `Chosen`, `ci_median`, `ci_mean`, `coverage`."""
    path = Path(run_path) / dataset_name / f"hof_{loss_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="Complexity")


@st.cache_data
def load_dataset_characteristics() -> pd.DataFrame:
    """Static per-dataset metadata from the OpenML-CTR23 suite (n_samples,
    n_features, missing_data, categorical_features, base-regressor R2/MSE),
    independent of any results run -- join on `dataset_name` to relate a
    run's per-dataset performance to what the dataset itself looks like.
    Numeric columns are comma-formatted strings (`n_samples`) or
    `"mean +/- std"` strings (the R2/MSE columns) in the source CSV, so both
    are parsed down to plain floats here.
    """
    path = REPO_ROOT / "results" / "OpenML-CTR23-statistics-500-estimators-10-fold-cv.csv"
    df = pd.read_csv(path)
    for col in ("n_samples", "n_features", "missing_data", "categorical_features"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(int)
    for col in df.columns:
        if col.startswith("R2_") or col.startswith("MSE_"):
            df[col] = df[col].astype(str).str.split(" +/- ", regex=False).str[0].astype(float)
    return df


def compute_pareto_dominance(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """For each dataset (row of `df`), find which of `methods` are
    Pareto-non-dominated on (lower `<method>_median` is better, higher
    `<method>_coverage` is better); tally non-dominated/dominated/"alone"
    (sole non-dominated method) counts per method across all datasets.

    Same dominance rule as `src/analysis/check_pareto_optimality.py`, just
    computed live from an already-loaded results.csv instead of requiring
    that script's separate `results-statistics.csv` to have been run.
    """
    counts = {m: {"non_dominated": 0, "dominated": 0, "alone": 0} for m in methods}

    for _, row in df.iterrows():
        points = {}
        for m in methods:
            median, coverage = row.get(f"{m}_median"), row.get(f"{m}_coverage")
            if median is None or coverage is None or pd.isna(median) or pd.isna(coverage):
                continue
            points[m] = (median, coverage)

        non_dominated = []
        for m, (median, coverage) in points.items():
            dominated = any(
                other_median <= median and other_coverage >= coverage
                for other_m, (other_median, other_coverage) in points.items()
                if other_m != m
            )
            counts[m]["dominated" if dominated else "non_dominated"] += 1
            if not dominated:
                non_dominated.append(m)

        if len(non_dominated) == 1:
            counts[non_dominated[0]]["alone"] += 1

    return pd.DataFrame.from_dict(counts, orient="index")


def early_stop_params(run_path: str) -> dict:
    """A run's `sr_params.early_stop_*` settings from its saved config.yaml
    snapshot -- read regardless of whether `early_stop` itself was on, since
    they're saved either way and are still useful as context (e.g. the
    `min_relative_improvement` value doubles as a sensible default
    `tolerance` for `compute_convergence_step`, since both express "how much
    relative improvement counts as negligible"). Note `chunk_size` here
    counts PySR outer iterations, not TensorBoard log steps (PySR logs
    several points per iteration, one per population), so it can't be used
    to replay `fit_with_early_stopping`'s block logic directly against a
    logged curve -- that's why `compute_convergence_step` uses a
    step-native definition instead. Defaults match
    `fit_with_early_stopping`'s own signature defaults.
    """
    defaults = {"chunk_size": 1, "patience": 3, "min_relative_improvement": 1e-3}
    config_path = Path(run_path) / "config.yaml"
    if not config_path.exists():
        return defaults
    with open(config_path) as f:
        config = yaml.safe_load(f)
    sr_params = (config or {}).get("sr_params", {})
    return {
        "chunk_size": sr_params.get("early_stop_chunk_size", defaults["chunk_size"]),
        "patience": sr_params.get("early_stop_patience", defaults["patience"]),
        "min_relative_improvement": sr_params.get(
            "early_stop_min_improvement", defaults["min_relative_improvement"]
        ),
    }


def tb_log_dir(run_path: str, dataset_name: str, loss_name: str) -> Path:
    return Path(run_path) / dataset_name / "tb_logs" / loss_name


@st.cache_data
def load_loss_curve(run_path: str, dataset_name: str, loss_name: str):
    """(steps, losses) for one dataset/loss's SR search, read from its
    TensorBoard log (`search/data/summaries/min_loss`, the best loss on the
    Pareto front, logged every iteration). Returns (None, None) if no log
    directory exists for this dataset/loss."""
    log_dir = tb_log_dir(run_path, dataset_name, loss_name)
    if not log_dir.exists():
        return None, None
    from src.utils.utils import read_tensorboard_scalar
    return read_tensorboard_scalar(str(log_dir), "search/data/summaries/min_loss")


def compute_convergence_step(steps, losses, tolerance=1e-3):
    """First step at which the search had already captured `1 - tolerance`
    of its total loss improvement (initial running-best loss down to the
    final one) -- e.g. `tolerance=1e-3` means "first step within 0.1% of
    the run's eventual best loss, relative to how much it improved in
    total." A convergence marker comparable across datasets/losses without
    needing to know how PySR's own TensorBoard step counter relates to its
    `niterations`/`populations` settings (steps aren't 1-per-iteration --
    PySR logs several points per iteration, one per population -- so a
    literal replay of `fit_with_early_stopping`'s iteration-block logic
    isn't meaningful against this axis).
    """
    steps = np.asarray(steps)
    running_best = np.minimum.accumulate(np.asarray(losses))
    initial, final = running_best[0], running_best[-1]
    total_improvement = initial - final
    if total_improvement <= 0:
        return steps[0]
    threshold = final + tolerance * total_improvement
    return steps[np.argmax(running_best <= threshold)]
