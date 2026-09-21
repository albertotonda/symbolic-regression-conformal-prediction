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
