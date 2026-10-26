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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import sympy
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import method_stats  # noqa: E402
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
    "sr_pinball",
]
METHOD_COLORS = dict(zip(_KNOWN_METHOD_ORDER, _CATEGORICAL_PALETTE))

# Shared figure sizing, so every page's plots stay consistent and there's
# one place to bump when "make the plots bigger" comes up again.
CELL_SIZE = 440  # px per subplot, kept equal on both axes so each cell is square
LEGEND_MARGIN = 180  # vertical space above a grid for the shared legend, clear of subplot titles
DETAIL_SIZE = 860  # px, single-item detail/full-size view


def show_results_folder(run_path: str) -> None:
    """Caption naming the results folder a page reads from."""
    st.caption(f"Results folder: `{Path(run_path).name}`")


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


def load_results(run_path: str) -> pd.DataFrame:
    """Load a run's results.csv (one row per dataset), with each SR method's
    columns taken from the Hall of Fame equation picked by `sr_selection()`."""
    return _load_results(run_path, sr_selection())


@st.cache_data
def _load_results(run_path: str, selection: str) -> pd.DataFrame:
    df = pd.read_csv(Path(run_path) / "results.csv")
    for i, name in df["dataset_name"].items():
        for method, complexity in _sr_swaps(run_path, name, selection).items():
            hof = load_hof(run_path, name, method[len("sr_"):])
            for suffix, column in [("mean", "ci_mean"), ("median", "ci_median"), ("coverage", "coverage")]:
                if f"{method}_{suffix}" in df.columns:
                    df.loc[i, f"{method}_{suffix}"] = hof.loc[complexity, column]
    return df


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


def has_calibration_data(run_path: str, dataset_name: str) -> bool:
    return (Path(run_path) / dataset_name / "calibration_data.csv").exists()


@st.cache_data
def load_calibration_data(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Raw calibration-set (y, y_pred, residuals) -- the calibration-side
    counterpart of `load_testing_data`. Lighter than
    `load_per_point_calibration`: no sigma columns required, just the file
    itself, since this doesn't need any method's difficulty score."""
    path = Path(run_path) / dataset_name / "calibration_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="index")


def load_intervals(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Long-format (method, index, lower_bound, upper_bound) for every
    method on this dataset's test set, SR methods following `sr_selection()`."""
    return _load_intervals(run_path, dataset_name, sr_selection())


@st.cache_data
def _load_intervals(run_path: str, dataset_name: str, selection: str) -> pd.DataFrame | None:
    path = _per_point_source_files(run_path, dataset_name)["intervals"]
    if not path.exists():
        return None
    intervals = pd.read_csv(path)
    for method, complexity in _sr_swaps(run_path, dataset_name, selection).items():
        hof_path = Path(run_path) / dataset_name / f"hof_intervals_{method[len('sr_'):]}.csv"
        if not hof_path.exists():
            continue
        hof = pd.read_csv(hof_path)
        replacement = hof[hof["complexity"] == complexity].drop(columns="complexity").assign(method=method)
        intervals = pd.concat([intervals[intervals["method"] != method], replacement[intervals.columns]],
                              ignore_index=True)
    return intervals


def _load_sigmas(run_path: str, dataset_name: str, split: str, selection: str) -> pd.DataFrame:
    """`methods_sigmas_<split>.csv` (split = "test" or "cal"), with each SR
    method's column taken from the selected Hall of Fame equation."""
    dataset_dir = Path(run_path) / dataset_name
    sigmas = pd.read_csv(dataset_dir / f"methods_sigmas_{split}.csv", index_col="index")
    for method, complexity in _sr_swaps(run_path, dataset_name, selection).items():
        hof_path = dataset_dir / f"hof_sigmas_{split}_{method[len('sr_'):]}.csv"
        if method in sigmas.columns and hof_path.exists():
            sigmas[method] = pd.read_csv(hof_path, index_col="index")[str(complexity)]
    return sigmas


def load_method_stats(run_path: str) -> pd.DataFrame:
    """`method_stats.dataset_stats` for every dataset in a run with
    `testing_data.csv` + `methods_intervals.csv`, stacked into one long
    table with `dataset_name` and `method` columns. Not cached itself
    (st.cache_data wouldn't notice edits to method_stats.py); the CSV
    loads it relies on are."""
    alpha = 1 - target_coverage(run_path)
    frames = []
    for name in sorted(load_results(run_path)["dataset_name"].unique()):
        if not has_interval_detail_data(run_path, name):
            continue
        stats = method_stats.dataset_stats(load_testing_data(run_path, name), load_intervals(run_path, name), alpha)
        frames.append(stats.rename_axis("method").reset_index().assign(dataset_name=name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    clear "not available" state instead of crashing. SR methods follow
    `sr_selection()` (current format only).
    """
    return _load_per_point(run_path, dataset_name, sr_selection())


@st.cache_data
def _load_per_point(run_path: str, dataset_name: str, selection: str) -> pd.DataFrame | None:
    paths = _per_point_source_files(run_path, dataset_name)

    if paths["legacy"].exists():
        return pd.read_csv(paths["legacy"])

    if not (paths["testing"].exists() and paths["sigmas"].exists() and paths["intervals"].exists()):
        return None

    testing = pd.read_csv(paths["testing"], index_col="index")
    sigmas = _load_sigmas(run_path, dataset_name, "test", selection)
    intervals = _load_intervals(run_path, dataset_name, selection)

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


def has_calibration_per_point_data(run_path: str, dataset_name: str) -> bool:
    dataset_dir = Path(run_path) / dataset_name
    return (dataset_dir / "calibration_data.csv").exists() and (dataset_dir / "methods_sigmas_cal.csv").exists()


def load_per_point_calibration(run_path: str, dataset_name: str) -> pd.DataFrame | None:
    """Per-calibration-point sigma + the base regressor's absolute residual,
    for one dataset -- the calibration-side counterpart of `load_per_point`.
    No `width`/`covered` columns: those aren't defined for calibration
    points in split conformal (the calibration set sets the quantile, it
    never gets its own prediction interval), so only `sigma_<method>` and
    `abs_residual` exist here. Only supports the current
    (`calibration_data.csv` + `methods_sigmas_cal.csv`) format -- there's no
    calibration-side equivalent of the older `per_point.csv`. SR methods
    follow `sr_selection()`.
    """
    return _load_per_point_calibration(run_path, dataset_name, sr_selection())


@st.cache_data
def _load_per_point_calibration(run_path: str, dataset_name: str, selection: str) -> pd.DataFrame | None:
    dataset_dir = Path(run_path) / dataset_name
    cal_path = dataset_dir / "calibration_data.csv"
    sigmas_path = dataset_dir / "methods_sigmas_cal.csv"
    if not (cal_path.exists() and sigmas_path.exists()):
        return None

    calibration = pd.read_csv(cal_path, index_col="index")
    sigmas = _load_sigmas(run_path, dataset_name, "cal", selection)

    df = pd.DataFrame(index=calibration.index)
    df["abs_residual"] = calibration["residuals"].abs()
    for method in sigmas.columns:
        df[f"sigma_{method}"] = sigmas[method]

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


def n_features_used(equation: str) -> int:
    """Number of distinct input features (`x0`, `x1`, ...) in a PySR
    equation string."""
    return len(set(re.findall(r"\bx\d+\b", equation)))


# logm(u) = log(|u| + 1e-8) in the SR config; the 1e-8 is dropped so that
# exp(c * logm(u)) displays as |u|**c
_SYMPY_LOCALS = {"logm": lambda u: sympy.log(sympy.Abs(u))}


@st.cache_data
def difficulty_equation(equation: str, digits: int = 4) -> str:
    """Difficulty score sigma = exp(equation) for a PySR equation (which
    predicts log sigma), simplified for display: top-level constants become
    a factor e^c, top-level c*log(u) terms become u**c, and every other term
    stays inside one exp(...). Floats are rounded to `digits` significant
    digits."""
    expr = sympy.expand(sympy.sympify(equation, locals=_SYMPY_LOCALS),
                        power_exp=False, power_base=False, log=False)
    factor, rest = sympy.Integer(1), []
    for term in sympy.Add.make_args(expr):
        coeff, body = term.as_coeff_Mul()
        if body == 1:
            factor *= sympy.exp(coeff)
        elif isinstance(body, sympy.log):
            factor *= body.args[0] ** coeff
        else:
            rest.append(term)
    sigma = factor * sympy.exp(sympy.Add(*rest))
    return str(sigma.xreplace({f: sympy.Float(f, digits) for f in sigma.atoms(sympy.Float)}))


def pysr_selections(hof: pd.DataFrame) -> dict[str, int]:
    """Complexity PySR's `get_best` would pick under each `model_selection`
    strategy, mirroring `pysr.sr.idx_model_selection`/`calculate_scores`.
    Score is -d(log loss)/d(complexity) between consecutive HoF entries (0
    for the simplest); "score" takes the max score, "best" the max score
    among losses <= 1.5x the minimum, "accuracy" the minimum loss."""
    hof = hof.sort_index()
    loss = hof["Loss"].to_numpy(dtype=float)
    complexity = hof.index.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = -np.log(loss[1:] / loss[:-1]) / np.diff(complexity)
    score = np.where(loss[1:] > 0, score, np.inf)
    score = pd.Series(np.concatenate([[0.0], score]), index=hof.index)
    near_best = hof["Loss"] <= 1.5 * hof["Loss"].min()
    return {
        "accuracy": int(hof["Loss"].idxmin()),
        "score": int(score.idxmax()),
        "best": int(score[near_best].idxmax()),
    }


# PySR `model_selection` strategies offered for the SR methods, as labels
SR_SELECTIONS = {
    "accuracy": "accuracy (lowest loss)",
    "score": "score (largest loss drop per complexity)",
    "best": "best (largest score among losses ≤ 1.5× min)",
}


def sr_selection() -> str:
    """The SR equation selection picked on the Home page; "accuracy" by
    default, which is the equation the experiment itself keeps."""
    return st.session_state.get("sr_selection", "accuracy")


@st.cache_data
def _sr_swaps(run_path: str, dataset_name: str, selection: str) -> dict[str, int]:
    """SR methods (`sr_<loss>`) whose selected Hall of Fame equation is not
    the one the run saved in results.csv/methods_*.csv (its `Chosen` row),
    mapped to the selected equation's complexity."""
    swaps = {}
    for loss in discover_losses(run_path, dataset_name):
        hof = load_hof(run_path, dataset_name, loss)
        if hof is None or hof.empty:
            continue
        complexity = pysr_selections(hof)[selection]
        chosen = hof.index[hof["Chosen"].astype(bool)] if "Chosen" in hof.columns else []
        if complexity not in chosen:
            swaps[f"sr_{loss}"] = complexity
    return swaps


def has_hof_per_point_data(run_path: str, dataset_name: str, loss_name: str) -> bool:
    dataset_dir = Path(run_path) / dataset_name
    return (
        (dataset_dir / "testing_data.csv").exists()
        and (dataset_dir / f"hof_sigmas_test_{loss_name}.csv").exists()
        and (dataset_dir / f"hof_intervals_{loss_name}.csv").exists()
    )


@st.cache_data
def load_hof_per_point(run_path: str, dataset_name: str, loss_name: str) -> pd.DataFrame | None:
    """Per-test-point sigma/width/coverage for *every* Hall-of-Fame equation
    (keyed by complexity, not method) + the base regressor's absolute
    residual, for one dataset/loss -- the per-equation analogue of
    `load_per_point` (which is per-method). Same three
    `{abs_residual, sigma_<complexity>, width_<complexity>, covered_<complexity>}`
    column shape, so the same per-point analysis logic works on either.
    """
    if not has_hof_per_point_data(run_path, dataset_name, loss_name):
        return None
    dataset_dir = Path(run_path) / dataset_name

    testing = pd.read_csv(dataset_dir / "testing_data.csv", index_col="index")
    sigmas = pd.read_csv(dataset_dir / f"hof_sigmas_test_{loss_name}.csv", index_col="index")
    intervals = pd.read_csv(dataset_dir / f"hof_intervals_{loss_name}.csv")

    df = pd.DataFrame(index=testing.index)
    df["abs_residual"] = testing["residuals"].abs()
    for col in sigmas.columns:
        complexity = int(col)
        eq_intervals = intervals.loc[intervals["complexity"] == complexity].set_index("index")
        df[f"sigma_{complexity}"] = sigmas[col]
        df[f"width_{complexity}"] = eq_intervals["upper_bound"] - eq_intervals["lower_bound"]
        df[f"covered_{complexity}"] = (
            (testing["y"] >= eq_intervals["lower_bound"]) & (testing["y"] <= eq_intervals["upper_bound"])
        )

    return df.reset_index(drop=True)


def hof_per_point_complexities(df: pd.DataFrame) -> list[int]:
    """Complexity values with per-point data in a `load_hof_per_point`
    frame, ascending."""
    return sorted(int(c[len("sigma_"):]) for c in df.columns if c.startswith("sigma_"))


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


@st.cache_data
def load_dataset_overview(run_path: str) -> pd.DataFrame:
    """One row per dataset in a run: OpenML-CTR23 metadata (target, size,
    feature types, CV R2 of a random forest) next to this run's own split
    sizes and base-regressor test R2. Metadata columns are NaN for datasets
    not in the CTR23 table."""
    results = load_results(run_path)
    characteristics = load_dataset_characteristics()
    overview = results[["dataset_name"] + (["r2"] if "r2" in results.columns else [])].merge(
        characteristics, on="dataset_name", how="left"
    )
    overview["n_calibration"] = [
        len(df) if (df := load_calibration_data(run_path, name)) is not None else None
        for name in overview["dataset_name"]
    ]
    overview["n_test"] = [
        len(df) if (df := load_testing_data(run_path, name)) is not None else None
        for name in overview["dataset_name"]
    ]
    columns = {
        "dataset_name": "Dataset",
        "target_name": "Target",
        "n_samples": "n samples",
        "n_features": "n features",
        "categorical_features": "Categorical features",
        "missing_data": "Missing values",
        "n_calibration": "n calibration",
        "n_test": "n test",
        "r2": "Test R² (this run)",
        "R2_RandomForestRegressor": "CV R² (RF, OpenML)",
    }
    overview = overview[[c for c in columns if c in overview.columns]].rename(columns=columns)
    return overview.sort_values("Dataset").reset_index(drop=True)


@st.cache_data(show_spinner="Fetching dataset description from OpenML…")
def load_dataset_description(dataset_name: str) -> dict | None:
    """The dataset's own OpenML description text, fetched live via the
    OpenML API: `task_id` looked up from the same CTR23 metadata table as
    `load_dataset_characteristics`, then `openml.tasks.get_task(task_id)
    .get_dataset()` -- the same task_id -> dataset path used by
    `src/utils/data.py`'s `load_and_preprocess_openml_task`. Cached across
    reruns since it's a network call; returns None if the dataset isn't in
    the CTR23 table or the API call fails (offline, dataset removed, etc.)
    so callers can show a clear fallback instead of crashing the page.
    """
    characteristics = load_dataset_characteristics()
    match = characteristics.loc[characteristics["dataset_name"] == dataset_name, "task_id"]
    if match.empty:
        return None
    task_id = int(match.iloc[0])

    import openml
    try:
        openml_dataset = openml.tasks.get_task(task_id).get_dataset()
    except Exception:
        return None

    return {
        "description": openml_dataset.description,
        "dataset_id": openml_dataset.dataset_id,
        "citation": getattr(openml_dataset, "citation", None),
    }


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


def _loss_curve_cache_path(run_path: str, dataset_name: str, loss_name: str) -> Path:
    return Path(run_path) / dataset_name / f"loss_curve_{loss_name}.csv"


@st.cache_data
def load_loss_curve(run_path: str, dataset_name: str, loss_name: str):
    """(steps, losses) for one dataset/loss's SR search (the best loss on
    the Pareto front, logged every iteration).

    Prefers the precomputed `loss_curve_<loss>.csv` (step,loss columns)
    that `run_sigma_sr.py` saves alongside its PNG plot, since reading it
    back from the raw TensorBoard log is expensive: the SR search logs many
    other tags too (per-complexity equation losses, full equation-string
    tensors, population-complexity histograms), and TensorBoard's
    EventAccumulator fully parses all of them on `Reload()` regardless of
    which single tag is actually wanted -- observed at ~7s and 100+MB per
    dataset on a real run, which multiplies fast across dozens of datasets.
    Falls back to that slow path for older runs that predate the CSV, and
    then writes it out as a cache so the next read of this same
    dataset/loss is fast too (best-effort -- silently skipped if the run
    folder isn't writable).

    Returns (None, None) if neither the CSV nor a TensorBoard log exists.
    """
    cache_path = _loss_curve_cache_path(run_path, dataset_name, loss_name)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return cached["step"].tolist(), cached["loss"].tolist()

    log_dir = tb_log_dir(run_path, dataset_name, loss_name)
    if not log_dir.exists():
        return None, None
    from src.utils.utils import read_tensorboard_scalar
    steps, losses = read_tensorboard_scalar(str(log_dir), "search/data/summaries/min_loss")

    try:
        pd.DataFrame({"step": steps, "loss": losses}).to_csv(cache_path, index=False)
    except OSError:
        pass
    return steps, losses


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
