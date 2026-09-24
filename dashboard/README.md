# Comparison dashboard

Interactive Streamlit app for comparing CP methods across datasets and
result runs. Self-contained: its own dependencies below, and it only reads
from `results-*/` folders and imports (never modifies) `src/utils/plotting`
for consistent method labels/colors.

## Setup

```
.venv/bin/pip install -r dashboard/requirements.txt
```

## Run

```
.venv/bin/streamlit run dashboard/app.py
```

Opens at http://localhost:8501. `app.py` is a thin entry point: it sets
page config once and declares the sidebar navigation
(`st.navigation`/`st.Page`, a flat list of pages -- no section headers,
since each page below is itself already a many-tab section, so a
one-page-per-header group would just be an extra empty layer), then hands
off to whichever page is selected. `home.py` is the landing page (pick a
results run, see the method leaderboard: Pareto-non-dominance counts
across the run's datasets).

Sidebar pages, each with several tabs:
- **Method Comparison** (`pages/method_comparison.py`) — a single **Dataset**
  selector above the tabs drives every per-dataset tab; Pareto's Grid view,
  Difficulty Heatmap, and Dataset Characteristics always show every dataset
  regardless of it (see the module docstring). Tabs:
  - Pareto — median width vs. coverage; Grid (all datasets) or Detail
    (single dataset, via the shared selector) radio
  - Sigma Relationships — sigma vs. residual (calibration or test) or width
    (test only), unbinned, one subplot per method
  - Interval Width — one subplot per method, per-point width vs. y_pred
    plus a sliding-window median: where each method spends its width
  - Difficulty Heatmap — coverage/width by residual-rank decile; Grid (all
    methods) or Detail (single method) radio; binned on residual rank
    rather than each method's own sigma rank, so columns are directly
    comparable method-to-method
  - Dataset Characteristics — method width ratio vs. dataset size/features/R2
  - Confidence Intervals — sanity-check view of actual/predicted/interval
    band for a handful of test points
  - Conditional Coverage — sliding-window empirical coverage, one subplot
    per method, along y_pred or own sigma quantile, each with the question
    it answers. Below it, mean width vs. worst-group coverage, one point
    per method, with groups from worst slab, y_pred bins or own-sigma bins.
    Worst slab needs `testing_features.csv` (see below)
- **Hall of Fame** (`pages/6_Hall_of_Fame_Tradeoff.py`) — one dataset/loss's
  SR equations at a time, five tabs: Trade-off (coverage vs. width, colored
  by complexity), Sigma vs Outcome (each equation's own sigma vs.
  residual/width, unbinned), Interval Width and Conditional Coverage (same
  views as Method Comparison's, one subplot per equation), Complexity x
  Decile (heatmap, rows = complexity, columns =
  residual-rank decile)
- **Training** (`pages/9_Training_Dynamics.py`) — Training Dynamics: SR
  loss-vs-iteration curves (from a precomputed `loss_curve_<loss>.csv`,
  falling back to the raw TensorBoard log for older runs) and a
  convergence-speed comparison across datasets

Streamlit tabs can't nest, so a page whose own view has a Grid/Detail split
(Pareto, Difficulty Heatmap) uses a radio button for that, not a second
level of `st.tabs`.

## Adding a new plot

If it's a variant of an existing section's view (e.g. another cross-method
comparison), add it as another `with tab_x: ...` block (wrapped in its own
`_render_*()` function, called immediately after) inside that section's
existing page script, following `method_comparison.py`'s or
`6_Hall_of_Fame_Tradeoff.py`'s pattern — don't create a new top-level page
per plot.

For a genuinely new section, add a new `dashboard/pages/<name>.py` script
and add it to the `pages` list in `app.py` (there's no automatic
`pages/`-folder discovery — `app.py` calling `st.navigation` takes over
sidebar rendering entirely, so an un-listed script would never appear).

Either way: read the run path back with `st.session_state.get("run_path")`
and reuse `dashboard/data.py` for loading/method metadata. Don't call
`st.set_page_config` inside a page (only `app.py` may call it, once) — set
the sidebar/tab label via `st.Page(..., title=...)` instead. Inside a tab
function, use `return` for an early exit ("no data for this"), never
`st.stop()` — `st.stop()` halts the *entire* script, which would blank out
every tab after it since all tabs run in one script pass.

## Worst-slab coverage

`conditional_coverage.py` implements worst-slab coverage (Cauchois et al.,
2021): search 1000 random directions in feature space for the slab holding
at least a given fraction of test points with the lowest coverage, on half
of the test set, and report its coverage on the other half. It needs the
normalized test features, saved by `src/run_sigma_sr.py` as
`testing_features.csv`. For runs made before that, rebuild them from the
same seeded OpenML split:

```
.venv/bin/python src/analysis/backfill_testing_features.py results-sigma-sr-full
```

Tests: `uv run --no-project --with pytest --with pandas --with numpy pytest dashboard/tests`
