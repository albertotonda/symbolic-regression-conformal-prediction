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
page config once and declares the grouped sidebar navigation
(`st.navigation`/`st.Page`), then hands off to whichever page is selected.
`home.py` is the landing page (pick a results run, see the method
leaderboard: Pareto-non-dominance counts across the run's datasets).

Sidebar sections, each one page with several tabs:
- **Method Comparison** (`pages/method_comparison.py`) — a single **Dataset**
  selector above the tabs drives every per-dataset tab; Pareto's Grid view,
  Difficulty Heatmap, and Dataset Characteristics always show every dataset
  regardless of it (see the module docstring). Tabs:
  - Pareto — median width vs. coverage; Grid (all datasets) or Detail
    (single dataset, via the shared selector) radio
  - Sigma Relationships — sigma vs. residual (calibration or test) or width
    (test only), unbinned, one subplot per method
  - Width by Residual Rank
  - Difficulty Heatmap — coverage/width by residual-rank decile; Grid (all
    methods) or Detail (single method) radio; binned on residual rank
    rather than each method's own sigma rank, so columns are directly
    comparable method-to-method
  - Sigma Ridgeline
  - Dataset Characteristics — method width ratio vs. dataset size/features/R2
  - Method Head-to-Head — per-point width comparison between two methods
  - Confidence Intervals — sanity-check view of actual/predicted/interval
    band for a handful of test points
  - Sigma vs Coverage — sliding-window empirical coverage over each
    method's own sorted sigma, one subplot per method
- **Hall of Fame** (`pages/6_Hall_of_Fame_Tradeoff.py`) — one dataset/loss's
  SR equations at a time, four tabs: Trade-off (coverage vs. width, colored
  by complexity), Sigma vs Outcome (each equation's own sigma vs.
  residual/width, unbinned), Sigma vs Coverage (sliding-window coverage per
  equation), Complexity x Decile (heatmap, rows = complexity, columns =
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
and add it to the `pages` dict in `app.py` (there's no automatic
`pages/`-folder discovery — `app.py` calling `st.navigation` takes over
sidebar rendering entirely, so an un-listed script would never appear).

Either way: read the run path back with `st.session_state.get("run_path")`
and reuse `dashboard/data.py` for loading/method metadata. Don't call
`st.set_page_config` inside a page (only `app.py` may call it, once) — set
the sidebar/tab label via `st.Page(..., title=...)` instead. Inside a tab
function, use `return` for an early exit ("no data for this"), never
`st.stop()` — `st.stop()` halts the *entire* script, which would blank out
every tab after it since all tabs run in one script pass.
