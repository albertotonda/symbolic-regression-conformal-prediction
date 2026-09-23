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

Sidebar sections:
- **Method Comparison** (cross-method, one run's datasets at a time):
  - Pareto: Width vs Coverage
  - Sigma Relationships — sigma vs. residual (calibration or test) or width
    (test only), unbinned, one subplot per method
  - Width by Residual Rank
  - Difficulty Heatmap — coverage/width by residual-rank decile (grid
    across all methods, or single-method detail); binned on residual rank
    rather than each method's own sigma rank, so columns are directly
    comparable method-to-method
  - Sigma Ridgeline
  - Dataset Characteristics — method width ratio vs. dataset size/features/R2
  - Method Head-to-Head — per-point width comparison between two methods
  - Confidence Intervals — sanity-check view of actual/predicted/interval
    band for a handful of test points
  - Sigma vs Coverage — sliding-window empirical coverage over each
    method's own sorted sigma, one subplot per method
- **Hall of Fame** (one dataset/loss's SR equations at a time), four tabs:
  Trade-off (coverage vs. width, colored by complexity), Sigma vs Outcome
  (each equation's own sigma vs. residual/width, unbinned), Sigma vs
  Coverage (sliding-window coverage per equation), Complexity x Decile
  (heatmap, rows = complexity, columns = residual-rank decile)
- **Training**: Training Dynamics — SR loss-vs-iteration curves (from a
  precomputed `loss_curve_<loss>.csv`, falling back to the raw TensorBoard
  log for older runs) and a convergence-speed comparison across datasets

## Adding a new plot

Add a new `dashboard/pages/<name>.py` script, then add it to the `pages`
dict in `app.py` under the section it belongs to (there's no more automatic
`pages/`-folder discovery — `app.py` calling `st.navigation` takes over
sidebar rendering entirely, so an un-listed script would never appear).
Read the run path back with `st.session_state.get("run_path")` and reuse
`dashboard/data.py` for loading/method metadata. Don't call
`st.set_page_config` inside the page itself (only `app.py` may call it,
once) — set the sidebar/tab label via `st.Page(..., title=...)` instead.
