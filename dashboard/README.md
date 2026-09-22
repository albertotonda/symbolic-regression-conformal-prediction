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

Opens at http://localhost:8501. Pick a results run on the home page (which
also shows a method leaderboard: Pareto-non-dominance counts across the
run's datasets), then open a plot from the sidebar:
- Pareto: Width vs Coverage
- Sigma Relationships — sigma vs. residual (calibration or test) or width
  (test only), unbinned, one subplot per method
- Width by Residual Rank
- Difficulty Heatmap — coverage/width by residual-rank decile (grid across
  all methods, or single-method detail); binned on residual rank rather
  than each method's own sigma rank, so columns are directly comparable
  method-to-method
- Sigma Ridgeline
- Hall of Fame Trade-off — every SR equation on the complexity/loss Pareto
  front, scattered by (coverage, width)
- Dataset Characteristics — method width ratio vs. dataset size/features/R2
- Method Head-to-Head — per-point width comparison between two methods
- Training Dynamics — SR loss-vs-iteration curves and a convergence-speed
  comparison across datasets
- Confidence Intervals — sanity-check view of actual/predicted/interval
  band for a handful of test points
- Sigma vs Coverage — sliding-window empirical coverage over each method's
  own sorted sigma, one subplot per method

## Adding a new plot

Add a new `dashboard/pages/N_<name>.py` script — Streamlit lists it in the
sidebar automatically. Read the run path back with
`st.session_state.get("run_path")` and reuse `dashboard/data.py` for
loading/method metadata.
