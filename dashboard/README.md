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

Opens at http://localhost:8501. Pick a results run on the home page, then
open a plot from the sidebar:
- Pareto: Width vs Coverage
- Sigma vs Residuals
- Width by Residual Rank
- Difficulty Heatmap
- Sigma Ridgeline

## Adding a new plot

Add a new `dashboard/pages/N_<name>.py` script — Streamlit lists it in the
sidebar automatically. Read the run path back with
`st.session_state.get("run_path")` and reuse `dashboard/data.py` for
loading/method metadata.
