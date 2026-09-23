# -*- coding: utf-8 -*-
"""Entry point: declares the sidebar navigation and hands off to the
selected page. `home.py` is the landing page (run picker + leaderboard);
`pages/` scripts read the run path back from st.session_state["run_path"].
Uses st.navigation (not the automatic pages/-folder sidebar) so page labels
are independent of filenames -- a flat list, not grouped under section
headers: Method Comparison, Hall of Fame, and Training Dynamics are each
already a many-tab section on their own (a one-page-per-header group would
just be an extra empty layer), and Dataset Analysis is a single self-
contained view that doesn't need one either.
"""

import streamlit as st

st.set_page_config(page_title="CP Methods Comparison", layout="wide")

pages = [
    st.Page("home.py", title="Home", default=True),
    st.Page("pages/dataset_analysis.py", title="Dataset Analysis"),
    st.Page("pages/method_comparison.py", title="Method Comparison"),
    st.Page("pages/6_Hall_of_Fame_Tradeoff.py", title="Hall of Fame"),
    st.Page("pages/9_Training_Dynamics.py", title="Training Dynamics"),
]

st.navigation(pages).run()
