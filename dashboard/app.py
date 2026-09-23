# -*- coding: utf-8 -*-
"""Entry point: declares the sidebar's grouped navigation and hands off to
the selected page. `home.py` is the landing page (run picker + leaderboard);
`pages/` scripts read the run path back from st.session_state["run_path"].
Grouped with st.navigation (not the automatic pages/-folder sidebar) so
method-comparison, Hall-of-Fame, and training-dynamics pages sit under their
own headers instead of one flat list.
"""

import streamlit as st

st.set_page_config(page_title="CP Methods Comparison", layout="wide")

pages = {
    "Overview": [
        st.Page("home.py", title="Home", default=True),
    ],
    "Method Comparison": [
        st.Page("pages/method_comparison.py", title="Method Comparison"),
    ],
    "Hall of Fame": [
        st.Page("pages/6_Hall_of_Fame_Tradeoff.py", title="Hall of Fame"),
    ],
    "Training": [
        st.Page("pages/9_Training_Dynamics.py", title="Training Dynamics"),
    ],
}

st.navigation(pages).run()
