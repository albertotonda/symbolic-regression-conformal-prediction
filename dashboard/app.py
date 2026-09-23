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
        st.Page("pages/1_Pareto_Width_vs_Coverage.py", title="Pareto: Width vs Coverage"),
        st.Page("pages/2_Sigma_Relationships.py", title="Sigma Relationships"),
        st.Page("pages/3_Width_by_Residual_Rank.py", title="Width by Residual Rank"),
        st.Page("pages/4_Difficulty_Heatmap.py", title="Difficulty Heatmap"),
        st.Page("pages/5_Sigma_Ridgeline.py", title="Sigma Ridgeline"),
        st.Page("pages/7_Dataset_Characteristics.py", title="Dataset Characteristics"),
        st.Page("pages/8_Method_Head_to_Head.py", title="Method Head-to-Head"),
        st.Page("pages/10_Confidence_Intervals.py", title="Confidence Intervals"),
        st.Page("pages/11_Sigma_vs_Coverage.py", title="Sigma vs Coverage"),
    ],
    "Hall of Fame": [
        st.Page("pages/6_Hall_of_Fame_Tradeoff.py", title="Hall of Fame"),
    ],
    "Training": [
        st.Page("pages/9_Training_Dynamics.py", title="Training Dynamics"),
    ],
}

st.navigation(pages).run()
