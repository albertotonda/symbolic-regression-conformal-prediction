import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import sys
    import os

    import matplotlib.pyplot as plt
    import matplotlib

    import pandas as pd
    from IPython.display import display

    import sympy

    from sklearn.ensemble import RandomForestRegressor

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    random_seed = 42
    return os, pd, plt, root_path, sympy


@app.cell
def _():
    return


@app.cell
def _(os, pd, plt, root_path, sympy):
    results_path = os.path.join(root_path, "results-sigma-sr-42_20260825-101216")
    methods = [
        "conformal_predictor",
        "normalized_cp_knn_dist",
        "normalized_cp_knn_std",
        "normalized_cp_knn_res",
        "normalized_cp_norm_var",
        "mondrian_cp",
        "symbolic_regression_mae",
        "symbolic_regression_mean_width",
        "symbolic_regression_pairwise_rank",
    ]

    df_results = pd.read_csv(os.path.join(results_path, "results.csv")).set_index("dataset_name")

    for name in df_results.index:
        df_equations = pd.read_csv(os.path.join(results_path, name, "symbolic_regression_mean_width.csv"))

        for candidate in df_equations["sympy_format"]:
            eq = sympy.sympify(candidate)
        x = df_equations["complexity"]
    
        # fig, ax = plt.subplots()
        # for m in methods:
        #     x = df_results.loc[name][m+"_coverage"]
        #     y = df_results.loc[name][m+"_median"]
        #     if "symbolic_regression" in m:
        #         ax.scatter(x, y, s=(120), label=m, edgecolors="red")
        #     else:
        #         ax.scatter(x, y, label=m)

        # ax.invert_xaxis()
        # ax.set_xlabel("coverage on the test set")
        # ax.set_ylabel("median amplitude of the confidence intervals")
        # ax.legend(loc="upper right")
    
        plt.show()

        break
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
