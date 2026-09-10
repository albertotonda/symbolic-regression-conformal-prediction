import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import sys
    import os

    import numpy as np
    import pandas as pd
    import sympy

    from crepes.extras import DifficultyEstimator, binning
    from crepes import WrapRegressor, ConformalRegressor

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    import openml

    from IPython.display import display
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    suite_id = 353
    random_seed = 42
    confidence = 0.95
    return (
        ConformalRegressor,
        DifficultyEstimator,
        RandomForestRegressor,
        StandardScaler,
        WrapRegressor,
        binning,
        confidence,
        display,
        mcolors,
        mo,
        np,
        openml,
        os,
        pd,
        plt,
        r2_score,
        random_seed,
        root_path,
        suite_id,
        sympy,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(StandardScaler, openml, random_seed, train_test_split):
    class Dataset:
        def __init__(self, task_id):
            task = openml.tasks.get_task(task_id, download_splits=True)
            dataset = task.get_dataset()
            X_raw, y_raw, categorical, features = dataset.get_data(target=dataset.default_target_attribute)

            self.X = X_raw.loc[y_raw.notna()]
            self.y = y_raw.dropna()

            for i, c in enumerate(self.X.columns):
                if categorical[i]:
                    self.X[c] = self.X[c].cat.codes

            self.X.dropna(axis=1, how="any", inplace=True)
        
            self.name : str          = dataset.name
            self.id : int            = task_id
            self.n_samples : int     = self.X.shape[0]
            self.n_features : int    = self.X.shape[1]
            self.categorical : bool  = any(categorical)
            self.missing_data : bool = self.X.isna().any().any()
            self.description : str   = dataset.description

        def scale_and_split_dataset(self):
            X = self.X.values
            y = self.y.values

            X_prop_train, X_test, y_prop_train, y_test = train_test_split(
                X, y, test_size=0.5, shuffle=True, random_state=random_seed
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_test, y_test, test_size=0.5, shuffle=True, random_state=random_seed
            )

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_prop_train = scaler_X.fit_transform(X_prop_train)
            X_cal        = scaler_X.transform(X_cal)
            X_test       = scaler_X.transform(X_test)

            y_prop_train = scaler_y.fit_transform(y_prop_train.reshape(-1, 1)).ravel()
            y_cal        = scaler_y.transform(y_cal.reshape(-1, 1)).ravel()
            y_test       = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            return X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test

        def __str__(self):
            return f"""
    Dataset {self.name} ({self.id}):
        Number of samples: {self.n_samples}
        Number of features: {self.n_features}
        Includes categorical features: {str(self.categorical)}
        Missing data: {self.missing_data}

        {self.description}
            """

    return (Dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset analysis
    """)
    return


@app.cell
def _(Dataset, display, openml, pd, suite_id):
    suite = openml.study.get_suite(suite_id)
    task_ids = [t for t in suite.tasks]

    dataset_dict = []

    for task_id in task_ids:
        dataset = Dataset(task_id)
        dataset_dict.append({
            "name": dataset.name,
            "id": dataset.id,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "categorical": dataset.categorical,
            "missing_data": dataset.missing_data,
        })

    df_datasets = pd.DataFrame(dataset_dict).set_index("name")
    display(df_datasets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Synthetic dataset

    $$f(X) = 2x_3 - 1.5x_4 + x_3 x_4 + \mathcal{N}(0,\sigma(X)) \quad x_i \in [-1, 1]$$
    $$\sigma(X) = \exp(2x_1 + 0.5x_2)$$
    """)
    return


@app.cell
def _():
    # synthetic_data_path = os.path.join(root_path, "results-test-sigma-dummy-42_20260904-154629")
    # df_synthetic_results = pd.read_csv(os.path.join(synthetic_data_path, "results.csv")).set_index("task_id")
    # df_synthetic_eq = pd.read_csv(os.path.join(synthetic_data_path, "synthetic_noise_test", "symbolic_regression_bin_crossfit_equations.csv"))

    # display(df_synthetic_eq)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CP computation
    """)
    return


@app.cell
def _(display, os, pd, root_path):
    data_folder = os.path.join(root_path, "results-sigma-sr-hybrid-loss")
    df_results = pd.read_csv(os.path.join(data_folder, "results.csv")).set_index("dataset_name")
    display(df_results)
    return data_folder, df_results


@app.cell
def _(
    Dataset,
    RandomForestRegressor,
    WrapRegressor,
    df_results,
    r2_score,
    random_seed,
):
    example_dataset = Dataset(int(df_results.iloc[0]["task_id"]))

    X_prop_train, X_cal, X_test, y_prop_train, y_cal, y_test = example_dataset.scale_and_split_dataset()

    base_regressor = WrapRegressor(RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed))
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')
    return (
        X_cal,
        X_prop_train,
        X_test,
        base_regressor,
        example_dataset,
        y_cal,
        y_cal_pred,
        y_prop_train,
        y_test,
        y_test_pred,
    )


@app.cell(hide_code=True)
def _(
    DifficultyEstimator,
    WrapRegressor,
    X_cal,
    X_prop_train,
    X_test,
    base_regressor,
    binning,
    confidence,
    df_results,
    example_dataset,
    np,
    random_seed,
    y_cal,
    y_prop_train,
    y_test,
):
    def compute_normalized_intervals(de, learner_prop, X_cal, y_cal, X_test, confidence):
        """
        Calibrate a normalized conformal regressor using an already-fitted
        DifficultyEstimator. Returns the confidence intervals for the test set,
        together with the difficulty estimates on the calibration and test sets.
        """
        sigmas_cal = de.apply(X_cal)

        regressor_norm = WrapRegressor(learner_prop)
        regressor_norm.calibrate(X_cal, y_cal, de=de)

        intervals = regressor_norm.predict_int(X_test, confidence=confidence)

        return intervals, sigmas_cal

    def _find_bin_thresholds_with_min_size(sigmas_cal_var, min_points, random_seed):
            """
            crepes.extras.binning's min_size parameter requests bins=len(values)//
            min_size equal-frequency bins, but with tied/duplicated difficulty
            scores (common for the ensemble-variance estimator) pd.qcut can still
            leave a handful of bins a few points short of min_size. So, rather than
            trusting the requested bin count outright, verify the actual per-bin
            counts and back off the number of bins until every one of them holds at
            least min_points calibration points.
            """
            number_of_bins = len(sigmas_cal_var) // min_points
            while number_of_bins > 1:
                assigned_bins, bin_thresholds = binning(
                    sigmas_cal_var, bins=number_of_bins, seed=random_seed)
                counts = np.bincount(assigned_bins.astype(int))
                if counts.min() >= min_points:
                    return bin_thresholds
                number_of_bins -= 1
            return np.array([-np.inf, np.inf])

    def compute_ci_stats(confidence_intervals):
        ci_amplitude_mean = np.nanmean((confidence_intervals[:,1] - confidence_intervals[:,0]))
        ci_amplitude_median = np.nanmedian((confidence_intervals[:,1] - confidence_intervals[:,0]))
        # this expression below is a bit of a mess, but it's 1 if the measured
        # value falls within the confidence intervals, and 0 otherwise (summed up, divided by n_samples)
        coverage = np.nansum([1 if (y_test[i] >= confidence_intervals[i,0] and
                               y_test[i] <= confidence_intervals[i,1]) else 0
                        for i in range(len(y_test))]) / len(y_test)

        return ci_amplitude_mean, ci_amplitude_median, coverage


    learner_prop = base_regressor.learner
    sigmas_comp = {}
    sigmas_train = {}
    sigmas_cal = {}
    sigmas_test = {}
    conf_intervals = {}

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    # Standard CP
    print("Computing CI for SCP...")
    base_regressor.calibrate(X_cal, y_cal)
    sigmas_comp["conformal_predictor"] = np.ones(len(X_cal))
    conf_intervals["conformal_predictor"] = base_regressor.predict_int(X_test, confidence=confidence)

    # KNN distance
    print("Computing CI for knn_dist NCP...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, scaler=True)
    de_knn_dist_oob = DifficultyEstimator()
    de_knn_dist_oob.fit(X=X_prop_train, scaler=True, oob=True)
    sigmas_train["normalized_cp_knn_dist"] = de_knn_dist_oob.apply()
    sigmas_cal["normalized_cp_knn_dist"] = de_knn_dist_oob.apply(X_cal)
    sigmas_test["normalized_cp_knn_dist"] = de_knn_dist_oob.apply(X_test)
    conf_intervals["normalized_cp_knn_dist"], sigmas_comp["normalized_cp_knn_dist"] = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN std
    print("Computing CI for knn_std NCP...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
    de_knn_std_oob = DifficultyEstimator()
    de_knn_std_oob.fit(X=X_prop_train, y=y_prop_train, scaler=True, oob=True)
    sigmas_train["normalized_cp_knn_std"] = de_knn_std_oob.apply()
    sigmas_cal["normalized_cp_knn_std"] = de_knn_std_oob.apply(X_cal)
    sigmas_test["normalized_cp_knn_std"] = de_knn_std_oob.apply(X_test)
    conf_intervals["normalized_cp_knn_std"], sigmas_comp["normalized_cp_knn_std"] = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN out-of-bag residuals
    print("Computing CI for knn_res NCP...")
    de_knn_res = DifficultyEstimator()
    de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
    de_knn_res_oob = DifficultyEstimator()
    de_knn_res_oob.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True, oob=True)
    sigmas_train["normalized_cp_knn_res"] = de_knn_res_oob.apply()
    sigmas_cal["normalized_cp_knn_res"] = de_knn_res_oob.apply(X_cal)
    sigmas_test["normalized_cp_knn_res"] = de_knn_res_oob.apply(X_test)
    conf_intervals["normalized_cp_knn_res"], sigmas_comp["normalized_cp_knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, confidence)

    # Random Forest variance
    print("Computing CI for var NCP...")
    de_var = DifficultyEstimator()
    de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
    de_var_oob = DifficultyEstimator()
    de_var_oob.fit(X=X_prop_train, learner=learner_prop, scaler=True, oob=True)
    sigmas_train["normalized_cp_norm_var"] = de_var_oob.apply()
    sigmas_cal["normalized_cp_norm_var"] = de_var.apply(X_cal) # For cal and test, use default (no oob) version, otherwise same oob trees are used instead of full model
    sigmas_test["normalized_cp_norm_var"] = de_var.apply(X_test)
    conf_intervals["normalized_cp_norm_var"], sigmas_comp["normalized_cp_norm_var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, confidence)

    # Mondrian CP using variance
    print("Computing CI for MCP...")
    min_points = int(1 / (1-confidence) - 1) + 1
    bin_thresholds = _find_bin_thresholds_with_min_size(sigmas_comp["normalized_cp_norm_var"], min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas_comp["mondrian_cp"] = np.ones(len(X_cal))
    conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=confidence)

    # augment input using sigmas
    X_train_sr = np.zeros((X_prop_train.shape[0], len(sigmas_train)), dtype=np.float32)
    X_cal_sr = np.zeros((X_cal.shape[0], len(sigmas_cal)), dtype=np.float32)
    X_test_sr = np.zeros((X_test.shape[0], len(sigmas_test)), dtype=np.float32)
    for i, key in enumerate(sigmas_train.keys()):
            X_train_sr[:,i] = sigmas_train[key]
            X_cal_sr[:,i] = sigmas_cal[key]
            X_test_sr[:,i] = sigmas_test[key]
    X_train_sr = np.concatenate((X_prop_train, X_train_sr), axis=1)
    X_cal_sr = np.concatenate((X_cal, X_cal_sr), axis=1)
    X_test_sr = np.concatenate((X_test, X_test_sr), axis=1)

    methods = ["conformal_predictor", "normalized_cp_knn_dist", "normalized_cp_knn_std", "normalized_cp_knn_res", "normalized_cp_norm_var", "mondrian_cp"]
    print("SANITY CHECK")
    for m in methods:
        mean, median, coverage = compute_ci_stats(conf_intervals[m])
        print(f"""
        {m}:
            mean: {mean} vs {df_results.loc[example_dataset.name, f"{m}_mean"]}
            median: {median} vs {df_results.loc[example_dataset.name, f"{m}_median"]}
            coverage: {coverage} vs {df_results.loc[example_dataset.name, f"{m}_coverage"]}
        """)
    return X_cal_sr, X_test_sr, X_train_sr, compute_ci_stats, learner_prop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Equation analysis
    """)
    return


@app.cell(hide_code=True)
def _(
    ConformalRegressor,
    DifficultyEstimator,
    X_cal,
    X_cal_sr,
    X_test,
    X_test_sr,
    X_train_sr,
    compute_ci_stats,
    confidence,
    data_folder,
    df_results,
    display,
    example_dataset,
    learner_prop,
    os,
    pd,
    sympy,
    y_cal,
):
    eq_dict = []

    for dataset_name in df_results.index:

        # TODO: remove this for full analysis
        if dataset_name != example_dataset.name:
            continue

        eq_df = pd.read_csv(os.path.join(data_folder, dataset_name, "symbolic_regression_bin_crossfit_equations.csv"))

        for idx, row in eq_df.iterrows():
            raw_eq = row["sympy_format"]
            eq = sympy.sympify(raw_eq)

            symbols = sorted(eq.free_symbols, key=lambda s: int(str(s)[1:]))
            expr_fn = sympy.lambdify(symbols, sympy.exp(eq), modules="numpy")
            f_eval = lambda X: expr_fn(*[X[:, int(str(s)[1:])] for s in symbols])

            de_sr = DifficultyEstimator()
            de_sr.fit(X_train_sr, f=f_eval, scaler=True)

            sigmas_cal_sr = de_sr.apply(X_cal_sr)
            sigmas_test_sr = de_sr.apply(X_test_sr)
            cr_sr = ConformalRegressor()
            cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

            cp_key = f"symbolic_regression_mean_width"
            ci = cr_sr.predict_int(
                learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=confidence
            )

            eq_mean, eq_median, eq_coverage = compute_ci_stats(ci)
            eq_dict.append({
                "dataset_name": dataset_name,
                "equation": raw_eq,
                "complexity": row["complexity"],
                "chosen": row["chosen"] == "<-- chosen",
                "interval_mean": eq_mean,
                "interval_median": eq_median,
                "coverage": eq_coverage,
                "sigmas": sigmas_cal_sr,
                "ci": ci
            })

    df_equations = pd.DataFrame(eq_dict)
    display(df_equations)
    return (df_equations,)


@app.cell(hide_code=True)
def _(df_equations, plt):
    fig1, ax1 = plt.subplots(1,3, sharex=True, figsize=(20,6))
    for idx1 in df_equations.index:
        row1 = df_equations.loc[idx1]
        if row1["chosen"]:
            ax1[0].scatter(row1["complexity"], row1["interval_mean"], s=150, label="Chosen Equation")
            ax1[1].scatter(row1["complexity"], row1["interval_median"], s=150, label="Chosen Equation")
            ax1[2].scatter(row1["complexity"], row1["coverage"], s=150, label="Chosen Equation")
        
        else:
            ax1[0].scatter(row1["complexity"], row1["interval_mean"])
            ax1[1].scatter(row1["complexity"], row1["interval_median"])
            ax1[2].scatter(row1["complexity"], row1["coverage"])
    
        ax1[0].set_ylabel("Interval Mean")
        ax1[1].set_ylabel("Interval Median")
        ax1[2].set_ylabel("Coverage")

        for pos1 in range(3):
            ax1[pos1].legend()
            ax1[pos1].set_xlabel("Equation Complexity")
    plt.show()
    return


@app.cell(hide_code=True)
def _(df_equations, example_dataset, plt):
    vmin = df_equations["complexity"].min()
    vmax = df_equations["complexity"].max()
    df_ds = df_equations[df_equations["dataset_name"] == example_dataset.name]

    fig2, ax2 = plt.subplots(1,2, figsize=(20,8))

    ax2[0].scatter(
        df_ds["coverage"],
        df_ds["interval_mean"],
        c=df_ds["complexity"],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        s=50,
        alpha=0.8,
    )
    ax2[0].set_ylabel("Interval Mean")
    sc2 = ax2[1].scatter(
        df_ds["coverage"],
        df_ds["interval_median"],
        c=df_ds["complexity"],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        s=50,
        alpha=0.8,
    )
    ax2[1].set_ylabel("Interval Median")

    chosen_df = df_ds[df_ds["chosen"]]
    if not chosen_df.empty:
        ax2[0].scatter(
            chosen_df["coverage"],
            chosen_df["interval_mean"],
            c=chosen_df["complexity"],
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            s=150,
            label="Chosen Equation",
        )
        ax2[1].scatter(
            chosen_df["coverage"],
            chosen_df["interval_median"],
            c=chosen_df["complexity"],
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            s=150,
            label="Chosen Equation",
        )

    for pos2 in range(2):
        ax2[pos2].legend()
        ax2[pos2].invert_xaxis()
        ax2[pos2].set_xlabel("Coverage")
        cbar2 = fig2.colorbar(sc2, ax=ax2[pos2])
        cbar2.set_label("Complexity")
    
    plt.show()
    return vmax, vmin


@app.cell(hide_code=True)
def _(df_equations, mcolors, np, plt, vmax, vmin, y_cal, y_cal_pred):
    abs_res_cal = np.abs(y_cal_pred - y_cal)
    sorted_indices = np.argsort(abs_res_cal)
    x3 = sorted_indices[np.linspace(start=0, stop=len(abs_res_cal), endpoint=False, num=50, dtype=int)]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    fig3, ax3 = plt.subplots()
    for idx3, row3 in df_equations.iterrows():
        if row3["chosen"]:
            plt.plot(
                abs_res_cal[x3], 
                row3["sigmas"][x3], 
                alpha=1,
                c=cmap(norm(row3["complexity"])),
                linewidth=3,
                label="chosen equation"
            )
        else:
            plt.plot(
                abs_res_cal[x3], 
                row3["sigmas"][x3], 
                alpha=0.6,
                c=cmap(norm(row3["complexity"])),
            )

    sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm3.set_array([])  # Dummy array for scalar mappable
    cbar3 = plt.colorbar(sm3, ax=plt.gca())
    cbar3.set_label("Complexity")
    ax3.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(plt):
    def plot_confidence_intervals(y, y_pred, y_pred_ci) :
        """

        """
        # sort y_test values from small to big, along with y_pred_ci
        # using a list is pretty slow, there is probably a smarter way to do this
        # with numpy arrays, but the data set sizes should be small, so who cares
        y_and_ci = []
        for i in range(0, len(y)) :
            y_and_ci.append([y[i], y_pred[i], y_pred_ci[i]])
        y_and_ci = sorted(y_and_ci, key=lambda x : x[0])

        fig, ax = plt.subplots()#figsize=(10,8))

        # plot measured values and point predictions for y
        x = range(0, len(y))
        ax.scatter(x, [x[0] for x in y_and_ci], marker='o', color='green', label="Measured values")
        ax.scatter(x, [x[1] for x in y_and_ci], marker='x', color='orange', label="Predictions")

        # visualize corresponding confidence intervals around point predictions
        ax.fill_between(x, [x[2][0] for x in y_and_ci], [x[2][1] for x in y_and_ci], color='orange', alpha=0.3)

        ax.set_xlabel("Samples sorted by increasing value of target")
        ax.set_ylabel("Value of target y")
        ax.legend(loc='best')

        return fig, ax

    return (plot_confidence_intervals,)


@app.cell
def _(df_equations, plot_confidence_intervals, plt, y_test, y_test_pred):
    for idx4, row4 in df_equations.iterrows():
        fig4, ax4 = plot_confidence_intervals(y_test[:25], y_test_pred[:25], row4["ci"])
        if row4["chosen"]:
            ax4.set_title("Chosen")
    plt.show()
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
