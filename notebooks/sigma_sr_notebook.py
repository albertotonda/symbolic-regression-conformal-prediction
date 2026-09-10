import marimo

__generated_with = "0.24.0"
app = marimo.App(layout_file="layouts/sigma_sr_notebook.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $\sigma$-SR: Normalized Conformal Prediction with SR-predictor for difficulty estimation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Motivations

    ### 1.1. Why conformal prediction?

    We want to associate an estimate of uncertainty to a point prediction $\hat{y}$. **Conformal Prediction (CP)** is a popular method to predict a confidence zone for a test target. We will focus on split CP, that carry a **finite-sample, distribution-free marginal coverage guarantee**:
    $$P(Y_{n+1} \in C_n(X_{n+1})) \geq 1 - \alpha$$
    with no assumption on the base model or the data-generating distribution, as long as the data is exchangeable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2. Why symbolic regression?

    The other motivation behind the use of symbolic regression is that usual difficulty estimators are **black boxes**, hardly interpretable and each representing different aspects of the model uncertainty: epistemic (model's inherent capabilities), aleatoric (noise in training data) or data sparsity.

    **Idea:** use Symbolic Regression to search for a *closed-form,
    human-readable formula* for difficulty estimation instead searching over algebraic expressions, guided by a fitness function.
    """)
    return


@app.cell(hide_code=True)
def _():
    import os
    import re
    import sys
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import jinja2

    from collections import defaultdict

    from scipy.stats import pearsonr

    from crepes import WrapRegressor, ConformalRegressor
    from crepes.extras import DifficultyEstimator, MondrianCategorizer, binning

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from pysr import PySRRegressor

    import sympy as sp

    # make src/ (this file's parent's parent) importable, so this script can be
    # run directly (e.g. `uv run src/experiments/run_sigma_sr.py`) regardless of
    # the current working directory
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # repo root, one level above src/ -- used in section 6 to locate the
    # precomputed results of the full 22-dataset sweep (run_sigma_sr.py) and
    # the OpenML-CTR23 dataset-statistics CSV
    repo_root = os.path.dirname(src_path)

    from data import load_and_preprocess_openml_task, get_benchmark_task_ids
    from evaluate import evaluate_and_plot_method, log_equations, plot_pareto, setup_results_folder, translations

    warnings.simplefilter(action='ignore', category=FutureWarning)
    sns.set_theme(style='darkgrid')
    print('All imports OK')
    return (
        DifficultyEstimator,
        RandomForestRegressor,
        StandardScaler,
        WrapRegressor,
        binning,
        load_and_preprocess_openml_task,
        np,
        os,
        pd,
        plt,
        r2_score,
        re,
        repo_root,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(load_and_preprocess_openml_task, plt):
    task_id = 361260 # miami_housing
    random_seed = 42
    confidence = 0.95

    df_X, df_y, task = load_and_preprocess_openml_task(task_id)
    dataset = task.get_dataset()

    print(f"Dataset  : {dataset.name}")
    print(f"Samples  : {df_X.shape[0]}")
    print(f"Features : {df_X.shape[1]}")
    print(f"Target   : min={df_y.min():.2f}  max={df_y.max():.2f}  mean={df_y.mean():.2f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(df_y.values, bins=60)
    ax.set_xlabel('Target value')
    ax.set_ylabel('Count')
    ax.set_title(f"Distribution of target in '{dataset.name}'")
    plt.show()
    return confidence, df_X, df_y, random_seed


@app.cell(hide_code=True)
def _(StandardScaler, df_X, df_y, random_seed, train_test_split):
    X = df_X.values
    y = df_y.values
    feature_names = list(df_X.columns)

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

    print(f'Proper training : {X_prop_train.shape[0]} samples')
    print(f'Calibration     : {X_cal.shape[0]} samples')
    print(f'Test            : {X_test.shape[0]} samples')
    return X_cal, X_prop_train, X_test, y_cal, y_prop_train, y_test


@app.cell(hide_code=True)
def _(
    RandomForestRegressor,
    WrapRegressor,
    X_cal,
    X_prop_train,
    X_test,
    r2_score,
    random_seed,
    y_prop_train,
    y_test,
):
    print("Training base regressor...")
    base_regressor = WrapRegressor(
        RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=random_seed)
    )
    base_regressor.fit(X_prop_train, y_prop_train)

    y_cal_pred = base_regressor.predict(X_cal)
    y_test_pred = base_regressor.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(f'R² on test set: {r2:.4f}')
    return (base_regressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Other Methods

    ### 2.0. Conformal Prediction setup

    The price to pay for confience intervals in conformal prediction is the use of a calibration set. In practice, we split the (exchangeable) dataset in a training set $Z_{train}$, a calibration set $Z_{cal}$ and a test set $Z_{test}$. The training set is used to train the base regressor that can be of any kind.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1. Standard CP

    In Standard Conformal Prediction, we can compute non-conformity scores from the calibration set:
    $$\delta_i = \vert y_i - \hat{y_i}\vert$$
    Choosing the desired uncertainty $\alpha$, we then compute the empirical $(1-\alpha)$-quantile $\delta_{1−\alpha}$ using ordered $\delta_i$. The confidence interval is:
    $$\mathcal{C}_{1-\alpha}(x)=[\hat{y}-\delta_{1-\alpha}, \hat{y}+\delta_{1-\alpha}]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2. Normalized CP

    The main argument against SCP is that the confidence bounds have a fixed width for all predictions. That can lead easy predictions to have useless confidence bounds (e. g. a prediction of $0.02 \pm 1.3$). In order to have adaptive confidence bounds, Normalized Conformal Prediction methods compute difficulty scores that scale the interval according to a difficulty estimation function $\sigma(x)$:
    $$\delta_i = \frac{\vert y_i - \hat{y_i}\vert}{\sigma(x_i) + \epsilon}$$
    The non-conformity scores are now "normalized" using the difficulty estimation function to compute a difficulty-agnostic $(1-\alpha)$-quantile $\delta_{1-\alpha}$. For each new test point, we compute the interval scaled by the estimated difficulty of the test point:
    $$q(x)=\delta_{1-\alpha} \cdot (\sigma(x)+\epsilon)$$
    and obtain the interval $\mathcal{C}_{1-\alpha}(x)=[\hat{y}-q(x), \hat{y}+q(x)]$. Common NCP difficulty estimators from the litterature are:
    * $\sigma^{dist}$: distance of $k$ nearest neighbors in training set (density estimation)
    * $\sigma^{std}$: standard deviation of $k$ neighbors in training set (variance estimation)
    * $\sigma^{res}$: sum of residuals of $k$ nearest neighbors in training set, using out-of-bag estimation to avoid overfitting (regressor performance estimation)
    * $\sigma^{var}$: variance of ensemble predictors on training set
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3. Mondrian CP

    Another apporach to adaptive intervals is Mondrian Conformal Prediction. The idea is that instead of computing a single $\delta_{1-\alpha}$, we split the dataset into bins according to some criterion and follow the SCP procedure on each bin separately. The original method sorts the training set using $\sigma^{var}(X_{train})$ to compute the bins, iteratively finding the highest number of bins so that each one can achieve $(1-\alpha)$ confidence on the calibration set.
    """)
    return


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
    np,
    random_seed,
    y_cal,
    y_prop_train,
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

    learner_prop = base_regressor.learner
    sigmas = {}
    conf_intervals = {}

    y_pred_oob = learner_prop.oob_prediction_
    residuals_prop_oob = y_prop_train - y_pred_oob

    # Standard CP
    print("Computing CI for SCP...")
    base_regressor.calibrate(X_cal, y_cal)
    sigmas["conformal_predictor"] = np.ones(len(X_cal))
    conf_intervals["conformal_predictor"] = base_regressor.predict_int(X_test, confidence=confidence)

    # KNN distance
    print("Computing CI for knn_dist NCP...")
    de_knn_dist = DifficultyEstimator()
    de_knn_dist.fit(X=X_prop_train, scaler=True)
    conf_intervals["normalized_cp_knn_dist"], sigmas["normalized_cp_knn_dist"] = compute_normalized_intervals(de_knn_dist, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN std
    print("Computing CI for knn_std NCP...")
    de_knn_std = DifficultyEstimator()
    de_knn_std.fit(X=X_prop_train, y=y_prop_train, scaler=True)
    conf_intervals["normalized_cp_knn_std"], sigmas["normalized_cp_knn_std"] = compute_normalized_intervals(de_knn_std, learner_prop, X_cal, y_cal, X_test, confidence)

    # KNN out-of-bag residuals
    print("Computing CI for knn_res NCP...")
    de_knn_res = DifficultyEstimator()
    de_knn_res.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True)
    conf_intervals["normalized_cp_knn_res"], sigmas["normalized_cp_knn_res"] = compute_normalized_intervals(de_knn_res, learner_prop, X_cal, y_cal, X_test, confidence)

    # Random Forest variance
    print("Computing CI for var NCP...")
    de_var = DifficultyEstimator()
    de_var.fit(X=X_prop_train, learner=learner_prop, scaler=True)
    conf_intervals["normalized_cp_norm_var"], sigmas["normalized_cp_norm_var"] = compute_normalized_intervals(de_var, learner_prop, X_cal, y_cal, X_test, confidence)

    # Mondrian CP using variance
    print("Computing CI for MCP...")
    min_points = int(1 / (1-confidence) - 1) + 1
    bin_thresholds = _find_bin_thresholds_with_min_size(sigmas["normalized_cp_norm_var"], min_points, random_seed)
    number_of_bins = len(bin_thresholds) - 1
    print(f"Number of Mondrian bins: {number_of_bins}")

    # the "mc" argument for calibrate() internally takes X as only parameter,
    # so recompute sigmas_var = de_var.apply(X) instead of using pre-computed ones
    def mondrian_categories(X):
        return binning(de_var.apply(X), bins=bin_thresholds, seed=random_seed)

    regressor_mond = WrapRegressor(learner_prop)
    regressor_mond.calibrate(X_cal, y_cal, mc=mondrian_categories)
    sigmas["mondrian_cp"] = np.ones(len(X_cal))
    conf_intervals["mondrian_cp"] = regressor_mond.predict_int(X_test, confidence=confidence)
    return de_var, learner_prop, residuals_prop_oob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proposed Approach

    ### 3.1. Symbolic Regression as an estimator for $\sigma(x)$

    The idea is that we use Symbolic Regression as a difficulty estimator to use
    in Normalized Conformal Prediction (NCP):
    $$\delta_i = \frac{\vert y_i - \hat{y_i}\vert}{\sigma_{SR}(x_i) + \epsilon}$$

    Since the symbolic regressor can accept any number of features, we can augment the feature space of the symbolic regressor using $\sigma$ from the other NCP methods. This method allows for a litteral expression of difficulty that can help target difficult features, and that can be trained using any objective function (differentiability isn't required) using any additional features like we did with other $\sigma$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2. How to avoid data leakage?

    The cost for this method is that we need extra data to train the SR and compute all the $\sigma$. We can either split further the training set to create a sub-calibration set, or keep the same training set and make use of the random forest regressor's out-of-bag predictions. We chose the second option to be able to use small-scale datasets.

    We are using the 4 NCP $\sigma$ described in section `2.2` as augmented features to the SR training: $\sigma^{dist}$, $\sigma^{std}$, $\sigma^{res}$ and $\sigma^{var}$. By excluding point $i$ from the $k$-nearest neighbor computations in $\sigma(x_i)$ and  using out-of-bag predictions as targets, we ensure that no data can leak into the symbolic regressor's training.
    """)
    return


@app.cell(hide_code=True)
def _(
    DifficultyEstimator,
    X_cal,
    X_prop_train,
    X_test,
    de_var,
    learner_prop,
    np,
    residuals_prop_oob,
    y_prop_train,
):
    print("Computing NCP sigmas for data augmentation...")
    sigmas_train = {}
    sigmas_cal = {}
    sigmas_test = {}

    # KNN distance
    de_knn_dist_oob = DifficultyEstimator()
    de_knn_dist_oob.fit(X=X_prop_train, scaler=True, oob=True)
    sigmas_train["knn_dist"] = de_knn_dist_oob.apply()
    sigmas_cal["knn_dist"] = de_knn_dist_oob.apply(X_cal)
    sigmas_test["knn_dist"] = de_knn_dist_oob.apply(X_test)

    # KNN std
    de_knn_std_oob = DifficultyEstimator()
    de_knn_std_oob.fit(X=X_prop_train, y=y_prop_train, scaler=True, oob=True)
    sigmas_train["knn_std"] = de_knn_std_oob.apply()
    sigmas_cal["knn_std"] = de_knn_std_oob.apply(X_cal)
    sigmas_test["knn_std"] = de_knn_std_oob.apply(X_test)

    # KNN out-of-bag residuals
    de_knn_res_oob = DifficultyEstimator()
    de_knn_res_oob.fit(X=X_prop_train, residuals=residuals_prop_oob, scaler=True, oob=True)
    sigmas_train["knn_res"] = de_knn_res_oob.apply()
    sigmas_cal["knn_res"] = de_knn_res_oob.apply(X_cal)
    sigmas_test["knn_res"] = de_knn_res_oob.apply(X_test)

    # Random Forest variance
    de_var_oob = DifficultyEstimator()
    de_var_oob.fit(X=X_prop_train, learner=learner_prop, scaler=True, oob=True)
    sigmas_train["var"] = de_var_oob.apply()
    sigmas_cal["var"] = de_var.apply(X_cal) # For cal and test, use default (no oob) version, otherwise same oob trees are used instead of full model
    sigmas_test["var"] = de_var.apply(X_test)

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

    y_log_abs_residual = np.log(np.abs(residuals_prop_oob))
    y_raw_residual = residuals_prop_oob
    return y_log_abs_residual, y_raw_residual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3. Defining the loss function

    We need to find the objective to guide the evolution in the symbolic regression. In order to maximize conditional coverage, we want to find the function $\sigma$ that approximates as close as possible the true difficulty that the base regressor has to
    predict the target corresponding to a datapoint $x$. This difficulty can
    originate from different sources: data sparsity, noisy training data
    (aleatoric) or model weakness (epistemic).

    We can note that for any given $x$, we only observe one residual, very sensitive to noise. That is the main motivation behind the feature augmentation with $\sigma$: they are a smoothed, more robust estimate of the difficulty of a point.

    * Element-wise MAE loss on absolute residuals:
    $$\mathcal{L} = \vert \log\vert\hat{r}\vert - \log\vert r\vert \vert$$
    As a starting point, we estimate the difficulty of the model using only the absolute residuals of the function. Simple loss in practice, no assumption about underlying data, but very sensitive to noise (mitigated by using log).
    * Mean interval width:
    $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N q(x_i)$$
    We directly try to minimize the mean interval width $q(x)$, since coverage is guaranteed by design. Requires running the calibration and interval computation at each step of the evolution.
    * Pairwise ranking loss:
    $$
    \Delta r_k = \log\big(|r_j| + \epsilon\big) - \log\big(|r_i| + \epsilon\big), \qquad
    \Delta\sigma_k = \log\sigma_j - \log\sigma_i
    $$

    $$
    s_k = \operatorname{sign}(\Delta r_k), \qquad w_k = |\Delta r_k|
    $$

    $$
    h_k = \max\big(0, m - s_k \Delta\sigma_k\big)
    $$

    $$
    \mathcal{L} = \frac{\displaystyle\sum_{k=1}^{\lfloor N/2\rfloor} w_k \cdot h_k}{\displaystyle\sum_{k=1}^{\lfloor N/2\rfloor} w_k}
    $$
    The idea is that we want to keep the relative ordering of the absolute residuals, since any scaling factor in $\sigma$ is canceled out in the interval computation. For that, we penalize when the order is wrong or not within a fixed margin $m$. Without the margin, a constant $\sigma$ would be stuck in a local optimum. We also add a weight by how much it was wrong ($\Delta\sigma$) and how far apart the residuals originally are ($\Delta r$). Again, we are working with log residuals to mitigate the effect of outliers.
    """)
    return


@app.cell(hide_code=True)
def _(y_log_abs_residual, y_raw_residual):
    mean_width_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual (not its log)
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        sigma = exp.(log_sigma)
        if any(sigma .<= zero(T)) || !all(isfinite.(sigma))
            return L(Inf)
        end

        # simulate the real conformal calibration step (empirical 95% quantile
        # of normalized residuals) on this batch, then score the resulting
        # mean interval width -- literally the downstream deliverable, not a
        # proxy for it. Manual sort-based quantile since Statistics.quantile
        # may not be in scope inside PySR's custom-loss eval context.
        scores = abs.(dataset.y) ./ sigma
        n = dataset.n
        sorted_scores = sort(scores)
        idx = clamp(ceil(Int, 0.95 * n), 1, n)
        q_hat = L(sorted_scores[idx])

        widths = q_hat .* sigma
        return L(sum(widths) / n)
    end
    """

    pairwise_ranking_loss_julia = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual (not its log)
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        if !all(isfinite.(log_sigma))
            return L(Inf)
        end

        # only the RELATIVE ordering of sigma across points matters for
        # downstream conformal efficiency (a constant rescale cancels out at
        # calibration), so reward correctly ranking pairs of points by true
        # residual size instead of matching a noisy pointwise target.
        # Consecutive-row pairing keeps this O(n) and deterministic across
        # every fitness evaluation (rows are already shuffled by the
        # upstream train/cal/test split, so this is as good as random
        # pairing without the run-to-run noise random sampling would add).
        # Pairs are weighted by |delta_resid| so near-ties (unreliable,
        # mostly-noise comparisons) contribute little.
        n = dataset.n
        npairs = div(n, 2)
        eps = L(1e-6)
        # NOTE: margin must be > 0. At margin=0, a constant tree gives
        # delta_sigma=0 for every pair, so hinge=0 for every pair and the
        # loss is EXACTLY 0 -- the global minimum, trivially and immediately
        # achieved by any constant. That's a degenerate optimum, not a
        # search-budget problem: no formula can ever score better than the
        # constant's 0, so there is zero selection pressure to leave it.
        # margin=0.1 makes a constant score exactly 0.1 (bad but beatable),
        # only reachable by 0 through genuine, sufficiently-separated ranking.
        margin = L(0.1)

        total_loss = zero(L)
        total_weight = zero(L)
        for k in 1:npairs
            i = 2k - 1
            j = 2k
            delta_resid = log(abs(L(dataset.y[j])) + eps) - log(abs(L(dataset.y[i])) + eps)
            delta_sigma = L(log_sigma[j]) - L(log_sigma[i])

            s = sign(delta_resid)
            weight = abs(delta_resid)

            hinge = max(zero(L), margin - s * delta_sigma)
            total_loss += weight * hinge
            total_weight += weight
        end

        return total_weight == zero(L) ? zero(L) : total_loss / total_weight
    end
    """

    sigma_losses = [
        ("mae", dict(elementwise_loss="L1DistLoss()"), y_log_abs_residual),
        ("mean_width", dict(loss_function=mean_width_loss_julia), y_raw_residual),
        ("pairwise_rank", dict(loss_function=pairwise_ranking_loss_julia), y_raw_residual),
    ]
    return


@app.cell(hide_code=True)
def _():
    # for loss_name, loss_kwargs, y_train_sr in sigma_losses:
    #     sigma_predictor = PySRRegressor(
    #         model_selection="score",
    #         tournament_selection_n=15, # default 15
    #         populations=31, # default 31
    #         population_size=30, # must be >= topn:=12 (default 27)
    #         niterations=50, # default 100
    #         binary_operators=["+", "-", "*", "/"],
    #         unary_operators=["sin", "cos", "log", "exp"],
    #         # nested_constraints={
    #         #     "sin": {"cos": 0, "sin": 0},
    #         #     "cos": {"cos": 0, "sin": 0},
    #         #     "log": {"log": 0},
    #         #     "exp": {"exp": 0}},
    #         temp_equation_file=True, # does not clutter directory with temporary files
    #         verbosity=1, # can also be set to 0, it should be ok
    #         random_state=random_seed,
    #         **loss_kwargs,
    #     )
    #     sigma_predictor.fit(X_train_sr, y_train_sr)
    #     log_equations(sigma_predictor, task_folder, f"symbolic_regression_{loss_name}")

    #     de_sr = DifficultyEstimator()
    #     de_sr.fit(X_train_sr, f=lambda X: np.exp(sigma_predictor.predict(X)), scaler=True)

    #     # WrapRegressor.calibrate()/.predict_int() feed the SAME X to both the
    #     # wrapped learner (needs the raw data) and de.apply() (needs the
    #     # sigma-augmented columns) — incompatible here, so calibrate manually via
    #     # the lower-level ConformalRegressor instead.
    #     sigmas_cal_sr = de_sr.apply(X_cal_sr)
    #     sigmas_test_sr = de_sr.apply(X_test_sr)

    #     cr_sr = ConformalRegressor()
    #     cr_sr.fit(y_cal - learner_prop.predict(X_cal), sigmas=sigmas_cal_sr)

    #     cp_key = f"symbolic_regression_{loss_name}"
    #     conf_intervals[cp_key] = cr_sr.predict_int(
    #         learner_prop.predict(X_test), sigmas=sigmas_test_sr, confidence=confidence
    #     )
    #     sigmas[cp_key] = sigmas_cal_sr

    # # per-method CI plot + coverage/amplitude stats, same helper
    # # run_interval_sr.py uses (stats get appended into results_dictionary)
    # for method, intervals in conf_intervals.items():
    #     evaluate_and_plot_method(method, intervals, y_test, y_test_pred,
    #                               dataset, task_folder, results_dictionary)

    # results_dictionary["task_id"].append(task_id)
    # results_dictionary["dataset_name"].append(dataset.name)
    # results_dictionary["r2"].append(r2)
    # last_task_methods = list(conf_intervals.keys())

    # # per-task Pareto plot across all methods computed for this task
    # fig2, ax2 = plot_pareto(last_task_methods, results_dictionary, translations=translations)
    # ax2.set_title(f"Performance of conformal prediction methods on dataset {dataset.name}")
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Results from the complete run (22 OpenML-CTR23 datasets, losses comparison)
    """)
    return


@app.cell(hide_code=True)
def _(os, pd, repo_root):
    results_run_folder = os.path.join(repo_root, "results-sigma-sr-42_20260825-101216")
    losses = ["mae", "mean_width", "pairwise_rank"]
    sr_keys = [f"symbolic_regression_{loss}" for loss in losses]
    comparison_keys = [
        "conformal_predictor",
        "normalized_cp_knn_dist",
        "normalized_cp_knn_std",
        "normalized_cp_knn_res",
        "normalized_cp_norm_var",
        "mondrian_cp"
    ]

    df_run_results = pd.read_csv(os.path.join(results_run_folder, "results.csv"))
    df_pareto_stats = pd.read_csv(os.path.join(results_run_folder, "results-statistics.csv"), index_col=0)

    df_openml_stats = pd.read_csv(
        os.path.join(repo_root, "results", "OpenML-CTR23-statistics-500-estimators-10-fold-cv.csv")
    )
    df_openml_stats["n_samples"] = df_openml_stats["n_samples"].astype(str).str.replace(",", "").astype(int)

    df_run_results = df_run_results.merge(
        df_openml_stats[["task_id", "n_samples", "n_features", "missing_data", "categorical_features"]],
        on="task_id",
    )
    df_run_results.head()
    return comparison_keys, df_run_results, losses, results_run_folder, sr_keys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1. Which sigma estimator does SR actually rely on?

    Of the 4 candidate difficulty signals available to the SR search
    ($\sigma^{dist}$, $\sigma^{std}$, $\sigma^{res}$, $\sigma^{var}$), how often
    does the chosen equation use each one, and how (alone, in combination)?
    """)
    return


@app.cell(hide_code=True)
def _(df_run_results, losses, mo, np, os, pd, plt, re, results_run_folder):
    # sigma columns are appended, in this order, right after the dataset's
    # own base features (see "Computing NCP sigmas for data augmentation" cell)
    AUGMENTED_FEATURE_NAMES = ["knn_dist", "knn_std", "knn_res", "var"]

    def _feature_indices(equation_str):
        return sorted({int(m) for m in re.findall(r"x(\d+)", str(equation_str))})

    _rows = []
    for _, _run_row in df_run_results.iterrows():
        _dataset_name = _run_row["dataset_name"]
        _n_base_features = int(_run_row["n_features"])
        for _loss_name in losses:
            _eq_path = os.path.join(
                results_run_folder, _dataset_name, f"symbolic_regression_{_loss_name}_equations.csv"
            )
            if not os.path.exists(_eq_path):
                continue
            _df_eq = pd.read_csv(_eq_path)
            _chosen = _df_eq[_df_eq["chosen"] == "<-- chosen"]
            if _chosen.empty:
                continue
            _chosen_row = _chosen.iloc[0]
            _indices = _feature_indices(_chosen_row["sympy_format"])
            _base_indices = [i for i in _indices if i < _n_base_features]
            _augmented_names = tuple(
                AUGMENTED_FEATURE_NAMES[i - _n_base_features]
                for i in _indices
                if i >= _n_base_features
            )
            _rows.append(
                {
                    "dataset_name": _dataset_name,
                    "loss": _loss_name,
                    "complexity": _chosen_row["complexity"],
                    "n_base_features_used": len(_base_indices),
                    "n_augmented_features_used": len(_augmented_names),
                    "augmented_features_used": _augmented_names,
                }
            )

    df_equation_composition = pd.DataFrame(_rows)

    # NOTE: Moneyball (task 361616) has columns dropped for missing data in
    # load_and_preprocess_openml_task, so its real base-feature count differs
    # from the OpenML-CTR23-statistics n_features used here -- its
    # base/augmented split is unreliable and should be read with that caveat.
    _summary_rows = []
    for _loss_name in losses:
        _df_loss = df_equation_composition[df_equation_composition["loss"] == _loss_name]
        for _feat_name in AUGMENTED_FEATURE_NAMES:
            _used_total = int(
                _df_loss["augmented_features_used"].apply(lambda feats: _feat_name in feats).sum()
            )
            _used_alone = int(
                _df_loss.apply(
                    lambda r: r["n_base_features_used"] == 0
                    and r["augmented_features_used"] == (_feat_name,),
                    axis=1,
                ).sum()
            )
            _summary_rows.append(
                {
                    "loss": _loss_name,
                    "augmented_feature": _feat_name,
                    "used_alone": _used_alone,
                    "used_total": _used_total,
                }
            )
    df_augmented_feature_summary = pd.DataFrame(_summary_rows)

    _n_pure_sigma = int((df_equation_composition["n_base_features_used"] == 0).sum())
    _n_total = len(df_equation_composition)

    # visualize: for each loss, how often is each of the 4 sigma estimators
    # used anywhere in the chosen equation vs. used as the sole feature?
    _fig, _axes = plt.subplots(1, len(losses), figsize=(5 * len(losses), 4), sharey=True)
    _x = np.arange(len(AUGMENTED_FEATURE_NAMES))
    for _ax, _loss_name in zip(_axes, losses):
        _df_loss_summary = df_augmented_feature_summary[
            df_augmented_feature_summary["loss"] == _loss_name
        ]
        _ax.bar(_x - 0.2, _df_loss_summary["used_total"], width=0.4, label="used anywhere")
        _ax.bar(_x + 0.2, _df_loss_summary["used_alone"], width=0.4, label="used alone")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(AUGMENTED_FEATURE_NAMES, rotation=30)
        _ax.set_title(_loss_name)
    _axes[0].set_ylabel("number of datasets (out of 22)")
    _axes[0].legend()
    plt.suptitle("Which sigma estimator does the chosen equation rely on?")
    plt.tight_layout()

    mo.vstack([
        mo.mpl.interactive(_fig),
        mo.md(f"{_n_pure_sigma}/{_n_total} chosen equations use zero base features (pure sigma-based)."),
        df_equation_composition["n_base_features_used"]
        .value_counts()
        .sort_index()
        .rename("n_equations_by_base_feature_count")
        ])
    return (df_equation_composition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2. Is any single sigma-SR loss consistently the best?

    Restricting the Pareto comparison to only the 3 sigma-SR methods (median
    width vs. coverage), how often is each one beaten by the other two?
    """)
    return


@app.cell(hide_code=True)
def _(df_run_results, mo, pd, plt, sr_keys):
    # A sigma-SR method's (median width, coverage) point for a dataset is
    # "dominated" if another sigma-SR method has BOTH narrower (or equal)
    # intervals AND equal-or-better coverage on that same dataset -- i.e. it
    # is strictly beaten. Every method DOES produce a result on every
    # dataset; "dominated"/"not dominated" is about whether that result is
    # beaten by a competitor, not about whether it exists.
    # Among the methods NOT dominated on a given dataset:
    #  - "sole best": only one method escapes domination -> a clear winner.
    #  - "tied best": more than one method escapes domination (e.g. one has
    #    narrower intervals, another has better coverage -- neither beats
    #    the other on both axes at once) -> no clear winner.
    def _is_dominated(point, other_point):
        return point[0] >= other_point[0] and point[1] <= other_point[1]

    _rows = []
    for _, _row in df_run_results.iterrows():
        _points = {m: (_row[f"{m}_median"], _row[f"{m}_coverage"]) for m in sr_keys}
        _non_dominated = [
            m for m in sr_keys
            if not any(_is_dominated(_points[m], _points[o]) for o in sr_keys if o != m)
        ]
        for m in sr_keys:
            if m not in _non_dominated:
                _category = "dominated"
            elif len(_non_dominated) == 1:
                _category = "sole best"
            else:
                _category = "tied best"
            _rows.append({"dataset_name": _row["dataset_name"], "method": m, "category": _category})

    df_sr_pareto_categories = pd.DataFrame(_rows)
    df_sr_pareto_summary = (
        df_sr_pareto_categories.groupby(["method", "category"]).size().unstack(fill_value=0)
        .reindex(columns=["dominated", "tied best", "sole best"], fill_value=0)
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _bottom = None
    for _category, _color in zip(
        ["dominated", "tied best", "sole best"], ["tab:red", "tab:orange", "tab:green"]
    ):
        _values = df_sr_pareto_summary[_category]
        _ax.bar(df_sr_pareto_summary.index, _values, bottom=_bottom, label=_category, color=_color)
        _bottom = _values if _bottom is None else _bottom + _values
    _ax.set_ylabel("number of datasets (out of 22)")
    _ax.set_title("Pareto standing among the 3 sigma-SR methods")
    _ax.legend()
    plt.xticks(rotation=10)
    plt.tight_layout()

    mo.vstack([
        mo.md("Out of the 22 datasets, how often is each sigma-SR method beaten by the other two?"),
        df_sr_pareto_summary,
        mo.mpl.interactive(_fig)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3. Does base-model accuracy explain interval width?

    Interval width should partly reflect how uncertain the base regressor
    itself is; does a higher R² on a dataset predict narrower intervals?
    """)
    return


@app.cell(hide_code=True)
def _(df_run_results, mo, np, pd, plt, sr_keys):
    from scipy.stats import pearsonr as _pearsonr
    from scipy.stats import zscore 

    # does a better base regressor (higher R^2) lead to narrower intervals and/or better coverage?
    dict_corr = {}
    _corr_rows = []
    for _m in sr_keys:
        dict_corr[_m] = df_run_results[["r2", f"{_m}_coverage", f"{_m}_median"]].copy()
        dict_corr[_m]["zscore"] = zscore(dict_corr[_m][f"{_m}_median"])
        dict_corr[_m] = dict_corr[_m][abs(dict_corr[_m]["zscore"]) < 3.0]
        _r_width, _p_width = _pearsonr(dict_corr[_m]["r2"], dict_corr[_m][f"{_m}_median"])
        _r_cov, _p_cov = _pearsonr(dict_corr[_m]["r2"], dict_corr[_m][f"{_m}_coverage"])
        _corr_rows.append(
            {
                "method": _m,
                "corr(R2, median width)": round(_r_width, 2),
                "p-value": round(_p_width, 3),
                "corr(R2, coverage)": round(_r_cov, 2),
                "p-value ": round(_p_cov, 3),
            }
        )
    df_r2_correlations = pd.DataFrame(_corr_rows)

    _fig, _axes = plt.subplots(1, len(sr_keys), figsize=(5 * len(sr_keys), 4), sharey=True)
    for _ax, _m in zip(_axes, sr_keys):
        _x = dict_corr[_m]["r2"]
        _y = dict_corr[_m][f"{_m}_median"]
        _ax.scatter(_x, _y, alpha=0.7)
        _slope, _intercept = np.polyfit(_x, _y, 1)
        _xs = np.linspace(_x.min(), _x.max(), 50)
        _ax.plot(_xs, _slope * _xs + _intercept, color="black", linestyle="--")
        _ax.set_xlabel("base regressor R²")
        _ax.set_title(_m.replace("symbolic_regression_", ""))
    _axes[0].set_ylabel("median interval width")
    plt.suptitle("Interval width vs. base regressor accuracy, per sigma-SR loss")
    plt.tight_layout()

    mo.vstack([
        df_r2_correlations,
        mo.mpl.interactive(_fig)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4. Does equation complexity track dataset characteristics?

    Do bigger, higher-dimensional, or harder-to-fit datasets push the SR
    search toward more complex equations?

    $\rightarrow$ Answer: **NO**
    """)
    return


@app.cell(hide_code=True)
def _(df_equation_composition, df_run_results, losses, mo, pd, plt):
    from scipy.stats import pearsonr as _pearsonr

    _df_complexity = df_equation_composition.merge(
        df_run_results[["dataset_name", "n_samples", "n_features", "r2"]], on="dataset_name"
    )
    _stats = ["n_samples", "n_features", "r2"]
    _complexity_corr_rows = []
    for _loss_name in losses:
        _df_loss = _df_complexity[_df_complexity["loss"] == _loss_name]
        for _stat in _stats:
            _r, _p = _pearsonr(_df_loss["complexity"], _df_loss[_stat])
            _complexity_corr_rows.append(
                {"loss": _loss_name, "vs": _stat, "corr": round(_r, 2), "p-value": round(_p, 3)}
            )
    df_complexity_correlations = pd.DataFrame(_complexity_corr_rows)

    print("Is chosen-equation complexity related to dataset size, dimensionality, or base R^2?")
    print(df_complexity_correlations.to_string(index=False))

    _fig, _axes = plt.subplots(len(losses), len(_stats), figsize=(4 * len(_stats), 3.5 * len(losses)))
    for _i, _loss_name in enumerate(losses):
        _df_loss = _df_complexity[_df_complexity["loss"] == _loss_name]
        for _j, _stat in enumerate(_stats):
            _ax = _axes[_i][_j]
            _ax.scatter(_df_loss[_stat], _df_loss["complexity"], alpha=0.7)
            if _i == len(losses) - 1:
                _ax.set_xlabel(_stat)
            if _j == 0:
                _ax.set_ylabel(f"{_loss_name}\ncomplexity")
    plt.suptitle("Chosen-equation complexity vs. dataset characteristics")
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5. Is realized coverage close to the 0.95 target?

    Split conformal prediction guarantees marginal coverage by construction;
    this checks that the guarantee actually holds in practice, and how much
    it varies dataset to dataset.
    """)
    return


@app.cell(hide_code=True)
def _(comparison_keys, confidence, df_run_results, mo, pd, plt, sr_keys):
    # simple question: on average, how close is each method's realized test
    # coverage to the 0.95 target, and how much does that vary dataset to dataset?
    _methods = comparison_keys + sr_keys
    _rows = []
    for _m in _methods:
        _coverage = df_run_results[f"{_m}_coverage"]
        _rows.append(
            {"method": _m, "mean_coverage": _coverage.mean(), "std_coverage": _coverage.std()}
        )
    df_coverage_summary = pd.DataFrame(_rows).sort_values("mean_coverage")

    print(f"Target coverage = {confidence}. Mean/std of realized coverage across the 22 datasets:")
    print(df_coverage_summary.round(4).to_string(index=False))

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _order = df_coverage_summary["method"].tolist()
    for _y, _m in enumerate(_order):
        _coverage = df_run_results[f"{_m}_coverage"]
        _ax.scatter(_coverage, [_y] * len(_coverage), alpha=0.35, color="tab:blue", s=25)
        _ax.scatter(_coverage.mean(), _y, color="black", marker="D", s=70, zorder=3)
    _ax.axvline(confidence, color="red", linestyle="--", label=f"target ({confidence})")
    _ax.set_yticks(range(len(_order)))
    _ax.set_yticklabels(_order)
    _ax.set_xlabel("coverage on test set")
    _ax.set_title("Realized coverage per method (diamond = mean)")
    _ax.legend()
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Conclusions

    **Equation composition**

    - 58/66 (88%) chosen equations reduce to a single pre-existing $\sigma$
      estimator ($\sigma^{var}$ or $\sigma^{res}$ dominate; $\sigma^{dist}$ is
      essentially never chosen) - SR mostly *selects and recalibrates* an
      existing difficulty signal rather than discovering new structure.
    - No significant link between equation complexity and dataset size,
      dimensionality, or base R².

    **Marginal coverage is not a differentiator**

    - All 9 methods land within ~0.5-1.6% of the 0.95 target, by construction
      of split conformal prediction; the real differences are in interval
      width and *conditional* reliability, not average coverage.

    **Loss comparison**

    Matching theory, all losses median width performance are correlated with base regressor R²: less reliable regressor leads to more uncertainty.
    - **mean_width**: least Pareto-dominated among the 3 SR losses.
    - **mae**: performance varies a lot; probably because we are fitting a noisy
      single-residual proxy.
    - **pairwise_rank**: never the sole Pareto-best method, likely a structural weakness (hinge/margin loss has zero gradient once ranking is roughly correct).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Discussion

    - Split CP doesn't necessarily guarantee the conditional coverage $P(Y_{n+1} \in C_n(X_{n+1}) \vert X_{n+1} = x) \geq 1 - \alpha$. In other words, the marginal coverage is averaged over X so can potentially have over-coverage in some regions and under-coverage in others. That's one of the motivation behind other methods like *Boosted Conformal Prediction Intervals* or Mondrian bins.
    - By increasing the margin in the **pairwise ranking loss**, we can expect the model to keep evolving the shape of the difficulty estimator to match the absolute residuals
    - The main problem of the average width loss is in-sample optimism in the width estimation: since each expression is evaluated by calibrating on the same training set as the SR, overfitting to noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Next steps

    Ideas to try:
    - Adding a Mondrian CP/contrast tree-style split inside SR training loop, then adding a penalizing term to the loss for bin coverage; the idea is to improve conditional coverage by guaranteeing coverage in sub-splits. Keep a low number of bins to avoid overfitting
    - Use 2-fold cross-fitting in the evolutionary loop, fitting one split on half A and calibrating on half B and vice versa; the goal is to fix overfitting in the loss

    For each evolutionary iteration:
    1. Eval $\sigma(X)$, compute $\delta(X)=\frac{\vert \hat{r}_{oob}\vert }{\sigma(X)}$
    2. Compute bins using scores from $\sigma(X)$ (ex: 4 bins)
    3. Random split training set into 2 folds. For each one:
        - Compute the $(1-\alpha)$-quantile $\delta_{1-\alpha}$
        - Eval on the other fold:

       $\ell_w = \frac{1}{M} \sum_{i=1}^M \delta_{1-\alpha}\sigma(x_i) \quad \ell_{cov} = \frac{1}{N_B} \sum_{i=1}^{N_B} \mathbb{1}(\delta_i \leq \delta_{1-\alpha})$

    4. Aggregate losses:

        $\mathcal{L} = \frac{1}{2} (\ell_{w,A} + \ell_{w,B}) + \lambda \sum_{k=1}^B (\frac{1}{2} (\ell_{cov, A} + \ell_{cov, B}) - (1 - \alpha))^2$
    """)
    return


if __name__ == "__main__":
    app.run()
