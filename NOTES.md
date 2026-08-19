# Notes

## Ideas to be developed
1. Use the boundaries of the Mondrian predictors as initial seeds for the values in the GP.
2. If extracting the sigmas from the Mondrian predictors is impossible, extract directly the lambdas.
3. After obtaining the final Pareto frontier, remove all solutions that do not respect the hard constraint of $(1-\alpha)$ coverage, and from those left, pick the one with the best value of the fitness function.
4. Idea to fix the theoretical miscoverage of the direct confidence bounds regression using symbolic regression: instead of directly predicting confidence bounds, predict difficulty estimation and calibrate on calibration set. Problem: difficulty estimator cannot be trained on the same training set as base regressor. Two ideas:
    * For large datasets, split into two sets (80/20 for example), train base regressor on 80% and difficulty estimator on 20%. Optionally, retrain base regressor after using 100% of data. Note: sigmas used in SR training and comparison are not the same (first are sigmas on 80% of data, second on 100%).
    * For Random Forests: use "free" data from OOB predictions to train difficulty estimator. Use all sigmas from OOB predictions for SR training, and full base predictor for comparison.
    * (explored further on 2026-08-14 and 2026-08-18 — see chronological notes below for rejected alternatives, dataset-size caveats, fitness-function design, and the exact OOB/leave-one-out mechanics)

## Chronological notes

### 2026-08-18
Worked out the exact leakage mechanics of crepes' `DifficultyEstimator`/
`WrapRegressor` internals needed to get the train/cal/test sigma pipeline
right for the SR-difficulty-estimator design (idea #4, OOB sub-bullet).

- KNN-based sigmas (`knn_dist`, `knn_std`, `knn_oob_res`): `de.apply()` with
  no `X` returns a cached, genuinely leave-one-out score (sklearn's
  self-excluding `kneighbors()` query, computed once at `fit(oob=True)`
  time); `de.apply(X)` with explicit `X` never checks `self.oob` — it's
  always an honest lookup against the fit set. So **one** estimator, fit
  once with `oob=True` on `X_prop_train`, safely covers both: `.apply()` for
  train-set (SR training-feature) sigmas, and `.apply(X_cal)` /
  `.apply(X_test)` for calibration/test sigmas — no self-match risk on the
  latter two since those rows were never in the KNN pool.
- Ensemble-variance sigma (`ensemble_var`) does *not* share that property:
  `apply(X)` branches on the instance's `self.oob` flag regardless of what
  `X` is, so an `oob=True`-fitted variance estimator replays
  bootstrap-membership masks by row *position* — meaningless, silently
  wrong (no error) when applied to `X_cal`/`X_test`. Needs a second
  estimator instance (`oob=False`) for calibration/test sigmas. Bonus:
  crepes always fits the MinMax scaler bounds via the OOB-masked formula
  whenever `scaler=True`, regardless of the `oob` flag, so the two
  instances end up with matching scaler ranges without extra work.
- Why variance needs OOB at all, mechanistically: unpruned RF trees
  memorize their in-bag rows, so mixing in-bag (falsely confident) and OOB
  tree predictions when measuring cross-tree variance on a *training* row
  understates true difficulty — the variance analog of KNN's
  self-distance-0 problem. Doesn't apply to `X_cal`/`X_test`: no tree ever
  saw them, so all trees are already "OOB" for those rows and plain
  full-ensemble variance is correct as-is.
- Confirmed no leakage from using `y_prop_train`-derived sigma features
  (`knn_std_oob`, `knn_oob_res`) to train an SR model whose target is also
  `y_prop_train`-derived (`|y_prop_train - y_pred_oob|`): leave-one-out/OOB
  guarantees a row's own label never enters its own feature, only its
  neighbors'/other rows'. The resulting feature-target correlation (local
  label dispersion ↔ own residual size) is the intended signal, not
  circularity.
- Net calibration-time procedure mirrors the existing
  `compute_normalized_intervals` pattern exactly: `X_prop_train` needs the
  OOB-safe treatment only because (unlike the existing pipeline) its
  sigmas double as SR training features; `X_cal`/`X_test` need no special
  treatment at all — plain `.apply()`, same as the existing
  `de.apply(X_cal)`/`de.apply(X_test)` calls, since they were never part of
  fitting anything (KNN index, RF ensemble, or the SR model itself).
- Practical upshot: the trained SR is structurally a crepes
  "function"-type difficulty estimator (`f=sr_model.predict` over
  `[X, sigma_features]`) and can be dropped into
  `regressor_norm.calibrate(X_cal, y_cal, de=...)` / `predict_int` the same
  way the existing `de` objects are, once its own sigma-features are
  computed for `X_cal`/`X_test` via the plain (non-OOB) objects.

### 2026-08-14
Revisited idea #4 (SR-predicted difficulty estimator) in detail.

- Split problem: the two options above still hold. Also considered and rejected
  two more: splitting the calibration set in two (shrinks/hurts the reliability
  of the conformal quantile itself, which is the more precious resource to
  protect) and full k-fold CV on the training set (retraining cost multiplies
  by k+1, too expensive given PySR's search cost). OOB (RandomForest base
  regressor) needs no split at all and stays preferred; the 80/20 hold-out is
  the general fallback for other base regressors, optionally retraining the
  base regressor on 100% afterward.
- Checked the split-fraction idea against our own OpenML-CTR23 stats: a 20%
  hold-out is risky for small/high-dimensional datasets — energy_efficiency
  (768 rows), forest_fires (517), cars (804), QSAR_fish_toxicity (908),
  socmob (1156), solar_flare (1066 rows / 10 features),
  student_performance_por (649 rows / 30 features), and especially
  geographical_origin_of_music (1059 rows / 116 features). Plenty of headroom
  on sarcos, diamonds, wave_energy, superconductivity, etc. Should gate the
  SR-sigma approach by dataset size/dimensionality rather than a fixed split
  fraction everywhere.
- ensemble_var (tree disagreement, epistemic only, never touches y) and
  OOB-residual-based sigma (total realized error: noise + bias + leftover
  epistemic spread) are not the same signal and can diverge — a region where
  trees agree but are systematically wrong looks "easy" to ensemble_var but
  isn't. Worth running as separate difficulty-estimator variants rather than
  assuming redundancy.
- Fitness function for the sigma regressor: the downstream conformal quantile
  q absorbs any constant/multiplicative miscalibration of sigma, so the loss
  only needs to get the *shape*/ranking of difficulty right, not the absolute
  scale — and should stay symmetric/coverage-neutral, unlike the existing
  asymmetric bound-predictor loss above (baking coverage into both sigma's
  fitness and the calibration step would double-count it).
  Ranked 4 candidate losses, best to worst:
    1. MAE on log(|residual|) — robust to outliers, no distributional or
       coverage assumptions, no custom Julia code needed (just a target
       transform + PySR's built-in loss).
    2. Pinball/quantile loss on log(|residual|) — MAE is just its tau=0.5
       case; same robustness, plus a tunable tau (diagnostic: how close the
       resulting q lands to 1).
    3. Scale-invariant log-variance loss, Var(log|r| - log(sigma)) —
       theoretically elegant (pure shape-matching) but squared-error-based
       (more outlier-sensitive) and needs a custom Julia loss.
    4. Gaussian NLL, r^2/(2*sigma^2) + log(sigma) — worst fit: assumes
       Gaussian residuals, most outlier-sensitive, no practical payoff since
       nothing downstream needs sigma to be a literal std dev.
  Recommendation: start with MAE on log(|residual|); try pinball at a couple
  of tau values as a second variant.
- Next steps: implement the chosen loss as a new difficulty-estimator variant
  alongside knn_oob_res/ensemble_var in fit_difficulty_estimators; gate by
  dataset size/dimensionality; compare against the existing direct-bound SR
  predictor and the paper's SRCP baselines on interval efficiency and
  held-out coverage (not in-sample, unlike the current check at
  symbolic_regression.py:118).

### 2024-06-04
We should probably go for a RandomForest with 500 trees.

### 2024-06-04
physiochemical_protein with high settings for PySR makes the program crash. Probably need some re-runs with lower settings.

### 2024-06-03
There are some issues that previously did not exist with KNN. Maybe it's due to scikit-learn version 1.5; I could try to create an environment with scikit-learn==1.4.2; No, actually it turns out that we just need to update threadpoolctl to the latest version.

### 2024-05-14
Easier approach to fitness: just fit the distance between predicted and true point. However, penalize more if the distance is smaller.

### 2024-05-13
What we would like to do:
1. minimize number of measured points outside of [predicted-ci, predicted+ci]
2. minimize amplitude of confidence intervals

Fitness function: predict value of the confidence interval; apply to predicted point: 
1. if measured point falls within confidence interval, -1 on first fitness (SATURATE ON COVERAGE REQUIRED)
2. if measured point is out, minimize delta out
3. then, minimize size of confidence intervals

## Resources
Libraries for conformal prediction: https://github.com/henrikbostrom/crepes
Examples: https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html#Investigating-the-prediction-intervals