# Notes

## Ideas to be developed
1. Use the boundaries of the Mondrian predictors as initial seeds for the values in the GP.
2. If extracting the sigmas from the Mondrian predictors is impossible, extract directly the lambdas.
3. After obtaining the final Pareto frontier, remove all solutions that do not respect the hard constraint of $(1-\alpha)$ coverage, and from those left, pick the one with the best value of the fitness function.
4. Idea to fix the theoretical miscoverage of the direct confidence bounds regression using symbolic regression: instead of directly predicting confidence bounds, predict difficulty estimation and calibrate on calibration set. Problem: difficulty estimator cannot be trained on the same training set as base regressor. Two ideas:
    * For large datasets, split into two sets (80/20 for example), train base regressor on 80% and difficulty estimator on 20%. Optionally, retrain base regressor after using 100% of data. Note: sigmas used in SR training and comparison are not the same (first are sigmas on 80% of data, second on 100%).
    * For Random Forests: use "free" data from OOB predictions to train difficulty estimator. Use all sigmas from OOB predictions for SR training, and full base predictor for comparison.

## Chronological notes

### 2026-09-23
Corrected a wrong assumption made mid-analysis while diagnosing why sigma
vs. `|residual|` looks non-monotonic even after binning (see the same-day
`method_comparison.py` change below): data augmentation with the 4
pre-computed sigma columns (`knn_dist`/`knn_std`/`knn_res`/`var`) has been
**off** since 2026-09-14 — SR trains purely on raw base features now, so
"SR mostly recovers/recalibrates one of the 4 existing sigma columns"
(2026-08-21/08-31 entries, also in Claude memory `sigma-sr-theory`/
`sigma-sr-findings`) describes the augmented-input regime, not the current
one. Any resemblance between the SR sigma's shape and a baseline sigma's
shape now has to come from both discovering similar structure independently
from raw features, not from SR literally having that column as an input to
select/rescale.

Also changed `dashboard/pages/method_comparison.py`'s "Sigma Relationships"
tab from an unbinned per-point scatter to the same equal-count-bins/median
pattern already used by "Width by Residual Rank" and the Difficulty Heatmap
(`N_BINS = 15`, bin on the target — `abs_residual` or `width_<method>` —
median sigma per bin, one line per method's own subplot). Confirms the
non-monotonicity/non-linearity in the sigma-vs-residual relationship is not
just unbinned-scatter noise: `var` in particular stays flat (or dips)
through low/mid residual bins and only rises sharply in the top 1-2 bins,
and `knn_dist` even trends down at the top bin on some datasets
(`airfoil_self_noise`, `concrete_compressive_strength`).

### 2026-09-14 (uncommitted as of this note)
Extended the Hall-of-Fame diagnostics from 2026-09-11 and made a first pass
at testing whether SR can find difficulty structure without leaning on the
4 pre-computed sigma proxies.

- **Turned off all 4 sigma-augmentation columns** in
  `src/configs/sigma-sr/default_config.yaml`. This only gates what's fed into the SR input
  matrix — the 4 baseline normalized-CP methods themselves are still always
  computed. With augmentation off, SR now fits purely on
  raw base features, directly testing the collapse mechanism (SR mostly re-selecting one of the 4 pre-smoothed sigmas rather than discovering new structure); with the proxies removed,
  SR has to either discover structure from raw features or fail to beat
  baseline width.
- Also in the same config: added `lambda_cov: 500` (now threaded as an
  explicit parameter into `bin_crossfit_loss_julia(confidence, lambda_cov)`
  in `utils/losses.py`, replacing what was previously an undefined
  `lambda_cov` reference inside the Julia loss string); dropped `sin`/`cos`
  from `unary_operators` (now just `log`/`exp`).
- **New diagnostic plots**, all keyed off a new
  `compute_binned_coverage_width` helper (`utils/evaluate.py`: quantile-bins
  test points by a difficulty score, reports per-bin coverage/median
  width/mean width):
  - `plot_equation_performance_vs_complexity` — every Hall-of-Fame equation
    for a loss, scattered by (coverage, width) and colored by complexity,
    chosen equation highlighted.
  - `plot_binned_sigma_metric` — coverage/width vs. binned difficulty score;
    used two ways: across every HoF equation (colored by complexity, only
    the chosen one labeled) and across CP methods (one line per method).
  - `plot_sigma_distributions` — violin plot of each method's difficulty
    score, normalized by its own median (raw scales aren't comparable
    across estimator types).
  - Test-set sigmas (`sigmas_test`) are now saved for `symbolic_regression_*`
    too (previously only cal-set `sigmas_comp` was), needed for the
    size-stratified test-set coverage plots above.

### 2026-09-11
Added the ability to pull the **entire** PySR
Hall of Fame and all equations from last generation 
per dataset/loss

### 2026-08-31
Reviewed `sigma_sr_notebook.py`'s results section (full 22-dataset sweep,
`results-sigma-sr-42_20260825-101216/`, comparing the `mae`/`mean_width`/
`pairwise_rank` sigma-SR losses against the 6 baseline CP methods) and
implemented its "Next steps" loss in `run_sigma_sr.py`.

- **Equation composition**: 58/66 (88%) chosen equations across the 3 losses
  reduce to a single pre-existing sigma estimator (`var`/`res` dominate,
  `dist` essentially never chosen) — SR mostly selects/recalibrates an
  existing difficulty signal rather than discovering new structure. No link
  between equation complexity and dataset size, dimensionality, or base R².
- **Marginal coverage is not a differentiator**: all 9 methods land within
  ~0.5-1.6% of the 0.95 target by construction of split CP; real differences
  are in interval width and conditional coverage, not average coverage.
- **Loss comparison**: median width correlates with base-regressor R² for
  all 3 losses (weaker regressor → wider intervals, as expected). `mean_width`
  is the least Pareto-dominated of the 3; `mae` performance varies a lot
  (noisy single-residual proxy); `pairwise_rank` is never the sole
  Pareto-best, likely a structural weakness (hinge/margin loss has zero
  gradient once ranking is roughly correct).
- **`mean_width`'s known weakness**: in-sample optimism — each candidate is
  calibrated and scored on the same training rows, so it overfits to noise.
- **Implemented next step**: `bin_crossfit_loss_julia` in `run_sigma_sr.py`. Fixes the
  in-sample optimism via 2-fold cross-fitting (calibrate the quantile on one
  fold, score width/coverage on the other) and adds an explicit per-bin
  (4 equal-frequency Mondrian bins on sigma) squared-coverage-deviation
  penalty, instead of relying on marginal coverage alone. Not yet run on the
  full 22-dataset sweep.

### 2026-08-21
Analyzed the first completed sigma-SR run end-to-end
(`results-sigma-sr-42_20260819-165935/`, 22 datasets, `run_sigma_sr.py`'s
MAE-on-`log|residual|` loss): aggregate performance, dataset-characteristic
correlations, chosen-equation composition, and the theoretical reason behind
the single-sigma collapse noticed while reading the equations.

- **Chosen-equation composition, all 22 datasets.** Result: 11/22
  collapse to `sigma_var` alone, 8/22 to `sigma_knn_res` alone, 1 to
  `sigma_knn_std` alone, 1 (pumadyn32nh) to base features only, 1 (Moneyball)
  mixes base+sigma. **`sigma_knn_dist` is never chosen, not once.** Datasets
  that settle on `knn_res` average 0.95x baseline width (net win); datasets
  that settle on `var` average 1.15x (net loss, including the single worst
  result, concrete_compressive_strength at 1.67x).
- **Confirmed empirically the 2026-08-18 entry's OOB/leave-one-out
  mechanics** — for the 7 datasets where the chosen equation is a *pure*
  `log(single_sigma)` (coefficient 1, no additive offset), the resulting
  SR-CP's mean/median/coverage on cal/test match the corresponding baseline
  NCP method's.
  This is the direct empirical signature of what 2026-08-18 established
  structurally: `oob=True` only changes `.apply()`'s no-argument
  (self-referential, training-set) behavior; `.apply(X_cal)`/`.apply(X_test)`
  on genuinely external data is identical whether or not the estimator was
  fit with `oob=True`. For `ensemble_var` specifically, `run_sigma_sr.py`
  already deliberately reuses the plain (non-OOB) `de_var` object for
  cal/test (see the inline comment at that call site). **Not leakage**: OOB protects the
  training side from self-referential leakage; cal/test rows were never at
  risk since they were never in any fit set to begin with.
- **Why this makes SR mostly "select + recalibrate one existing NCP sigma"
  rather than discover new structure** The fitting target, `log|residual_i|` for a single OOB point, is
  one noisy sample of that point's local error scale. The four candidate sigma columns are already
  locally averaged (KNN-neighbor or cross-tree averaging), smoothed
  estimates of the same latent quantity while raw base features are not.
  Fitting any reasonably-smooth symbolic function to a noisy pointwise
  target is itself an implicit smoother (nearby x's pulled toward similar
  `f(x)`), but needs enough local data density to work; with too little,
  PySR can't out-perform an explicitly pre-smoothed proxy, so it just
  recovers one via a monotonic transform. `model_selection
  ="best"` + default `parsimony` then prunes any second, mutually-correlated
  sigma column since it rarely earns enough loss reduction to clear the
  complexity penalty. This directly explains the dataset-scale sensitivity
  found in the same analysis: more data lets
  SR's own implicit smoothing get good enough to compete with, or beat, the
  pre-smoothed sigmas.
- **Clarified what "difficulty"/"sigma" actually means across the four
  estimators — not all the same kind of quantity.** `ensemble_var` is a
  literal statistical variance (of the RF estimator's own predictions across
  trees — epistemic/model uncertainty only, never touches `y`); `knn_oob_res`
  is a local average of *total realized error* (aleatoric + epistemic
  conflated, can't be told apart from a residual alone); `knn_std` is local
  label dispersion (aleatoric-leaning, but also picks up real unmodeled
  curvature of the true function within the neighborhood — not pure noise);
  `knn_dist` is a pure sparsity/extrapolation proxy (never touches `y` or
  residuals at all). The SR sigma-predictor's own target aims at the same
  thing as `knn_oob_res`. Worth remembering when reasoning about *why* a
  given sigma wins on a given dataset: it's not only "smoothed vs. not," it
  can also be "which underlying source of difficulty (epistemic / aleatoric
  / sparsity) actually dominates for this dataset."
- **Candidate next steps to test (none implemented yet)**:
  1. `model_selection="accuracy"` or a lower `parsimony`, to see whether
     relaxing the complexity penalty alone produces genuinely richer
     (multi-sigma or sigma+feature) equations, and at what generalization
     cost on the small datasets.
  2. Decorrelate the four sigma columns (residualize each against the
     others) before feeding to SR, so a combination isn't just combining
     redundant copies of the same latent signal.
  3. Pre-smooth the SR *target* itself (e.g. a mild KNN-average of
     `log|residual|`, independent of the four existing sigma constructions)
     to lower the noise floor and give raw features a fairer shot.
  4. Two-stage/boosted SR fit (fit on the target, then fit a second SR model
     on the residual using the remaining columns) to force multi-term
     structure without fighting the parsimony pressure directly.
  5. Revisit a coverage-aware loss for the sigma-estimator itself (in the
     spirit of `loss_function_julia_penalize_smaller` in the direct-bound
     predictor, `src/run_interval_sr.py`), rather than a symmetric proxy loss
     on `log|residual|` — optimizes the actual downstream CP quality instead
     of a noisy intermediate target.

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
  held-out coverage (not in-sample, unlike the current check in
  run_symbolic_regression, pipeline.py).

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