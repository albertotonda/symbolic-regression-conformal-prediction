# Notes

## Ideas to be developed
1. Use the boundaries of the Mondrian predictors as initial seeds for the values in the GP.
2. If extracting the sigmas from the Mondrian predictors is impossible, extract directly the lambdas.
3. After obtaining the final Pareto frontier, remove all solutions that do not respect the hard constraint of $(1-\alpha)$ coverage, and from those left, pick the one with the best value of the fitness function.
4. Idea to fix the theoretical miscoverage of the direct confidence bounds regression using symbolic regression: instead of directly predicting confidence bounds, predict difficulty estimation and calibrate on calibration set. Problem: difficulty estimator cannot be trained on the same training set as base regressor. Two ideas:
    * For large datasets, split into two sets (80/20 for example), train base regressor on 80% and difficulty estimator on 20%. Optionally, retrain base regressor after using 100% of data. Note: sigmas used in SR training and comparison are not the same (first are sigmas on 80% of data, second on 100%).
    * For Random Forests: use "free" data from OOB predictions to train difficulty estimator. Use all sigmas from OOB predictions for SR training, and full base predictor for comparison.
5. Separate the "level effect" from the rest of the difficulty (not started, still under discussion). In `results-sigma-sr-42-base`, the SR sigma is strongly tied to the predicted value: median |Spearman(sigma, y_pred)| ≈ 0.5 on the test split, up to 0.95+ on fifa/diamonds, vs ≈ 0.2 with `knn_dist`. The SR mostly finds the feature that best tracks y_pred: noise proportional to the value for skewed positive targets (prices, wages, counts), smaller errors near a bound (cpu_activity, physiochemical_protein, wine grades), and RF shrinkage on extreme values.
    * **Priority, simple option: normalized target.** Run the SR on relative residuals, log|r| − log|y_pred| (i.e. |r| / |y_pred|), and deploy sigma(x) = |y_pred| · exp(g(x)). g then reads as relative difficulty. Caveats: y_pred must be in raw units (the pipeline standardizes y, so standardized y_pred can be ≈ 0 or negative), and it suits positive targets far from 0; use |y_pred| + c as a safeguard. It assumes errors exactly proportional to the value, so it can over-correct (additive noise part, bounded targets).
    * More general method, which doesn't depend on the target distribution:
        * Fit a 1-D level curve h(y_pred) on the OOB residuals with the same loss: log|r| ≈ h(y_pred). Keep it simple: low-degree polynomial/spline, or PySR on y_pred alone to stay symbolic.
        * Run the SR on log|r| − h(y_pred), on the features as now → g(x).
        * Deploy sigma(x) = exp(h(y_pred) + g(x)); calibration and coverage unchanged.
        * h adapts: increasing for skewed targets, shrinking near a bound, flat when there's no level effect (then it reduces to the current method). y_pred stays standardized, so no division by y_pred.
        * Reading: h = how difficulty depends on the predicted value, g = what the features add on top. Comparing the loss explained by h vs g measures per dataset how "level-driven" difficulty is.
    * Alternative: add y_pred as an extra SR input (simpler, but h and g end up mixed in one equation and y_pred costs complexity).
    * To check first (both options): subtracting log|y_pred| or h from the target only works as an offset if the custom Julia loss depends on (prediction − target) alone.

## Chronological notes

### 2026-10-27 — Density/outlier scores vs SR sigma (`results-sigma-sr-42-base`)

Question: does the SR difficulty score just find rare regions of feature space, or does it add information? `src/run_density_comparison.py` rebuilds the run's splits, fits KDE (Scott bandwidth), Isolation Forest (200 trees), LOF (k = `ncp_knn_k`) and GMM (1–10 components by BIC) on the training features only, and computes Spearman(rarity score, sigma of the PySR "score" equation) on cal + test. Hyperparameters are standard defaults, not tuned. OC-SVM was dropped: it estimates the support, not the density inside it. Results in `density_spearman.csv` (run level) and `density_scores.csv` (per dataset).

- Weak agreement overall: median Spearman KDE 0.17, GMM 0.16, Isolation Forest 0.14, LOF 0.07.
- Strong agreement (≥ 0.5 for one method): socmob (GMM 0.79), energy_efficiency, brazilian_houses, cpu_activity, miami_housing, diamonds (GMM 0.58). There, difficulty is largely "rare points are harder".
- Clearly negative (SR sigma higher in dense regions): superconductivity (≈ −0.65 for KDE/IF/GMM), naval_propulsion_plant (KDE −0.58), health_insurance (≈ −0.35), auction_verification (≈ −0.25). Difficulty is driven by something other than rarity (e.g. the level effect, idea 5).
- Near zero on many datasets (grid_stability, kin8nm, cps88wages, geographical_origin_of_music, physiochemical_protein). geographical_origin_of_music was read as "sparse regions" in the 2026-10-22 interpretation, but no density method agrees.
- KDE, Isolation Forest and GMM mostly agree with each other; LOF stays near 0, probably because it measures density relative to the neighbours rather than global rarity.
- Next: check which features drive each score (separate experiment), and whether KDE bandwidth / GMM components change the conclusions.

### 2026-10-22 — Qualitative interpretation of the HoF equations (`results-sigma-sr-42-base`)

For all 35 datasets, the PySR "score" equation and the most complex HoF equation were interpreted against the OpenML context, feature/target distributions and binned |residual| per feature decile (cal+test). Results are in `notebooks/equation_interpretation.py` (data: `notebooks/data/equation_interpretation_42_base.json`; analyses written by LLM agents, to be checked).

- Score equations: all interpretable (30 yes, 5 partly); verdicts 18 expected, 14 plausible, 2 unexpected (Moneyball, pumadyn32nh), 1 unexplained (kin8nm).
- Most complex: 11/35 not interpretable (22 partly, 2 yes); 6 expected, 15 plausible, 1 unexpected (pumadyn32nh), 2 unexplained (kin8nm, white_wine). Often adds little held-out ranking power, and sometimes less (forest_fires, geographical_origin_of_music, naval).
- Recurring mechanisms: width follows the predicted level of a heavy-tailed target (fifa, diamonds, kings_county, cps88wages, video_transcoding); sparse regions of feature space (geo music, space_ga); bounded targets (cpu_activity, physiochemical_protein).
- Recurring artifacts: `1/x` or `logm` on standardized categorical codes act as lookup tables that depend on the arbitrary code order (Moneyball, solar_flare, fps_benchmark, energy_efficiency); `logm` singularities create V-shapes and notches (pumadyn32nh, diamonds, white_wine).
- Homoscedastic datasets (pumadyn32nh, Moneyball) give near-constant sigma, as they should, but SR still fits small noise shapes.

### 2026-10-19 — Why use SR for NCP

The main question that this project answers is: *"How to compute explainable prediction intervals for uncertainty estimation (and why)?"*. My answer to why having an expression for the prediction interval is that it gives an explainable estimate of difficulty, which is auditable, tunable, and encapsulates all kinds of noise. 

A concrete example: you want to model the carbon emissions of a car depending on speed, fuel type, car size, car weight, origin of the car, etc. You use a Random Forest base regressor, to which you add conformal prediction to compute prediction intervals. You choose NCP for both the theoretical coverage guarantee and the adaptive intervals. You notice that the difficulty increases with car size, i.e. *predictions are less precise for high car sizes*. 

**Your question is:** *Why should I use symbolic regression for my intervals, when you have the option to use black-box methods?*

Using symbolic regression, you obtain an expression of the difficulty function like:
$$\sigma(x)= 2+0.004 \cdot \text{car size}^2 + 4 \cdot [\text{origin}] + \log(\text{speed})$$
Having a literal expression can have multiple advantages over other methods:
1. **Interpretability and Auditability:** You can directly inspect how individual features drive uncertainty. This allows domain experts to challenge the relationships (ex: verifying whether carbon emission uncertainty should scale quadratically with car size) and test if they generalize to new data.
2. **Understanding feature relationships.** The expression can capture nonlinear effects and interactions, helping us understand how prediction difficulty changes across different combinations of features. This can also suggest where additional data collection might be useful.
3. **Flexible feature combinations.** Symbolic regression can discover functional relationships between features without requiring us to specify the complete mathematical form in advance. The effect of one feature can depend on the values of others.
4. **Robustness via Parsimony:** Estimating prediction difficulty from residuals is a noisy task. While black-box models often overfit to this noise and yield erratic intervals, SR’s preference for simple mathematical structures provides a smoothed estimate, robust to unseen data.

### 2026-10-02

- With `sr_train > 0`, data augmentation only uses knn_dist and knn_std
  (sigmas applied to the SR split, no out-of-bag). `sigma_knn_res` and
  `sigma_var` are skipped with a message; they are only used with
  `sr_train: 0`.
- Mondrian CP now runs for every predictor: `mondrian_cp` bins on the RF
  variance sigmas (RandomForestRegressor), `mondrian_cp_knn_std` bins on the
  knn_std sigmas (other predictors).

### 2026-09-28

- `run_sigma_sr.py` now works with any `predictor_model` (RF, SVR, XGB,
  LinearRegression), set up through `split` (train/sr_train/cal/test in %).
  `sr_train: 0` keeps the old setup (SR trained on the RF's out-of-bag
  residuals, RF only). `sr_train > 0` trains the SR and the extra regressors
  on residuals from a separate SR split that the base model never saw.
  `knn_res`/`var`/`mondrian_cp` are only run for RF.
- Normalized CP baselines (knn_dist, knn_std, knn_res, var) are always
  fitted on the train split and calibrated on sr_train + cal, so they use
  as much non-training data as the SR does. Their augmentation sigmas are
  computed on sr_train only (out-of-bag when sr_train is 0). knn_res uses
  out-of-bag residuals on the train split (oob_score is always on for RF),
  so it is skipped for other models. standard_cp and mondrian_cp are also calibrated on
  sr_train + cal (Mondrian bins are sized on that set).
- Bug (also on main): with data augmentation turned on, the SR/extra-regressor
  intervals sent the augmented X to the base regressor, which crashed.
  `compute_normalized_intervals` now takes the difficulty estimator's
  inputs separately (`X_cal_de`/`X_test_de`).

### 2026-09-25

- Mondrian CP audit: rebuilding `mondrian_cp` intervals from the saved
  files of `results-sigma-sr-comparison` with a separate per-bin quantile
  gives exactly the saved intervals and coverage (all 35 datasets).
- Bug: `int(1/(1-0.95) - 1) + 1` gave 19, not 20 (float 1/0.05 =
  19.999...), so bins held 19 points. Still valid, since 19 is the
  minimum for a finite 95% quantile. Fixed with
  `cp_methods.mondrian_min_bin_size`.
- The bins are as small as possible (~20 points at 95%), so each bin's
  half-width is its max |residual|: valid coverage, noisy widths. Mondrian
  often beats `var` because it fits a free step-shaped width over sigma
  instead of assuming width is proportional to sigma.
- `mondrian_cp` sigmas are saved as ones; the bin ids aren't saved.

### 2026-09-24 (branch `global-q-loss`)
Aligned the `bin_crossfit` loss with what gets deployed, after tracing
stepped/stratified SR widths (e.g. Moneyball, complexity 9,
`0.0036 * logm(...)` in `results-sigma-sr-full`) to two mismatches:

- **Per-bin q in the loss vs. one global q at deployment.** With a q per
  sigma bin, sigma's magnitude between bins doesn't matter (each bin's q
  absorbs it), so SR only had to rank points into 4 bins. The Moneyball
  equation's raw sigma spreads only 1.18x -- almost standard CP -- yet
  scored well. The loss now computes one global conformal quantile
  (finite-sample rank ceil((1-alpha)(m+1)), as in crepes) per cross-fit
  direction; the 4 sigma-rank bins only measure conditional coverage for
  the penalty.
- **Min-max scaling at deployment.** `DifficultyEstimator.fit(f=...,
  scaler=True)` min-max scales sigma to [0, 1] on the training set, clips,
  and adds beta=0.01. That shifts sigma, changing its ratios: Moneyball's
  1.18x raw spread became 101x deployed. Across the 35 chosen equations the
  deployed p95/p5 spread was a median 1.9x the raw one (up to 101x on
  energy_efficiency; geographical_origin_of_music 1.00x -> 12.8x), with up
  to 27% of test points clipped. SR sigma is now deployed with
  `scaler=False, beta=0`, i.e. exactly exp(equation). Baseline kNN/variance
  estimators keep their scaler.
- The cross-fit split is now a fixed permutation per seed (sorted hashes of
  (index, seed)) instead of `rand(n)` on every call, so all candidate
  equations are scored on the same split.
- `tests/test_losses.py` evaluates the Julia loss on synthetic data with
  known sigma (log sigma = x0): oracle 2.97 < constant 4.08 << flattened
  oracle (x0 * 0.01, same ranking, no spread) 17.1; scale-invariant;
  width term matches global normalized CP.
- Replaced the `mae` loss (L1 on log|residual|, i.e. the conditional
  median) with `pinball`: pinball loss at tau = confidence on the same
  log|residual| target, so the equation estimates the log of the
  |residual| quantile a conformal interval at that level needs, not the
  median. `pinball_loss_julia` in `utils/losses.py`; enable with
  `loss_functions: [bin_crossfit, pinball]`. A 15-iteration PySR fit on
  log sigma = x0 data recovers `x0 + 0.673` (true: x0 + log 1.96 = 0.673).
  Old runs' `mae` results keep their name (`sr_mae`).
- To do: `lambda_cov: 500` was set when the per-bin penalty barely did
  anything (in-loss Mondrian almost guaranteed per-bin coverage); re-tune
  with `scripts/run_lambda_cov_sweep.sh` before a full rerun, then compare
  against `results-sigma-sr-full`.

### 2026-09-24
Added a Conditional Coverage tab to the dashboard (Method Comparison and
Hall of Fame; replaces "Residuals vs Coverage"): one sliding-window
coverage plot with a selectable x-axis -- y_pred, own sigma, or the
worst-slab projection of the test features. |residual| and y were tried
and dropped: both condition on the outcome.

- Why: coverage binned by |residual| is descriptive only. A point is
  uncovered exactly when |r|/sigma > q, so the top residual bins are
  under-covered for any method, even an oracle sigma, and a random sigma
  also flattens that curve. Conditional coverage should be checked on
  groups defined before seeing y: y_pred bins, feature slabs.
- Worst-slab coverage (Cauchois et al., 2021): 1000 random directions,
  slabs of at least 20% (adjustable) of points, searched on one half of
  the test set and measured on the other half. Per-method mode is the
  fair metric (each method gets the same adversarial search); shared mode
  evaluates every method inside the reference method's worst slab, which
  by construction makes the reference look worst there.
- Also in the tab: an "Own sigma" x-axis (coverage along each method's
  own sigma quantile, the self-consistency check), and a mean width vs.
  worst-group coverage chart (one point per method/equation; groups =
  worst slab, y_pred bins or own-sigma bins, each at least 20% of points,
  worst group picked on one half and measured on the other). y and
  |residual| aren't offered as groups: they condition on the outcome.
  On small datasets (e.g. forest_fires, 130 test points) held-out groups
  have ~13 points, so these numbers are noise; the tab warns below 100.
- Later the same day: the worst-slab projection was dropped as a plot
  x-axis (a random mix of features has no physical meaning); only the
  worst-slab coverage number remains, in the width vs. worst-group chart.
  The shared-direction mode went with it. Sigma Ridgeline and Method
  Head-to-Head tabs removed.
- "Width by Residual" became an Interval Width tab (Method Comparison and
  Hall of Fame): width vs. y_pred (where each method spends width), or
  right-sizing -- points grouped by their own width, 95% quantile of
  |residual| vs. median half-width, diagonal = right-sized. Binning width
  by |residual| was dropped: a noisy |r| flattens the trend. On
  superconductivity, right-sizing shows SR too narrow for its narrowest
  intervals (q95 = 2.0x half-width) and too wide for its widest (0.67x);
  knn_res stays within 0.75-1.5x; knn_dist is badly mis-sized
  (4.6x to 0.05x).
- Right-sizing (own-sigma x-axis) later removed from the Interval Width
  tab too; it now shows width vs. y_pred only.
- Coverage per Decile and Hall of Fame's Coverage per Decile now bin
  columns by y_pred decile (shared across methods/equations) or own-sigma
  decile, instead of residual-rank decile (outcome-conditioned).
- `run_sigma_sr.py` now saves `testing_features.csv`;
  `src/analysis/backfill_testing_features.py` rebuilt it for
  `results-sigma-sr-full` (all 35 datasets matched the saved y_test).
- First look, per-method WSC: superconductivity -- knn_res 0.946, SR
  0.881, standard 0.884, knn_dist 0.816; wave_energy -- mondrian 0.939,
  SR 0.921, knn_dist 0.868.
- Open issue noticed while reviewing losses: `bin_crossfit` calibrates
  each of its 4 sigma bins with its own quantile inside the loss, but the
  deployed SR-CP uses one global quantile (`compute_normalized_intervals`).
  Training and deployment don't match.

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

Also changed `dashboard/pages/method_comparison.py`'s "Sigma vs Residuals"
tab from an unbinned per-point scatter to the same equal-count-bins/median
pattern already used by "Width by Residual Rank" and the Coverage per Decile
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