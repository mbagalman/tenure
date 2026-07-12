# Changelog

All notable changes to Tenure are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once the public API is frozen at v1.0.

Until v1.0 the API is still settling and minor (0.x) releases may make small breaking changes.
Audit check IDs (TNR001-TNR005, VAL001-VAL003) are a stable public contract even pre-1.0.

## [Unreleased]

### Added

- **Example gallery** (`docsite/gallery/`): complete, narrated analyses whose code blocks are
  executed by the test suite on every commit (doc-rot gate). Launch set: *The $10 mistake*
  (left truncation with the full warn -> attest -> block -> fix audit escalation, checked
  against known truth), *A 3-year LTV from 2 years of data* (KM truncation vs parametric and
  hybrid projection, landing within ~1.5% of truth), and *Real data: Telco churn* (the IBM
  dataset ingested with its assumptions stated out loud -- the tenure-to-dates recipe, the
  survivorship attestation, and the classic contract-type results). Gallery images regenerate
  deterministically via `scripts/build_gallery_images.py`; the Telco CSV is user-downloaded,
  never vendored.

### Fixed

- Quickstart wording: omitting `event_observed_from` makes TNR001 **warn and ask** whether
  event history is complete; the block fires when you answer `includes_pre_entry_churners=False`.
  The tutorial previously claimed the omission alone would raise `AuditBlockedError`.

## [0.5.1] -- 2026-07-11

The hardening patch: every finding from four rounds of external review of the v0.5 surface, all
verified empirically before fixing. Two corrections change reported statistics (the stratified
C-index and the log-rank degrees of freedom) -- if you scored stratified models or ran log-rank
tests with sparse groups on v0.5.0, re-run them.

### Fixed

- `retention_at` (and `summarize`) no longer crash on duplicate horizons like ``[30, 30]``
  (review) -- horizons dedupe order-preserving; asking for a horizon twice means asking once.
- `churn_risk_scores` on a `TimeVaryingCox` now raises a clear error pointing at
  `TimeVaryingCox.risk_scores()` / `predict_survival(path)` (review) -- it previously crashed
  with a raw `AttributeError` deep in lifelines (a time-varying fitter cannot produce the
  per-subject survival-at-horizon column from a single covariate row).

- `summarize(horizons=[])` no longer returns an empty table (review): the retention-first inner
  merge annihilated the RMST/LTV rows against an empty retention frame. The table now assembles
  on a base frame of the groups, so an empty-horizons report carries RMST/LTV with no retention
  columns.
- `summarize(horizons="30")` now means a single 30-unit horizon (review) -- the scalar string
  previously iterated to characters and silently produced 3-unit and 0-unit retention. Scalars
  coerce the same way `retention_at` already coerced them.
- `plot_survival` on a user-supplied `ax` anchors the caveat stamp and the hybrid splice note to
  that axes (review) -- they were written with `fig.text` at the corners of the whole figure,
  spraying text across unrelated subplots in a user's grid. Self-managed figures are unchanged.

- `logrank_test` degrees of freedom are now the covariance matrix's actual rank, not a hardcoded
  ``n_groups - 1`` (review): a group never at risk at any event time (all censored before the
  first event) contributes zero variance, and testing the lower-rank statistic against the larger
  df inflated the p-value (overly conservative). A zero-rank covariance (no two groups ever at
  risk together) now raises with guidance. Per-group risk-set counting also switched to a single
  `np.bincount` pass per event time (was an O(rows x groups) mask loop).
- `hybrid_survival` now rejects splicing curves with different ``time_unit``s (review) -- a daily
  empirical curve with a monthly model tail previously spliced silently and misaligned every
  tenure.
- `ParametricSurvival` queries at ``NaN`` tenures now propagate ``NaN`` (review) -- ``NaN > 0``
  is ``False``, so they previously coerced to a silent S = 1.0.

- Stratified Cox models are now scored with the **stratified C-index** (within-stratum pairs,
  pooled by pair count) in both `cross_validate` and `concordance` (review). A stratified model's
  partial hazard carries no baseline, so ranking it across strata assumed the shared baseline the
  model explicitly rejects -- cross-strata pairs are no longer compared, and results record
  `pair_restriction: "within_stratum"`. Note the semantics: the stratified variable's effect lives
  in the baselines and is deliberately not part of the C-index.
- `panel_folds` is now invariant to dataframe row order (review): customer ids are sorted before
  the seeded shuffle, so the same records always produce the same folds.
- Fold designs are built directly from the parent's canonical rows (review): float tenures are
  carried bit-for-bit (no tenure -> date -> tenure round trip) and the parent's covariate
  mappings are inherited, keeping every fold's encoded column space identical.
- The entry-aware concordance documents its O(events x rows) cost (review) -- exact and fine at
  fold sizes, a known bottleneck on very large cohorts.
- `logrank_test` documents the same cost class (review): O(unique event times x rows), which for
  continuous date-derived tenures (every event time unique) degrades to O(events x rows) on very
  large cohorts.

## [0.5.0] -- 2026-07-11

The adoption release: the deferred P2s pulled forward so choosing Tenure over hand-rolled KM + Cox
costs nothing in capability. Group-comparison inference, principled projection past the data
window, the PH-violation remedy, and error bars on validation -- each with the same honesty
guarantees (delayed-entry correctness, no silent extrapolation, leakage asserted) as the rest of
the library. Also the first release with this documentation site.

### Added

- Panel-aware cross-validation (DV4-7, deferred from v0.4): `cross_validate(factory, design, k=5)`
  fits a fresh Cox-family model per fold and returns a per-fold C-index table with mean and spread;
  `panel_folds` partitions by customer id (all of a customer's intervals travel together,
  disjointness asserted); `ensure_panel_safe` is the VAL003 leakage guard for hand-built splits.
  The per-fold C-index is delayed-entry aware -- events are compared only against customers at
  risk at that moment -- and reproduces lifelines' `concordance_index` exactly (tie conventions
  included) when there is no delayed entry. Cross-sectional by design: complements, never
  replaces, `temporal_holdout`.
- `hybrid_survival(km, model)` -- hybrid (spliced) survival curves: empirical Kaplan-Meier up to
  each group's supported horizon, the model's conditional tail beyond, rescaled to meet exactly at
  the splice boundary. Long-horizon RMST/LTV use every observed event AND a principled tail. Each
  `HybridGroupCurve` records its boundary and source curves; `plot_survival` marks the boundary
  with a dotted line and a "data ends, model tail begins" note; CIs exist only on the empirical
  segment. A step-curve tail cannot launder a flat tail into extrapolation -- the hybrid stays
  truncated where the tail model's own support ends.
- `CoxPH(strata=[...])` -- stratified Cox, the standard remedy when
  `proportional_hazards_test` flags a categorical covariate: refit the same design with the
  offender stratified (its own baseline hazard per level, no coefficient, no PH assumption).
  The PH-violation warning now names the raw covariate and the exact `strata=` call. Prediction,
  `profile_grid`, `churn_risk_scores`, and all business outputs work unchanged, with each
  profile's curve drawn from its own stratum's baseline. Coefficients reference-matched to a bare
  lifelines stratified fit (with delayed entry).
- `ParametricSurvival(distribution=...)` -- parametric survival models (`weibull` default,
  `exponential`, `lognormal`, `loglogistic`) wrapping lifelines. Unlike Kaplan-Meier, a fitted
  distribution is defined at every tenure, so `rmst` / `survival_weighted_ltv` / `retention_at`
  extrapolate past observed support (`truncated=False`) for principled long-horizon LTV. Presents
  the same multi-group `SurvivalFunction` interface (A3), honors delayed entry, and exposes fitted
  parameters via `.params_` (Weibull `shape` reads the hazard trend). Survival functions are
  evaluated in closed form and reference-matched to lifelines across all four distributions.
- `logrank_test(design, by=...)` -- the log-rank test for group comparison, with a `LogRankReport`
  (per-group observed/expected table, chi-square statistic, degrees of freedom, p-value,
  `significant(alpha)`). Left-truncation aware: risk sets are built from the delayed-entry times,
  so a window-cut cohort is compared correctly. Reference-matched to lifelines on the no-entry case.
- Documentation site (MkDocs Material): tutorials, the bias-audit catalog, and autodoc API
  reference. A [scope page](scope.md) draws the contractual vs. non-contractual boundary.

## [0.4.0] -- 2026-06-08

Out-of-time validation and predictive metrics. Validation is a separate layer over predictions plus
a held-out design; it never reaches into estimator internals.

### Added

- `temporal_holdout(design, cutoff)` -- the out-of-time train/test split. Training is censored at
  the cutoff (no post-cutoff event leaks); the test cohort is the at-risk-at-cutoff set with
  post-cutoff outcomes on an evaluation clock. Single-spell and interval designs handled uniformly.
- `TestCohort` -- the evaluation-clock cohort produced by the split.
- `random_split(design)` -- the footgun, kept but warned: emits **VAL001** because a random split
  of a survival panel leaks future information.
- `concordance(model, test_cohort)` -- Harrell's C-index on the evaluation clock, model-agnostic
  across Cox-family, survival-function/KM, and raw risk arrays.
- `brier(model, test_cohort, times)` and `integrated_brier(...)` -- time-dependent IPCW Brier
  score and Integrated Brier Score, hand-rolled to keep the core dependency-light (no compiled
  extras). **VAL002** flags horizons beyond model support.
- `calibration(model, test_cohort, horizon)` -- reliability-diagram data: predicted survival by
  risk bin vs. Kaplan-Meier-observed survival. `plot_calibration(result)` renders it.
- `ValidationResult` contract (tidy `.table` + `.metadata`); the VAL001/002/003 ids are kept out of
  the design-time TNR registry by design.

## [0.3.1] -- 2026-06-07

Hardening pass: six correctness fixes, all covered by tests.

### Fixed

- `from_event_dates` now parses churn dates strictly -- a present-but-unparseable value raises
  instead of being silently coerced to "active" (which had inflated retention/LTV).
- `GroupCurve.effective_horizon` no longer collapses to 0 for all-censored cohorts, so RMST/LTV are
  positive and run to the supported horizon.
- `TimeVaryingCox.predict_survival` uses lifelines' centered partial hazard, consistent with its
  mean-centered baseline (was biased by a constant factor). Re-verified against an independent
  `CoxPHFitter` oracle rather than the implementation's own formula.
- Estimators call `ensure_estimable(design)`: fitting a design with unmapped statuses raises unless
  it has been audited, and a blocked strict audit leaves the design unfittable (a caught
  `AuditBlockedError` cannot bypass the guard).
- `encode_covariates` raises on unknown categorical levels instead of silently folding them into
  the baseline.

## [0.3.0] -- 2026-06-07

The time-varying data model -- the highest-architectural-impact release. Adds the interval
data shape; formal recurrent-event and multi-state estimators remain post-v1.0.

### Added

- `StudyDesign.from_intervals` -- the counting-process (start-stop) constructor: one row per
  (subject, interval), time-varying covariates, terminal-only event. Extends the canonical table
  additively (interval start/stop *are* the canonical entry/exit tenures).
- `TimeVaryingCox` -- leakage-safe time-varying Cox wrapping lifelines' `CoxTimeVaryingFitter`;
  `summary`, per-interval `risk_scores`, and `predict_survival(path)` via baseline-hazard
  integration along a covariate path.
- `landmark(design, landmark_time)` -- a lighter-weight alternative that builds a static landmark
  design (at-risk subjects, covariates as of the landmark, delayed entry) consumable by CoxPH/KM.
- `naive_vs_corrected_immortal_demo` -- the immortal-time payoff: a static "ever-upgraded" Cox shows
  an illusory protective effect while the time-varying model recovers the truth.

### Changed

- **Immortal-time prevention.** TNR004 now short-circuits to **pass** on interval designs: the
  future-looking attribute is encoded 0-before / 1-after, so the bias is structurally prevented,
  not merely warned about.

## [0.2.0] -- 2026-06-07

Risk modeling with static covariates: from "how is the cohort retaining?" to "which customers are
at risk, and why?"

### Added

- `CoxPH` -- Cox proportional hazards (static covariates) wrapping lifelines; `predict_survival`
  at covariate profiles produces the same `SurvivalFunction` the business outputs consume.
- `churn_risk_scores` -- per-customer risk score, survival at horizon, and cohort percentile.
- `CoxPH.proportional_hazards_test()` (Schoenfeld residuals) and `plot_log_log_survival` -- the PH
  diagnostic surface.
- `NelsonAalen` estimator + `CumulativeHazardFunction` and `plot_cumulative_hazard`.
- `StudyDesign.covariate_cols` / `covariate_mappings` / `encode_covariates`.

## [0.1.0] -- 2026-06-07

The correctness MVP. Theme: correct retention and LTV analysis for contractual subscription churn,
with the study-design audit as the hero feature.

### Added

- `StudyDesign` with two explicit, mutually-exclusive input schemas (`from_status`,
  `from_event_dates`) normalizing to one canonical internal table.
- The pluggable bias audit with stable public check IDs: **TNR001** (left-truncation),
  **TNR002** (time-origin), **TNR003** (exit mislabeling / informative censoring),
  **TNR004** (immortal-time, warn), **TNR005** (weak/over-extrapolated horizon, output-time).
  Block-by-default with a `strictness="warn"` opt-out; a clean cohort returns all-pass, zero
  warnings.
- `KaplanMeier` (delayed-entry aware, group comparison, CIs) behind a multi-group survival
  interface; the `SurvivalFunction` / `GroupCurve` abstraction the business layer consumes.
- Business outputs: `retention_at`, `rmst` (truncate-and-relabel, never silent extrapolation),
  period-correct `survival_weighted_ltv`, and `summarize` / `SummaryReport` carrying audit
  provenance.
- `plot_survival` (KM curves, CI bands, at-risk table, caveat stamp when a block is bypassed).
- `RetentionStudy` / `RetentionResult` -- the guided high-level workflow.
- Synthetic SVOD dataset (`load_svod_demo`, `svod_demo_truth`) and the headline
  `naive_vs_corrected_demo` pinning the left-truncation LTV dollar gap as a regression gate.
- Packaging: MIT license, pyproject/hatchling, ruff, pytest, GitHub Actions (Linux + Windows),
  Python 3.10+.

[Unreleased]: https://github.com/mbagalman/tenure/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/mbagalman/tenure/releases/tag/v0.5.1
[0.5.0]: https://github.com/mbagalman/tenure/releases/tag/v0.5.0
[0.4.0]: https://github.com/mbagalman/tenure/releases/tag/v0.4.0
[0.3.1]: https://github.com/mbagalman/tenure/releases/tag/v0.3.1
[0.3.0]: https://github.com/mbagalman/tenure/releases/tag/v0.3.0
[0.2.0]: https://github.com/mbagalman/tenure/releases/tag/v0.2.0
[0.1.0]: https://github.com/mbagalman/tenure/releases/tag/v0.1.0
