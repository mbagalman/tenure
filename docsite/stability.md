# API stability and deprecation policy

This page is the contract v1.0 freezes. It exists so you can build on Tenure without re-reading
changelogs defensively: what is listed as covered will not break out from under you within a
major version.

## What the guarantee covers

1. **Every name exported by `tenure`** (`tenure.__all__`). Nothing is removed or renamed within
   a major version except through the deprecation procedure below.
2. **Signatures.** Parameters are never removed, renamed, or re-ordered within a major version.
   New parameters are always keyword-only with a default, so existing calls keep working.
3. **Tidy-frame columns.** The column names (and order) of every documented DataFrame return --
   `retention_at`, `rmst`, `survival_weighted_ltv`, `SummaryReport.table`, `RiskScores.table`,
   `ValidationResult.table`, `LogRankReport.table`, and the rest -- are part of the API.
4. **Check and footgun IDs.** `TNR001`-`TNR005` and `VAL001`-`VAL003` are permanent, citable
   identifiers; a pasted audit or validation report stays meaningful forever.
5. **Exception types.** Documented failure modes keep raising the documented exception types
   (`TenureValidationError`, `AuditBlockedError`, ...).
6. **Defaults.** Default parameter values change only at a minor version, always with a
   changelog entry.

## What it deliberately does not cover

Some public surface must stay free to evolve; relying on these is at your own risk:

- **`.fitter`** on `CoxPH` / `TimeVaryingCox` is an escape hatch to the underlying
  [lifelines](https://lifelines.readthedocs.io/) object. It is guaranteed to *exist*; its type
  and behavior follow whatever lifelines version you have installed.
- **Direct constructors** of `StudyDesign`, `SurvivalFunction`, and the curve classes are
  plumbing for the factory paths (`from_event_dates` / `from_status` / `from_intervals`,
  fitted estimators). Build through the factories; the constructors' keyword lists may change.
- **`.metadata` dictionaries** carry an additive-only guarantee: existing keys are never
  removed or renamed within a major version, but new keys may appear at any time.
- **`TestCohort.paths`** (the per-customer post-cutoff panel) is provisional, reserved for the
  future time-varying validation metrics. `TestCohort.table` *is* covered.
- **Demo functions** (`naive_vs_corrected_demo`, ...) freeze their result-dict *keys*; the
  numeric values are regression anchors pinned to seeds and dependency versions, not promises.
- **Message text** -- exception messages, warning wording, docstrings, and report prose may
  improve at any time. Match on types and IDs, never on message strings.

## Naming conventions (so the API reads predictably)

- `estimator` = anything producing survival curves (a fitted `KaplanMeier`, a `SurvivalFunction`,
  a hybrid); `model` = a fitted predictive model being validated; `design` = a `StudyDesign`.
- `horizon` is a scalar time; `horizons` a sequence of them; `times` a metric evaluation grid.
- Grouped estimators fit with `fit(data, by=...)`; covariate models fit with `fit(design)` and
  take their columns from the design's `covariate_cols`. The asymmetry is the statistics, not
  an oversight.

## The deprecation procedure

1. The release that deprecates something keeps it fully working and emits a
   `DeprecationWarning` naming the replacement.
2. It stays working, with the warning, for **at least one further minor release**.
3. Removal happens no earlier than the minor release after that, and the
   [changelog](changelog.md) lists every deprecation and removal with its replacement.

Pre-1.0 releases (everything up to and including the current version) are not bound by this
policy -- it takes effect at v1.0.0.
