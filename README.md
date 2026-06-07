# Tenure

**Audit-first survival analysis for B2C customer churn.**

Tenure's thesis: the hard, value-adding part of churn survival analysis is *not* the
estimators -- `lifelines` already nails those. It is getting the **study design** right.
Tenure makes the statistically correct design the default and makes biased designs hard
to produce by accident, via a plain-language **study-design audit** that runs *before*
any number is returned.

> **Status: v0.3 (alpha).** v0.1 = the audit + Kaplan-Meier + retention/LTV. v0.2 adds risk
> modeling (Cox PH, churn-risk scoring, PH diagnostics). v0.3 adds the time-varying data model
> (interval / counting-process schema, time-varying Cox) that *prevents* immortal-time bias, plus
> landmarking. The public API is settling; minor changes are still possible. The distribution name
> on PyPI is not final.

## Why this vs lifelines?

`lifelines` gives you correct *estimators* and assumes you have already built a statistically
valid risk set. In practice that assumption is where most business churn analyses quietly go
wrong -- left-truncation inflates retention and LTV, window-as-origin and immortal-time bias
fabricate effects, informative censoring skews curves. Tenure wraps lifelines for the math and
adds the layer it is missing: a **study-design audit** that makes the correct design the default
and the biased one hard to produce by accident, plus business outputs (retention %, RMST, LTV $)
that carry their audit caveats. Run it *before* you trust a curve.

## What the audit catches (TNR001-TNR005)

| Check | Bias | Default |
|---|---|---|
| **TNR001** | Left-truncation / delayed entry -- event history that does not reach back to a customer's origin (a "Window-Cut" study, e.g. a billing migration) must be modeled with delayed entry, or retention/LTV are biased upward. | block |
| **TNR002** | Time-origin confusion -- using the observation-window start as t=0 instead of true signup. | block |
| **TNR003** | Event/censoring mislabeling -- unmapped exit statuses (status schema), and a warning when a non-churn exit is mapped to `censored` (informative censoring). | block / warn |
| **TNR004** | Immortal-time -- a covariate level that only appears for higher-tenure customers (a data-driven quantile shift test). On an interval/time-varying design the bias is structurally prevented, so this passes. | warn |
| **TNR005** | Weak / over-extrapolated horizon -- RMST/LTV past the supported horizon are truncated-and-relabeled rather than read off the flat KM tail. | warn |

Each check is bypassable with `strictness="warn"` and clearable with an explicit attestation
(e.g. `attest_origin_correct=True`) when you know the design is genuinely fine.

## Quickstart

The one-liner that shows why this exists -- the LTV dollars a naive analysis over-states when
it mishandles left-truncation:

```python
import tenure

result = tenure.naive_vs_corrected_demo()        # synthetic SVOD data, seed 0
print(f"naive LTV:      ${result['naive_ltv']:.2f}")
print(f"corrected LTV:  ${result['corrected_ltv']:.2f}")
print(f"true LTV:       ${result['true_ltv']:.2f}")
print(f"over-statement: ${result['ltv_dollar_diff']:.2f} per customer")
print(result["audit"].to_markdown())             # the TNR001 warning that explains the gap
# naive LTV:      $101.16
# corrected LTV:  $90.81
# true LTV:       $90.96
# over-statement: $10.35 per customer
```

Or drive the pieces yourself -- audit a study design, fit Kaplan-Meier, summarize:

```python
df = tenure.load_svod_demo(with_left_truncation=True)
study = tenure.StudyDesign.from_event_dates(
    df, id_col="customer_id", origin_col="signup_date", churn_date_col="churn_date",
    active_as_of="2026-05-31", analysis_start="2024-01-01",
    event_observed_from="2024-01-01",            # model delayed entry -> audit passes
    group_cols=["plan"],
)
report = tenure.audit(study)                      # raises on a blocking design (default)
km = tenure.KaplanMeier().fit(study, by="plan")
summary = tenure.summarize(km, period_margin=12.0, ltv_horizon=365.0, audit_report=report)
print(summary.to_markdown())
```

## Risk modeling (v0.2): Cox, scoring, diagnostics

Move from "how is the cohort retaining" to "which customers are at risk, and why." Declare
covariates on the study design; Cox plugs into the same outputs as Kaplan-Meier.

```python
study = tenure.StudyDesign.from_event_dates(
    df, id_col="customer_id", origin_col="signup_date", churn_date_col="churn_date",
    active_as_of="2026-05-31", covariate_cols=["plan", "tenure_days_at_signup"],
)
cox = tenure.CoxPH().fit(study)

# Survival curves at covariate profiles -> consumed by retention_at / rmst / survival_weighted_ltv:
curves = cox.predict_survival(cox.profile_grid("plan"))
print(tenure.retention_at(curves, [90, 365]))

# Per-customer churn-risk scores (partial hazard) + ranking:
scores = tenure.churn_risk_scores(cox, horizon=365.0)
print(scores.table.sort_values("risk_score", ascending=False).head())

# Is the proportional-hazards assumption respected?
diag = cox.proportional_hazards_test()          # warns if violated; tidy report on .table
print(diag.ok, diag.violations)
# tenure.plot_log_log_survival(km_by_plan)      # visual PH check
```

Nelson-Aalen cumulative hazard is also available (`tenure.NelsonAalen`, `plot_cumulative_hazard`).

## Time-varying covariates (v0.3): the immortal-time fix

Customer risk is driven by behavior that changes over the lifetime -- usage, plan, support
contacts, promos. A static analysis that classifies customers by a *future-looking* attribute
("ever upgraded") credits the upgraded group with the immortal person-time before the upgrade and
invents a protective effect. v0.3 fixes this at the data model: an interval (counting-process)
schema where covariates vary per interval, and a `TimeVaryingCox` that consumes it. The upgrade is
encoded `0` before it happens and `1` after, so it cannot leak future survival.

```python
result = tenure.naive_vs_corrected_immortal_demo()    # the upgrade truly has no effect (HR = 1)
print(f"naive (static ever-upgraded) HR: {result['naive_hazard_ratio']:.3f}")   # ~0.62, illusory
print(f"corrected (time-varying)     HR: {result['corrected_hazard_ratio']:.3f}")  # ~1.02, truth
# naive audit warns TNR004; the interval-design audit passes it (the bias cannot occur).
```

Build an interval design and fit the time-varying Cox:

```python
study = tenure.StudyDesign.from_intervals(
    panel, id_col="customer_id", origin_col="signup_date",
    interval_start_col="period_start", interval_end_col="period_end", event_col="churned",
    covariate_cols=["plan", "monthly_usage", "promo_active"],   # may change each interval
)
tv = tenure.TimeVaryingCox().fit(study)
print(tv.summary)                       # covariate, coef, hazard_ratio, p_value
print(tv.risk_scores().head())          # per-interval, time-varying partial hazard ratio

# Survival for one hypothetical customer's covariate PATH -> a SurvivalFunction the outputs consume:
path = tenure.StudyDesign.from_intervals(one_customer_panel, id_col=..., covariate_cols=[...])
curve = tv.predict_survival(path)
print(tenure.rmst(curve, horizon=365))
```

**Landmarking.** For a static model that still avoids immortal-time, `tenure.landmark(study, L)`
keeps only subjects at risk at tenure `L`, fixes their covariates to the value as of `L`, and
returns a single-interval design (delayed entry at `L`) that `CoxPH` / `KaplanMeier` consume
unchanged.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## License

MIT
