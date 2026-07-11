# Retention and LTV

Once you have an audited [`StudyDesign`](../reference/study-design.md) and a fitted estimator, the
business-output layer turns a survival curve into the numbers a business actually asks for:
retention at horizons, restricted mean survival time, and a period-correct lifetime value.

These outputs consume a **survival abstraction**, not estimator internals. That means
Kaplan-Meier, Cox at a covariate profile, and a time-varying Cox path all feed the same functions.

## Setup

```python
import tenure

df = tenure.load_svod_demo(with_left_truncation=True)
study = tenure.StudyDesign.from_event_dates(
    df, id_col="customer_id", origin_col="signup_date", churn_date_col="churn_date",
    active_as_of="2026-05-31", analysis_start="2024-01-01",
    event_observed_from="2024-01-01", group_cols=["plan"],
)
report = tenure.audit(study)
km = tenure.KaplanMeier().fit(study, by="plan")
```

## Retention at horizons

[`retention_at`](../reference/outputs.md) reports survival at each horizon, per group, with
confidence intervals and a support flag:

```python
print(tenure.retention_at(km, [30, 90, 180, 365]))
```

The `supported` column is `False` when a horizon exceeds the group's supported window -- the
retention there is read off the flat Kaplan-Meier tail and should be treated with caution. This is
the [TNR005](../audit-catalog.md) output-time guard in action.

## Restricted Mean Survival Time (RMST)

RMST is the average time-in-subscription through a horizon -- the area under the survival curve. It
is well defined even when the median survival is not reached:

```python
print(tenure.rmst(km, horizon=365))
```

Tenure never silently extrapolates a flat tail. The integral runs only to a per-group **effective
horizon**; if your requested horizon outran support, the result is flagged `truncated=True` and
`effective_horizon` reports where the integration actually stopped.

## Survival-weighted LTV

[`survival_weighted_ltv`](../reference/outputs.md) weights each period's contribution margin by the
probability the customer is still subscribed at that point:

```python
print(tenure.survival_weighted_ltv(
    km,
    period_margin=12.0,    # contribution margin per period
    horizon=365.0,         # horizon, in the study's time unit (days here)
    period="month",        # the margin's period -- reconciled against the time unit
    discount_rate=0.0,     # optional NPV discounting
))
```

!!! warning "Period-correct by construction (FR-BO-3)"
    LTV reconciles the survival time unit against the margin period. You **cannot** accidentally
    multiply a daily survival probability by a monthly margin -- a classic and silent error. State
    your `period` and Tenure does the alignment.

## One report for all of it

[`summarize`](../reference/outputs.md) bundles retention, RMST, and LTV into a single
[`SummaryReport`](../reference/outputs.md) that carries the audit provenance:

```python
summary = tenure.summarize(
    km,
    period_margin=12.0,
    ltv_horizon=365.0,
    horizons=[30, 90, 180, 365],
    period="month",
    currency="USD",
    audit_report=report,
)
print(summary.to_markdown())
```

The report's `.table` (a tidy DataFrame) and `.metadata` (verdict, horizons, currency) are
backend-neutral, so they slot into a deck or a downstream pipeline without depending on a pandas
index.

## Plot it

```python
ax = tenure.plot_survival(km, audit_report=report)  # KM curves + CI bands + at-risk table
```

Pass the audit report through when plotting. If the audit was bypassed with `strictness="warn"`,
the chart is stamped with a concise caveat note so the caveat travels into whatever deck the chart
lands in.

## Next steps

- [Risk modeling with Cox](risk-modeling.md) -- per-customer risk and the same outputs from a model.
- [Out-of-time validation](validation.md) -- prove the numbers generalize forward.
