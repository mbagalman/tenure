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

## Are the groups actually different? The log-rank test

Grouped curves invite the question "do these plans really retain differently, or is the gap just
noise?" [`logrank_test`](../reference/estimators.md) is the standard answer -- a hypothesis test of
whether the group survival curves differ:

```python
result = tenure.logrank_test(study, by="plan")
print(result.summary)     # chi2, df, p-value, and the plain-language verdict
print(result.table)       # per group: n, observed events, expected-under-the-null
```

A small p-value says the curves differ by more than sampling variation; a group whose `observed`
events fall well below `expected` retained better than the pooled average. The test groups exactly
as `KaplanMeier.fit(..., by=...)` does and **honors delayed entry** -- unlike a bare
`lifelines` log-rank, it builds each risk set from the left-truncation-aware entry times, so a
window-cut cohort is compared correctly rather than biased.

## Restricted Mean Survival Time (RMST)

RMST is the average time-in-subscription through a horizon -- the area under the survival curve. It
is well defined even when the median survival is not reached:

```python
print(tenure.rmst(km, horizon=365))
```

Tenure never silently extrapolates a flat tail. The integral runs only to a per-group **effective
horizon**; if your requested horizon outran support, the result is flagged `truncated=True` and
`effective_horizon` reports where the integration actually stopped.

## Projecting past your data: parametric survival

Kaplan-Meier truncates because it is non-parametric -- it genuinely knows nothing past the last
observed event. When you want a *principled* projection beyond your data window (a 3-year LTV from
one year of history, say), fit a parametric model instead. [`ParametricSurvival`](../reference/estimators.md)
fits a distribution (`weibull` by default, or `exponential` / `lognormal` / `loglogistic`) and
presents the **same interface** as Kaplan-Meier, so the business outputs consume it unchanged:

```python
para = tenure.ParametricSurvival("weibull").fit(study, by="plan")
print(para.params_)                              # per-group scale + shape
print(tenure.rmst(para, horizon=1095))           # 3-year RMST -- truncated=False
print(tenure.survival_weighted_ltv(para, period_margin=12.0, horizon=1095.0, period="month"))
```

Because a fitted distribution is defined at every tenure, `rmst` and `survival_weighted_ltv` now
reach the full horizon (`truncated=False`) instead of stopping at the effective horizon. The
Weibull `shape` parameter reads the hazard directly: `> 1` means churn risk **rises** with tenure,
`< 1` means it **falls**, and `== 1` is the memoryless exponential.

!!! warning "You are opting into a model"
    Extrapolation is the whole point here, but it is only as good as the distribution's fit. Beyond
    each group's `last_event_time` the curve is the model's projection, not evidence. Use a
    parametric fit when you deliberately want to project; use Kaplan-Meier with truncate-and-relabel
    when you want to report only what the data supports.

## Best of both: hybrid (spliced) curves

A pure parametric curve replaces your data *everywhere* -- even inside the window where the
Kaplan-Meier estimate is the honest choice. [`hybrid_survival`](../reference/estimators.md) splices
the two: **empirical up to each group's supported horizon, the model's conditional tail beyond**,
rescaled so the segments meet exactly (the model contributes shape, the data contributes level):

```python
km = tenure.KaplanMeier().fit(study, by="plan")
para = tenure.ParametricSurvival("weibull").fit(study, by="plan")
hyb = tenure.hybrid_survival(km, para)

print(tenure.rmst(hyb, horizon=1095))       # empirical area + model tail, truncated=False
tenure.plot_survival(hyb)                    # dotted line marks where data ends, model begins
```

Every curve records its **splice boundary** and both source curves, so nobody downstream can
mistake projection for evidence: `plot_survival` draws a dotted line at each boundary with a note,
and confidence intervals exist only on the empirical segment (the model tail is a point estimate).
Inside the data window the hybrid *is* the Kaplan-Meier curve, CI band and all.

One honesty guarantee worth knowing: the tail model's own support still applies. Splicing a
step-curve tail (e.g. a Cox profile curve, whose baseline ends with the training data) does not
turn its flat tail into a projection -- the hybrid stays truncated where that model's support
ends. Use a parametric tail when you want true extrapolation.

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
