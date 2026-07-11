# Time-varying covariates

Customer risk is driven by behavior that changes over the lifetime -- usage, plan, support
contacts, promos. A static analysis that classifies customers by a *future-looking* attribute
("ever upgraded") credits the upgraded group with all the person-time *before* the upgrade and
invents a protective effect out of nothing. This is **immortal-time bias**, and it is one of the
most common silent errors in churn modeling.

v0.3 fixes it at the data model: an interval (counting-process) schema where covariates vary per
interval, consumed by a `TimeVaryingCox`. The future-looking attribute is encoded `0` before it
happens and `1` after, so it cannot leak future survival.

## See the bias, then the fix

```python
import tenure

result = tenure.naive_vs_corrected_immortal_demo()
print(f"naive (static ever-upgraded) HR: {result['naive_hazard_ratio']:.3f}")   # ~0.62, illusory
print(f"corrected (time-varying)     HR: {result['corrected_hazard_ratio']:.3f}")  # ~1.02, the truth
```

In this synthetic data the upgrade truly has **no effect** (true hazard ratio = 1.0). The naive
static model reports `HR ~ 0.62`, a fabricated 38% protective effect. The time-varying model
recovers `HR ~ 1.02`. And the audit behaves accordingly: the naive design **warns**
[TNR004](../audit-catalog.md), while the interval design **passes** it -- the bias structurally
cannot occur.

## Build an interval design

The third study-design constructor, `from_intervals`, takes a panel with one row per
customer-interval. Covariates may differ on each row; the event is terminal (it can only happen on
a customer's last interval):

```python
study = tenure.StudyDesign.from_intervals(
    panel,
    id_col="customer_id",
    origin_col="signup_date",
    interval_start_col="period_start",
    interval_end_col="period_end",
    event_col="churned",
    covariate_cols=["plan", "monthly_usage", "promo_active"],   # may change each interval
)
```

The interval start/stop *are* the canonical entry/exit tenures -- the schema extends the
one-row-per-subject model additively, so counting-process Kaplan-Meier and the business outputs all
keep working.

## Fit the time-varying Cox

```python
tv = tenure.TimeVaryingCox().fit(study)
print(tv.summary)                # covariate, coef, hazard_ratio, p_value
print(tv.risk_scores().head())   # per-interval, time-varying partial hazard ratio
```

## Survival for a covariate path

To get a survival curve for a *hypothetical* customer whose covariates follow a known path, build a
single-subject interval design and call `predict_survival`. The result is a
[`SurvivalFunction`](../reference/estimators.md) the business outputs consume directly:

```python
path = tenure.StudyDesign.from_intervals(
    one_customer_panel,
    id_col="customer_id", origin_col="signup_date",
    interval_start_col="period_start", interval_end_col="period_end",
    event_col="churned", covariate_cols=["plan", "monthly_usage", "promo_active"],
)
curve = tv.predict_survival(path)
print(tenure.rmst(curve, horizon=365))
```

Under the hood this integrates lifelines' Breslow baseline cumulative hazard along the path; a
constant path reduces exactly to the static-Cox curve.

## Landmarking: a simpler alternative

If you want a static model that still avoids immortal-time, [`landmark`](../reference/estimators.md)
is the lighter-weight route. It keeps only subjects still at risk at tenure `L`, fixes their
covariates to the value as of `L`, and returns a single-interval design (with delayed entry at `L`)
that `CoxPH` or `KaplanMeier` consume unchanged:

```python
landmarked = tenure.landmark(study, landmark_time=90)
cox = tenure.CoxPH().fit(landmarked)
```

This answers "given what I know about a customer at day 90, how do they retain from there?" without
the look-ahead that creates the bias.

## Calendar effects ride along

Because Tenure keeps tenure-time and calendar-time distinct, calendar covariates (seasonality,
macro events) can sit on the interval rows alongside the tenure-varying ones -- a dual clock -- with
no special handling.

## Next steps

- [Out-of-time validation](validation.md) -- prove the model generalizes forward in time.
- [The bias audit](../audit-catalog.md) -- how TNR004 prevention works.
