# Risk modeling with Cox

Kaplan-Meier answers "how is the cohort retaining?" Cox proportional hazards answers "**which**
customers are at risk, and **why**?" You declare covariates on the study design, and Cox plugs into
the same outputs as Kaplan-Meier.

## Fit a Cox model

Declare covariates with `covariate_cols`; everything else about the design is unchanged:

```python
import tenure

df = tenure.load_svod_demo(with_left_truncation=False)
study = tenure.StudyDesign.from_event_dates(
    df, id_col="customer_id", origin_col="signup_date", churn_date_col="churn_date",
    active_as_of="2026-05-31",
    covariate_cols=["plan", "channel"],
)
tenure.audit(study)

cox = tenure.CoxPH().fit(study)
```

## Survival curves at covariate profiles

A fitted Cox model produces a survival curve for any covariate profile, and those curves are the
same [`SurvivalFunction`](../reference/estimators.md) the business outputs consume. `profile_grid`
builds a tidy grid varying one covariate while holding the rest at reference values:

```python
curves = cox.predict_survival(cox.profile_grid("plan"))
print(tenure.retention_at(curves, [90, 365]))
print(tenure.rmst(curves, horizon=365))
```

This is the [A3 estimator interface](../why-tenure.md) at work: N independent Kaplan-Meier fits and
one Cox model evaluated at several profiles present an identical interface to `retention_at`,
`rmst`, and `survival_weighted_ltv`.

## Per-customer churn-risk scores

[`churn_risk_scores`](../reference/outputs.md) scores each customer in a cohort with the fitted
model:

```python
scores = tenure.churn_risk_scores(cox, horizon=365.0)
print(scores.table.sort_values("risk_score", ascending=False).head())
```

The tidy `.table` has, per customer: `risk_score` (the Cox partial hazard ratio `exp(beta^T X)`,
higher = riskier), `survival_at_horizon` (predicted survival probability at the horizon), and
`risk_percentile` (rank within the cohort, in `[0, 1]`). To score a *fresh* cohort rather than the
training data, pass it as the `design` argument.

## Check the proportional-hazards assumption

Cox assumes hazards are proportional over time. Tenure gives you the diagnostic directly:

```python
diag = cox.proportional_hazards_test()   # Schoenfeld-residual test; warns if violated
print(diag.ok, diag.violations)
print(diag.table)                         # tidy [covariate, test_statistic, p_value, status]
```

For a visual check, the log-log survival plot should show roughly parallel curves under
proportional hazards:

```python
km_by_plan = tenure.KaplanMeier().fit(study, by="plan")
tenure.plot_log_log_survival(km_by_plan)
```

## When PH fails: the stratified Cox

If the test flags a categorical covariate, the standard remedy is to **stratify** on it: each
stratum gets its own baseline hazard (no proportionality assumed between them), while the other
covariates still share one set of coefficients. Refit the *same* design with one argument:

```python
strat = tenure.CoxPH(strata=["plan"]).fit(study)
strat.proportional_hazards_test()        # plan is gone from the test -- nothing left to violate
curves = strat.predict_survival(strat.profile_grid("plan"))   # per-stratum baselines
```

The stratified covariate no longer has a coefficient (you give up its hazard ratio -- that is the
trade), but predictions, [`churn_risk_scores`](../reference/outputs.md), and every business output
work unchanged, and each profile's curve now uses its own stratum's baseline hazard. For a
*numeric* violator, bin it into a categorical column first, or move to a
[time-varying design](time-varying.md).

## Nelson-Aalen cumulative hazard

The cumulative-hazard analogue of Kaplan-Meier is also available:

```python
na = tenure.NelsonAalen().fit(study, by="plan")
print(na.cumulative_hazard_at([90, 365]))
tenure.plot_cumulative_hazard(na)
```

## Next steps

- [Time-varying covariates](time-varying.md) -- when risk drivers change over the lifetime.
- [Out-of-time validation](validation.md) -- C-index and Brier score for the Cox model.
