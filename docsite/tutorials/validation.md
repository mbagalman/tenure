# Out-of-time validation

A model that fits the past can still fail on the future. v0.4 evaluates predictions the **honest**
way -- out-of-time -- and makes the biased way (a random split that leaks future information) hard
to do by accident. Validation is a separate layer that operates over predictions plus a held-out
design; it never reaches into estimator internals.

## Setup

Validation needs a study whose customers span the calendar cutoff -- some active before it, with
outcomes observed after it. We use the full historical cohort (with delayed entry modeled) and
declare covariates so a Cox model has something to fit:

```python
import tenure

df = tenure.load_svod_demo(with_left_truncation=True)
study = tenure.StudyDesign.from_event_dates(
    df, id_col="customer_id", origin_col="signup_date", churn_date_col="churn_date",
    active_as_of="2026-05-31", analysis_start="2024-01-01",
    event_observed_from="2024-01-01", covariate_cols=["plan", "channel"],
)
tenure.audit(study)
```

## The temporal holdout

[`temporal_holdout`](../reference/validation.md) splits at a calendar cutoff. Training is censored
at the cutoff (no post-cutoff event can leak into it), and the test cohort is the set of customers
still active at the cutoff, scored on what actually happened *after* it -- on the evaluation clock:

```python
train, test = tenure.temporal_holdout(study, cutoff="2025-01-01")
cox = tenure.CoxPH().fit(train)
```

The cutoff must fall inside the cohort's active span: pick a date by which some customers have
signed up and are still at risk, with enough follow-up remaining after it to observe outcomes. A
cutoff before everyone's entry (or after everyone has already exited) leaves nothing to train or
test on and raises a clear error.

!!! warning "Why not a random split?"
    `tenure.random_split(study)` exists, but it **warns** ([VAL001](../reference/validation.md)).
    Splitting a survival panel at random lets the model see a customer's future when predicting
    their past. For churn, validation must be forward-in-time.

## Discrimination: the concordance index

Do higher-risk customers actually churn sooner? The C-index answers that
(`0.5` = chance, `1.0` = perfect):

```python
print(tenure.concordance(cox, test).estimate)
```

C-index wraps lifelines' Harrell concordance on the evaluation clock, oriented so higher predicted
risk corresponds to shorter survival.

## Accuracy: time-dependent Brier score and IBS

The Brier score measures squared error between predicted survival and observed outcome at a set of
times; the Integrated Brier Score (IBS) summarizes it across a horizon (lower is better). Tenure
uses inverse-probability-of-censoring weighting (IPCW) so right-censoring does not bias the score:

```python
print(tenure.brier(cox, test, [30, 60, 90]).table)
print(tenure.integrated_brier(cox, test, [30, 60, 90]).estimate)
```

!!! note "Time grids and extrapolation"
    `times` must be a positive, strictly increasing grid. Validating a tenure-clock model
    out-of-time always extrapolates a little (the oldest active customer sits at the edge of
    support at the cutoff), so `n_extrapolated` / `pct_extrapolated` are always recorded in
    `.metadata`; the [VAL002](../reference/validation.md) warning only fires when a material
    fraction of scored cells are beyond the fitted model's support.

## Calibration: predicted vs observed

Calibration asks whether a predicted 70% survival really corresponds to 70% observed survival.
[`calibration`](../reference/validation.md) bins customers by predicted survival at a horizon and
compares each bin against the Kaplan-Meier-observed survival (censoring-correct):

```python
cal = tenure.calibration(cox, test, horizon=90)
print(cal.metadata["calibration_error"])   # support-weighted |predicted - observed|
tenure.plot_calibration(cal)                # reliability diagram vs the diagonal
```

## Everything carries its provenance

Every metric returns a [`ValidationResult`](../reference/validation.md): a tidy `.table` plus
`.metadata` recording the prediction time, model type, train/test sizes, censoring method, and any
`VAL00x` support warnings. A validation report travels with the context needed to trust it.

## Model support

- **C-index** works for Cox-family models (partial hazard on covariates as of the cutoff) and for
  overall survival curves / Kaplan-Meier (where it reduces to chance, by construction).
- **Brier / IBS** and **calibration** support `CoxPH` and overall survival curves today.
  Time-varying-Cox accuracy metrics are a later addition.

The C-index wraps lifelines; the Brier score and IBS are implemented directly to keep the core
dependency-light (no compiled extras).

## Next steps

- [The bias audit](../audit-catalog.md) -- the design-time checks that precede all of this.
- [API reference: Validation](../reference/validation.md) -- full signatures.
