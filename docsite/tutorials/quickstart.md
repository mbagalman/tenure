# Quickstart: the LTV gap

This tutorial shows, in under five minutes, why Tenure exists. We will reproduce the dollar gap a
naive churn analysis opens up when it mishandles left-truncation, then build the corrected analysis
by hand so you can see every step.

## 1. The headline demo

Tenure ships a deterministic demo on synthetic SVOD data (seed 0) with a known ground truth:

```python
import tenure

result = tenure.naive_vs_corrected_demo()
print(f"naive LTV:      ${result['naive_ltv']:.2f}")
print(f"corrected LTV:  ${result['corrected_ltv']:.2f}")
print(f"true LTV:       ${result['true_ltv']:.2f}")
print(f"over-statement: ${result['ltv_dollar_diff']:.2f} per customer")
```

```text
naive LTV:      $101.16
corrected LTV:  $90.81
true LTV:       $90.96
over-statement: $10.35 per customer
```

The naive analysis over-states lifetime value by **$10.35 per customer** -- not because the
estimator is wrong, but because the *study design* silently assumed older customers were observed
from signup when they were not. The corrected analysis lands within a few cents of the truth.

To see the explanation the audit produces:

```python
print(result["audit"].to_markdown())
```

This prints the TNR001 finding -- the left-truncation warning that explains the gap.

## 2. Build it yourself

Now the same analysis using the low-level pieces, so the workflow is concrete. First, load the
demo data (with deliberate left-truncation baked in) and declare an explicit study design:

```python
df = tenure.load_svod_demo(with_left_truncation=True)

study = tenure.StudyDesign.from_event_dates(
    df,
    id_col="customer_id",
    origin_col="signup_date",
    churn_date_col="churn_date",      # null churn date = still active
    active_as_of="2026-05-31",
    analysis_start="2024-01-01",
    event_observed_from="2024-01-01", # event recording starts here -> model delayed entry
    group_cols=["plan"],
)
```

The key line is `event_observed_from`. It tells Tenure that churn events were only *recorded*
starting 2024-01-01, so any customer who signed up earlier must enter the risk set with delayed
entry at the tenure they were first observed. Without it, the audit would **block**.

## 3. Audit before you trust a curve

```python
report = tenure.audit(study)          # raises AuditBlockedError on a blocking design
print(report.to_markdown())
```

Because we modeled delayed entry, the audit passes. Had we omitted `event_observed_from`, this call
would raise `AuditBlockedError` with a plain-language explanation of the left-truncation problem and
how to fix it. That is the whole point: the dangerous design is the one that does not run.

## 4. Fit and summarize

The audited design now flows into a Kaplan-Meier fit and the business outputs:

```python
km = tenure.KaplanMeier().fit(study, by="plan")
summary = tenure.summarize(
    km,
    period_margin=12.0,    # $/period contribution margin
    ltv_horizon=365.0,     # horizon in days
    audit_report=report,   # provenance travels with the output
)
print(summary.to_markdown())
```

The [`SummaryReport`](../reference/outputs.md) carries retention at standard horizons, RMST, and a
period-correct survival-weighted LTV -- each annotated with the audit verdict that produced it.

## What just happened

The naive path and the corrected path used the **same estimator**. The only difference was the
study design: whether older, late-observed customers entered the risk set correctly. That single
design decision moved LTV by 10%. Tenure's job is to make the correct decision the default and the
incorrect one hard to make by accident.

## Next steps

- [Retention and LTV](retention-and-ltv.md) -- the business outputs in depth.
- [Risk modeling with Cox](risk-modeling.md) -- move from cohort retention to per-customer risk.
- [The bias audit](../audit-catalog.md) -- the full TNR001-TNR005 catalog.
