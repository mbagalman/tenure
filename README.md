# Tenure

**Audit-first survival analysis for B2C customer churn.**

Tenure's thesis: the hard, value-adding part of churn survival analysis is *not* the
estimators -- `lifelines` already nails those. It is getting the **study design** right.
Tenure makes the statistically correct design the default and makes biased designs hard
to produce by accident, via a plain-language **study-design audit** that runs *before*
any number is returned.

> **Status: pre-alpha (Phase 0 skeleton).** Not yet released. The distribution name on
> PyPI is not final. APIs will change.

## What the audit catches (v0.1, in progress)

- **TNR001 -- Left-truncation / delayed entry.** The subtle one: having an older
  customer's record is not enough; if your event history does not reach back to their
  origin (a "Window-Cut" study, e.g. a billing-system migration), they must be modeled
  with delayed entry or your retention and LTV are biased upward.
- More checks (event/censoring mislabeling, time-origin confusion, immortal-time,
  horizon support) land across the v0.1 milestones.

## Quickstart (preview)

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

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## License

MIT
