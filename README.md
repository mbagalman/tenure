# Tenure

**Audit-first survival analysis for B2C customer churn.**

Tenure's thesis: the hard, value-adding part of churn survival analysis is *not* the
estimators -- `lifelines` already nails those. It is getting the **study design** right.
Tenure makes the statistically correct design the default and makes biased designs hard
to produce by accident, via a plain-language **study-design audit** that runs *before*
any number is returned.

> **Status: v0.1 (alpha).** The public API is settling; minor changes are still possible. The
> distribution name on PyPI is not final.

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
| **TNR004** | Immortal-time -- a covariate level that only appears for higher-tenure customers (a data-driven quantile shift test). | warn |
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

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## License

MIT
