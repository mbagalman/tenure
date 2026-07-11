# Tenure

**Audit-first survival analysis for B2C customer churn.**

Tenure's thesis: the hard, value-adding part of churn survival analysis is *not* the
estimators -- [`lifelines`](https://lifelines.readthedocs.io/) already nails those. It is getting
the **study design** right. Tenure makes the statistically correct design the default and makes
biased designs hard to produce by accident, via a plain-language **study-design audit** that runs
*before* any number is returned.

!!! note "Status: v0.4 (alpha)"
    The public API is settling; minor changes are still possible before v1.0. The distribution
    name on PyPI is not yet final.

## The 30-second pitch

`lifelines` gives you correct *estimators* and assumes you have already built a statistically
valid risk set. In practice that assumption is where most business churn analyses quietly go
wrong: left-truncation inflates retention and LTV, window-as-origin and immortal-time bias
fabricate effects, informative censoring skews curves. Tenure wraps lifelines for the math and
adds the layer it is missing -- a [**study-design audit**](audit-catalog.md) that makes the correct
design the default and the biased one hard to produce by accident, plus business outputs
(retention %, RMST, LTV $) that carry their audit caveats.

```python
import tenure

# The headline: dollars a naive analysis over-states by mishandling left-truncation.
result = tenure.naive_vs_corrected_demo()
print(f"naive LTV:      ${result['naive_ltv']:.2f}")     # $101.16  (over-stated)
print(f"corrected LTV:  ${result['corrected_ltv']:.2f}")  # $90.81   (close to truth)
print(f"true LTV:       ${result['true_ltv']:.2f}")        # $90.96
```

## Where to go next

<div class="grid cards" markdown>

-   __Why Tenure?__

    The bias problem in business churn data, and how the audit-first design addresses it.

    [:octicons-arrow-right-24: Why Tenure](why-tenure.md)

-   __Install it__

    Pip install and the supported Python versions.

    [:octicons-arrow-right-24: Installation](installation.md)

-   __Quickstart__

    Run the naive-vs-corrected demo and read your first audit report.

    [:octicons-arrow-right-24: Quickstart](tutorials/quickstart.md)

-   __The bias audit__

    The TNR001-TNR005 check catalog -- what each catches and how to clear it.

    [:octicons-arrow-right-24: Audit catalog](audit-catalog.md)

</div>

## What you can do with it

| Capability | Entry points |
|---|---|
| Define a churn study with explicit semantics | [`StudyDesign.from_event_dates`](reference/study-design.md), `from_status`, `from_intervals` |
| Audit the design for bias before fitting | [`audit`](reference/audit.md) |
| Kaplan-Meier retention curves | [`KaplanMeier`](reference/estimators.md), [`plot_survival`](reference/plotting.md) |
| Business outputs (retention %, RMST, LTV $) | [`retention_at`, `rmst`, `survival_weighted_ltv`, `summarize`](reference/outputs.md) |
| Cox risk modeling + per-customer scoring | [`CoxPH`](reference/estimators.md), [`churn_risk_scores`](reference/outputs.md) |
| Time-varying covariates (immortal-time *prevented*) | [`TimeVaryingCox`, `landmark`](reference/estimators.md) |
| Out-of-time validation (C-index, Brier/IBS, calibration) | [`temporal_holdout`, `concordance`, `brier`, `calibration`](reference/validation.md) |

## License

[MIT](https://github.com/mbagalman/tenure/blob/main/LICENSE).
