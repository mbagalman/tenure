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

```python
import tenure

# Synthetic SVOD data with a deliberate left-truncation trap baked in:
df = tenure.load_svod_demo(with_left_truncation=True)

# A naive study design -- old customers present, pre-entry churners excluded,
# delayed entry NOT modeled:
study = tenure.StudyDesign.from_event_dates(
    df,
    id_col="customer_id",
    origin_col="signup_date",
    churn_date_col="churn_date",
    active_as_of="2026-05-31",
    analysis_start="2024-01-01",
    includes_pre_entry_churners=False,
)

report = tenure.audit(study, strictness="warn")  # 'block' (default) would raise
print(report.to_markdown())
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
