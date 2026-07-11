# The bias audit

The audit is Tenure's hero feature. You build a [`StudyDesign`](reference/study-design.md), then
call [`audit`](reference/audit.md) on it *before* fitting anything. The audit runs a registry of
checks with **stable, versioned public IDs** and returns a human-readable
[`AuditReport`](reference/audit.md) classifying the design as block / warn / pass.

```python
report = tenure.audit(study)          # raises AuditBlockedError on a blocking design (default)
print(report.to_markdown())            # plain-language explanation of every check
```

## How strictness works

By default the three success-criterion biases **block**: a blocking design raises
`AuditBlockedError` and cannot be fit, so a bias cannot slip through silently. You have two
deliberate escape hatches, both explicit:

- **`strictness="warn"`** on the audit call downgrades blocks to warnings. When you bypass a block
  this way, any chart you produce is stamped with a caveat note so the caveat travels into decks
  (`FR-RC-7`).
- **An attestation** on the `StudyDesign` constructor clears a specific check when you *know* the
  design is genuinely fine (for example, a legitimately late covariate is not immortal-time). This
  is how Tenure avoids "crying wolf" without letting you ignore a real problem by accident.

A correctly designed cohort returns **all-pass with zero warnings**. That is a tested guarantee,
not an aspiration.

## The checks

TNR001-TNR004 are *design-time* checks run by `audit(study)` before any fit. TNR005 is an
*output-time* guard applied during `summarize` / `rmst` / `survival_weighted_ltv` -- it catches
horizons that outrun the data's support at the moment you compute a number.

| Check | Bias | Default |
|---|---|---|
| **TNR001** | Left-truncation / delayed entry | block |
| **TNR002** | Time-origin / observation-window confusion | block |
| **TNR003** | Event / censoring mislabeling (+ informative censoring) | block / warn |
| **TNR004** | Immortal-time / future-looking covariate | warn |
| **TNR005** | Weak / over-extrapolated horizon | output-time; warn |

### TNR001 -- Left-truncation / delayed entry

**What it catches.** Event history that does not reach back to a customer's origin. Tenure
distinguishes two study types:

- *Full Historical Cohort* -- event history is complete back to every customer's signup. Fine; no
  delayed entry needed.
- *Window-Cut* -- event recording begins at some later date (`event_observed_from`, e.g. a billing
  migration). Older customers were observed late and **must** enter the risk set with delayed entry
  at tenure `event_observed_from - origin`, or retention and LTV are biased upward.

The check keys on observation completeness, **not** a naive `origin < analysis_start` date
comparison.

**How to clear it.** Pass `event_observed_from=` so delayed entry is modeled. The coarse fallback
attestation is `includes_pre_entry_churners=`. See the
[quickstart](tutorials/quickstart.md) for the worked example.

### TNR002 -- Time-origin confusion

**What it catches.** Using the observation-window start as `t = 0` instead of the customer's true
signup. Detecting this requires a real origin column; absent one, it falls back to an attestation.

**How to clear it.** Provide a genuine `origin_col`. If your origin truly is correct in a setup the
check cannot see, attest with `attest_origin_correct=True`.

### TNR003 -- Event / censoring mislabeling

**What it catches.** Two things:

1. **(block)** Any exit status not explicitly mapped to an intent. With `from_status`, you supply a
   `status_map` declaring each status as `event`, `censored`, or `exclude`. Unmapped statuses block
   until you decide.
2. **(warn)** A non-churn exit (upgrade, forced migration) mapped to `censored`. This invokes the
   independent-censoring assumption, which is often false here and biases retention pessimistically.

**How to clear it.** Map every status intent. The warning is informational -- heed it or document
why the censoring is independent.

### TNR004 -- Immortal-time / future-looking covariate

**What it catches.** A covariate level that only appears for higher-tenure customers (a data-driven
quantile-shift heuristic on `group_cols` / `covariate_cols`): if `min(tenure | X=1)` sits well
above zero, early churners could not have had `X=1`, which is the signature of immortal-time. The
language is "consistent with" -- it is a heuristic, not a verdict.

**How to clear it.**

- If the attribute is genuinely a late-but-origin-legitimate fact (e.g. an annual plan that can
  only exist after month one), attest with `attest_invariant_covariates=["annual_plan"]`.
- **Better:** move to a [time-varying / interval design](tutorials/time-varying.md). On an interval
  design TNR004 short-circuits to **pass** -- the bias is structurally prevented because the
  attribute is encoded `0` before it occurs and `1` after, so it cannot leak future survival. This
  is prevention, not warning.

This check warns only for single-spell designs.

### TNR005 -- Weak / over-extrapolated horizon

**What it catches.** Reading a retention/RMST/LTV number at a horizon that outruns the data's
support -- past the last event time, or where the at-risk count is too thin. This is an
*output-time* guard, not a design-time check, because it depends on the horizon you ask for.

**How it behaves.** Rather than silently reading a flat Kaplan-Meier tail, RMST and LTV are
**truncated-and-relabeled** to a per-group effective horizon, and the result is flagged
(`truncated=True`, `supported=False`). You always know which numbers are extrapolated.

## Related guards

- **Duplicate customer IDs** (win-backs) violate Kaplan-Meier's independence assumption and
  **block by default** (`FR-SD-7`). Pass a per-spell unique key as `id_col`, or choose an explicit
  `dedup_policy` (`"keep-first"` is unbiased; `"keep-most-recent"` warns -- it drops earlier spells,
  a selection bias). Recurrent events are represented by the
  [interval schema](tutorials/time-varying.md).
- **Period-correct LTV** (`FR-BO-3`): LTV reconciles the survival time unit against the margin
  period, so you cannot accidentally multiply daily survival by a monthly margin.

## Adding your own checks

The audit is a pluggable registry: a new check is a new module plus a `@register` decoration, with
no edits to the core. Check IDs are a public contract and are versioned. See the
[audit API reference](reference/audit.md).
