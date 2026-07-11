# Why Tenure?

## The problem is the study design, not the estimator

Survival analysis has excellent open-source estimators. [`lifelines`](https://lifelines.readthedocs.io/),
`scikit-survival`, and `lifetimes` all implement Kaplan-Meier, Cox proportional hazards, and the
rest correctly. If your only problem is "fit a Kaplan-Meier curve to a clean risk set," those
libraries are complete and Tenure has nothing to add.

But in business churn analysis the risk set is almost never clean to begin with, and *building*
it correctly -- the study design -- is the error-prone part those libraries leave entirely to you.
The math is right; the inputs are quietly wrong; the output looks plausible. That is the worst
failure mode, because nothing complains.

## The biases that hide in business churn data

Four design mistakes recur in real subscription data. Each produces a confident, wrong number.

### Left-truncation / delayed entry

Your event log often does not reach back to every customer's true signup. A billing-system
migration, a data-warehouse cutover, an analytics tool installed last year -- any of these means
older customers were only *observed* starting from some later date. If you treat their full tenure
as observed, you implicitly assume they could have churned during a period you never watched. They
could not, because if they had, they would not be in your data. This **inflates retention and LTV
upward**: the survivors are over-represented.

The fix is delayed entry (left-truncation handling): each such customer enters the risk set at the
tenure they were first observed, not at tenure zero. Tenure's audit ([TNR001](audit-catalog.md))
detects the window-cut pattern and requires you to model it.

### Time-origin confusion

The clock for churn survival starts at the customer's **signup**, not at the start of your
observation window. Using the window start as `t = 0` mislabels everyone's tenure and conflates
calendar time with tenure time. Tenure keeps origin, calendar time, and tenure time as distinct
concepts throughout ([TNR002](audit-catalog.md)).

### Event / censoring mislabeling

A customer who *upgraded* and one who *cancelled* are not the same exit. If an upgrade or a
forced migration is silently labeled as "churned," you over-count churn; if it is labeled
"censored" without thought, you assume that exit is independent of churn risk (the
independent-censoring assumption), which is often false and biases the curve. Tenure forces you to
declare the intent of every exit status and warns when a non-churn exit is censored
([TNR003](audit-catalog.md)).

### Immortal-time bias

Classifying customers by a *future-looking* attribute -- "ever upgraded," "ever contacted
support" -- credits the eventual-upgraders with all the person-time *before* they upgraded, during
which they could not yet have the attribute and (often) could not yet have churned. This fabricates
a protective effect out of nothing. Tenure *warns* about the signature in a static design
([TNR004](audit-catalog.md)) and *structurally prevents* it once you move to a
[time-varying design](tutorials/time-varying.md), where the attribute is encoded `0` before the
event and `1` after.

## What Tenure adds

Tenure is a **hybrid** library by deliberate design (see the project's architecture decisions):

- It **wraps** the mature estimators (`lifelines`) rather than reinventing them. The math is not
  the value.
- It **builds the gap**: explicit study-design semantics, a pluggable bias audit with stable
  public check IDs, period-correct LTV, and out-of-time validation.

The audit is the hero feature. It runs *before* any number is returned, speaks plain language,
blocks the dangerous designs by default, and -- importantly -- stays quiet on a correctly designed
cohort. That last property ("no crying wolf") is a tested guarantee: a clean cohort returns
all-pass, zero warnings.

## What Tenure is not (yet)

Tenure's beachhead is **contractual subscription businesses** (streaming/SVOD-like), where churn is
observed directly. It is intentionally focused:

- It does **not** yet do non-contractual / latent-attrition models (BG/NBD, Pareto/NBD) -- those
  are post-v1.0.
- It does **not** include competing-risks or multi-state estimators yet (the data model can
  represent them; the estimators come later).
- It is **DataFrame-pure**: pandas in, you own your I/O. No warehouse connectors.

See the [installation guide](installation.md) to get started, or jump to the
[quickstart](tutorials/quickstart.md).
