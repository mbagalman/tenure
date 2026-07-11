# Scope: when to use Tenure (and when not to)

Tenure is deliberately focused. Using it outside its scope will not error -- it will hand you a
confident, wrong number, which is exactly the failure mode the library exists to prevent. So the
single most important thing to get right before you start is whether your churn problem is the kind
Tenure models.

## The one question that decides it

**Is there an observable, unambiguous moment the customer relationship ends?**

This is the *contractual vs. non-contractual* distinction from the survival-analysis and customer-lifetime-value
literature, and it -- not billing cadence, industry, or data volume -- is what determines whether
Tenure is the right tool.

- **Yes, there is a real end event** (a cancellation, a non-renewal, an account closure, a
  disconnect): your churn is **contractual**. Tenure is built for this.
- **No -- the customer simply goes quiet and might return** (no purchase in 90 days, an idle app):
  your churn is **non-contractual / latent attrition**. Tenure does not model this yet; forcing it
  to will bias your results. See [below](#not-yet-non-contractual-latent-attrition).

## In scope: contractual subscription churn

Tenure's beachhead is **streaming/SVOD** (the reference case: explicit cancel/non-renew events,
monthly billing, rich time-varying engagement). But nothing in the library is streaming-specific.
The design semantics are calendar-date based, not tied to a billing cycle, so any business with a
recorded end-of-relationship event fits:

| Business | Why it fits |
|---|---|
| Streaming / SVOD | Explicit cancel or non-renewal event; the reference vertical. |
| SaaS subscriptions | Cancellation date is recorded and unambiguous. |
| Telecom / broadband | Disconnect / port-out is a real, dated event. |
| Memberships, insurance, gym, box subscriptions | Lapse or non-renewal is observed. |
| Banking / brokerage accounts | Account closure is a genuine, dated end -- even though start and end dates are irregular. |

Irregular start and end dates are **not** a problem: `StudyDesign` takes arbitrary origin and exit
dates and never assumes a fixed cycle. A bank account that opens on a random Tuesday and closes
years later is as valid an input as a monthly SVOD subscription.

## Not yet: non-contractual / latent attrition

If your only signal that a customer left is **absence of activity**, Tenure is the wrong tool
today. There is no true "churn event" -- only silence -- and the customer can always come back.

Common examples:

- **Retail / grocery** -- "we consider you lost after 90 days without a purchase."
- **E-commerce** -- inferred lapse from order recency.
- **Free / ad-supported apps and gaming** -- inactivity, not cancellation.

!!! danger "Do not fabricate a churn date from an inactivity rule"
    You *can* feed a "90 days idle = churned on day 90" date into `from_event_dates`, and it will
    run. Do not. Survival estimators (Kaplan-Meier, Cox) assume the event is a genuine, permanent,
    unambiguous state transition. An inferred cutoff invents a hard boundary the math does not
    support, baking in bias -- and Tenure's audit does **not** yet catch this misuse, because every
    check assumes you are already in the contractual setting.

The correct tools for latent attrition are **latent-dropout models** -- BG/NBD, Pareto/NBD, and
Gamma-Gamma for the monetary side. These are on the [roadmap](https://github.com/mbagalman/tenure)
as post-v1.0 and feed the same LTV/reporting surface when they land.

## The three cases, side by side

A useful way to place your own problem:

1. **Streaming with a clear renewal/cancel cadence** (Starz, Netflix) -- *fully supported*, the
   reference case.
2. **Subscription that starts and ends at any time** (a bank account, a SaaS seat) -- *fully
   supported*; irregular dates are fine because churn is still a real, observed event.
3. **Any business that only infers loss from inactivity** (a grocer's 90-day rule) -- *not
   supported yet*; this is non-contractual, and belongs to the post-v1.0 latent-attrition models.

## Other current limitations

Even within contractual churn, a few things are intentionally out of scope for now:

- **DataFrame-pure.** Tenure takes a pandas DataFrame and returns tidy, backend-neutral objects.
  You own reading and writing your data; there are no warehouse connectors.
- **No competing-risks or multi-state *estimators* yet.** The interval data model can *represent*
  repeat spells and multiple exit types, but cause-specific / Fine-Gray / multi-state models are
  post-v1.0. Today, non-churn exits are handled by explicit event/censored/exclude mapping and the
  [TNR003](audit-catalog.md) warning, not by a competing-risks model.
- **Contractual assumption throughout.** Every audit check and estimator assumes churn is directly
  observed. That assumption is the boundary of this page.

If your problem is contractual, head to the [quickstart](tutorials/quickstart.md). If it is not,
Tenure is not your tool yet -- and it is better to know that before you have a chart.
