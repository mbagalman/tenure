# Example gallery

Complete, runnable analyses -- each one a story with a beginning (a business question), a
middle (the trap the audit catches), and an end (the corrected number, checked against ground
truth where the data is synthetic).

Every code block on these pages is executed by the test suite on every commit, so the gallery
cannot silently rot.

| Example | What it teaches |
|---|---|
| [The $10 mistake](the-ltv-gap.md) | Left truncation: how a warehouse migration silently inflates LTV, and the audit escalation (warn -> attest -> block -> fix) that stops it. Synthetic data with known truth. |
| [A 3-year LTV from 2 years of data](three-year-ltv.md) | Principled projection: Kaplan-Meier truncates at the data's edge; parametric and hybrid curves reach the horizon -- and land within ~1% of the known truth. |
| [Real data: Telco churn](telco-churn.md) | The most famous churn dataset, ingested with its assumptions stated out loud: the tenure-to-dates recipe, the survivorship question the audit makes you answer, and the classic contract-type results. |

!!! note "Running the examples"
    The synthetic examples run as-is (`pip install` Tenure and paste). The Telco example needs
    one local file -- download instructions are on its page.
