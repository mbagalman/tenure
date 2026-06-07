"""NFR-PERF-1: performance is a tracked benchmark, not a release gate.

A small moderate-size sanity runs by default (catches gross regressions / crashes on bigger
data). The full 1e5 / 1e6 benchmarks print timings and run only when TENURE_PERF is set.
"""

from __future__ import annotations

import os
import time

import pytest

import tenure


def _fit_and_ltv(n: int, seed: int = 0):
    df = tenure.load_svod_demo(with_left_truncation=True, seed=seed, n=n)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    km = tenure.KaplanMeier().fit(design)
    return tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0)


def test_moderate_size_completes():
    assert _fit_and_ltv(20_000).iloc[0]["ltv"] > 0


@pytest.mark.skipif(
    not os.environ.get("TENURE_PERF"), reason="set TENURE_PERF=1 to run the perf benchmark"
)
@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_perf_benchmark(n):
    start = time.perf_counter()
    out = _fit_and_ltv(n)
    elapsed = time.perf_counter() - start
    print(f"\n[perf] fit + LTV on n={n:,}: {elapsed:.2f}s")
    assert out.iloc[0]["ltv"] > 0
