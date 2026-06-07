"""NFR-SIM-1: across many seeds, the corrected pipeline's 365-day LTV is closer to the known
data-generating process than the naive pipeline's (aggregate MAE). Tracked benchmark.
"""

from __future__ import annotations

import warnings

import numpy as np

import tenure

_N_SEEDS = 50


def test_corrected_beats_naive_on_aggregate_ltv_mae():
    naive_errors = []
    corrected_errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(_N_SEEDS):
            df = tenure.load_svod_demo(with_left_truncation=True, seed=seed, n=1500)
            result = tenure.naive_vs_corrected_demo(df=df)
            true_ltv = result["true_ltv"]
            naive_errors.append(abs(result["naive_ltv"] - true_ltv))
            corrected_errors.append(abs(result["corrected_ltv"] - true_ltv))

    mae_naive = float(np.mean(naive_errors))
    mae_corrected = float(np.mean(corrected_errors))

    assert mae_corrected < mae_naive
    # The naive bias is material, not noise-level -- the whole point of the library.
    assert mae_naive > 3.0
