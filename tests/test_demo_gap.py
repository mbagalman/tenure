"""The headline demo: naive vs corrected LTV gap from mishandled left-truncation (AC-3a)."""

from __future__ import annotations

import numpy as np

import tenure

# Deterministic for seed 0 (PCG64 stream + lifelines are platform-stable). This fixed gap is
# the regression gate (NFR-CORR-3): a real change to the pipeline shifts it.
EXPECTED_DOLLAR_GAP = 10.349240


def test_dollar_gap_is_the_fixed_anchor():
    result = tenure.naive_vs_corrected_demo()
    assert np.isclose(result["ltv_dollar_diff"], EXPECTED_DOLLAR_GAP, atol=1e-4)


def test_naive_overstates_ltv():
    result = tenure.naive_vs_corrected_demo()
    assert result["naive_ltv"] > result["corrected_ltv"]
    assert result["ltv_dollar_diff"] > 5.0  # a material, not noise-level, gap


def test_corrected_recovers_ground_truth():
    result = tenure.naive_vs_corrected_demo()
    true_ltv = result["true_ltv"]
    # The corrected pipeline is far closer to truth than the naive one...
    assert abs(result["corrected_ltv"] - true_ltv) < abs(result["naive_ltv"] - true_ltv)
    # ...and within a couple percent of the closed-form value.
    assert abs(result["corrected_ltv"] - true_ltv) / true_ltv < 0.02


def test_naive_design_audit_flags_left_truncation():
    result = tenure.naive_vs_corrected_demo()
    assert any(r.check_id == "TNR001" for r in result["audit"].warnings)


def test_demo_is_deterministic():
    a = tenure.naive_vs_corrected_demo()
    b = tenure.naive_vs_corrected_demo()
    assert a["ltv_dollar_diff"] == b["ltv_dollar_diff"]


def test_demo_accepts_explicit_dataframe():
    df = tenure.load_svod_demo(with_left_truncation=True, seed=0)
    result = tenure.naive_vs_corrected_demo(df=df)
    assert np.isclose(result["ltv_dollar_diff"], EXPECTED_DOLLAR_GAP, atol=1e-4)
