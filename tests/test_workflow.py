from __future__ import annotations

import pandas as pd
import pytest

import tenure

_COMMON = dict(
    id_col="customer_id",
    origin_col="signup_date",
    churn_date_col="churn_date",
    active_as_of="2026-05-31",
    analysis_start="2024-01-01",
)


def _clean_study(**kwargs):
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    return tenure.RetentionStudy.from_event_dates(df, group_cols=["plan"], **_COMMON, **kwargs)


def _naive_study(strictness):
    df = tenure.load_svod_demo(with_left_truncation=True, seed=2)
    return tenure.RetentionStudy.from_event_dates(
        df, includes_pre_entry_churners=False, strictness=strictness, **_COMMON
    )


def test_run_returns_bundled_result():
    result = _clean_study().run()
    assert result.audit.clean
    assert set(result.curves.groups) == {"basic", "standard", "premium"}
    summary = result.summary(period_margin=12.0, ltv_horizon=365.0)
    assert "ltv" in summary.table.columns


def test_audit_runs_before_numbers_and_blocks():
    with pytest.raises(tenure.AuditBlockedError):
        _naive_study("block").run()


def test_warn_strictness_emits_warning_and_proceeds():
    with pytest.warns(UserWarning, match="TNR001"):
        result = _naive_study("warn").run()
    assert any(r.check_id == "TNR001" for r in result.audit.warnings)
    assert result.curves.groups  # numbers were still produced


def test_tiers_agree():  # AC-10 / NFR-API-1
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    result = tenure.RetentionStudy.from_event_dates(df, group_cols=["plan"], **_COMMON).run()
    workflow_ltv = (
        result.ltv(period_margin=12.0, horizon=365.0).sort_values("group").reset_index(drop=True)
    )
    design = tenure.StudyDesign.from_event_dates(df, group_cols=["plan"], **_COMMON)
    km = tenure.KaplanMeier().fit(design, by="plan")
    primitive_ltv = (
        tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0)
        .sort_values("group")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(workflow_ltv, primitive_ltv)


def test_no_stdout_writes(capsys):
    _clean_study().run().summary(period_margin=12.0, ltv_horizon=365.0)
    assert capsys.readouterr().out == ""


def test_summary_carries_audit_provenance():
    summary = _clean_study().run().summary(period_margin=12.0, ltv_horizon=365.0)
    assert summary.metadata["audit_verdict"] == "clean (no findings)"


def test_from_status_workflow_end_to_end():
    df = pd.DataFrame(
        {
            "cid": ["a", "b", "c"],
            "start": ["2025-01-01"] * 3,
            "exit": ["2025-03-01", "2025-06-01", "2025-06-01"],
            "status": ["churn", "active", "active"],
        }
    )
    study = tenure.RetentionStudy.from_status(
        df,
        id_col="cid",
        origin_col="start",
        exit_col="exit",
        status_col="status",
        status_map={"churn": "event", "active": "censored"},
        active_as_of="2025-06-01",
    )
    result = study.run()
    assert result.audit.clean
    assert result.curves.groups == ["overall"]
