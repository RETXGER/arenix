"""CLI / export_json round-trip — mock orchestrator çıktısı geçerli JSON olmalı."""

import json
import os
import tempfile

import pytest

from arenix_engine import (
    Orchestrator,
    SessionConfig,
    AttackerRole,
    TargetRole,
    AnalyzerRole,
    ObserverRole,
    ContextTracker,
    ArenixAnalyzerV2,
    build_adapter,
    export_json,
)


@pytest.fixture()
def orchestrator_mock():
    adapter = build_adapter("mock", "mock-all")
    cfg = SessionConfig(
        session_id="pytest-export-session",
        industry="default",
        attacker_provider="mock",
        attacker_model="mock-a",
        target_provider="mock",
        target_model="mock-t",
        analyzer_provider="mock",
        analyzer_model="mock-z",
        observer_provider="mock",
        observer_model="mock-o",
        max_turns=2,
        stop_on_break=False,
        export_json_path="arenix_report.json",
    )
    attacker = AttackerRole(adapter, max_retries=2, profile="balanced", max_turns=cfg.max_turns)
    target = TargetRole(adapter, max_retries=2)
    tracker = ContextTracker()
    analyzer = AnalyzerRole(ArenixAnalyzerV2(industry="default"), tracker)
    observer = ObserverRole(adapter, max_retries=2)
    return Orchestrator(
        config=cfg,
        attacker=attacker,
        target=target,
        analyzer=analyzer,
        observer=observer,
        tracker=tracker,
    )


def test_export_json_writes_parseable_round_trip(orchestrator_mock):
    result = orchestrator_mock.run()
    assert "raw_report" in result
    assert "analysis_report" in result
    assert "session" in result

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "arenix_report_test.json")
        export_json(result, path)
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)

    assert "session" in loaded
    assert "analysis_report" in loaded
    assert loaded["session"]["session_id"] == "pytest-export-session"
    assert isinstance(loaded["analysis_report"], dict)
    assert "status" in loaded["analysis_report"]
    assert "raw_report" in loaded
    assert isinstance(loaded["raw_report"], dict)
