"""print_summary konsolda cp1254 (strict) altında UnicodeEncodeError vermesin."""

import io

import pytest

from arenix_engine import print_summary


def _minimal_summary_payload(**kwargs):
    trend = kwargs.get("trend")
    return {
        "session": {
            "session_id": "test-session",
            "industry": "default",
            "enable_semantic_analysis": False,
            "export_json_path": "out.json",
        },
        "turn_records": [
            {
                "turn_id": 1,
                "attack_pressure": 10.0,
                "compromise_score": 20.0,
                "resilience_score": 80.0,
                "status": "SAFE",
                "observer_confirmed_break": False,
            }
        ],
        "analysis_report": {
            "status": "SAFE",
            "attack_detected": False,
            "model_compromised": False,
            "model_under_pressure": False,
            "total_turns": 1,
            "overall_attack_pressure": 10.0,
            "overall_compromise_score": 20.0,
            "max_compromise_score": 20.0,
            "average_resilience": 80.0,
            "confidence_score": 90.0,
            "vulnerability_level": "low",
            "total_latency_ms": 0,
            "total_tokens": 0,
            "break_point": None,
            "vulnerabilities_found": ["Ascii only note"],
            "recommendations": ["Ascii only rec"],
            "trend": trend,
        },
    }


def test_safe_console_print_fallback_on_strict_cp1254(monkeypatch):
    """Stdout cp1254 + errors strict iken emoji basımı yakalanır, replace ile yazılır."""
    import arenix_engine as ae

    class ExplodingStdout:
        encoding = "cp1254"

        def flush(self):  # noqa: ANN001
            pass

        def write(self, s: str, _called=[]):  # noqa: ANN001
            if "\U0001f525" in s or "\u2705" in s or "\u274c" in s:
                raise UnicodeEncodeError("cp1254", s, 0, 1, "emoji")
            # strict cp1254 path for ascii-only remainder
            s.encode(self.encoding)

    monkeypatch.setattr("sys.stdout", ExplodingStdout())

    ae._safe_console_print("\U0001f525 smoke")
    ae._safe_console_print("ASCII only")


def test_print_summary_no_error_with_strict_cp1254_stdout(monkeypatch):
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1254", newline="", errors="strict")
    monkeypatch.setattr("sys.stdout", stream)

    print_summary(_minimal_summary_payload())
    stream.flush()
    out = buf.getvalue()
    assert b"ARENIX" in out or b"SECURITY" in out


def test_print_summary_trend_branch_strict_cp1254(monkeypatch):
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1254", newline="", errors="strict")
    monkeypatch.setattr("sys.stdout", stream)

    print_summary(
        _minimal_summary_payload(
            trend={
                "attack_trend": "stable",
                "defense_trend": "stable",
                "compromise_velocity": 0.5,
                "risk_acceleration": 0.0,
                "estimated_break_turn": None,
            }
        )
    )
    stream.flush()
    assert len(buf.getvalue()) > 0
