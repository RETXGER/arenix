import pytest


api = pytest.importorskip("api")
if not getattr(api, "HAS_FASTAPI", False):
    pytest.skip("FastAPI is not installed in this environment", allow_module_level=True)


def test_sanitize_error_message_redacts_known_key_formats():
    raw = (
        "openai=sk-abcDEF1234567890XYZ "
        "anthropic=sk-ant-abcdef1234567890 "
        "gemini=AIzaSyAABBCCDDEEFF001122334455 "
        "auth=Bearer super-secret-token-123456"
    )
    sanitized = api._sanitize_error_message(raw)

    assert "sk-abcDEF1234567890XYZ" not in sanitized
    assert "sk-ant-abcdef1234567890" not in sanitized
    assert "AIzaSyAABBCCDDEEFF001122334455" not in sanitized
    assert "Bearer super-secret-token-123456" not in sanitized
    assert sanitized.count("[REDACTED]") >= 4


def test_sanitize_error_message_keeps_normal_text():
    raw = "request failed with timeout after 30 seconds"
    assert api._sanitize_error_message(raw) == raw
