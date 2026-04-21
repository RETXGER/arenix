import os
import types

from arenix_engine import CustomEndpointAdapter, build_adapter


def test_build_adapter_supports_custom_provider():
    adapter = build_adapter("custom", "company-safe-v1")
    assert isinstance(adapter, CustomEndpointAdapter)


def test_custom_adapter_resolves_model_alias(monkeypatch):
    monkeypatch.setenv("ARENIX_CUSTOM_MODEL_ALIASES", '{"company-safe-v1":"gateway-model-prod"}')
    adapter = CustomEndpointAdapter("custom", "company-safe-v1")
    assert adapter._resolve_runtime_model_name() == "gateway-model-prod"


def test_custom_adapter_build_client_uses_headers_and_timeout(monkeypatch):
    captured = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ARENIX_CUSTOM_BASE_URL", "https://example-gateway/v1")
    monkeypatch.setenv("ARENIX_CUSTOM_API_KEY", "secret-key")
    monkeypatch.setenv("ARENIX_CUSTOM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ARENIX_CUSTOM_HEADERS", '{"X-Tenant-ID":"acme","X-Org":"security"}')
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: types.SimpleNamespace(OpenAI=DummyOpenAI) if name == "openai" else __import__(name),
    )

    adapter = CustomEndpointAdapter("custom", "company-safe-v1")
    adapter._build_client()

    assert captured["api_key"] == "secret-key"
    assert captured["base_url"] == "https://example-gateway/v1"
    assert captured["timeout"] == 45.0
    assert captured["default_headers"]["X-Tenant-ID"] == "acme"
    assert captured["default_headers"]["X-Org"] == "security"
