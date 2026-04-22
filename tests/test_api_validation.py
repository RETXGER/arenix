import pytest


api = pytest.importorskip("api")
if not getattr(api, "HAS_FASTAPI", False):
    pytest.skip("FastAPI is not installed in this environment", allow_module_level=True)

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client():
    return TestClient(api.app)


def test_start_run_rejects_unknown_provider(client):
    payload = {
        "attacker_provider": "unknown-provider",
        "attacker_model": "x-model",
    }
    resp = client.post("/api/v1/run", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert "attacker_provider" in body["detail"]


def test_start_run_requires_custom_base_url(client):
    payload = {
        "attacker_provider": "custom",
        "attacker_model": "company-safe-v1",
        "target_provider": "custom",
        "target_model": "company-safe-v1",
    }
    resp = client.post("/api/v1/run", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert "base_url" in body["detail"]


def test_tournament_rejects_invalid_attack_profile(client):
    payload = {
        "models": [{"provider": "mock", "model_name": "mock"}],
        "attack_profile": "ultra",
    }
    resp = client.post("/api/v1/tournament", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert "attack_profile" in body["detail"]


def test_tournament_requires_custom_base_url(client):
    payload = {
        "models": [{"provider": "custom", "model_name": "company-safe-v1"}],
        "attack_profile": "balanced",
    }
    resp = client.post("/api/v1/tournament", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert "base_url" in body["detail"]
