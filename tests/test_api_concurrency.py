import os
import threading
import time

import pytest


api = pytest.importorskip("api")
if not getattr(api, "HAS_FASTAPI", False):
    pytest.skip("FastAPI is not installed in this environment", allow_module_level=True)


def test_temporary_env_is_isolated_across_threads():
    original = os.environ.get("OPENAI_API_KEY")
    seen = []

    def worker(name: str, value: str, hold_ms: int):
        with api._temporary_env({"OPENAI_API_KEY": value}):
            # Record what this worker sees while its override is active.
            seen.append((name, os.environ.get("OPENAI_API_KEY")))
            time.sleep(hold_ms / 1000.0)

    t1 = threading.Thread(target=worker, args=("t1", "key-one", 120))
    t2 = threading.Thread(target=worker, args=("t2", "key-two", 40))

    t1.start()
    # Stagger start so thread 2 attempts to enter while thread 1 holds lock.
    time.sleep(0.02)
    t2.start()
    t1.join()
    t2.join()

    assert ("t1", "key-one") in seen
    assert ("t2", "key-two") in seen

    restored = os.environ.get("OPENAI_API_KEY")
    if original is None:
        assert restored is None
    else:
        assert restored == original


def test_provider_env_overrides_sets_expected_keys():
    gemini = api._provider_env_overrides("gemini", "g-key")
    assert gemini["GEMINI_API_KEY"] == "g-key"
    assert gemini["GOOGLE_API_KEY"] == "g-key"

    custom = api._provider_env_overrides("custom", "c-key", "https://gateway.example/v1")
    assert custom["ARENIX_CUSTOM_API_KEY"] == "c-key"
    assert custom["ARENIX_CUSTOM_BASE_URL"] == "https://gateway.example/v1"
