from __future__ import annotations

import covenant.llm.client as client_module
from covenant.llm.client import Client


def _clear() -> None:
    client_module._exhausted.clear()


def test_every_model_is_offered_while_none_has_refused():
    _clear()
    c = Client(backend="gemini", model="a", alternates=("b", "c"))
    assert c._models() == ["a", "b", "c"]


def test_a_model_that_refused_for_the_day_is_skipped():
    _clear()
    c = Client(backend="gemini", model="a", alternates=("b",))
    client_module._exhausted["a"] = client_module.time.monotonic()
    assert c._models() == ["b"]


def test_the_mark_expires_so_a_run_can_never_lock_itself_out():
    _clear()
    c = Client(backend="gemini", model="a", alternates=("b",))
    # marked longer ago than the retry interval -- a day may have rolled over, or the refusal was
    # misread as daily; either way the model must be probed again rather than written off
    client_module._exhausted["a"] = (
        client_module.time.monotonic() - client_module.EXHAUSTED_RETRY_AFTER - 1
    )
    assert c._models() == ["a", "b"]


def test_with_every_model_spent_the_chain_is_still_returned():
    _clear()
    c = Client(backend="gemini", model="a", alternates=("b",))
    now = client_module.time.monotonic()
    client_module._exhausted.update({"a": now, "b": now})
    # the call has to reach the provider and fail with its real error, not raise IndexError here
    assert c._models() == ["a", "b"]


def test_a_per_minute_refusal_is_not_remembered():
    _clear()

    class Response:
        text = '{"error":{"message":"Quota exceeded","details":[{"quotaId":"PerMinute"}]}}'

    class Error(Exception):
        response = Response()

    client_module._mark_exhausted_if_daily("a", Error())
    assert "a" not in client_module._exhausted


def test_a_key_swapped_into_the_env_file_is_picked_up_without_a_restart(tmp_path, monkeypatch):
    _clear()
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=first\n")
    monkeypatch.setattr(client_module, "_ENV_PATH", str(env))
    monkeypatch.setattr(client_module, "_last_seen_key", {})
    assert client_module.gemini_key() == "first"

    # the key burns out and is replaced while the process keeps running
    env.write_text("GEMINI_API_KEY=second\n")
    assert client_module.gemini_key() == "second"


def test_swapping_the_key_clears_what_was_learned_about_the_old_one(tmp_path, monkeypatch):
    _clear()
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=first\n")
    monkeypatch.setattr(client_module, "_ENV_PATH", str(env))
    monkeypatch.setattr(client_module, "_last_seen_key", {})
    client_module.gemini_key()
    client_module._exhausted["gemini-3.1-flash-lite"] = client_module.time.monotonic()

    env.write_text("GEMINI_API_KEY=second\n")
    client_module.gemini_key()
    # the new key has its own untouched allowance; the old key's exhaustion says nothing about it
    assert client_module._exhausted == {}


def test_reading_the_same_key_twice_does_not_clear_anything(tmp_path, monkeypatch):
    _clear()
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=same\n")
    monkeypatch.setattr(client_module, "_ENV_PATH", str(env))
    monkeypatch.setattr(client_module, "_last_seen_key", {})
    client_module.gemini_key()
    client_module._exhausted["m"] = client_module.time.monotonic()
    client_module.gemini_key()
    assert "m" in client_module._exhausted


def test_a_daily_refusal_is_remembered():
    _clear()

    class Response:
        text = '{"error":{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}'

    class Error(Exception):
        response = Response()

    client_module._mark_exhausted_if_daily("a", Error())
    assert "a" in client_module._exhausted
