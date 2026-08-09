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


class FakeResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload or {}
        self.status_code = status
        self.text = text or str(payload)
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")


def test_anthropic_is_called_over_plain_http_with_no_sdk(monkeypatch):
    _clear()
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, body=json)
        return FakeResponse({"content": [{"type": "text", "text": '{"ok":1}'}]})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(client_module, "_ENV_PATH", "")
    c = Client(backend="anthropic", model="claude-sonnet-5")
    assert c.complete("sys", "usr", max_tokens=64) == '{"ok":1}'
    assert sent["url"].endswith("/messages")
    assert sent["headers"]["x-api-key"] == "k"
    assert sent["headers"]["anthropic-version"] == client_module.ANTHROPIC_VERSION
    assert sent["body"]["system"] == "sys"


def test_anthropic_reasoning_blocks_do_not_displace_the_answer(monkeypatch):
    _clear()
    monkeypatch.setattr(
        client_module.httpx,
        "post",
        lambda *a, **k: FakeResponse(
            {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ANS"}]}
        ),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(client_module, "_ENV_PATH", "")
    assert Client(backend="anthropic", model="m").complete("s", "u") == "ANS"


def test_openai_goes_to_the_openai_endpoint(monkeypatch):
    _clear()
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers)
        return FakeResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(client_module, "_ENV_PATH", "")
    assert Client(backend="openai", model="gpt-4o").complete("s", "u") == "hi"
    assert sent["url"].startswith(client_module.OPENAI_BASE_URL)
    assert sent["headers"]["Authorization"] == "Bearer k"


def test_a_model_wanting_max_completion_tokens_is_retried_with_it(monkeypatch):
    _clear()
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if "max_tokens" in json:
            return FakeResponse(status=400, text="Unsupported parameter: use max_completion_tokens")
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(client_module, "_ENV_PATH", "")
    assert Client(backend="openai", model="m").complete("s", "u", max_tokens=99) == "ok"
    assert "max_tokens" in calls[0] and calls[1]["max_completion_tokens"] == 99


def test_a_model_refusing_a_custom_temperature_is_retried_without_it(monkeypatch):
    _clear()
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("temperature") == 0.5:
            return FakeResponse(status=400, text="'temperature' does not support 0.5")
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(client_module, "_ENV_PATH", "")
    assert Client(backend="openai", model="m").complete("s", "u", temperature=0.5) == "ok"
    assert "temperature" not in calls[1]


def test_an_unrelated_400_is_not_silently_retried():
    assert client_module._repair_payload({"max_tokens": 1}, "model not found") is None


def test_a_daily_refusal_is_remembered():
    _clear()

    class Response:
        text = '{"error":{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}'

    class Error(Exception):
        response = Response()

    client_module._mark_exhausted_if_daily("a", Error())
    assert "a" in client_module._exhausted
