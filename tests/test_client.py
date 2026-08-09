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


def test_a_daily_refusal_is_remembered():
    _clear()

    class Response:
        text = '{"error":{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}'

    class Error(Exception):
        response = Response()

    client_module._mark_exhausted_if_daily("a", Error())
    assert "a" in client_module._exhausted
