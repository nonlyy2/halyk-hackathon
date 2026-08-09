from __future__ import annotations

from covenant.ingest.documents import UNBOUNDED_CONCURRENCY, concurrency


def test_an_explicit_number_is_used_as_given(monkeypatch):
    monkeypatch.setenv("COVENANT_CONCURRENCY", "12")
    assert concurrency() == 12


def test_zero_means_unbounded_not_sequential(monkeypatch):
    # the natural reading of "no limit" -- and the opposite of what max(1, 0) used to give,
    # which silently made a paid endpoint run one call at a time
    monkeypatch.setenv("COVENANT_CONCURRENCY", "0")
    assert concurrency() == UNBOUNDED_CONCURRENCY


def test_a_negative_value_is_read_the_same_way(monkeypatch):
    monkeypatch.setenv("COVENANT_CONCURRENCY", "-1")
    assert concurrency() == UNBOUNDED_CONCURRENCY


def test_an_unset_or_unparseable_value_falls_back_to_a_safe_default(monkeypatch):
    monkeypatch.delenv("COVENANT_CONCURRENCY", raising=False)
    assert concurrency() == 4
    monkeypatch.setenv("COVENANT_CONCURRENCY", "lots")
    assert concurrency() == 4
