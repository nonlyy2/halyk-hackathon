from __future__ import annotations

from conftest import txn

from covenant.analyze.binding import (
    _aggregate_bindings,
    _sanitise,
    bindable_terms,
    mask_threshold,
)


def test_bindable_terms_excludes_tag_aggregates_and_doc_figures():
    cov = {
        "formula": "revenue - operating_expenses + related_party_payments + group_capex",
        "doc_figures": {"group_capex": {"amount_usd": 1.0}},
    }
    assert bindable_terms(cov) == ["operating_expenses", "revenue"]


def test_bindable_terms_covers_the_precondition_too():
    cov = {"formula": "financing / ebitda_proxy", "precondition": "financing > 4000000"}
    assert bindable_terms(cov) == ["ebitda_proxy", "financing"]


def test_bindable_terms_survives_an_unparseable_formula():
    assert bindable_terms({"formula": "revenue - (operating"}) == []


def test_mask_threshold_hides_the_limit_in_every_spelling_used():
    clause = "не менее 1.20x от суммы, но не более $3,000,000.00 за период"
    assert "1.20" not in mask_threshold(clause, 1.2)
    assert "3,000,000.00" not in mask_threshold(clause, 3000000.0)


def test_mask_threshold_leaves_other_numbers_alone():
    clause = "разовыми признаются статьи в сумме не менее $300,000.00; лимит 0.08x"
    masked = mask_threshold(clause, 0.08)
    assert "300,000.00" in masked
    assert "0.08" not in masked


def test_mask_threshold_is_a_noop_without_a_numeric_threshold():
    assert mask_threshold("текст", None) == "текст"


def test_sanitise_drops_hallucinated_ids_and_unrequested_terms():
    payload = {"terms": {"6.1": {"revenue": ""}}, "clauses": {}, "transactions": []}
    data = {
        "6.1": {
            "revenue": {"txn_ids": ["TXN-A-0001", "TXN-A-9999"], "reason": "sales"},
            "invented_term": {"txn_ids": ["TXN-A-0001"]},
        },
        "6.9": {"revenue": {"txn_ids": ["TXN-A-0001"]}},
    }
    out = _sanitise(data, payload, {"TXN-A-0001"})
    assert out == {"6.1": {"revenue": {"txn_ids": ["TXN-A-0001"], "reason": "sales"}}}


def _sample(ids):
    return {"6.1": {"revenue": {"txn_ids": ids, "reason": "sales settlement"}}}


def test_aggregate_keeps_a_row_two_of_three_samples_agree_on():
    voted = _aggregate_bindings([_sample(["A", "B"]), _sample(["A", "B"]), _sample(["A", "C"])])
    assert voted["6.1"]["revenue"]["txn_ids"] == ["A", "B"]


def test_aggregate_drops_a_row_only_one_sample_of_three_named():
    voted = _aggregate_bindings([_sample(["A"]), _sample(["A"]), _sample(["A", "C"])])
    assert voted["6.1"]["revenue"]["txn_ids"] == ["A"]


def test_aggregate_reports_the_share_of_rows_the_samples_split_on():
    voted = _aggregate_bindings([_sample(["A", "B"]), _sample(["A"]), _sample(["A"])])
    # two ids seen in total, one of which (B) only some samples named
    assert voted["6.1"]["revenue"]["disagreement"] == 0.5
    assert voted["6.1"]["revenue"]["samples"] == 3


def test_aggregate_of_unanimous_samples_reports_no_disagreement():
    voted = _aggregate_bindings([_sample(["A"])] * 3)
    assert voted["6.1"]["revenue"]["disagreement"] == 0.0


def test_txn_helper_builds_a_usable_row():
    assert txn("TXN-A-0001", -100.0, "rent").amount_usd == -100.0
