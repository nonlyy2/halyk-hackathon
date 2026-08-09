from __future__ import annotations

import pytest
from conftest import txn

from covenant.analyze.review import apply_correction, build_worksheet, review_cell
from covenant.scoring.compute import ComputeResult, compute_covenant

LEDGER = [
    txn("TXN-X-0001", -4_204_663.19, "operating_costs", description="Kiln servicing and operating"),
    txn("TXN-X-0002", -8_599_812.02, "rent"),
    txn("TXN-X-0003", 6_918_204.37, "sales_revenue", description="Cement despatch settlement"),
]
ROLES = {
    "operating_costs": "operating_expenses",
    "rent": "operating_expenses",
    "sales_revenue": "revenue",
}
COV = {
    "formula": "revenue - operating_expenses",
    "comparison": ">=",
    "threshold": 1_000_000.0,
    "variables": {"revenue": "sales", "operating_expenses": "opex"},
}


class StubClient:
    """Stands in for an LLM: returns a canned JSON body and records what it was asked."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.last_user: str | None = None

    def complete_json(self, system, user, **kw):
        self.last_user = user
        return self.reply


def test_the_worksheet_itemises_every_term_with_its_rows_and_subtotal():
    result = compute_covenant(COV, ROLES, LEDGER, None)
    sheet = build_worksheet(COV, ROLES, LEDGER, None, result)
    assert sheet["terms"]["revenue"]["subtotal_usd"] == pytest.approx(6_918_204.37)
    assert sheet["terms"]["operating_expenses"]["subtotal_usd"] == pytest.approx(-12_804_475.21)
    assert sheet["status"] == result.status


def test_the_worksheet_shows_which_rows_a_bound_term_holds():
    binding = {"operating_expenses": {"txn_ids": ["TXN-X-0001"]}}
    result = compute_covenant(COV, ROLES, LEDGER, binding)
    sheet = build_worksheet(COV, ROLES, LEDGER, binding, result)
    term = sheet["terms"]["operating_expenses"]
    assert term["resolved_by"] == "bound"
    assert [r["txn_id"] for r in term["transactions"]] == ["TXN-X-0001"]


def test_a_confirming_critic_changes_nothing():
    client = StubClient({"verdict": "confirm"})
    result = compute_covenant(COV, ROLES, LEDGER, None)
    assert review_cell("clause", COV, ROLES, LEDGER, None, result, [], client) is None


def test_a_correction_is_kept_only_for_terms_this_covenant_has():
    client = StubClient(
        {
            "verdict": "correct",
            "terms": {
                "operating_expenses": {"txn_ids": ["TXN-X-0001"], "reason": "the operating line"},
                "some_other_term": {"txn_ids": ["TXN-X-0002"]},
            },
        }
    )
    result = compute_covenant(COV, ROLES, LEDGER, None)
    corrected = review_cell("clause", COV, ROLES, LEDGER, None, result, [], client)
    assert set(corrected) == {"operating_expenses"}
    assert corrected["operating_expenses"]["txn_ids"] == ["TXN-X-0001"]


def test_a_hallucinated_id_is_dropped_and_an_emptied_term_refused():
    client = StubClient(
        {"verdict": "correct", "terms": {"operating_expenses": {"txn_ids": ["TXN-X-9999"]}}}
    )
    result = compute_covenant(COV, ROLES, LEDGER, None)
    # every id was invented, so nothing survives -- and an emptied term must not read as zero
    assert review_cell("clause", COV, ROLES, LEDGER, None, result, [], client) is None


def test_the_critic_is_shown_the_rows_no_term_currently_claims():
    client = StubClient({"verdict": "confirm"})
    binding = {
        "operating_expenses": {"txn_ids": ["TXN-X-0001"]},
        "revenue": {"txn_ids": ["TXN-X-0003"]},
    }
    result = compute_covenant(COV, ROLES, LEDGER, binding)
    review_cell("clause", COV, ROLES, LEDGER, binding, result, ["flagged"], client)
    assert "TXN-X-0002" in client.last_user


def test_an_applied_correction_is_recomputed_not_taken_on_trust():
    corrected = {"operating_expenses": {"txn_ids": ["TXN-X-0001"]}}
    applied = apply_correction(COV, ROLES, LEDGER, None, corrected)
    assert applied is not None
    merged, result = applied
    assert merged["operating_expenses"]["txn_ids"] == ["TXN-X-0001"]
    assert result.actual == pytest.approx(2_713_541.18)
    assert result.status == "COMPLIANT"


def test_a_correction_that_will_not_evaluate_is_refused():
    broken = {**COV, "formula": "revenue - ebitda"}  # ebitda resolves to nothing
    assert (
        apply_correction(broken, ROLES, LEDGER, None, {"revenue": {"txn_ids": ["TXN-X-0003"]}})
        is None
    )


def test_review_never_invents_a_result_of_its_own():
    # the critic returns only a row selection; status and actual always come back from compute
    corrected = {"revenue": {"txn_ids": ["TXN-X-0003"]}}
    _merged, result = apply_correction(COV, ROLES, LEDGER, None, corrected)
    assert isinstance(result, ComputeResult)
