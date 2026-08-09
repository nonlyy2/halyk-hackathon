from __future__ import annotations

from covenant.cli import validate_submission

TEMPLATE = {
    "answers": {
        "P1": {"6.1": {}, "6.2": {}},
        "P2": {"6.1": {}},
    }
}


def _submission(**answers) -> dict:
    return {"team": "t", "contact_email": "e@x.com", "answers": answers}


def _cell(status="COMPLIANT", actual=1.0, evidence=None) -> dict:
    return {"status": status, "actual": actual, "evidence_txn_id": evidence}


def test_a_complete_well_typed_submission_has_no_problems():
    sub = _submission(P1={"6.1": _cell(), "6.2": _cell()}, P2={"6.1": _cell()})
    assert validate_submission(sub, TEMPLATE) == []


def test_a_missing_cell_is_reported():
    sub = _submission(P1={"6.1": _cell()}, P2={"6.1": _cell()})
    assert validate_submission(sub, TEMPLATE) == ["P1.6.2: missing"]


def test_a_whole_missing_scenario_is_reported():
    sub = _submission(P1={"6.1": _cell(), "6.2": _cell()})
    assert validate_submission(sub, TEMPLATE) == ["P2.6.1: missing"]


def test_a_status_outside_the_two_allowed_values_is_reported():
    sub = _submission(P1={"6.1": _cell(status="breach"), "6.2": _cell()}, P2={"6.1": _cell()})
    assert validate_submission(sub, TEMPLATE) == ["P1.6.1: status is 'breach'"]


def test_a_non_numeric_or_negative_actual_is_reported():
    for actual in (None, "1.0", float("nan"), float("inf"), -1.0, True):
        sub = _submission(P1={"6.1": _cell(actual=actual), "6.2": _cell()}, P2={"6.1": _cell()})
        problems = validate_submission(sub, TEMPLATE)
        assert problems and problems[0].startswith("P1.6.1: actual is"), actual


def test_a_key_the_template_does_not_have_is_reported():
    sub = _submission(P1={"6.1": _cell(), "6.2": _cell(), "6.9": _cell()}, P2={"6.1": _cell()})
    assert validate_submission(sub, TEMPLATE) == ["P1.6.9: not a template key"]


def test_the_top_level_identity_fields_are_required():
    sub = _submission(P1={"6.1": _cell(), "6.2": _cell()}, P2={"6.1": _cell()})
    sub["team"] = "  "
    sub["contact_email"] = ""
    assert validate_submission(sub, TEMPLATE) == ["team is empty", "contact_email is empty"]


def test_a_null_evidence_id_is_perfectly_valid():
    # the key holds null for every ratio and aggregate test, so null must not be flagged
    sub = _submission(P1={"6.1": _cell(evidence=None), "6.2": _cell()}, P2={"6.1": _cell()})
    assert validate_submission(sub, TEMPLATE) == []
