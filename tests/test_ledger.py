from __future__ import annotations

from covenant.ingest.ledger import _scenario_id_from_txn_id


def test_the_usual_numbering_is_parsed():
    assert _scenario_id_from_txn_id("TXN-P1-0039") == "P1"


def test_a_borrower_numbering_its_rows_differently_is_still_parsed():
    # one dataset numbers a borrower's rows TXN-KC-CAP-29; the case's rule is only that the id
    # begins with the scenario id, and a stricter reading dropped that borrower entirely
    assert _scenario_id_from_txn_id("TXN-KC-CAP-29") == "KC"
    assert _scenario_id_from_txn_id("TXN-KC-FIN-05") == "KC"


def test_a_two_letter_scenario_id_is_parsed():
    assert _scenario_id_from_txn_id("TXN-B2-0001") == "B2"


def test_something_that_is_not_a_transaction_id_yields_nothing():
    assert _scenario_id_from_txn_id("ACC-7801") is None
    assert _scenario_id_from_txn_id("TXN-") is None
