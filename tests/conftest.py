from __future__ import annotations

from covenant.analyze.enrich import EnrichedTxn


def txn(txn_id: str, amount: float, category: str, **kw) -> EnrichedTxn:
    """An enriched row with everything defaulted, so a test states only what it is about."""
    fields = {
        "txn_id": txn_id,
        "date": "2025-06-01",
        "counterparty": "Counterparty LLP",
        "description": "a payment",
        "amount": amount,
        "currency": "USD",
        "amount_usd": amount,
        "category": category,
        "raw_category": category,
        "related_party": False,
        "unrestricted_sub_transfer": False,
        "excluded_period": False,
        "exclusion_reason": None,
        "reclass_reason": None,
        "amount_corrected": False,
    }
    fields.update(kw)
    return EnrichedTxn(**fields)
