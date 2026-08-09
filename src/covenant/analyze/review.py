"""Step F -- a second opinion on the cells whose own signals say they are fragile.

Everything upstream commits to an answer without ever seeing what that answer is made of: the
binder picks rows without knowing the limit, the spec writes a formula without knowing which rows
it will land on, and compute.py evaluates without an opinion about either. That separation is what
keeps the pipeline honest, and it also means nothing has yet looked at a finished cell as a whole.

This stage does exactly that, and only where it is worth an LLM call: a value sitting a hair from
its threshold, a sum resting on one row, a related-party covenant that matched nobody, a term the
binding samples split on. The critic is shown the clause and the worked calculation and may
correct the row selection -- never the status, never the number. The correction is then recomputed
by the same deterministic evaluator, and accepted only if it evaluates.
"""

from __future__ import annotations

import json

from covenant.analyze.binding import bindable_terms
from covenant.analyze.enrich import EnrichedTxn
from covenant.llm.client import Client
from covenant.scoring.compute import (
    ComputeResult,
    _metric_expr,
    bound_ids,
    compute_covenant,
    variable_rows,
)

REVIEW_PROMPT = """You are reviewing one finished financial-covenant calculation for a corporate \
borrower, to catch a term that was made up of the wrong transactions.

You will be given the covenant clause verbatim, the formula that was evaluated, and a worksheet: \
for every term, the transactions that were summed into it and the resulting subtotal, then the \
computed value, the threshold and the status.

Judge ONE thing: does each term contain exactly the transactions the clause's own wording says it \
should? Specifically look for:
- rows swept in because a textbook chart of accounts would file them under the term, when the \
clause means the line item the borrower's statements carry under that caption. A term totalling \
several times the borrower's revenue has almost certainly been read too widely.
- an inflow counted as revenue that is not a sale -- a refund, rebate, credit note, sublet rent, \
interest income, deposit returned, tax reversal.
- a row the clause's own definition, or an auditor's disclosed reclassification, plainly puts \
inside the term but which is missing from it.
- a term that is empty when the ledger clearly contains rows answering to it.

Do NOT change the status, the threshold, the formula or any amount -- you cannot, and you are not \
being asked to. Do NOT adjust a selection because you think the borrower ought to comply or ought \
to breach; whether the result is a breach is none of your concern and is not a reason to move a \
row. Leave a term exactly as it is unless the clause's wording makes it plainly wrong.

Respond with strict JSON only, no markdown code fences, no commentary:
{"verdict": "confirm"} if every term is right as it stands, otherwise
{"verdict": "correct", "terms": {"<term name>": {"txn_ids": ["TXN-..."], "reason": "..."}}}
listing ONLY the terms you are changing, each with its complete corrected transaction list.
"""


def _row(e: EnrichedTxn) -> dict:
    row = {
        "txn_id": e.txn_id,
        "date": e.date,
        "counterparty": e.counterparty,
        "description": e.description,
        "amount_usd": None if e.amount_usd != e.amount_usd else round(e.amount_usd, 2),
    }
    if e.reclass_reason:
        row["reclassified"] = e.reclass_reason
    if e.excluded_period:
        row["excluded_from_period"] = True
    if e.related_party:
        row["related_party"] = True
    return row


def build_worksheet(
    cov: dict,
    roles: dict,
    enriched: list[EnrichedTxn],
    binding: dict | None,
    result: ComputeResult,
) -> dict:
    """The finished calculation, itemised -- what the critic reads and what a human eyeballs."""
    terms: dict[str, dict] = {}
    for name in bindable_terms(cov):
        kind, rows = variable_rows(name, cov, roles, enriched, binding)
        terms[name] = {
            "resolved_by": kind,
            "subtotal_usd": round(
                sum(e.amount_usd for e in rows if e.amount_usd == e.amount_usd), 2
            ),
            "transactions": [_row(e) for e in rows],
        }
    return {
        "formula": _metric_expr(cov),
        "terms": terms,
        "computed_value": result.actual,
        "threshold": cov.get("threshold"),
        "comparison": cov.get("comparison"),
        "status": result.status,
    }


def _unbound_ledger(cov: dict, enriched: list[EnrichedTxn], binding: dict | None) -> list[dict]:
    """Rows no term of this covenant currently claims -- the pool a missing row can come from."""
    claimed = {i for name in bindable_terms(cov) for i in (bound_ids(binding, name) or [])}
    return [_row(e) for e in enriched if e.txn_id not in claimed]


def review_cell(
    clause_text: str,
    cov: dict,
    roles: dict,
    enriched: list[EnrichedTxn],
    binding: dict | None,
    result: ComputeResult,
    flags: list[str],
    client: Client,
) -> dict | None:
    """Corrected {term: {"txn_ids": [...]}} for this covenant, or None to keep what we have."""
    payload = {
        "clause": clause_text,
        "worksheet": build_worksheet(cov, roles, enriched, binding, result),
        "transactions_not_in_any_term": _unbound_ledger(cov, enriched, binding),
        "why_this_cell_was_flagged": flags,
    }
    data = client.complete_json(
        REVIEW_PROMPT, json.dumps(payload, ensure_ascii=False), max_tokens=4096
    )
    if data.get("verdict") != "correct":
        return None

    known = {e.txn_id for e in enriched}
    allowed = set(bindable_terms(cov))
    corrected: dict[str, dict] = {}
    for term, entry in (data.get("terms") or {}).items():
        if term not in allowed or not isinstance(entry, dict):
            continue
        ids = entry.get("txn_ids")
        if not isinstance(ids, list):
            continue
        kept = [i for i in dict.fromkeys(map(str, ids)) if i in known]
        if not kept:
            # An emptied term is the one correction that cannot be checked by recomputing: it
            # evaluates fine and quietly reads as zero. Refuse it rather than trust it.
            continue
        corrected[term] = {"txn_ids": kept, "reason": str(entry.get("reason", "")), "by": "review"}
    return corrected or None


def apply_correction(
    cov: dict,
    roles: dict,
    enriched: list[EnrichedTxn],
    binding: dict | None,
    corrected: dict,
) -> tuple[dict, ComputeResult] | None:
    """(merged binding, recomputed cell) -- or None if the correction does not evaluate.

    A correction is only ever accepted on evidence that it computes; the critic never gets to
    assert a status or a number directly.
    """
    merged = {**(binding or {}), **corrected}
    try:
        return merged, compute_covenant(cov, roles, enriched, merged)
    except Exception:  # noqa: BLE001 -- a correction that breaks the cell is simply not applied
        return None
