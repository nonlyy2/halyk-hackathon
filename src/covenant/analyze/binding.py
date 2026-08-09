"""Step D2 -- bind each covenant term to the transactions that actually constitute it.

`spec.py` decides a covenant's SHAPE (formula, threshold, comparison, period, carve-out) from the
clause text. It decides the term's MEMBERSHIP from a list of category name strings alone -- it never
sees a description or an amount -- and its prompt asks for a mapping by economic substance. That is
the wrong question to ask of a ledger built like this one: a clause capping "Операционные расходы
по аудированной отчётности Заёмщика" means the line item the borrower's statements carry under that
name, and the ledger deliberately surrounds it with far larger rent, payroll, marketing and
insurance rows that any textbook chart of accounts would also call operating costs. Bucketing by
substance sweeps all of them in and the reported metric is off by an order of magnitude.

So membership is asked here instead, as a reading task over the actual rows, with the clause's own
words in view. The model only SELECTS rows; compute.py still does every sum, comparison and
rounding.

Two deliberate restrictions:

- the two built-in tag aggregates are never bound. Related-party membership follows from the KYC
  ownership table by exact name match (enrich.py), and this dataset seeds lexically similar decoy
  counterparties precisely to defeat anything looser -- a model reading names would fall for them.
- doc_figures are never bound. They are figures from the statements that correspond to no
  transaction at all, which is what makes them doc_figures.

The clause's own threshold is masked before the model sees it, so row selection cannot be steered
towards a convenient verdict. Definitions, materiality floors and any other numbers stay intact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from covenant.analyze.enrich import EnrichedTxn
from covenant.analyze.spec import RESERVED_AGGREGATES, _extract_names, _model_tag
from covenant.llm.client import Client

BIND_PROMPT = """You are identifying which of a corporate borrower's ledger transactions make up \
each term of a financial covenant, so that the amounts can be summed afterwards. You do not \
compute anything and you do not judge compliance -- you only select rows.

You will be given:
1. "clauses": the covenant clause text, verbatim, keyed by clause number. The clause's own numeric \
limit has been replaced with «[предел]» -- it is deliberately withheld, and you must not try to \
infer it or select rows so as to produce any particular outcome.
2. "terms": for each clause number, the named terms to resolve, each with a one-sentence \
description of what it sums.
3. "transactions": every transaction on this borrower's ledger, with txn_id, date, counterparty, \
description, amount_usd (negative = outflow), currency, category (assigned earlier from the \
description alone), raw_category (before any auditor override), and flags: related_party, \
unrestricted_sub_transfer, excluded_period, reclass_reason, amount_corrected.

For EVERY term of EVERY clause, return the txn_ids that belong to it.

How to decide membership:
- A covenant term names a LINE ITEM as the borrower's own books label it -- not an accounting \
category in the abstract. Prefer transactions whose DESCRIPTION names that line item. Where a \
clause caps "operating expenses as determined from the audited financial statements", it means the \
rows the statements report under that caption, which is typically the borrower's own operating \
activity, not every cost it happens to incur.
- This ledger deliberately contains large, unrelated costs -- rent, payroll, marketing, insurance, \
telecom, utilities, taxes, interest -- alongside the one line a covenant is about. Do NOT sweep the \
whole expense side into a term merely because a textbook chart of accounts would also file those \
costs under it. A term whose total comes to several times the borrower's revenue is a term you have \
read too widely.
- Where the clause DEFINES its own term (what "Revenue" / "Operating expenses" / "Adjusted EBITDA" \
comprises for that covenant), the clause's wording governs, over any accounting convention.
- The captions of a set of financial statements are DISTINCT line items, reported side by side: an \
operating-expenses caption is the cost of running the borrower's own operations -- servicing, \
maintenance, production, repair, cleaning and remediation works on its own plant and facilities -- \
and is reported separately from rent, payroll, marketing, insurance, telecom, utilities, taxes and \
interest, each of which has its own caption. Do not fold one caption into another. If the ledger \
contains a row whose description names the caption outright, that row is the caption's core and the \
burden is on you to justify adding anything else.
- You are given ALL of this agreement's clauses together for a reason: they measure DIFFERENT \
things. A payment that another clause singles out and caps in its own right -- a related-party \
retainer, an intra-group transfer, a capital acquisition -- is that item, and does not also belong \
inside a broad expense caption that a different clause measures.
- An auditor's reclassification (reclass_reason set) moves a transaction into or out of a term \
regardless of what its description says. That is the whole point of the disclosure.
- Revenue is the borrower's own sales / turnover / throughput / handling settlement. An inflow that \
is not a sale -- a refund, a rebate, a credit note, sublet rent, interest income, a deposit \
returned, recovered payroll, a tax reversal -- is NOT revenue, however large it is.
- Capital expenditure is an actual acquisition or construction of a capital asset. Capitalised \
interest is interest, despite the word "capital".
- Sign is irrelevant to membership: a refund of a cost belongs to that cost's line item, and the \
summing afterwards handles the sign.
- If genuinely no transaction belongs to a term, return an empty list. Do not pad it with the \
nearest-looking row.

For each term also give "reason": one sentence, citing the clause's own wording and what the \
selected rows have in common. If you excluded rows that a broad reading would have included, say \
why in that sentence.

Respond with strict JSON only, no markdown code fences, no commentary:
{"<clause number>": {"<term name>": {"txn_ids": ["TXN-..."], "reason": "..."}}}
Include every clause number and every term you were given, exactly once each.
"""

CACHE_DIR = Path(".cache/bindings")

# Below this share of the samples, a txn_id is not carried into the voted binding.
_VOTE_SHARE = 0.5
# Above this share of the borrower's whole ledger, a "term" is not a line item -- see _too_broad.
_MAX_TERM_SHARE = 0.5
# ...but only once the ledger is big enough for the share to mean anything.
_MIN_LEDGER_FOR_SHARE = 10


def bindable_terms(cov: dict) -> list[str]:
    """The formula/precondition names this stage may resolve: everything except the two tag
    aggregates (Python-derived from the KYC table) and this covenant's doc_figures (which
    correspond to no transaction at all)."""
    reserved = set(RESERVED_AGGREGATES) | set((cov.get("doc_figures") or {}).keys())
    names: set[str] = set()
    for expr in filter(None, [cov.get("formula"), cov.get("precondition")]):
        try:
            names |= _extract_names(expr)
        except (ValueError, SyntaxError):
            continue
    return sorted(names - reserved)


_MASK = "«[предел]»"


def mask_threshold(clause_text: str, threshold) -> str:
    """Hide the covenant's own limit in the clause text shown to the binder.

    Selecting rows and knowing the number they must clear are two jobs that should not meet: a model
    that can see the limit can shop for a membership that lands on the comfortable side of it. Only
    the covenant's own threshold is masked, in the spellings these clauses use, so definitions and
    materiality floors are untouched.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return clause_text
    value = abs(float(threshold))
    # Only spellings that read back as the same number: "{:,.0f}" of 0.08 is "0", and blanking every
    # "0" in the clause would take the materiality floors and the period dates with it.
    spellings = {
        s
        for s in (f"{value:,.2f}", f"{value:.2f}", f"{value:,.0f}", f"{value:g}")
        if float(s.replace(",", "")) == value
    }
    masked = clause_text
    for spelling in sorted(spellings, key=len, reverse=True):
        # not preceded or followed by more digits, so 300,000.00 is not hit while masking 0.00
        masked = re.sub(
            rf"(?<![\d.,]){re.escape(spelling)}(x)?(?![\d])",
            _MASK,
            masked,
        )
    return re.sub(rf"\$\s?{re.escape(_MASK)}", _MASK, masked)


def _txn_rows(enriched: list[EnrichedTxn]) -> list[dict]:
    return [
        {
            "txn_id": e.txn_id,
            "date": e.date,
            "counterparty": e.counterparty,
            "description": e.description,
            "amount_usd": None if e.amount_usd != e.amount_usd else round(e.amount_usd, 2),
            "currency": e.currency,
            "category": e.category,
            "raw_category": e.raw_category,
            "related_party": e.related_party,
            "unrestricted_sub_transfer": e.unrestricted_sub_transfer,
            "excluded_period": e.excluded_period,
            "reclass_reason": e.reclass_reason,
            "amount_corrected": e.amount_corrected,
        }
        for e in enriched
    ]


def _payload(spec: dict, clauses: dict[str, str], enriched: list[EnrichedTxn]) -> dict | None:
    clause_texts: dict[str, str] = {}
    terms: dict[str, dict[str, str]] = {}
    for key, cov in (spec.get("covenants") or {}).items():
        wanted = bindable_terms(cov)
        if not wanted:
            continue
        text = clauses.get(key) or cov.get("metric") or ""
        clause_texts[key] = mask_threshold(text, cov.get("threshold"))
        # The spec's own one-line description of each term. It carries the term's SCOPE, which the
        # clause often states only for the concept and not for the name: a covenant capping
        # transfers as a fraction of total capital expenditure names both, and without
        # "total capital expenditure incurred by the Borrower" the binder collapses the denominator
        # onto the numerator's single row. What it must never carry is a COMPOSITION -- see the
        # constraint on "variables" in SPEC_PROMPT, which exists because one such description
        # enumerated an item its clause never mentions and the binder duly included it.
        variables = cov.get("variables") or {}
        terms[key] = {name: variables.get(name, "") for name in wanted}
    if not terms:
        return None
    return {
        "clauses": clause_texts,
        "terms": terms,
        "transactions": _txn_rows(enriched),
    }


def bind_terms(
    spec: dict,
    clauses: dict[str, str],
    enriched: list[EnrichedTxn],
    client: Client,
    temperature: float = 0.0,
) -> dict[str, dict[str, dict]]:
    """{covenant_key: {term: {"txn_ids": [...], "reason": str}}} -- one call per scenario."""
    payload = _payload(spec, clauses, enriched)
    if payload is None:
        return {}
    data = client.complete_json(
        BIND_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        max_tokens=8192,
        temperature=temperature,
    )
    return _sanitise(data, payload, {e.txn_id for e in enriched})


def is_a_line_item(selected: int, ledger_size: int) -> bool:
    """Is this selection still a line item, or has it become the whole ledger?

    A covenant term names one line of the borrower's accounts, and every term these clauses use
    resolves to a handful of rows. A selection covering most of the ledger is not a generous
    reading of the clause -- it is the failure this stage exists to prevent, arrived at from the
    other side: the model fell back on "every cost is an operating expense" despite being told not
    to, and the metric is then wrong by an order of magnitude in the opposite direction.

    Counted in rows rather than dollars, so one large transaction cannot sway it, and only applied
    to a ledger big enough for a share to mean anything. Checked where the binding is USED rather
    than where it is produced, so it also covers a binding already sitting in the cache.
    """
    if ledger_size < _MIN_LEDGER_FOR_SHARE:
        return True
    return selected <= _MAX_TERM_SHARE * ledger_size


def _sanitise(data: dict, payload: dict, known_ids: set[str]) -> dict[str, dict[str, dict]]:
    """Keep only the clause/term pairs that were asked for, and only real transaction ids.

    A hallucinated id would silently contribute nothing to its sum, which reads exactly like a term
    the model decided is empty -- so it is dropped here, where it can still be counted and reported,
    rather than downstream where it would be invisible.
    """
    out: dict[str, dict[str, dict]] = {}
    for key, wanted_terms in payload["terms"].items():
        answered = data.get(key)
        if not isinstance(answered, dict):
            continue
        bound: dict[str, dict] = {}
        for term in wanted_terms:
            entry = answered.get(term)
            if not isinstance(entry, dict):
                continue
            ids = entry.get("txn_ids")
            if not isinstance(ids, list):
                continue
            kept = [i for i in dict.fromkeys(map(str, ids)) if i in known_ids]
            bound[term] = {"txn_ids": kept, "reason": str(entry.get("reason", ""))}
        if bound:
            out[key] = bound
    return out


def bind_terms_voted(
    spec: dict,
    clauses: dict[str, str],
    enriched: list[EnrichedTxn],
    client: Client,
    votes: int = 3,
    temperature: float = 0.4,
) -> dict[str, dict[str, dict]]:
    """Majority-vote membership per transaction across `votes` independent samples.

    Voting on the whole selected SET would need the samples to agree exactly to agree at all; a
    single extra row in one sample would discard it entirely. Membership is a per-row decision, so
    that is where the vote belongs: a txn_id is in the term if at least half the samples put it
    there. `disagreement` records the share of ids the samples split on, which is the signal the
    review stage triages on.
    """
    if votes <= 1:
        return bind_terms(spec, clauses, enriched, client, temperature=0.0)

    samples: list[dict] = []
    for _ in range(votes):
        try:
            samples.append(bind_terms(spec, clauses, enriched, client, temperature=temperature))
        except Exception:  # noqa: BLE001 -- a failed sample just doesn't vote
            continue
    if not samples:
        raise RuntimeError("all binding samples failed")
    return _aggregate_bindings(samples)


def _aggregate_bindings(samples: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for key in sorted({k for s in samples for k in s}):
        terms = sorted({t for s in samples for t in s.get(key, {})})
        bound: dict[str, dict] = {}
        for term in terms:
            present = [s[key][term] for s in samples if term in s.get(key, {})]
            if not present:
                continue
            counts = Counter(i for entry in present for i in entry["txn_ids"])
            quorum = len(present) * _VOTE_SHARE
            voted = [i for i, n in counts.items() if n >= quorum]
            split = sum(1 for _, n in counts.items() if 0 < n < len(present))
            bound[term] = {
                # keep the order of the first sample that mentioned each id, for readability
                "txn_ids": [i for i in dict.fromkeys(_ordered_ids(present)) if i in set(voted)],
                "reason": _modal_reason(present),
                "disagreement": round(split / len(counts), 3) if counts else 0.0,
                "samples": len(present),
            }
        if bound:
            out[key] = bound
    return out


def _ordered_ids(entries: list[dict]) -> list[str]:
    return [i for entry in entries for i in entry["txn_ids"]]


def _modal_reason(entries: list[dict]) -> str:
    reasons = [e.get("reason", "") for e in entries if e.get("reason")]
    return Counter(reasons).most_common(1)[0][0] if reasons else ""


def save_binding(sid: str, binding: dict, client: Client, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sid}__{_model_tag(client)}.json"
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2))
    return path


def load_binding(sid: str, client: Client, cache_dir: Path = CACHE_DIR) -> dict | None:
    path = cache_dir / f"{sid}__{_model_tag(client)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
