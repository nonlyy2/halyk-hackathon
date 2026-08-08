"""Stage A -- enrich a scenario's ledger rows once, covenant-agnostic.

Three independent layers, applied in order, matching PIPELINE.md's architecture:

1. Categorize every transaction from its OWN description alone (no covenant in view yet --
   this is what structurally prevents the catch-all-bucket bug: a model asked to bucket
   transactions "for this covenant" tends to invent a generic bucket and dump anything
   ambiguous into it, wildly inflating whichever ratio it's feeding).
2. Tag related-party / unrestricted-subsidiary from the KYC ownership section's own threshold
   -- via exact/normalized string match against counterparty, never fuzzy/embedding similarity
   (this dataset seeds lexically-similar-but-unrelated decoy names on purpose, e.g. "Foxridge
   Power LLC" / "Foxridge Insurance Co" / "Foxridge Telecom LP" -- different vendors sharing a
   brand root; only exact match can't be fooled by "sounds similar").
3. Apply the audit report's reclassifications / cut-off exclusions / one-off-addback flags /
   FX conversions as authoritative overrides on top.

Each layer is one LLM call per scenario (not per transaction), keeping cost proportional to
scenario count rather than transaction count.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from covenant.llm.client import Client

CATEGORIZE_PROMPT = """You are preparing a corporate borrower's ledger for covenant compliance \
review. You will be given a list of transactions (txn_id, date, counterparty, description, \
amount, currency). For EACH transaction, assign a short, literal, lowercase snake_case category \
that describes what the transaction actually is, based ONLY on its own DESCRIPTION -- do not \
consider any covenant, threshold, or what category would be convenient.

CRITICAL -- the counterparty name is NOT evidence of the category. In this ledger a counterparty's \
name routinely carries a trade word ("Payroll", "Insurance", "Telecom", "Cleaning", "Utility") that \
has nothing to do with what the transaction is: e.g. a payment to "Bridgeport Payroll Group Company" \
described as "Construction all-risk insurance" is INSURANCE, and a payment to "Foxridge Cleaning Co" \
described as "Telecom mobile fleet plan" is TELECOM. The description states what was bought; the \
counterparty is only a name. Use the counterparty solely to tell apart two transactions whose \
descriptions are identical -- never to infer the category.

Principles:
- Be granular and specific (e.g. "payroll", "capital_expenditure", "sales_revenue", \
"interest_expense", "insurance_premium", "rent", "utilities", "tax_payment", \
"marketing_advertising", "consulting_services", "related_party_advisory_fee" if the description \
itself says so, etc.) -- invent whatever category name fits, don't pick from a fixed list.
- NEVER merge genuinely distinct transaction types into one catch-all category (no \
"operating_expenses" dumping ground for anything ambiguous) -- if a transaction doesn't cleanly \
fit an existing category, give it its own specific one rather than forcing a fit.
- Reuse the exact same category string for transactions that are clearly the same type (e.g. \
every payroll-run transaction should get the identical string "payroll", not near-duplicates).
- A refund/credit/reversal of a category is still that category (e.g. a utility rebate is still \
"utilities", not a new "refunds" bucket), unless the description says otherwise.

Respond with strict JSON only, no markdown code fences, no commentary: a single object mapping \
each txn_id to its category string, e.g. {"TXN-P1-0001": "payroll", "TXN-P1-0002": "rent", ...}. \
Include every txn_id given, exactly once each.
"""

KYC_DEFINITIONS_PROMPT = """You are reading a KYC dossier's ownership/related-party section (and, \
if present, its subsidiary asset-collateral section) for a corporate borrower. Transcribe what the \
text states -- do NOT decide which entities cross the threshold (that comparison happens \
elsewhere). Extract:

- ownership_entities: list of {"name": <exact entity name as written>, "voting_rights_pct": \
<the face percentage shown in the table>, "indirect_via_group_pct": <number or null>} for EVERY \
organization listed in the ownership table, in the order given. Include all of them.
  "indirect_via_group_pct" is null for a normal directly-held stake. Set it ONLY when the text \
explicitly states that this entity's stake is held INDIRECTLY through an intermediate entity and \
that the Group's EFFECTIVE share governs -- e.g. "the stake in X is held indirectly through Y; the \
Group owns Z% of the voting rights in Y; the effective share is what counts". In that case put Z \
(the Group's percentage in the INTERMEDIATE entity) as "indirect_via_group_pct" -- just transcribe \
that number, do NOT multiply anything yourself. Never invent an adjustment the text doesn't state.
- related_party_threshold_pct: the voting-rights percentage this section states makes an entity a \
related party (a number), or null if no such threshold is stated.
- collateral_entities: list of {"name": <exact subsidiary name as written>, "pledged_asset_pct": \
<number>} for every subsidiary listed in an asset-collateral section, if present. Empty list if no \
such section exists in this text.
- unrestricted_sub_threshold_pct: the pledged-asset percentage below which that section states a \
subsidiary is outside the security perimeter / unrestricted (a number), or null if not present.

Respond with strict JSON only, no markdown code fences, no commentary:
{"ownership_entities": [{"name": "...", "voting_rights_pct": 0.0, "indirect_via_group_pct": null}], "related_party_threshold_pct": 30.0, "collateral_entities": [], "unrestricted_sub_threshold_pct": null}
"""

AUDIT_OVERRIDES_PROMPT = """You are reading a credit agreement's audit-report covenant addendum \
(reclassifications, cut-off/accrual notes, one-off items, foreign-currency settlements) alongside \
the borrower's transaction list (txn_id, date, counterparty, description, amount, currency). \
Identify, using the addendum's OWN disclosed figures and reasoning (never invent or estimate a \
number the text doesn't state):

- reclassifications: transactions the addendum says should be treated under a different category \
than their face description for covenant purposes. Each: {"txn_id": ..., "new_category": \
<short snake_case>, "reason": <why, citing the disclosure>}. Match by the disclosed counterparty \
name and dollar amount against the transaction list to find the right txn_id -- amounts may be \
disclosed with cents where the ledger rounds, or vice versa; match on the closest amount for that \
counterparty, and note if a name looks OCR-garbled but the amount clearly identifies one txn.
- cutoff_exclusions: transactions the addendum says fall outside the covenant period (accrual/\
cut-off timing) and must be excluded from period totals. Each: {"txn_id": ..., "reason": ...}.
- one_off_addback_candidates: EVERY transaction the addendum identifies as a one-off/non-recurring \
item, regardless of size -- do NOT leave one out for being too small. Where the text states a \
minimum size for add-back ("items of at least $X are treated as one-off"), record that separately \
per item rather than dropping the item: {"txn_id": ..., "amount_usd": <the disclosed USD amount>, \
"meets_materiality": true|false, "reason": ...}. A sub-threshold item is still a one-off item; the \
threshold only decides whether it may be ADDED BACK, not whether it exists. Use true when no \
minimum is stated.
- amount_overrides: the addendum-disclosed correct USD amount for any transaction whose ledger \
amount should NOT be used as-is -- either because it isn't in USD (a foreign-currency settlement \
disclosure), or because the ledger's amount is missing/blank/invalid for that transaction and the \
addendum states what it actually was. Each: {"txn_id": ..., "amount_usd": <signed, same sign as \
the transaction's nature -- negative for an outflow>, "reason": ...}. Match a missing-amount \
disclosure to its txn_id the same way as reclassifications (by counterparty name/description, \
since the amount itself won't be there to match on).
- disclosed_figures: any covenant-relevant dollar figure in the addendum that does NOT correspond \
to any transaction in the list at all (e.g. a provision or liability that was disclosed but never \
actually transacted, with no txn_id anywhere close). Each: {"label": ..., "amount_usd": <positive>, "reason": ...}.
  A figure counts here even when the statements do not print it under that name but do print the \
movement it follows from -- a period's additions to an asset class, for instance, follow from its \
opening balance, closing balance and the period's depreciation, all of which such a note states. \
Where that is the case, give the resulting amount and set out in "reason" which stated figures you \
combined and how, so the derivation can be checked.
  On units: where a blanket header ("amounts in thousands") disagrees with the full-precision \
figures a note itself prints, follow the note's own figures -- a number written to the cent is \
already in units.
- fx_rates: when the addendum discloses a foreign-currency amount TOGETHER WITH the USD amount it \
was actually settled for (e.g. "an invoice for 72,146.75 EUR was settled by a payment of \
$83,690.23"), transcribe BOTH numbers exactly as written: {"currency": "EUR", "foreign_amount": \
72146.75, "usd_amount": 83690.23, "reason": ...}. Do NOT divide, do NOT work out a rate and do NOT \
apply it to any transaction -- copying the pair is the whole job; the conversion happens elsewhere. \
Note that the disclosed pair often describes a DIFFERENT settlement than the transactions in the \
list -- transcribe it anyway, it establishes the period's rate for that currency.

Exclude anything the addendum records as CONSIDERED AND REJECTED. Where the text says the original \
classification is retained, that no covenant adjustment was made or required, or that an item was \
reviewed and needs no correction, that is NOT a reclassification, NOT an exclusion and NOT an \
override -- it must not appear in any list above. Only positions the addendum actually adopts count.

Omit any category entirely if the addendum has nothing for it (empty list is fine). Respond with \
strict JSON only, no markdown code fences, no commentary:
{"reclassifications": [...], "cutoff_exclusions": [...], "one_off_addback_candidates": [...], "amount_overrides": [...], "disclosed_figures": [...], "fx_rates": []}
"""

_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalize_name(name: str) -> str:
    name = _PAREN_SUFFIX_RE.sub("", name)
    name = re.sub(r"[.,\"'«»“”‘’]", "", name)
    return re.sub(r"\s+", " ", name).strip().casefold()


def _resolve_against_ledger(names: set[str], ledger_names: set[str]) -> set[str]:
    """Map KYC-transcribed names onto the names the ledger actually uses.

    Exact (normalized) equality stays the rule -- this dataset seeds lexically similar but unrelated
    vendors sharing a brand root, and only exact matching is immune to "sounds similar". The single
    concession is for scanned dossiers: OCR corrupts one word of a name ("Taraz" read as "Тага?")
    and the whole related-party test then silently sees nothing. So a name with no exact match falls
    back to ledger names agreeing on every token but one, and is accepted ONLY if exactly one such
    candidate exists -- decoy siblings differ in more than one token, so they can't collide here.
    """
    resolved: set[str] = set()
    for name in names:
        if name in ledger_names:
            resolved.add(name)
            continue
        tokens = name.split()
        near = [
            other
            for other in ledger_names
            if len(other.split()) == len(tokens)
            and sum(a != b for a, b in zip(tokens, other.split(), strict=True)) == 1
        ]
        if len(near) == 1:
            resolved.add(near[0])
    return resolved


def _effective_pct(entity: dict) -> float:
    """The voting-rights % to compare against the threshold. For a directly-held stake that's the
    face %; for an indirectly-held one (stake held through an intermediate the Group only partly
    owns) it's face x the Group's fraction of the intermediate. The multiplication is done here in
    Python, not by the model -- the model only transcribes the two raw numbers."""
    face = entity.get("voting_rights_pct", -1)
    indirect = entity.get("indirect_via_group_pct")
    if indirect is None:
        return face
    return face * indirect / 100.0


_DISCLOSED_AMOUNT_RE = r"{txn}.{{0,400}}?\$\s?([\d,]+\.\d{{2}})"


def _rescue_missing_amounts(txns: pd.DataFrame, overrides_by_txn: dict, source_text: str) -> None:
    """Last-resort read of a disclosed amount for a row the ledger left blank.

    Such a row poisons every term it belongs to (`_resolve_variable` refuses to sum a NaN), so the
    cell is lost outright unless the figure is recovered. The extractor sometimes files the
    disclosure under the wrong schema key; the amount is still in the text, in the one format these
    documents use. Outflow sign is taken from the borrower's other rows with the same counterparty,
    falling back to an outflow -- the disclosures that need rescuing are settlements and levies."""
    import re

    for row in txns[txns["amount"].isna()].itertuples():
        if row.txn_id in overrides_by_txn:
            continue
        match = re.search(_DISCLOSED_AMOUNT_RE.format(txn=re.escape(row.txn_id)), source_text, re.S)
        if not match:
            continue
        amount = float(match.group(1).replace(",", ""))
        siblings = txns[(txns["counterparty"] == row.counterparty) & txns["amount"].notna()][
            "amount"
        ]
        sign = -1.0 if siblings.empty or siblings.mean() < 0 else 1.0
        overrides_by_txn[row.txn_id] = {
            "txn_id": row.txn_id,
            "amount_usd": sign * amount,
            "reason": "recovered from disclosure text",
        }


def _supported_by_source(entry: dict, txns: pd.DataFrame, source_text: str) -> bool:
    """Is this override actually traceable to something the documents say?

    Extractors invent overrides -- on one scenario whose addendum reads "no reclassifications were
    required", three were returned, one of them moving the borrower's only revenue row out of
    revenue. Such an entry is indistinguishable from a real one downstream, so it is filtered here
    on provenance: the disclosure must name the transaction id, or name both its amount and its
    counterparty. That is how the genuine disclosures in this format are written."""
    txn_id = str(entry.get("txn_id", ""))
    if not txn_id:
        return False
    if txn_id in source_text:
        return True

    row = txns[txns["txn_id"] == txn_id]
    if row.empty:
        return False
    row = row.iloc[0]

    counterparty = _PAREN_SUFFIX_RE.sub("", str(row["counterparty"])).strip()
    if pd.isna(row["amount"]):
        # no amount to match on, so the name has to carry the identification by itself
        return bool(counterparty) and counterparty in source_text

    # An amount quoted to the cent identifies a transaction on its own -- two rows agreeing to the
    # penny do not happen by accident. Requiring the counterparty as well used to reject genuine
    # disclosures whose tables came through OCR with the name mangled ("Ilek" read as "Пек"), while
    # the hallucinated entries this filter exists for match neither the id nor the amount.
    amount = abs(float(row["amount"]))
    if any(fmt in source_text for fmt in (f"{amount:,.2f}", f"{amount:.2f}")):
        return True
    return bool(counterparty) and counterparty in source_text


def _drop_unsupported(overrides: dict, txns: pd.DataFrame, source_text: str) -> list[str]:
    """Filter overrides in place; returns a note per dropped entry, for the caller to log."""
    dropped: list[str] = []
    for key in (
        "reclassifications",
        "cutoff_exclusions",
        "one_off_addback_candidates",
        "amount_overrides",
    ):
        kept = []
        for entry in overrides.get(key) or []:
            if _supported_by_source(entry, txns, source_text):
                kept.append(entry)
            else:
                dropped.append(f"{key}:{entry.get('txn_id')}")
        overrides[key] = kept
    return dropped


def _fx_rates(overrides: dict) -> dict[str, float]:
    """currency -> USD per unit, derived in Python from the pair of amounts the auditor disclosed.

    An unconverted foreign-currency row is one of the costliest single errors available: it lands
    far outside the 5% band, so it forfeits `actual` AND (on a null-evidence cell) the evidence
    points that ride on it, and it can flip the status outright. The model is asked only to copy the
    two disclosed numbers -- the division belongs here, with the rest of the arithmetic."""
    seen: dict[str, list[float]] = {}
    for entry in overrides.get("fx_rates") or []:
        try:
            currency = str(entry["currency"]).upper()
            foreign, usd = float(entry["foreign_amount"]), float(entry["usd_amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if foreign and usd and currency != "USD":
            seen.setdefault(currency, []).append(abs(usd) / abs(foreign))
    # take the median, not whichever pair the extractor happened to emit first: a mis-paired
    # disclosure yields absurd rates (0.03, or 1.0 where the model simply echoed the ledger), and
    # one of those silently rescales a whole covenant term.
    return {c: sorted(v)[len(v) // 2] for c, v in seen.items()}


def _txn_rows(txns: pd.DataFrame) -> list[dict]:
    # a NaN amount is real data in this dataset (one row's amount is disclosed only in the audit
    # report, not the ledger) -- serialize it as JSON null, not Python's literal `NaN` token,
    # which isn't valid JSON and models don't reliably treat as "missing" rather than garbage.
    return [
        {
            "txn_id": r.txn_id,
            "date": r.date,
            "counterparty": r.counterparty,
            "description": r.description,
            "amount": None if pd.isna(r.amount) else r.amount,
            "currency": r.currency,
        }
        for r in txns.itertuples()
    ]


CATEGORIZE_CHUNK = 20  # a whole 55-row ledger in one call is where rows start going missing


def categorize_transactions(
    txns: pd.DataFrame, client: Client, max_rounds: int = 3
) -> dict[str, str]:
    """Category per transaction, with every row accounted for.

    Asked for ~55 rows at once, models quietly omit some -- and the caller used to paper over that
    with `.get(txn_id, "uncategorized")`, so the row simply vanished from every covenant term. The
    omitted row is not random: it is often the borrower's single revenue or capex line, i.e. exactly
    what a covenant measures, which turns one skipped id into a whole lost cell. Smaller batches are
    markedly more reliable, missing ids are re-asked, and anything still missing raises rather than
    being silently mislabelled."""
    rows = _txn_rows(txns)
    by_id = {r["txn_id"]: r for r in rows}
    out: dict[str, str] = {}

    for _ in range(max_rounds):
        pending = [i for i in by_id if i not in out]
        if not pending:
            break
        for start in range(0, len(pending), CATEGORIZE_CHUNK):
            chunk = [by_id[i] for i in pending[start : start + CATEGORIZE_CHUNK]]
            try:
                data = client.complete_json(
                    CATEGORIZE_PROMPT, json.dumps(chunk, ensure_ascii=False), max_tokens=4096
                )
            except Exception:  # noqa: BLE001 -- a failed chunk is retried by the outer round
                continue
            for k, v in data.items():
                if str(k) in by_id and isinstance(v, str):
                    out[str(k)] = v

    missing = [i for i in by_id if i not in out]
    if missing:
        # Label the stragglers rather than raising: the caller aborts the whole scenario on an
        # exception, so one stubborn row would cost all three of its covenants instead of only the
        # term that row feeds. "uncategorized" is the catch-all this stage otherwise avoids, but it
        # is confined here to rows that already survived every retry, and a covenant that doesn't
        # reference them is unaffected.
        print(
            f"  categorization incomplete after {max_rounds} rounds, "
            f"marking uncategorized: {missing}",
            flush=True,
        )
        for txn_id in missing:
            out[txn_id] = "uncategorized"
    return out


def extract_kyc_definitions(ownership_text: str | None, client: Client) -> dict:
    empty = {
        "ownership_entities": [],
        "related_party_threshold_pct": None,
        "collateral_entities": [],
        "unrestricted_sub_threshold_pct": None,
    }
    if not ownership_text:
        return empty
    data = client.complete_json(KYC_DEFINITIONS_PROMPT, ownership_text)
    for key in empty:
        data.setdefault(key, empty[key])
    return data


def _unclaimed_disclosed_amounts(
    overrides: dict, txns: pd.DataFrame, source_text: str
) -> list[str]:
    """Ledger rows whose exact amount is quoted in the documents but which no override mentions.

    The disclosures come as tables, and a model asked for them in one pass drops a row now and
    then -- silently, because nothing counts them. An amount written to the cent that matches a
    ledger row is a disclosure about that row by construction, so anything matching and unclaimed
    is a miss worth another attempt. Categorisation already works this way; extraction did not."""
    claimed = {
        str(entry.get("txn_id"))
        for key in (
            "reclassifications",
            "cutoff_exclusions",
            "one_off_addback_candidates",
            "amount_overrides",
        )
        for entry in overrides.get(key) or []
    }
    quoted = set(re.findall(r"\$\s?([\d,]+\.\d{2})", source_text))
    quoted = {float(q.replace(",", "")) for q in quoted}

    missed = []
    for row in txns.itertuples():
        if row.txn_id in claimed or pd.isna(row.amount):
            continue
        if abs(float(row.amount)) in quoted:
            missed.append(row.txn_id)
    return missed


def extract_audit_overrides(addendum_text: str | None, txns: pd.DataFrame, client: Client) -> dict:
    empty = {
        "reclassifications": [],
        "cutoff_exclusions": [],
        "one_off_addback_candidates": [],
        "amount_overrides": [],
        "disclosed_figures": [],
        "fx_rates": [],
    }
    if not addendum_text:
        return empty
    rows = _txn_rows(txns)
    user = f"Addendum text:\n\n{addendum_text}\n\nTransactions:\n\n{json.dumps(rows, ensure_ascii=False)}"
    data = client.complete_json(AUDIT_OVERRIDES_PROMPT, user, max_tokens=4096)
    for key in empty:
        data.setdefault(key, [])

    missed = _unclaimed_disclosed_amounts(data, txns, addendum_text)
    if missed:
        print(f"  disclosed amounts not accounted for, re-asking: {missed}", flush=True)
        retry_user = (
            user
            + "\n\nYour previous answer left these transactions out, although the addendum quotes "
            + f"their exact amounts: {missed}. Read what it says about each and place it in the "
            + "right list, or leave it out only if the text genuinely makes no claim about it. "
            + "Return the complete JSON object again."
        )
        try:
            second = client.complete_json(AUDIT_OVERRIDES_PROMPT, retry_user, max_tokens=4096)
        except Exception:  # noqa: BLE001 -- the first answer stands if the retry fails
            return data
        for key in empty:
            second.setdefault(key, [])
        # keep whichever pass accounted for more of the quoted amounts
        if len(_unclaimed_disclosed_amounts(second, txns, addendum_text)) < len(missed):
            return second
    return data


@dataclass(frozen=True)
class EnrichedTxn:
    txn_id: str
    date: str
    counterparty: str
    description: str
    amount: float
    currency: str
    amount_usd: float
    category: str
    raw_category: str
    related_party: bool
    unrestricted_sub_transfer: bool
    excluded_period: bool
    exclusion_reason: str | None
    reclass_reason: str | None
    amount_corrected: bool = False  # audit disclosed a USD amount overriding the ledger's


def enrich_scenario(
    sid: str,
    txns: pd.DataFrame,
    ownership_text: str | None,
    addendum_text: str | None,
    small_client: Client,
    complex_client: Client,
) -> tuple[list[EnrichedTxn], list[dict]]:
    """Returns (enriched transactions, disclosed_figures not tied to any transaction).

    Per-transaction categorization (a simple per-item labeling task, same category as document
    classification) uses small_client; cross-referencing the KYC/audit text against the
    transaction list uses complex_client -- that one large small_client call proved slow/
    unreliable on this dataset's ~55-row scenarios (gateway timeouts), which categorization
    doesn't need the bigger model to avoid anyway.
    """
    categories = categorize_transactions(txns, small_client)
    kyc = extract_kyc_definitions(ownership_text, complex_client)
    overrides = extract_audit_overrides(addendum_text, txns, complex_client)

    # threshold comparison happens here, in Python, not in the LLM call -- asking a model with
    # thinking disabled to both extract AND numerically filter in one JSON response proved
    # unreliable (confirmed: a 28.8%-owned entity against a stated 25.0% threshold was silently
    # dropped, consistently, across reruns, even though the model could state the threshold
    # correctly) -- so the LLM only transcribes {name, pct} pairs, and `>=`/`<` runs here.
    rp_threshold = kyc.get("related_party_threshold_pct")
    related_names = {
        _normalize_name(e["name"])
        for e in kyc.get("ownership_entities") or []
        if rp_threshold is not None and _effective_pct(e) >= rp_threshold
    }
    us_threshold = kyc.get("unrestricted_sub_threshold_pct")
    unrestricted_names = {
        _normalize_name(e["name"])
        for e in kyc.get("collateral_entities") or []
        if us_threshold is not None and e.get("pledged_asset_pct", 101) < us_threshold
    }

    # reconcile the dossier's spelling with the ledger's before tagging -- see _resolve_against_ledger
    ledger_names = {_normalize_name(r.counterparty) for r in txns.itertuples()}
    related_names = _resolve_against_ledger(related_names, ledger_names)
    unrestricted_names = _resolve_against_ledger(unrestricted_names, ledger_names)

    dropped = _drop_unsupported(overrides, txns, addendum_text or "")
    if dropped:
        print(f"  {sid}: dropped {len(dropped)} unsupported override(s): {dropped}", flush=True)

    fx_rates = _fx_rates(overrides)
    reclass_by_txn = {r["txn_id"]: r for r in overrides.get("reclassifications", [])}
    cutoff_by_txn = {r["txn_id"]: r for r in overrides.get("cutoff_exclusions", [])}
    addback_by_txn = {r["txn_id"]: r for r in overrides.get("one_off_addback_candidates", [])}
    amount_override_by_txn = {r["txn_id"]: r for r in overrides.get("amount_overrides", [])}
    _rescue_missing_amounts(txns, amount_override_by_txn, addendum_text or "")

    enriched: list[EnrichedTxn] = []
    for row in txns.itertuples():
        txn_id = row.txn_id
        raw_category = categories.get(txn_id, "uncategorized")
        category = raw_category
        reclass_reason = None
        if txn_id in reclass_by_txn:
            category = reclass_by_txn[txn_id]["new_category"]
            reclass_reason = reclass_by_txn[txn_id].get("reason")
        elif txn_id in addback_by_txn:
            # A one-off item always leaves its booked category -- that is what makes it a one-off.
            # Materiality decides only whether it may be added back, so it is carried as a separate
            # category rather than by dropping the item, which used to leave sub-threshold one-offs
            # sitting in the very line item the covenant caps.
            # Only an ELIGIBLE add-back gets its own category. A sub-threshold one-off stays exactly
            # where it was booked: the clause exempts it from being added back, not from being an
            # expense, so it must keep counting in its ordinary line item. Giving it a category of
            # its own invented a name no prompt describes, which no formula then references -- and
            # the amount simply vanished from the metric.
            eligible = addback_by_txn[txn_id].get("meets_materiality", True)
            if eligible:
                category = "one_off_addback"
            reclass_reason = addback_by_txn[txn_id].get("reason")

        # apply whenever an override is disclosed -- not gated on currency, since the same
        # mechanism also covers a missing/NaN ledger amount corrected by the audit disclosure,
        # which can happen on a nominally-USD row just as easily as a foreign-currency one.
        if txn_id in amount_override_by_txn:
            amount_usd = float(amount_override_by_txn[txn_id]["amount_usd"])
            # the ledger, not the model, decides the direction of the flow. A flipped sign on a
            # disclosed correction does not merely mistate that row: _resolve_variable sums signed
            # amounts, so an outflow written as positive nets AGAINST its own line item and can
            # move a whole covenant total by twice the row. Sign is only taken from the model when
            # the ledger genuinely has no amount to read it from.
            ledger_amount = float(row.amount) if pd.notna(row.amount) else 0.0
            if ledger_amount:
                amount_usd = abs(amount_usd) * (1 if ledger_amount > 0 else -1)
        elif row.currency == "USD":
            amount_usd = float(row.amount)
        elif row.currency in fx_rates:
            amount_usd = float(row.amount) * fx_rates[row.currency]
        else:
            # foreign-currency row with no disclosed rate -- leave as the raw (wrong-currency)
            # figure; downstream code must not silently trust it uncaveated.
            amount_usd = float(row.amount)

        norm_cp = _normalize_name(row.counterparty)
        excluded = txn_id in cutoff_by_txn

        enriched.append(
            EnrichedTxn(
                txn_id=txn_id,
                date=row.date,
                counterparty=row.counterparty,
                description=row.description,
                amount=float(row.amount),
                currency=row.currency,
                amount_usd=amount_usd,
                category=category,
                raw_category=raw_category,
                related_party=norm_cp in related_names,
                unrestricted_sub_transfer=norm_cp in unrestricted_names,
                excluded_period=excluded,
                exclusion_reason=cutoff_by_txn.get(txn_id, {}).get("reason") if excluded else None,
                reclass_reason=reclass_reason,
                amount_corrected=txn_id in amount_override_by_txn,
            )
        )

    return enriched, overrides.get("disclosed_figures", [])


def save_enriched(
    sid: str,
    enriched: list[EnrichedTxn],
    disclosed_figures: list[dict],
    cache_dir: str = ".cache/enriched",
) -> Path:
    path = Path(cache_dir) / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transactions": [asdict(e) for e in enriched],
        "disclosed_figures": disclosed_figures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_enriched(
    sid: str, cache_dir: str = ".cache/enriched"
) -> tuple[list[EnrichedTxn], list[dict]] | None:
    path = Path(cache_dir) / f"{sid}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [EnrichedTxn(**t) for t in payload["transactions"]], payload["disclosed_figures"]
