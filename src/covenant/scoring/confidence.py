"""Per-cell confidence signals -- a diagnostic layer, not a tuner.

On the private set there is no ground truth, so "robustness" reduces to knowing which of your
answered cells to trust and which to hand-review. Every signal here is computed deterministically
from artifacts you already have (the spec, the enriched ledger, the computed result) -- no LLM
calls, and nothing that reads ground truth, so it can neither cost much nor overfit.

The signals target the ways a cell can be quietly wrong:
- threshold_margin: how close the computed value sits to the covenant's threshold. A cell at
  0.041 vs a 0.04 limit could flip with a single misclassified transaction; one at 9.0 vs 0.15
  is unshakeable. Small margin -> low confidence.
- top_txn_share: how much of the deciding sum rides on ONE transaction. If a single row is 80% of
  the numerator, the whole verdict rests on that one row's category/tag being right.
- structural flags: a related-party covenant that matched zero related parties despite a stated
  threshold; a foreign-currency transaction with no disclosed USD rate; a compute error. Each is a
  concrete "this cell is probably missing something" signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from covenant.analyze.enrich import EnrichedTxn
from covenant.scoring.compute import (
    _formula_names,
    _period_filtered,
    _resolve_all,
    _safe_eval,
    compute_covenant,
)

# margins tighter than this (relative to threshold) mean one misclassification could flip the
# verdict; between the two, it's shaky but not knife-edge.
_LOW_MARGIN = 0.05
_MED_MARGIN = 0.15
# a single transaction contributing more than this share of the deciding sum makes the cell
# hostage to that one row's classification.
_DOMINANCE = 0.60


@dataclass(frozen=True)
class CellConfidence:
    level: str  # "high" | "medium" | "low" | "error"
    signals: dict
    flags: list[str]


def _participating_txns(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> list[EnrichedTxn]:
    names = _formula_names(cov)
    out = []
    for e in _period_filtered(enriched, cov):
        role = roles.get(e.category, e.category)
        if (
            role in names
            or e.category in names
            or (e.related_party and "related_party_payments" in names)
            or (e.unrestricted_sub_transfer and "unrestricted_sub_transfers" in names)
        ):
            out.append(e)
    return out


def cell_confidence(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> CellConfidence:
    try:
        result = compute_covenant(cov, roles, enriched)
    except Exception as exc:  # noqa: BLE001 -- a compute failure is itself the lowest-confidence signal
        return CellConfidence(level="error", signals={}, flags=[f"compute_error: {exc}"])

    signals: dict = {}
    flags: list[str] = []
    names = _formula_names(cov)

    # threshold margin -- how far the computed value is from the limit, relative to the limit
    threshold = cov.get("threshold")
    margin = None
    precondition = cov.get("precondition")
    if precondition and result.status == "COMPLIANT":
        # a springing covenant that didn't trigger is COMPLIANT by construction, not by margin
        signals["springing_not_triggered"] = True
    elif threshold:
        try:
            computed = _safe_eval(
                cov["formula"], _resolve_all(cov["formula"], cov, roles, enriched)
            )
            margin = abs(computed - threshold) / abs(threshold)
        except Exception:  # noqa: BLE001
            margin = None
    signals["threshold_margin_pct"] = round(margin * 100, 1) if margin is not None else None

    # single-transaction dominance of the deciding sum
    txns = _participating_txns(cov, roles, enriched)
    signals["n_participating_txns"] = len(txns)
    contribs = sorted(
        (abs(e.amount_usd) for e in txns if e.amount_usd == e.amount_usd), reverse=True
    )
    total = sum(contribs)
    top_share = contribs[0] / total if total > 0 and contribs else None
    signals["top_txn_share_pct"] = round(top_share * 100, 1) if top_share is not None else None

    # structural flags
    if "related_party_payments" in names and not any(e.related_party for e in enriched):
        flags.append("related_party_covenant_but_zero_matches")
    if "unrestricted_sub_transfers" in names and not any(
        e.unrestricted_sub_transfer for e in enriched
    ):
        flags.append("unrestricted_sub_covenant_but_zero_matches")
    if any(e.currency != "USD" for e in txns):
        flags.append("foreign_currency_transaction_in_scope")
    carve = cov.get("carve_out")
    if isinstance(carve, dict) and carve.get("kind") == "discretionary":
        # status computed from the raw threshold, but the clause permits a discretionary exception
        # we can't evaluate from the data -- the verdict could be COMPLIANT even if we say BREACH.
        flags.append("discretionary_carve_out_not_evaluable")

    # roll up to a level
    level = "high"
    if margin is not None and margin < _MED_MARGIN:
        level = "low" if margin < _LOW_MARGIN else "medium"
    if top_share is not None and top_share > _DOMINANCE and level == "high":
        level = "medium"
    if flags:
        level = "low"
    return CellConfidence(level=level, signals=signals, flags=flags)
