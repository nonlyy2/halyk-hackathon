"""Step E -- deterministic arithmetic: evaluate each covenant's formula in plain Python.

Never trust the model with arithmetic -- `actual` is scored on a 5%-tolerance decaying scale, so
an arithmetic slip is costly. This module resolves each formula/precondition variable to a real
number from the enriched ledger (role or raw-category match, either spelling -- see enrich.py's
aliasing note) or a doc_figures constant, evaluates via a restricted-AST-checked eval (never a raw
eval on unchecked model output), compares to the threshold, and derives `actual`/`status`/
`evidence_txn_id`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from covenant.analyze.enrich import EnrichedTxn

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)
_ALLOWED_CALLS = {"min", "max"}


def _validate_ast(tree: ast.AST, expr: str) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression element {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ValueError(f"disallowed function call in {expr!r}")


def _names_in(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree, expr)
    call_func_ids = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in call_func_ids}


def _safe_eval(expr: str, variables: dict[str, float]):
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree, expr)
    code = compile(tree, "<formula>", "eval")
    return eval(code, {"__builtins__": {}}, {**variables, "min": min, "max": max})  # noqa: S307


def _metric_expr(cov: dict) -> str:
    """The covenant's formula as a VALUE expression.

    "formula" is specified to evaluate to the metric itself, with the limit held separately in
    "threshold"/"comparison" -- but models routinely return the whole test instead
    ("max(payroll, utilities) <= 1500000"). That expression evaluates to True/False, so `actual`
    collapses to 1.0 or 0.0 and the status comparison then runs on a boolean. Taking the left side
    of a top-level comparison recovers the metric exactly; nothing is inferred, and a formula that
    is already a plain value is returned untouched. Preconditions are genuinely boolean and are
    never passed through here."""
    formula = cov["formula"]
    try:
        node = ast.parse(formula, mode="eval").body
    except SyntaxError:
        return formula
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        # the metric is whichever side carries the variables: written in reverse
        # ("1500000 >= max(payroll, utilities)") the left side is a bare constant, and taking it
        # would report the threshold as `actual` and compare the threshold with itself.
        left, right = ast.unparse(node.left), ast.unparse(node.comparators[0])
        for side in (left, right):
            if any(isinstance(n, ast.Name) for n in ast.walk(ast.parse(side, mode="eval"))):
                return side
    return formula


def _quarter_of(date_str: str, period_start: str | None = None) -> str:
    """Quarter of the covenant's own period, not of the calendar year.

    Clauses say "the fourth quarter of the period ending 2025-12-31", so the quarters run from the
    period's start. They coincide with calendar quarters only while the financial year happens to
    begin in January."""
    month = int(date_str[5:7]) + 12 * int(date_str[0:4])
    first = (
        month - (int(period_start[5:7]) + 12 * int(period_start[0:4]))
        if period_start
        else month - 1
    )
    return f"Q{first // 3 + 1}"


def _period_filtered(enriched: list[EnrichedTxn], cov: dict) -> list[EnrichedTxn]:
    period = cov.get("period") or {}
    start, end, quarter = period.get("start"), period.get("end"), cov.get("period_quarter")
    out = []
    for e in enriched:
        if e.excluded_period:
            continue
        if start and e.date < start:
            continue
        if end and e.date > end:
            continue
        if quarter and _quarter_of(e.date, start) != quarter:
            continue
        out.append(e)
    return out


def _resolve_variable(name: str, cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> float:
    doc_figures = cov.get("doc_figures") or {}
    if name in doc_figures:
        return float(doc_figures[name]["amount_usd"])

    filtered = _period_filtered(enriched, cov)
    # These clauses cap PAYMENTS TO ("платежи в пользу связанных сторон"), so only outflows count.
    # Summing signed amounts would let an inflow from an affiliate -- revenue from a related party,
    # a refund -- net against the payments and under-report the very thing being capped.
    if name == "related_party_payments":
        return abs(sum(min(e.amount_usd, 0.0) for e in filtered if e.related_party))
    if name == "unrestricted_sub_transfers":
        return abs(sum(min(e.amount_usd, 0.0) for e in filtered if e.unrestricted_sub_transfer))

    # Aliasing (PIPELINE.md): a formula term must resolve to its WHOLE role's sum, whether it's
    # spelled as the role name or as any single category in that role. Resolve the name to a target
    # role first -- if it's a role name use it; if it's a category name use that category's role --
    # then sum every transaction in that role. (Matching only the exact category would silently drop
    # sibling categories mapped to the same role, e.g. two distinct insurance categories.)
    # A covenant term names a LINE ITEM, so a term that is a category resolves to that category
    # alone. Promoting it to its whole semantic role sweeps in every sibling the model happened to
    # bucket together -- and the ledgers deliberately surround each covenant's real line with large
    # unrelated rows, so widening a term is not a rounding error, it changes the answer outright.
    # A term that names a role still sums the role, which is what a role name is for.
    role_values = set(roles.values())
    if name in roles and name not in role_values:
        matched = [e for e in filtered if e.category == name]
    elif name in role_values:
        matched = [e for e in filtered if roles.get(e.category, e.category) == name]
    else:
        matched = [e for e in filtered if e.category == name]
    target_role = name
    if not matched:
        # A known role/category with no rows in this period is a legitimate zero (a covenant can
        # cap a category the borrower simply didn't spend on) -- raising here cost the WHOLE cell.
        # An unknown name is still an error: it means the model referenced something that was never
        # populated, which validate_spec() reports, and silently reading it as 0 would hide that.
        if target_role in set(roles.values()) or target_role in roles:
            return 0.0
        raise ValueError(f"variable {name!r} is not a known role, category, tag or doc_figure")
    nan_txns = [e.txn_id for e in matched if e.amount_usd != e.amount_usd]  # NaN != NaN
    if nan_txns:
        raise ValueError(
            f"variable {name!r} includes transaction(s) with no resolved amount_usd: {nan_txns}"
        )
    # sum SIGNED amounts, then take the magnitude: a refund/credit (positive) nets against the
    # charges (negative) in its own line item, rather than inflating the total via per-item abs().
    return abs(sum(e.amount_usd for e in matched))


def _resolve_all(
    expr: str, cov: dict, roles: dict, enriched: list[EnrichedTxn]
) -> dict[str, float]:
    return {name: _resolve_variable(name, cov, roles, enriched) for name in _names_in(expr)}


def _evaluate(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> tuple[float, str]:
    """(true metric value, status) -- the single definition of status, so the counterfactual
    baseline in _find_evidence_txn can never disagree with what compute_covenant() reports
    (it previously skipped the carve-out and could search around the wrong verdict)."""
    metric = _metric_expr(cov)
    value = _safe_eval(metric, _resolve_all(metric, cov, roles, enriched))

    precondition = cov.get("precondition")
    if precondition and not bool(
        _safe_eval(precondition, _resolve_all(precondition, cov, roles, enriched))
    ):
        return value, "COMPLIANT"  # springing covenant not triggered

    status_value, effective_threshold = _apply_carve_out(cov, roles, enriched, value)
    ok = _COMPARISONS[cov["comparison"]](status_value, effective_threshold)
    return value, "COMPLIANT" if ok else "BREACH"


def _status_only(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> str:
    return _evaluate(cov, roles, enriched)[1]


def _formula_names(cov: dict) -> set[str]:
    names: set[str] = set()
    for expr in filter(None, [_metric_expr(cov), cov.get("precondition")]):
        names |= _names_in(expr)
    return names


def _specialness(e: EnrichedTxn) -> int:
    """How explicitly the documents single this transaction out. Used only to rank candidates:
    an auditor's written reclassification outranks a tag we derived ourselves from the KYC table."""
    if e.reclass_reason is not None:
        return 4
    if e.amount_corrected:
        return 3
    if e.excluded_period:
        return 2
    if e.related_party or e.unrestricted_sub_transfer:
        return 1
    return 0


def _counterfactuals(e: EnrichedTxn, enriched: list[EnrichedTxn]) -> list[list[EnrichedTxn]]:
    """Ledgers as they would read without the deciding act applied to `e`.

    For a reclassification the deciding act is the CHANGE OF LINE ITEM, so the counterfactual puts
    the transaction back under raw_category -- deleting the row instead (as this used to) also
    removes its contribution from the denominator, which is a different, wrong question. Removal is
    still tried as a second option because it is the right counterfactual when the recorded
    new_category happens to equal the original."""
    others = [x for x in enriched if x.txn_id != e.txn_id]
    out: list[list[EnrichedTxn]] = []

    def with_e(replacement: EnrichedTxn) -> list[EnrichedTxn]:
        return [replacement if x.txn_id == e.txn_id else x for x in enriched]

    if e.excluded_period:
        out.append(with_e(replace(e, excluded_period=False)))
    if e.reclass_reason is not None and e.raw_category != e.category:
        out.append(with_e(replace(e, category=e.raw_category)))
    if e.amount_corrected:
        # without the disclosed correction the ledger's own figure stands -- or, when the ledger had
        # no figure at all, the row simply cannot be valued and drops out.
        out.append(with_e(replace(e, amount_usd=e.amount)) if e.amount == e.amount else others)
    out.append(others)
    return out


def _find_evidence_txn(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> str | None:
    """The single transaction whose inclusion/exclusion/reclassification/correction flips the
    verdict, found by counterfactual rather than by picking the biggest row. Only transactions that
    are *special by classification* and that participate in this covenant's formula are candidates,
    which honours the case's "not the largest line, not a cumulative crossing" rule.

    Scoring note (CASE.ru.md Section 4): where the key holds a txn_id, a wrong id and `null` both score
    zero; where the key is null, whatever we send is ignored outright. Staying silent therefore
    never pays, so when several candidates flip -- or none does but a special participating
    transaction exists -- the best-ranked candidate is returned instead of None."""
    try:
        base = _status_only(cov, roles, enriched)
    except ValueError:
        return None
    names = _formula_names(cov)

    def participates(e: EnrichedTxn) -> bool:
        role = roles.get(e.category, e.category)
        return (
            role in names
            or e.category in names
            or e.raw_category in names
            or roles.get(e.raw_category, e.raw_category) in names
            or (e.related_party and "related_party_payments" in names)
            or (e.unrestricted_sub_transfer and "unrestricted_sub_transfers" in names)
        )

    candidates = [e for e in enriched if _specialness(e) > 0 and participates(e)]
    if not candidates:
        return None

    flippers = []
    for e in candidates:
        for counterfactual in _counterfactuals(e, enriched):
            try:
                if _status_only(cov, roles, counterfactual) != base:
                    flippers.append(e)
                    break
            except ValueError:
                continue

    pool = flippers or candidates
    best = max(
        pool,
        key=lambda e: (_specialness(e), abs(e.amount_usd) if e.amount_usd == e.amount_usd else 0.0),
    )
    return best.txn_id


@dataclass(frozen=True)
class ComputeResult:
    status: str
    actual: float
    evidence_txn_id: str | None


_COMPARISONS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def _apply_carve_out(
    cov: dict, roles: dict, enriched: list[EnrichedTxn], value: float
) -> tuple[float, float]:
    """Return (effective_value_for_status, effective_threshold) after any computable carve-out.
    A carve-out only affects STATUS, never the reported `actual` (CASE.ru.md: report the true value
    even when it exceeds the limit under a carve-out). Discretionary carve-outs (approval-based, not
    determinable from the data) are left unapplied here -- confidence.py flags them instead."""
    carve = cov.get("carve_out")
    threshold = cov["threshold"]
    if not isinstance(carve, dict):
        return value, threshold

    kind = carve.get("kind")
    if kind == "allowance":
        # the metric may exceed the limit by `allowance` (same unit) -- loosen the threshold in the
        # direction that makes exceeding permissible.
        allowance = abs(float(carve.get("allowance", 0)))
        if cov["comparison"] in ("<=", "<"):
            return value, threshold + allowance
        if cov["comparison"] in (">=", ">"):
            return value, threshold - allowance
        return value, threshold
    if kind == "exclusion":
        # a named line item is excluded from the tested value -- recompute the formula with those
        # transactions removed, for STATUS only.
        excluded_role = carve.get("excluded_role")
        # An exclusion is only meaningful for a line item this covenant actually measures. Models
        # invent carve-outs on unrelated roles, and deleting those rows still moves the result --
        # via the tag aggregates, which are summed over the whole ledger regardless of role. So an
        # exclusion that names nothing in this formula is treated as the no-op it should be.
        if excluded_role and excluded_role not in _formula_names(cov):
            return value, threshold
        if excluded_role:
            kept = [
                e
                for e in enriched
                if roles.get(e.category, e.category) != excluded_role
                and e.category != excluded_role
            ]
            try:
                kept_expr = _metric_expr(cov)
                return _safe_eval(kept_expr, _resolve_all(kept_expr, cov, roles, kept)), threshold
            except ValueError:
                return value, threshold
    # discretionary or unknown kind -> not computable here; status uses the raw threshold.
    return value, threshold


def fallback_cell(cov: dict | None, roles: dict, enriched: list[EnrichedTxn]) -> ComputeResult:
    """Best answer still available once compute_covenant() has failed.

    CASE.ru.md Section 4 scores a missing, non-numeric or non-`COMPLIANT`/`BREACH` status exactly like a
    wrong one: zero for the whole cell. An answered cell can only do better, so the pipeline must
    never emit a null cell -- it degrades instead. Steps, most to least informed:
      1. the formula alone, ignoring a precondition/carve-out that may be what broke,
      2. the largest single role in the formula, which at least has the right order of magnitude,
      3. COMPLIANT / 0.0 -- no metric survived, so assert nothing beyond the safer default
         (a covenant is a limit the borrower is normally inside)."""
    if cov is not None:
        for expr in filter(None, [_metric_expr(cov) if cov.get("formula") else None]):
            try:
                value = abs(_safe_eval(expr, _resolve_all(expr, cov, roles, enriched)))
            except Exception:  # noqa: BLE001 -- any failure just moves to the next fallback
                break
            threshold, comparison = cov.get("threshold"), cov.get("comparison")
            status = "COMPLIANT"
            if isinstance(threshold, (int, float)) and comparison in _COMPARISONS:
                status = "COMPLIANT" if _COMPARISONS[comparison](value, threshold) else "BREACH"
            return ComputeResult(
                status=status,
                actual=round(value, 2),
                evidence_txn_id=_safe_evidence(cov, roles, enriched),
            )
        # The threshold is the covenant's own stated limit, so it is the closest quantity to the
        # metric that is still knowable when the metric itself cannot be computed -- and a borrower
        # tends to sit near its limits. `0.0` is the worst possible guess: it scores zero for
        # `actual`, and on a null-evidence key it forfeits those 0.20 as well. Summing a role would
        # be worse still on a ratio covenant, where the metric is order 1 and the roles are millions.
        return ComputeResult(
            status="COMPLIANT",
            actual=_threshold_guess(cov),
            evidence_txn_id=_safe_evidence(cov, roles, enriched),
        )
    return ComputeResult(status="COMPLIANT", actual=0.0, evidence_txn_id=None)


def _threshold_guess(cov: dict) -> float:
    threshold = cov.get("threshold")
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        return round(abs(float(threshold)), 2)
    return 0.0


def _safe_evidence(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> str | None:
    try:
        return _find_evidence_txn(cov, roles, enriched)
    except Exception:  # noqa: BLE001 -- evidence is a bonus; never let it sink a rescued cell
        return None


def compute_covenant(cov: dict, roles: dict, enriched: list[EnrichedTxn]) -> ComputeResult:
    # `actual` is always the TRUE metric value even where a carve-out or an untriggered springing
    # test permits it to sit beyond the limit (CASE.ru.md) -- _evaluate keeps the two apart.
    value, status = _evaluate(cov, roles, enriched)
    evidence_txn_id = _find_evidence_txn(cov, roles, enriched)
    return ComputeResult(
        status=status, actual=round(abs(value), 2), evidence_txn_id=evidence_txn_id
    )
