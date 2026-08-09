"""Step B/C/D -- turn each covenant clause into a structured, computable spec.

One LLM call per scenario produces:

- "roles": every enriched-ledger category mapped to a short semantic role (capital_expenditure,
  operating_expenses, revenue, ...). Inventory-driven, not free-form invention -- the model is
  given the exact list of categories that actually occur in this scenario's enriched ledger and
  must map every one of them, never invent a category that isn't in the list. This is what
  structurally prevents the catch-all-bucket bug (PIPELINE.md): committing to "what is this
  transaction" already happened in enrich.py, before any covenant was in view; this stage only
  assigns each already-fixed category to a role.
- "covenants": per covenant key, a formula over role names (or the two built-in tag aggregates,
  or a doc_figures name) plus threshold/comparison/precondition/period/evidence-type.

The formula is a plain arithmetic expression evaluated later by compute.py via a safe AST walk --
never trusted to the model to compute. validate_spec() below flags any formula/precondition
variable that resolves to nothing (neither a role, a tag aggregate, nor a doc_figures entry)
*before* compute.py runs, so a silently-empty variable can't quietly compute a wrong answer.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

from covenant.llm.client import Client

SPEC_PROMPT = """You are converting a credit agreement's financial covenant clauses into a \
structured, computable specification for a specific borrower.

You will be given:
1. The full text of each covenant clause (keyed by its clause number, e.g. "6.1").
2. An inventory of transaction categories that actually occur in this borrower's enriched \
ledger -- each transaction was independently categorized from its own description, before any \
covenant was in view, and audit-report reclassifications/one-off flags have already been applied.
3. Which built-in tag aggregates have at least one transaction in this ledger: \
"related_party_payments" (sum of USD amounts of transactions tagged related-party) and/or \
"unrestricted_sub_transfers" (sum of USD amounts of transactions tagged as an unrestricted-\
subsidiary transfer). You may reference these two names directly in any formula/variables/\
precondition without mapping them to a role -- they are always available if listed as present.
4. Any figures disclosed in supporting documents that do NOT correspond to any transaction at \
all (e.g. a provision that was never actually transacted, or a parent company's consolidated \
figure) -- given as a list of {label, amount_usd, reason}.
5. Any reclassification/cut-off/addback events already applied while building the ledger -- given \
as a list of {resulting_category, reason}, each naming the role-mapped category a transaction now \
sits in because of the disclosed override. If a covenant's formula uses that category's role, this \
is your signal for "evidence": "single_transaction" (see below) -- it is the ONLY signal for it.

Produce two top-level keys:

"roles": an object mapping EVERY category in the given inventory (as the key, verbatim) to a \
short snake_case role name describing what line item it belongs to for covenant purposes (e.g. \
"revenue", "capital_expenditure", "operating_expenses", "payroll", "interest_expense", "rent", \
"insurance", "utilities", "taxes", "marketing"). Multiple categories may map to the same role. \
Every category in the given inventory MUST appear as a key exactly once -- do not omit any, do \
not add categories that weren't given.
Assign each category by its ECONOMIC SUBSTANCE -- what the item IS, not the keywords in its name \
and not whether its ledger amount is an inflow or outflow. Where a covenant clause you were given \
DEFINES one of its terms (what "Revenue" / "Operating expenses" / "Adjusted EBITDA" comprises for \
that covenant), that stated definition governs which categories belong to that role -- read the \
clause text and honour it. Two accounting-substance rules that always hold:
(1) capitalised interest (interest added to the carrying value of an asset) is still INTEREST -- \
map any "capitalised interest" category to "interest_expense", NOT to "capital_expenditure", \
despite the word "capital" in its name. Reserve "capital_expenditure" for actual acquisitions/\
construction/transfers of capital assets.
(2) the borrower's own sales / turnover / handling / distribution settlement (its core operating \
income) is the "revenue" role -- NEVER map a sales-settlement category to any expense role, even \
though its ledger amount is a large inflow. Do not treat other inflows that are not sales (e.g. \
"co-operative funding received", rebates, refunds, interest income) as "revenue".

"covenants": an object keyed by each covenant number given, each an object with:
  - "metric": one sentence describing what's being tested.
  - "period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} -- the covenant's own stated period.
  - "period_quarter": if the covenant applies only to one quarter of that period (not the whole \
thing), which one as "Q1"/"Q2"/"Q3"/"Q4", else null.
  - "variables": object mapping each name used in "formula"/"precondition" to a one-sentence \
description of what it sums or represents. This is documentation only, NOT a place to define a \
new name -- every name here must ALSO be a role (a "roles" value), one of the two built-in tag \
aggregates, or a doc_figures key. Never invent a composite/derived name here (e.g. "ebitda", \
"total_expenses", "net_income") that doesn't independently resolve to one of those three things.
State the term's SCOPE -- what kind of thing it is and how widely it reaches ("the Borrower's total \
capital expenditure for the period", "the Borrower's own sales revenue"). Never enumerate WHICH \
items compose it: "sum of operating costs and management advisory retainers" asserts a composition \
the clause did not state, and downstream stages read this description as if the clause had. If the \
clause itself lists what the term comprises, that list already lives in the clause text and does \
not need repeating here.
  - "formula": a plain arithmetic expression (+, -, *, /, min(...), max(...), parentheses only -- \
no other functions) using ONLY role names, the two built-in tag aggregates, and/or doc_figures \
names -- the same restriction as "variables" above. If the clause's metric is a composite like \
EBITDA, write the composite out inline from primitive role names rather than naming it ("ebitda" \
resolves to nothing). Build that expression from THE CLAUSE'S OWN WORDS and nothing else: where it \
says EBITDA means "Выручка за вычетом Операционных расходов", the formula is exactly \
"revenue - operating_expenses" -- subtracting payroll, interest or any other term the clause does \
not name invents a definition the document never gave, and every term you add that the clause \
omits makes the reported value wrong. Where the clause itemises what the metric comprises, include \
exactly those items. Do not reference any name that isn't independently one of those three kinds. \
Must evaluate to the metric's actual value (not a boolean).
Write out a composite the CLAUSE ITSELF defines ("EBITDA means Revenue less Operating expenses"), \
but do NOT expand an accounting CAPTION the clause simply names -- "Operating expenses", "Revenue", \
"Capital expenditure" are single line items of the borrower's statements, not sums for you to \
assemble. Where a clause names such a caption, give some category that role name in "roles" and \
reference that ONE name in the formula; enumerating the categories you believe make up the caption \
("rent + insurance + marketing + utilities + ...") asserts a composition the clause never stated \
and is wrong whenever the borrower's statements report the caption more narrowly. Which \
transactions the caption covers is decided later, against the clause's own wording.
  - "comparison": one of "<=", "<", ">=", ">", "==" -- how the formula's value relates to \
"threshold" when COMPLIANT.
  - "threshold": the numeric threshold, exactly as stated (no unit symbols).
  - "threshold_unit": "usd" or "ratio".
  - "precondition": if this is a "springing" covenant that only applies when some condition \
holds, a plain boolean comparison expression (e.g. "financing_receipts > 4000000") over the same \
kind of variable names, else null.
  - "carve_out": null if the clause states no exception. Otherwise a structured object describing \
the stated exception that lets the true value exceed the raw threshold while still COMPLIANT \
(the reported "actual" is always the true value; the carve-out only affects status):
      * a numeric allowance -> {"kind": "allowance", "allowance": <number in the SAME unit as \
threshold>, "note": "..."}  (e.g. "may exceed the limit by up to $200,000" -> allowance 200000).
      * an excluded line item -> {"kind": "exclusion", "excluded_role": "<role or category name \
whose transactions are excluded from the tested value>", "note": "..."}  (e.g. "excluding \
intra-group transfers").
      * a discretionary/approval-based exception that CANNOT be evaluated from the ledger or \
documents (e.g. "except amounts expressly approved by the Lender in writing") -> \
{"kind": "discretionary", "note": "..."}.
    Choose "discretionary" whenever whether the exception applies depends on information not \
present in the transaction data.
  - "evidence": "single_transaction" ONLY if one of the given reclassification/cut-off/addback \
events' resulting_category matches a role used in THIS covenant's formula -- that is the only \
basis for this value; do not infer it from clause wording or from how few transactions a category \
happens to contain. If no given event's resulting_category is used by this covenant's formula, \
"evidence" is "aggregate", full stop, even for a ratio or sum that currently has only one row.
  - "doc_figures": object mapping any name (used in "formula") that must come from the given \
disclosed-figures list rather than the ledger, to {"amount_usd": <number>, "source": <one \
sentence>}. Empty object if the covenant needs nothing beyond roles/tags.
  - "notes": list of any other caveats worth recording (empty list if none).

Never invent category names, thresholds, dollar figures, or definitions the source text doesn't \
state. Read the clause's own wording for the formula shape -- do not assume a "standard" covenant \
structure exists.

Respond with strict JSON only, no markdown code fences, no commentary: {"roles": {...}, "covenants": {...}}
"""

CACHE_DIR = Path(".cache/specs")


def _model_tag(client: Client) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"{client.backend}-{client.model}")


def build_spec(
    clauses: dict[str, str],
    categories: list[str],
    available_tags: list[str],
    disclosed_figures: list[dict],
    reclassification_events: list[dict],
    client: Client,
    temperature: float = 0.0,
    correction: str = "",
) -> dict:
    payload = {
        "clauses": clauses,
        "category_inventory": sorted(set(categories)),
        "available_tag_aggregates": available_tags,
        "disclosed_figures": disclosed_figures,
        "reclassification_events": reclassification_events,
    }
    user = json.dumps(payload, ensure_ascii=False) + correction
    data = client.complete_json(SPEC_PROMPT, user, max_tokens=8192, temperature=temperature)
    data.setdefault("roles", {})
    data.setdefault("covenants", {})
    return _split_inflows_from_expense_roles(_sanitise_identifiers(_coerce_numbers(data)))


def build_spec_voted(
    clauses: dict[str, str],
    categories: list[str],
    available_tags: list[str],
    disclosed_figures: list[dict],
    reclassification_events: list[dict],
    client: Client,
    votes: int = 3,
    temperature: float = 0.5,
) -> dict:
    """Self-consistency: sample `votes` independent spec generations at a nonzero temperature and
    majority-vote each decision. Spec generation is the pipeline's noisiest LLM stage -- a single
    greedy generation lands in whatever basin the model happens to prefer, and a small wording
    change can flip an unrelated scenario's role mapping. Voting keeps the modal (most agreed-upon)
    decision per category and per covenant field, which is far more stable across reruns.

    Voting is applied field-by-field on the compute-critical decisions (role per category; and per
    covenant: formula, comparison, threshold, threshold_unit, precondition, period_quarter,
    evidence, doc_figures). Descriptive-only fields (metric/variables/notes/period) are taken from
    the sample that agrees most with the voted roles, so they stay internally consistent."""
    if votes <= 1:
        # still goes through _repair_spec below -- an unresolvable variable kills the whole cell,
        # and that is just as true of a single greedy sample as of a voted one.
        spec = build_spec(
            clauses, categories, available_tags, disclosed_figures, reclassification_events, client
        )
        return _repair_spec(
            spec,
            clauses,
            categories,
            available_tags,
            disclosed_figures,
            reclassification_events,
            client,
            temperature,
        )

    samples: list[dict] = []
    for _ in range(votes):
        try:
            samples.append(
                build_spec(
                    clauses,
                    categories,
                    available_tags,
                    disclosed_figures,
                    reclassification_events,
                    client,
                    temperature=temperature,
                )
            )
        except Exception:  # noqa: BLE001 -- a failed sample just doesn't vote
            continue
    if not samples:
        raise RuntimeError("all spec-generation samples failed")
    spec = samples[0] if len(samples) == 1 else _aggregate_specs(samples, sorted(set(categories)))
    return _repair_spec(
        spec,
        clauses,
        categories,
        available_tags,
        disclosed_figures,
        reclassification_events,
        client,
        temperature,
    )


def _repair_spec(
    spec: dict,
    clauses: dict[str, str],
    categories: list[str],
    available_tags: list[str],
    disclosed_figures: list[dict],
    reclassification_events: list[dict],
    client: Client,
    temperature: float,
) -> dict:
    """One corrective round when validate_spec() finds formula variables that resolve to nothing.

    Such a covenant cannot be computed at all, so the cell is lost outright -- yet the failure is
    detected deterministically and is usually the same slip every time: the model names a composite
    ("ebitda", "total_expenses") the prompt told it to write out via primitive roles. Feeding that
    verdict back and resampling is far cheaper than losing the cell, and it only ever replaces a
    covenant whose variables now all resolve, so a repair can never make a spec less computable."""
    problems = validate_spec(spec)
    if not problems:
        return spec

    correction = (
        "\n\nYour previous answer is REJECTED for these covenants: "
        + json.dumps(problems, ensure_ascii=False)
        + '. Each listed name resolves to nothing: it is not a role you assigned in "roles", not '
        'one of the built-in tag aggregates, and not a key of that covenant\'s "doc_figures". '
        "Rewrite those formulas using ONLY names that are. A composite such as EBITDA must be "
        'spelled out from primitive roles (e.g. "revenue - operating_expenses"), never referenced '
        "by its name. Return the complete corrected JSON object."
    )
    try:
        repaired = build_spec(
            clauses,
            categories,
            available_tags,
            disclosed_figures,
            reclassification_events,
            client,
            temperature=temperature,
            correction=correction,
        )
    except Exception:  # noqa: BLE001 -- a failed repair just leaves the original spec in place
        return spec

    fixed = validate_spec(repaired)
    for key, cov in repaired.get("covenants", {}).items():
        if key in problems and key not in fixed:
            spec.setdefault("covenants", {})[key] = cov
    return spec


def _mode(values: list) -> object:
    """Most common value; ties broken by first appearance (Counter preserves insertion order)."""
    counts = Counter(json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values)
    winner_key = counts.most_common(1)[0][0]
    return json.loads(winner_key)


def _aggregate_specs(samples: list[dict], category_inventory: list[str]) -> dict:
    # roles: majority vote the role per category
    voted_roles: dict[str, str] = {}
    for cat in category_inventory:
        votes = [s["roles"].get(cat) for s in samples if s.get("roles", {}).get(cat)]
        if votes:
            voted_roles[cat] = _mode(votes)

    # the sample whose roles agree most with the voted roles supplies descriptive covenant fields
    def agreement(sample: dict) -> int:
        return sum(1 for c, r in voted_roles.items() if sample.get("roles", {}).get(c) == r)

    representative = max(samples, key=agreement)

    voted_field_keys = [
        "formula",
        "comparison",
        "threshold",
        "threshold_unit",
        "precondition",
        "period_quarter",
        "evidence",
        "doc_figures",
    ]
    all_keys = sorted({k for s in samples for k in s.get("covenants", {})})
    voted_covenants: dict[str, dict] = {}
    for key in all_keys:
        present = [s["covenants"][key] for s in samples if key in s.get("covenants", {})]
        base = dict(representative.get("covenants", {}).get(key, present[0]))
        for field in voted_field_keys:
            vals = [c[field] for c in present if field in c]
            if vals:
                base[field] = _mode(vals)
        voted_covenants[key] = base

    return {"roles": voted_roles, "covenants": voted_covenants}


_ALLOWED_AST_NODES = (
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


def _extract_names(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    call_func_ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"disallowed expression element {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ValueError(f"disallowed function call in {expr!r}")
            call_func_ids.add(id(node.func))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in call_func_ids}


# The model routinely says, in prose, that the formula it just wrote is not the thing the clause
# asks for -- "capital_expenditure is used as a proxy for such transfers", "financing receipts are
# not represented in the available data, so the formula reflects only EBITDA", "condition cannot be
# evaluated from available data". Those sentences land in "metric" and "notes", which nothing reads
# back, while the approximation they describe lands in "formula", which decides the answer. Every
# one of the spec defects found on the private set was confessed this way first.
#
# These are hedges in the model's own descriptive voice, not facts about any dataset: they carry no
# borrower, threshold, category or covenant number, and they read the same whatever the documents
# say. Matching them costs nothing and needs no second model.
_HEDGE_RE = re.compile(
    r"\bproxy\b|\bapproximat|\bcannot be (evaluated|determined|computed)|\bnot (represented|"
    r"available|disclosed|present) in|\bunable to\b|\bassumed?\b|\bnot expressed\b|\bplaceholder\b"
    r"|\bне (может быть|представлен|раскрыт)|\bприближ|\bдопущени",
    re.IGNORECASE,
)


def self_reported_approximations(spec: dict) -> dict[str, str]:
    """covenant_key -> the sentence in which the model admitted its formula is an approximation.

    A diagnostic, never an automatic edit: the admission says the answer is doubtful, not what the
    right answer would have been.
    """
    found: dict[str, str] = {}
    for key, cov in (spec.get("covenants") or {}).items():
        if not isinstance(cov, dict):
            continue
        notes = cov.get("notes") or []
        text = " ".join([str(cov.get("metric") or ""), *(str(n) for n in notes)])
        match = _HEDGE_RE.search(text)
        if match:
            start = max(0, match.start() - 60)
            found[key] = text[start : match.end() + 90].strip()
    return found


def validate_spec(spec: dict) -> dict[str, list[str]]:
    """covenant_key -> list of variable names that resolve to nothing (neither a role, a
    built-in tag aggregate, nor a doc_figures entry). Non-empty means: don't trust that
    covenant's compute() result -- something the formula references was never populated."""
    role_values = set(spec.get("roles", {}).values())
    role_keys = set(spec.get("roles", {}).keys())
    known_base = role_values | role_keys | {"related_party_payments", "unrestricted_sub_transfers"}

    problems: dict[str, list[str]] = {}
    for key, cov in spec.get("covenants", {}).items():
        known = known_base | set(cov.get("doc_figures", {}).keys())
        missing: set[str] = set()
        for expr in filter(None, [cov.get("formula"), cov.get("precondition")]):
            try:
                names = _extract_names(expr)
            except (ValueError, SyntaxError) as exc:
                # SyntaxError as well as ValueError: a formula that is not valid Python at all
                # ("revenue - (operating") raises SyntaxError out of ast.parse, and catching only
                # ValueError let it escape validate_spec and abort the entire scenario -- three
                # cells lost to one malformed string, when the point of this function is to report
                # exactly that as a per-covenant problem.
                missing.add(f"(unparseable expression: {exc})")
                continue
            missing |= names - known
        if missing:
            problems[key] = sorted(missing)
    return problems


def save_spec(sid: str, spec: dict, client: Client, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sid}__{_model_tag(client)}.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    return path


_INFLOW_MARKERS = ("_income", "_rebate", "_refund", "_recovery", "_credit", "_received")


def _split_inflows_from_expense_roles(spec: dict) -> dict:
    """Keep an inflow category out of the expense role it mirrors.

    A category named for money coming back -- interest income, an insurance rebate, a tax refund --
    shares a role with its expense counterpart often enough to matter, and once it does the formula
    can no longer tell them apart: _resolve_variable sums the role signed, so the inflow quietly
    cancels part of the very cost the covenant caps. Giving it a role of its own keeps both
    addressable, and a formula that genuinely wants the net can still name each side.
    """
    roles = spec.get("roles") or {}
    for category, role in list(roles.items()):
        if role.endswith("_expense") and category.lower().endswith(_INFLOW_MARKERS):
            roles[category] = f"{category}_role"
    return spec


def _sanitise_identifiers(spec: dict) -> dict:
    """Rename doc_figures keys that are not valid Python identifiers, in the formulas too.

    Models label a disclosed figure the way the document phrases it -- "Aggregate severance program
    obligation" -- and drop that phrase straight into the formula. It is several bare names with no
    operator between them, so the expression will not parse at all and the covenant is lost whole,
    even though every number needed to compute it is present and correct."""
    for cov in (spec.get("covenants") or {}).values():
        figures = cov.get("doc_figures") or {}
        renames = {
            key: re.sub(r"\W+", "_", key.strip()).strip("_").lower()
            for key in figures
            if not key.isidentifier()
        }
        if not renames:
            continue
        cov["doc_figures"] = {renames.get(k, k): v for k, v in figures.items()}
        for field in ("formula", "precondition"):
            expr = cov.get(field)
            if not expr:
                continue
            for old, new_name in sorted(renames.items(), key=lambda kv: -len(kv[0])):
                expr = expr.replace(old, new_name)
            cov[field] = expr
    return spec


_NUMERIC_RE = re.compile(r"^\s*\$?\s*(-?[\d,]*\.?\d+)\s*[x%]?\s*$")


def _as_number(value):
    """A threshold the model wrote as text, read back as the number it plainly is.

    "0.08", "$3,000,000.00" and "1.20x" are all the threshold spelled the way the clause prints it.
    Left as strings they are not merely untidy: compute.py compares the metric against the threshold,
    and float-vs-str raises, so the covenant falls through to its fallback and loses the comparison
    it already had. Anything that does not read as a plain number is left exactly as it is.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    match = _NUMERIC_RE.match(value)
    if not match:
        return value
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return value


def _coerce_numbers(spec: dict) -> dict:
    for cov in (spec.get("covenants") or {}).values():
        if not isinstance(cov, dict):
            continue
        if "threshold" in cov:
            cov["threshold"] = _as_number(cov["threshold"])
        carve = cov.get("carve_out")
        if isinstance(carve, dict) and "allowance" in carve:
            carve["allowance"] = _as_number(carve["allowance"])
        for figure in (cov.get("doc_figures") or {}).values():
            if isinstance(figure, dict) and "amount_usd" in figure:
                figure["amount_usd"] = _as_number(figure["amount_usd"])
    return spec


RESERVED_AGGREGATES = ("related_party_payments", "unrestricted_sub_transfers")


def _unshadow_reserved_roles(spec: dict) -> dict:
    """Stop a role from stealing the name of a built-in tag aggregate.

    compute.py answers those two names from the tag, before any role lookup. So a model that also
    uses one as a ROLE name causes two silent errors at once: the role's transactions disappear
    from whatever term should have contained them, and the formula reads the tag total instead --
    zero, if tagging happened to fail. Renaming the role keeps both meanings addressable."""
    roles = spec.get("roles") or {}
    if not any(r in RESERVED_AGGREGATES for r in roles.values()):
        return spec
    spec["roles"] = {c: (f"{r}_role" if r in RESERVED_AGGREGATES else r) for c, r in roles.items()}
    return spec


def load_spec(sid: str, client: Client, cache_dir: Path = CACHE_DIR) -> dict | None:
    path = cache_dir / f"{sid}__{_model_tag(client)}.json"
    if not path.exists():
        return None
    return _unshadow_reserved_roles(
        _split_inflows_from_expense_roles(
            _sanitise_identifiers(_coerce_numbers(json.loads(path.read_text())))
        )
    )
