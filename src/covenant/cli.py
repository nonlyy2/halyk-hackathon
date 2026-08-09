"""Single entrypoint for the whole pipeline: `covenant <stage>`.

The stages were previously seven separate scripts, each re-declaring the same arguments, its own
copy of the "authoritative documents" lookup, and its own cwd-relative default paths. They are
subcommands here so that the shared pieces are shared for real.

    covenant classify --data ./data
    covenant enrich   --data ./data
    covenant spec     --data ./data --votes 3
    covenant bind     --data ./data --votes 3
    covenant build    --data ./data --team NAME --contact-email you@example.com --review
    covenant doctor   --data ./data          # preflight: what is missing before you submit
    covenant score    --data ./data          # dev only, needs ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from covenant.analyze.binding import (
    bind_terms_voted,
    bindable_terms,
    is_a_line_item,
    load_binding,
    save_binding,
)
from covenant.analyze.classify import _detected_markers, classify_document
from covenant.analyze.clauses import (
    extract_clause_text,
    extract_covenant_addendum,
    extract_ownership_section,
    find_covenant_keys,
    llm_clause_texts,
    ownership_section_found,
)
from covenant.analyze.documents_index import DocumentIndex
from covenant.analyze.enrich import enrich_scenario, load_enriched, save_enriched
from covenant.analyze.review import apply_correction, build_worksheet, review_cell
from covenant.analyze.spec import build_spec_voted, load_spec, save_spec, validate_spec
from covenant.config import Paths, resolve
from covenant.ingest.documents import concurrency, extract_documents
from covenant.ingest.ledger import load_ledger
from covenant.ingest.matching import match_accounts, match_company_names
from covenant.llm.client import get_client
from covenant.scoring.compute import compute_covenant, fallback_cell, narrow_absurd_cost_roles
from covenant.scoring.confidence import cell_confidence

_print_lock = threading.Lock()


def _say(message: str) -> None:
    """One line at a time -- stages run several scenarios concurrently and interleave otherwise."""
    with _print_lock:
        print(message, flush=True)


def _in_parallel(items: list, work) -> None:
    """Run `work(item)` over items, never letting one failure take down the batch.

    A scenario that raises has always cost only itself here (cmd_enrich caught it inline); doing it
    in one place means every stage behaves the same way, and the traceback still names the scenario.
    """

    def guarded(item):
        try:
            work(item)
        except Exception as exc:  # noqa: BLE001 -- one bad scenario must not kill the batch
            _say(f"{item}: ERROR {exc}")

    workers = min(concurrency(), len(items)) or 1
    if workers == 1:
        for item in items:
            guarded(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(guarded, items))


def _scenarios(paths: Paths, requested: list[str]) -> list[str]:
    template = json.loads(paths.template.read_text())
    return requested or sorted(template["answers"])


def _docs(paths: Paths) -> dict:
    return {d.doc_id: d for d in extract_documents(str(paths.documents), str(paths.text_cache))}


def _template_keys(paths: Paths, sid: str) -> list[str]:
    return sorted(json.loads(paths.template.read_text())["answers"].get(sid, {}))


def _agreement_text(docs: dict, index: DocumentIndex, sid: str, article: str) -> str | None:
    """The authoritative credit agreement whose article this borrower's covenants actually sit in."""
    agreements = index.authoritative(sid, "credit_agreement")
    if not agreements:
        return None
    return max(
        (docs[r.doc_id].text for r in agreements),
        key=lambda t: len(find_covenant_keys(t, article)),
    )


def _clause_texts(text: str, keys: list[str], client=None) -> tuple[dict[str, str], list[str]]:
    """(clause text per key, keys the deterministic slicers could not find).

    A key with no text is a covenant with no wording, which loses the cell outright -- so when a
    client is supplied the missing ones are recovered with a single call over the whole agreement.
    """
    clauses: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        try:
            clauses[key] = extract_clause_text(text, key, keys)
        except ValueError:
            missing.append(key)
    if missing and client is not None:
        _say(f"  clause headings not found for {missing} -- recovering from the full agreement")
        try:
            clauses.update(llm_clause_texts(text, missing, client))
        except Exception as exc:  # noqa: BLE001 -- the cells still get their spec, just no wording
            _say(f"  clause recovery failed: {exc}")
    return clauses, missing


def cmd_classify(args, paths: Paths) -> None:
    docs = _docs(paths)
    ledger = load_ledger(str(paths.ledger))
    in_scope = set(json.loads(paths.template.read_text())["answers"])
    client = get_client("small")
    _say(f"model: {client.model}")

    paths.classifications.parent.mkdir(parents=True, exist_ok=True)
    done = {
        r.doc_id for r in DocumentIndex.load(paths.classifications).all() if r.model == client.model
    }
    write_lock = threading.Lock()

    def run(batch, label: str) -> None:
        pending = [m for m in batch if m.doc_id not in done]
        _say(f"{label}: {len(batch)} candidates, {len(pending)} to classify")
        with paths.classifications.open("a") as fh:

            def one(match) -> None:
                c = classify_document(docs[match.doc_id], client)
                row = {
                    "doc_id": match.doc_id,
                    "model": client.model,
                    "scenario_ids": match.scenario_ids,
                    "account_ids": match.account_ids,
                    "company_name": c.company_name,
                    "doc_type": c.doc_type,
                    "is_authoritative_for_covenants": c.is_authoritative_for_covenants,
                    "reason": c.reason,
                }
                with write_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add(match.doc_id)
                _say(
                    f"[{label}] {match.doc_id} {match.scenario_ids} "
                    f"{c.doc_type} auth={c.is_authoritative_for_covenants}"
                )

            _in_parallel(pending, one)

    by_account = [
        m
        for m in match_accounts(list(docs.values()), ledger)
        if any(s in in_scope for s in m.scenario_ids)
    ]
    run(by_account, "by-account")

    # Second pass, once borrower names are known: a covenant can point at a figure that lives only
    # in a document carrying no account id, such as a parent's consolidated statements.
    names = DocumentIndex.load(paths.classifications).borrower_names()
    by_name = [
        m
        for m in match_company_names(list(docs.values()), names, {m.doc_id for m in by_account})
        if any(s in in_scope for s in m.scenario_ids)
    ]
    run(by_name, "by-name")
    _say("done")


def _enrichment_sources(
    docs: dict, index: DocumentIndex, sid: str
) -> tuple[str | None, str | None]:
    ownership = (
        "\n\n".join(
            s
            for r in index.authoritative(sid, "kyc_dossier")
            if (s := extract_ownership_section(docs[r.doc_id].text))
        )
        or None
    )

    audit = index.authoritative(sid, "audit_report")
    audit_ids = {r.doc_id for r in audit}
    # A report whose disclosures sit under an unrecognised heading still contributes: falling
    # back to its whole text is how a parent's consolidated statements stay reachable.
    sections = [
        extract_covenant_addendum(docs[r.doc_id].text) or docs[r.doc_id].text for r in audit
    ]
    # Corrections are not confined to the audit report -- a treasury memo can be the only place
    # a ledger amount is stated. Any document naming this borrower's own transactions counts,
    # unless it carries a supersession or draft marker.
    for row in index.for_scenario(sid):
        if row.doc_id in audit_ids:
            continue
        text = docs[row.doc_id].text
        if _detected_markers(text):
            continue
        other_authoritative = row.is_authoritative and row.doc_type not in (
            "credit_agreement",
            "kyc_dossier",
        )
        if f"TXN-{sid}-" in text or other_authoritative:
            sections.append(text)

    return ownership, "\n\n".join(filter(None, sections)) or None


def cmd_enrich(args, paths: Paths) -> None:
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    ledger = load_ledger(str(paths.ledger))
    small, complex_ = get_client("small"), get_client("complex")
    _say(f"small={small.model} complex={complex_.model}")

    def one(sid: str) -> None:
        if not args.fresh and load_enriched(sid, str(paths.enriched)) is not None:
            _say(f"{sid}: cached, skipping")
            return
        ownership, addendum = _enrichment_sources(docs, index, sid)
        enriched, disclosed = enrich_scenario(
            sid,
            ledger.transactions_for(sid),
            ownership,
            addendum,
            small,
            complex_,
            votes=args.votes,
        )
        save_enriched(sid, enriched, disclosed, str(paths.enriched))
        _say(
            f"{sid}: {len(enriched)} txns, "
            f"{sum(e.related_party for e in enriched)} related_party, "
            f"{sum(1 for e in enriched if e.reclass_reason)} reclassified, "
            f"{len(disclosed)} disclosed_figures"
        )

    _in_parallel(_scenarios(paths, args.scenarios), one)


def cmd_spec(args, paths: Paths) -> None:
    template = json.loads(paths.template.read_text())
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    client = get_client("complex")
    _say(f"model={client.model}")

    def one(sid: str) -> None:
        if not args.fresh and load_spec(sid, client, paths.specs) is not None:
            _say(f"{sid}: cached, skipping")
            return
        loaded = load_enriched(sid, str(paths.enriched))
        if loaded is None:
            _say(f"{sid}: ERROR no enriched cache -- run `covenant enrich` first")
            return
        enriched, disclosed = loaded

        # The article number comes from the template's own keys ("6.1" -> article 6). Hardcoding it
        # would cost every cell at once on a set whose covenants sit under a different article, and
        # the template states it for free.
        articles = {k.split(".")[0] for k in template["answers"].get(sid, {})} or {"6"}
        article = sorted(articles)[0]
        text = _agreement_text(docs, index, sid, article)
        if text is None:
            _say(f"{sid}: ERROR no authoritative credit agreement")
            return
        # answer what the template asks for, even if a heading for it was not located
        keys = sorted(set(find_covenant_keys(text, article)) | set(_template_keys(paths, sid)))
        clauses, _missing = _clause_texts(text, keys, client)

        categories = [e.category for e in enriched] + [e.raw_category for e in enriched]
        tags = [
            name
            for name, present in [
                ("related_party_payments", any(e.related_party for e in enriched)),
                ("unrestricted_sub_transfers", any(e.unrestricted_sub_transfer for e in enriched)),
            ]
            if present
        ]
        events = [
            {"resulting_category": e.category, "reason": e.reclass_reason}
            for e in enriched
            if e.reclass_reason
        ]

        spec = build_spec_voted(
            clauses,
            categories,
            tags,
            disclosed,
            events,
            client,
            votes=args.votes,
            temperature=args.vote_temperature,
        )
        # carried in the cache so `bind`, `--review` and `doctor` read the same wording the spec was
        # built from, instead of re-slicing the agreement and possibly landing on different text.
        spec["clauses"] = clauses
        problems = validate_spec(spec)
        save_spec(sid, spec, client, paths.specs)
        _say(
            f"{sid}: {len(spec['covenants'])} covenants "
            f"{'PROBLEMS: ' + str(problems) if problems else 'ok'}"
        )

    _in_parallel(_scenarios(paths, args.scenarios), one)


def _spec_clauses(paths: Paths, docs: dict, index: DocumentIndex, spec: dict, sid: str) -> dict:
    """The clause text this scenario's covenants were written from.

    Normally the spec cache carries it. A spec cached before it did gets the clauses re-sliced here,
    deterministically and without a call -- binding and reviewing against a one-line "metric"
    summary instead of the clause's own wording gives up most of what those stages are for.
    """
    clauses = spec.get("clauses") or {}
    if clauses:
        return clauses
    keys = sorted(spec.get("covenants") or {}) or _template_keys(paths, sid)
    article = sorted({k.split(".")[0] for k in keys} or {"6"})[0]
    text = _agreement_text(docs, index, sid, article)
    if text is None:
        return {}
    return _clause_texts(text, keys)[0]


def cmd_bind(args, paths: Paths) -> None:
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    client = get_client("complex")
    _say(f"model={client.model}")

    def one(sid: str) -> None:
        if not args.fresh and load_binding(sid, client, paths.bindings) is not None:
            _say(f"{sid}: cached, skipping")
            return
        spec = load_spec(sid, client, paths.specs)
        loaded = load_enriched(sid, str(paths.enriched))
        if spec is None or loaded is None:
            _say(f"{sid}: ERROR needs both a spec and an enriched cache")
            return
        binding = bind_terms_voted(
            spec,
            _spec_clauses(paths, docs, index, spec, sid),
            loaded[0],
            client,
            votes=args.votes,
            temperature=args.vote_temperature,
        )
        save_binding(sid, binding, client, paths.bindings)
        bound = sum(len(terms) for terms in binding.values())
        wanted = sum(len(bindable_terms(c)) for c in (spec.get("covenants") or {}).values())
        _say(f"{sid}: bound {bound}/{wanted} terms across {len(binding)} covenants")

    _in_parallel(_scenarios(paths, args.scenarios), one)


def _cell(cov, roles, enriched, binding):
    """The computed cell, degrading rather than failing -- a null cell scores like a wrong one."""
    if cov is None:
        return fallback_cell(None, {}, [])
    try:
        return compute_covenant(cov, roles, enriched, binding)
    except Exception as exc:  # noqa: BLE001
        print(f"  {exc} -- falling back", file=sys.stderr)
        return fallback_cell(cov, roles, enriched, binding)


def _reviewed(sid, key, cov, roles, enriched, binding, result, client):
    """(result, binding) after a second opinion, unchanged if the critic confirms or fails."""
    confidence = cell_confidence(cov, roles, enriched, binding)
    if confidence.level == "high":
        return result, binding
    reasons = confidence.flags + [f"confidence={confidence.level}", str(confidence.signals)]
    try:
        corrected = review_cell(
            cov.get("clause_text", ""), cov, roles, enriched, binding, result, reasons, client
        )
    except Exception as exc:  # noqa: BLE001 -- the reviewed cell is a bonus, never a risk
        _say(f"  {sid}.{key}: review failed ({exc}) -- keeping the computed cell")
        return result, binding
    if not corrected:
        return result, binding
    applied = apply_correction(cov, roles, enriched, binding, corrected)
    if applied is None:
        _say(f"  {sid}.{key}: correction did not evaluate -- keeping the computed cell")
        return result, binding
    new_binding, new_result = applied
    _say(
        f"  {sid}.{key}: reviewed {sorted(corrected)} "
        f"{result.status} {result.actual} -> {new_result.status} {new_result.actual}"
    )
    return new_result, new_binding


def cmd_build(args, paths: Paths) -> None:
    template = json.loads(paths.template.read_text())
    client = get_client("complex")  # resolves the model tag on the spec/binding caches
    review_client = client if args.review else None
    submission = {
        "team": args.team,
        "contact_email": args.contact_email,
        "model": client.model,
        "answers": {},
    }
    worksheets: dict[str, dict] = {}
    # only needed to re-slice clause text for a spec cached before it carried any, and only when
    # the critic is going to read it -- extracting every document otherwise costs a cold run for
    # nothing, since the rest of this stage is pure arithmetic over the caches.
    docs = _docs(paths) if args.review else {}
    index = DocumentIndex.load(paths.classifications) if args.review else DocumentIndex([])

    def flush() -> None:
        # allow_nan=False: the default emits the bare token NaN, which is not valid JSON and would
        # make the affected cells unscoreable. Written after every scenario so that a crash late in
        # a run still leaves a submittable file behind.
        Path(args.out).write_text(
            json.dumps(submission, ensure_ascii=False, indent=2, allow_nan=False)
        )

    for sid in template["answers"]:
        spec = load_spec(sid, client, paths.specs)
        loaded = load_enriched(sid, str(paths.enriched))
        bindings = load_binding(sid, client, paths.bindings) or {}
        clauses = _spec_clauses(paths, docs, index, spec, sid) if spec and review_client else {}
        # Only terms the binding actually resolves count as bound. A selection the line-item guard
        # rejects falls back to the role map, and that role then still needs narrowing -- treating
        # it as bound leaves the wide reading in place with nothing left to correct it.
        ledger_size = len(loaded[0]) if loaded else 0
        bound_terms = {
            term
            for terms in bindings.values()
            for term, entry in terms.items()
            if entry.get("txn_ids") and is_a_line_item(len(entry["txn_ids"]), ledger_size)
        }
        roles = (
            narrow_absurd_cost_roles(spec["roles"], loaded[0], bound_terms)
            if spec and loaded
            else {}
        )

        cells: dict[str, dict] = {}
        sheets: dict[str, dict] = {}
        for key in template["answers"][sid]:
            cov = (spec or {}).get("covenants", {}).get(key) if loaded else None
            enriched = loaded[0] if loaded else []
            binding = bindings.get(key) or {}
            result = _cell(cov, roles, enriched, binding)
            if cov is not None and review_client is not None:
                cov = {**cov, "clause_text": clauses.get(key, "")}
                result, binding = _reviewed(
                    sid, key, cov, roles, enriched, binding, result, review_client
                )
            actual = (
                result.actual
                if isinstance(result.actual, (int, float)) and math.isfinite(result.actual)
                else 0.0
            )
            cells[key] = {
                "status": result.status,
                "actual": actual,
                "evidence_txn_id": result.evidence_txn_id,
            }
            if cov is not None:
                sheets[key] = build_worksheet(cov, roles, enriched, binding, result)

        submission["answers"][sid] = cells
        worksheets[sid] = sheets
        flush()

    paths.worksheets.parent.mkdir(parents=True, exist_ok=True)
    paths.worksheets.write_text(json.dumps(worksheets, ensure_ascii=False, indent=2))
    print(f"wrote {args.out} and {paths.worksheets}")
    if args.score:
        cmd_score(args, paths)


def cmd_doctor(args, paths: Paths) -> None:
    """Everything that could quietly cost cells, before the submission is sent.

    Reads only artefacts the run already produced -- never ground truth -- so it says the same
    things on the private set as on the public one. That is the point: on the day, the ranked list
    of what to look at by hand has to come from the data itself.
    """
    template = json.loads(paths.template.read_text())
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    client = get_client("complex")
    problems = 0

    for sid in sorted(template["answers"]):
        notes: list[str] = []
        keys = _template_keys(paths, sid)
        article = sorted({k.split(".")[0] for k in keys} or {"6"})[0]

        by_type = {
            t: len(index.authoritative(sid, t))
            for t in ("credit_agreement", "audit_report", "kyc_dossier")
        }
        if not by_type["credit_agreement"]:
            notes.append("no authoritative credit agreement")
        if not by_type["audit_report"]:
            notes.append("no authoritative audit report")

        text = _agreement_text(docs, index, sid, article)
        if text is not None:
            _clauses, missing = _clause_texts(text, keys)
            if missing:
                notes.append(f"clause headings not found: {missing}")

        kyc = index.authoritative(sid, "kyc_dossier")
        if kyc and not any(ownership_section_found(docs[r.doc_id].text) for r in kyc):
            notes.append("KYC ownership heading not found (fell back to the whole dossier)")

        loaded = load_enriched(sid, str(paths.enriched))
        if loaded is None:
            notes.append("no enriched cache")
        else:
            enriched = loaded[0]
            # An unconverted foreign-currency row is only worth reporting where a covenant actually
            # sums it -- most ledgers carry one or two that no clause touches, and a preflight that
            # cries wolf on nine scenarios out of twelve does not get read. cell_confidence() raises
            # it per cell, in scope, below.
            nan_rows = [e.txn_id for e in enriched if e.amount_usd != e.amount_usd]
            if nan_rows:
                notes.append(f"rows with no resolved amount: {nan_rows}")

        spec = load_spec(sid, client, paths.specs)
        binding = load_binding(sid, client, paths.bindings) or {}
        if spec is None:
            notes.append("no spec cache")
        else:
            spec_problems = validate_spec(spec)
            if spec_problems:
                notes.append(f"unresolvable formula variables: {spec_problems}")
            for key in keys:
                cov = spec.get("covenants", {}).get(key)
                if cov is None:
                    notes.append(f"{key}: no covenant in the spec")
                    continue
                wanted, got = set(bindable_terms(cov)), set(binding.get(key, {}))
                if wanted - got:
                    notes.append(f"{key}: unbound terms {sorted(wanted - got)}")
                if loaded is not None:
                    wide = {
                        term: len(entry.get("txn_ids") or [])
                        for term, entry in binding.get(key, {}).items()
                        if not is_a_line_item(len(entry.get("txn_ids") or []), len(loaded[0]))
                    }
                    if wide:
                        notes.append(
                            f"{key}: term(s) too wide to be a line item, using the role map "
                            f"instead: {wide} of {len(loaded[0])} rows"
                        )
                if loaded is not None:
                    c = cell_confidence(cov, spec.get("roles", {}), loaded[0], binding.get(key))
                    if c.level != "high":
                        notes.append(f"{key}: confidence={c.level} {c.flags or c.signals}")

        problems += len(notes)
        header = f"{sid}: docs {by_type}"
        print(header if not notes else f"{header}\n  " + "\n  ".join(notes))

    print()
    print(f"{problems} thing(s) worth a look" if problems else "nothing flagged")


def cmd_score(args, paths: Paths) -> None:
    from covenant.scoring.rubric import format_report, score_submission

    if not paths.ground_truth.exists():
        print(f"no {paths.ground_truth} -- scoring skipped", file=sys.stderr)
        return
    print()
    print(format_report(score_submission(args.out, str(paths.ground_truth))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="covenant", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="dataset directory (or set COVENANT_DATA)")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for name, fn, wants_scenarios in [
        ("classify", cmd_classify, False),
        ("enrich", cmd_enrich, True),
        ("spec", cmd_spec, True),
        ("bind", cmd_bind, True),
    ]:
        sub = subparsers.add_parser(name)
        sub.set_defaults(fn=fn)
        if wants_scenarios:
            sub.add_argument("scenarios", nargs="*")
            sub.add_argument("--fresh", action="store_true", help="ignore the existing cache")
        if name == "enrich":
            sub.add_argument(
                "--votes",
                type=int,
                default=1,
                help="categorise this many times and take the modal category per row",
            )
        if name in ("spec", "bind"):
            sub.add_argument(
                "--votes", type=int, default=3, help="self-consistency samples to majority-vote"
            )
            sub.add_argument(
                "--vote-temperature", type=float, default=0.5 if name == "spec" else 0.4
            )

    build = subparsers.add_parser("build")
    build.set_defaults(fn=cmd_build)
    build.add_argument("--out", default="submission.json")
    build.add_argument("--team", required=True)
    build.add_argument("--contact-email", required=True)
    build.add_argument(
        "--review", action="store_true", help="second-opinion pass on low-confidence cells"
    )
    build.add_argument("--score", action="store_true", help="dev only: score against ground truth")

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(fn=cmd_doctor)

    score = subparsers.add_parser("score")
    score.set_defaults(fn=cmd_score)
    score.add_argument("--out", default="submission.json")

    args = parser.parse_args(argv)
    started = time.monotonic()
    args.fn(args, resolve(args.data))
    print(f"[{args.stage}] {time.monotonic() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
