"""Single entrypoint for the whole pipeline: `covenant <stage>`.

The stages were previously seven separate scripts, each re-declaring the same arguments, its own
copy of the "authoritative documents" lookup, and its own cwd-relative default paths. They are
subcommands here so that the shared pieces are shared for real.

    covenant classify --data ./data
    covenant enrich   --data ./data
    covenant spec     --data ./data --votes 3
    covenant build    --data ./data --team NAME --contact-email you@example.com
    covenant score    --data ./data          # dev only, needs ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from covenant.analyze.classify import _detected_markers, classify_document
from covenant.analyze.clauses import (
    extract_clause_text,
    extract_covenant_addendum,
    extract_ownership_section,
    find_covenant_keys,
)
from covenant.analyze.documents_index import DocumentIndex
from covenant.analyze.enrich import enrich_scenario, load_enriched, save_enriched
from covenant.analyze.spec import build_spec_voted, load_spec, save_spec, validate_spec
from covenant.config import Paths, resolve
from covenant.ingest.documents import extract_documents
from covenant.ingest.ledger import load_ledger
from covenant.ingest.matching import match_accounts, match_company_names
from covenant.llm.client import get_client
from covenant.scoring.compute import compute_covenant, fallback_cell


def _scenarios(paths: Paths, requested: list[str]) -> list[str]:
    template = json.loads(paths.template.read_text())
    return requested or sorted(template["answers"])


def _docs(paths: Paths) -> dict:
    return {d.doc_id: d for d in extract_documents(str(paths.documents), str(paths.text_cache))}


def cmd_classify(args, paths: Paths) -> None:
    docs = _docs(paths)
    ledger = load_ledger(str(paths.ledger))
    in_scope = set(json.loads(paths.template.read_text())["answers"])
    client = get_client("small")
    print(f"model: {client.model}", flush=True)

    paths.classifications.parent.mkdir(parents=True, exist_ok=True)
    done = {
        r.doc_id for r in DocumentIndex.load(paths.classifications).all() if r.model == client.model
    }

    def run(batch, label: str) -> None:
        pending = [m for m in batch if m.doc_id not in done]
        print(f"{label}: {len(batch)} candidates, {len(pending)} to classify", flush=True)
        with paths.classifications.open("a") as fh:
            for i, match in enumerate(pending, 1):
                c = classify_document(docs[match.doc_id], client)
                fh.write(
                    json.dumps(
                        {
                            "doc_id": match.doc_id,
                            "model": client.model,
                            "scenario_ids": match.scenario_ids,
                            "account_ids": match.account_ids,
                            "company_name": c.company_name,
                            "doc_type": c.doc_type,
                            "is_authoritative_for_covenants": c.is_authoritative_for_covenants,
                            "reason": c.reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()
                done.add(match.doc_id)
                print(
                    f"[{label} {i}/{len(pending)}] {match.doc_id} {match.scenario_ids} "
                    f"{c.doc_type} auth={c.is_authoritative_for_covenants}",
                    flush=True,
                )

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
    print("done", flush=True)


def cmd_enrich(args, paths: Paths) -> None:
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    ledger = load_ledger(str(paths.ledger))
    small, complex_ = get_client("small"), get_client("complex")
    print(f"small={small.model} complex={complex_.model}", flush=True)

    for sid in _scenarios(paths, args.scenarios):
        if not args.fresh and load_enriched(sid, str(paths.enriched)) is not None:
            print(f"{sid}: cached, skipping", flush=True)
            continue

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

        try:
            enriched, disclosed = enrich_scenario(
                sid,
                ledger.transactions_for(sid),
                ownership,
                "\n\n".join(filter(None, sections)) or None,
                small,
                complex_,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad scenario must not kill the batch
            print(f"{sid}: ERROR {exc}", flush=True)
            continue

        save_enriched(sid, enriched, disclosed, str(paths.enriched))
        print(
            f"{sid}: {len(enriched)} txns, "
            f"{sum(e.related_party for e in enriched)} related_party, "
            f"{sum(1 for e in enriched if e.reclass_reason)} reclassified, "
            f"{len(disclosed)} disclosed_figures",
            flush=True,
        )


def cmd_spec(args, paths: Paths) -> None:
    docs = _docs(paths)
    index = DocumentIndex.load(paths.classifications)
    client = get_client("complex")
    print(f"model={client.model}", flush=True)

    for sid in _scenarios(paths, args.scenarios):
        if not args.fresh and load_spec(sid, client, paths.specs) is not None:
            print(f"{sid}: cached, skipping", flush=True)
            continue
        agreements = index.authoritative(sid, "credit_agreement")
        if not agreements:
            print(f"{sid}: ERROR no authoritative credit agreement", flush=True)
            continue
        loaded = load_enriched(sid, str(paths.enriched))
        if loaded is None:
            print(f"{sid}: ERROR no enriched cache -- run `covenant enrich` first", flush=True)
            continue
        enriched, disclosed = loaded

        text = max(
            (docs[r.doc_id].text for r in agreements), key=lambda t: len(find_covenant_keys(t))
        )
        keys = find_covenant_keys(text)
        clauses = {k: extract_clause_text(text, k, keys) for k in keys}
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

        try:
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
        except Exception as exc:  # noqa: BLE001
            print(f"{sid}: ERROR {exc}", flush=True)
            continue

        problems = validate_spec(spec)
        save_spec(sid, spec, client, paths.specs)
        print(
            f"{sid}: {len(spec['covenants'])} covenants "
            f"{'PROBLEMS: ' + str(problems) if problems else 'ok'}",
            flush=True,
        )


def cmd_build(args, paths: Paths) -> None:
    template = json.loads(paths.template.read_text())
    client = get_client("complex")  # only to resolve the model tag on the spec cache
    submission = {
        "team": args.team,
        "contact_email": args.contact_email,
        "model": client.model,
        "answers": {},
    }

    for sid in template["answers"]:
        spec = load_spec(sid, client, paths.specs)
        loaded = load_enriched(sid, str(paths.enriched))
        cells: dict[str, dict] = {}
        for key in template["answers"][sid]:
            # A null cell scores exactly like a wrong one, so every failure path still answers.
            if spec is None or loaded is None:
                result = fallback_cell(None, {}, [])
            else:
                cov = spec["covenants"].get(key)
                try:
                    result = (
                        compute_covenant(cov, spec["roles"], loaded[0])
                        if cov
                        else fallback_cell(None, {}, [])
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"{sid}.{key}: {exc} -- falling back", file=sys.stderr)
                    result = fallback_cell(cov, spec["roles"], loaded[0])
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
        submission["answers"][sid] = cells

    # allow_nan=False: the default emits the bare token NaN, which is not valid JSON and would make
    # the affected cells unscoreable.
    Path(args.out).write_text(json.dumps(submission, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"wrote {args.out}")
    if args.score:
        cmd_score(args, paths)


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
    ]:
        sub = subparsers.add_parser(name)
        sub.set_defaults(fn=fn)
        if wants_scenarios:
            sub.add_argument("scenarios", nargs="*")
            sub.add_argument("--fresh", action="store_true", help="ignore the existing cache")
        if name == "spec":
            sub.add_argument(
                "--votes", type=int, default=3, help="self-consistency samples to majority-vote"
            )
            sub.add_argument("--vote-temperature", type=float, default=0.5)

    build = subparsers.add_parser("build")
    build.set_defaults(fn=cmd_build)
    build.add_argument("--out", default="submission.json")
    build.add_argument("--team", required=True)
    build.add_argument("--contact-email", required=True)
    build.add_argument("--score", action="store_true", help="dev only: score against ground truth")

    score = subparsers.add_parser("score")
    score.set_defaults(fn=cmd_score)
    score.add_argument("--out", default="submission.json")

    args = parser.parse_args(argv)
    args.fn(args, resolve(args.data))


if __name__ == "__main__":
    main()
