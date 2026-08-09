"""Build a perturbed copy of a dataset, with its key transformed the same way.

The public set is 36 cells and we can see its key, which makes it a very poor judge of whether a
change generalises: any edit kept because the score went up may simply have fitted those 36. The
private set answers that question once, for real, with no second attempt.

A perturbation is the nearest thing to a held-out set that can be built without one. Change
something the private set could plausibly do differently -- number the covenants under a different
article, caption the clauses another way, renumber the accounts, denominate the ledger a thousand
times larger -- while keeping the answer derivable from the old key by a known transformation. A
pipeline that reads the documents will score the same. A pipeline that has memorised this dataset's
shape will not, and the gap says exactly what it memorised.

Deliberately NOT perturbed: anything whose correct answer stops being derivable. Rewording a
transaction description changes what the covenant should measure, so it cannot be checked
mechanically and is left alone -- a test whose expected output is a guess proves nothing.

    uv run python tools/perturb.py relabel  --data data --out /tmp/pert-relabel
    uv run python tools/perturb.py rescale  --data data --out /tmp/pert-rescale --factor 1000
    uv run python tools/perturb.py reformat --data data --out /tmp/pert-reformat

Then run the pipeline against `--data <out>` and score as usual: the transformed key is written to
the output directory as `ground_truth.json`, so `covenant build --score` compares against it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import pandas as pd

from covenant.config import resolve
from covenant.ingest.documents import _cache_path, extract_documents

# ---------------------------------------------------------------------------------------------
# text rewriting helpers
# ---------------------------------------------------------------------------------------------

_MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")


def _scale_money(text: str, factor: float) -> str:
    """Multiply every dollar figure in a document by `factor`, keeping the written format.

    Only `$`-prefixed figures: a ratio ("1.20x"), an ownership percentage ("46.8%") and a date
    ("2025-01-01") must all stay exactly as they are, and none of them carries a dollar sign.
    """

    def one(match: re.Match[str]) -> str:
        value = float(match.group(1).replace(",", "")) * factor
        return f"${value:,.2f}"

    return _MONEY_RE.sub(one, text)


def _renumber_article(text: str, old: str, new: str) -> str:
    """Move the covenants to a different article number, headings and cross-references alike."""
    text = text.replace(f"Статья {old}", f"Статья {new}")
    return re.sub(rf"(Пункт\s+){old}\.(\d+)", rf"\g<1>{new}.\2", text)


def _recaption_clauses(text: str) -> str:
    """Print the clause headings the way a differently-templated agreement might.

    "Пункт 6.1" -> "6.1." -- the caption the deterministic slicer looks for first disappears, and
    only the generic numbered-heading fallback can still find the clause.
    """
    return re.sub(r"Пункт\s+(\d+\.\d+)", r"\1.", text)


_HEADING_SWAPS = {
    "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ": "РАСКРЫТИЯ ДЛЯ ЦЕЛЕЙ ПРОВЕРКИ ОБЯЗАТЕЛЬСТВ",
    "Бенефициарное владение и контроль": "Структура участия и контроля",
    "Идентификация и проверка сведений": "Установление и сверка данных клиента",
}


def _reheading(text: str) -> str:
    """Rename the section headings every slicer keys on, leaving the sections themselves intact."""
    for old, new in _HEADING_SWAPS.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------------------------
# the perturbations
# ---------------------------------------------------------------------------------------------


def _relabel_maps(ledger: pd.DataFrame, seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """scenario_id -> new scenario_id, and account_id -> new account_id."""
    rng = random.Random(seed)
    scenarios = sorted({t.split("-")[1] for t in ledger["txn_id"]})
    # a shape the public set never uses, so a hardcoded "P<n>"/"B<n>" cannot accidentally match
    new_scenarios = [f"Z{i + 41}" for i in range(len(scenarios))]
    rng.shuffle(new_scenarios)
    accounts = sorted(set(ledger["account_id"]))
    new_accounts = [f"ACC-{9100 + i}" for i in range(len(accounts))]
    rng.shuffle(new_accounts)
    return dict(zip(scenarios, new_scenarios, strict=True)), dict(
        zip(accounts, new_accounts, strict=True)
    )


def perturb(kind: str, data: Path, out: Path, factor: float, seed: int) -> None:
    paths = resolve(data)
    ledger = pd.read_csv(paths.ledger)
    template = json.loads(paths.template.read_text())
    key = json.loads(paths.ground_truth.read_text()) if paths.ground_truth.exists() else None

    scenario_map: dict[str, str] = {}
    account_map: dict[str, str] = {}
    txn_map: dict[str, str] = {}

    if kind == "relabel":
        scenario_map, account_map = _relabel_maps(ledger, seed)
        # renumber the transactions too, so nothing can key on a position or an id it saw before
        for sid, new_sid in scenario_map.items():
            rows = ledger[ledger["txn_id"].str.startswith(f"TXN-{sid}-")]
            for offset, txn_id in enumerate(rows["txn_id"], start=1):
                txn_map[txn_id] = f"TXN-{new_sid}-{offset + 100:04d}"
        ledger["txn_id"] = ledger["txn_id"].map(lambda t: txn_map.get(t, t))
        ledger["account_id"] = ledger["account_id"].map(lambda a: account_map.get(a, a))
        ledger = ledger.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    elif kind == "rescale":
        ledger["amount"] = ledger["amount"] * factor
    elif kind != "reformat":
        raise SystemExit(f"unknown perturbation {kind!r}")

    out.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out / "master_ledger_2025.csv", index=False)

    # documents: copied so the pipeline sees a normal dataset, with the text cache pre-populated
    # from the perturbed text so no PDF is re-rendered and no OCR is re-run
    documents_out = out / "documents"
    documents_out.mkdir(exist_ok=True)
    cache_out = out / ".cache" / "text"
    cache_out.mkdir(parents=True, exist_ok=True)

    docs = extract_documents(str(paths.documents), str(paths.text_cache))
    for doc in docs:
        source = paths.documents / f"{doc.doc_id}.pdf"
        if not source.exists():
            continue
        shutil.copy2(source, documents_out / source.name)
        text = doc.text
        if kind == "relabel":
            for old, new in account_map.items():
                text = text.replace(old, new)
            for old, new in txn_map.items():
                text = text.replace(old, new)
        elif kind == "rescale":
            text = _scale_money(text, factor)
        elif kind == "reformat":
            # 26, not 8: these agreements already run to article 25, so a lower number would
            # collide with an existing heading and the test would be measuring my own collision
            # rather than the pipeline.
            text = _reheading(_recaption_clauses(_renumber_article(text, "6", "26")))
        target = _cache_path(documents_out / source.name, cache_out)
        target.write_text(json.dumps({"text": text, "method": doc.method}, ensure_ascii=False))

    # template and key, transformed the same way
    def remap_cells(cells: dict, is_key: bool) -> dict:
        out_cells = {}
        for cell_key, cell in cells.items():
            new_key = f"26.{cell_key.split('.', 1)[1]}" if kind == "reformat" else cell_key
            cell = dict(cell)
            if is_key:
                if kind == "rescale" and _is_money(cell["actual"]):
                    cell["actual"] = round(cell["actual"] * factor, 2)
                if kind == "relabel" and cell.get("evidence_txn_id"):
                    cell["evidence_txn_id"] = txn_map.get(
                        cell["evidence_txn_id"], cell["evidence_txn_id"]
                    )
            out_cells[new_key] = cell
        return out_cells

    def _is_money(actual: float) -> bool:
        # A ratio covenant's `actual` is a pure number and does not move when the ledger is
        # redenominated; a dollar amount does. Nothing in the key says which is which, so it is
        # taken from the magnitude -- crude, but these two populations are orders apart.
        return abs(actual) >= 1000

    template["answers"] = {
        scenario_map.get(sid, sid): remap_cells(cells, is_key=False)
        for sid, cells in template["answers"].items()
    }
    (out / "submission_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2)
    )

    if key is not None:
        key["scenarios"] = {
            scenario_map.get(sid, sid): {
                "covenants": remap_cells(scenario["covenants"], is_key=True)
            }
            for sid, scenario in key["scenarios"].items()
        }
        (out / "ground_truth.json").write_text(json.dumps(key, ensure_ascii=False, indent=2))

    print(f"wrote {kind} dataset to {out}")
    if scenario_map:
        print(f"  scenarios: {scenario_map}")
    if kind == "rescale":
        print(f"  amounts and $ thresholds scaled by {factor}")
    if kind == "reformat":
        print("  article 6 -> 26, clause captions dropped, section headings renamed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["relabel", "rescale", "reformat"])
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    perturb(args.kind, Path(args.data), Path(args.out), args.factor, args.seed)


if __name__ == "__main__":
    main()
