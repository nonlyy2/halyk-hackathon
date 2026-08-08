"""One place that answers "which documents speak for this borrower".

Each stage used to carry its own copy of this lookup -- five near-identical helpers, one of which
returned the first match and the others a list, so a scenario whose disclosures were split across
two audit reports was handled differently depending on which stage asked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Classification:
    doc_id: str
    model: str
    scenario_ids: list[str]
    account_ids: list[str]
    company_name: str
    doc_type: str
    is_authoritative: bool
    reason: str

    @classmethod
    def from_row(cls, row: dict) -> Classification:
        return cls(
            doc_id=row["doc_id"],
            model=row.get("model", ""),
            scenario_ids=row.get("scenario_ids", []),
            account_ids=row.get("account_ids", []),
            company_name=row.get("company_name", ""),
            doc_type=row.get("doc_type", "other"),
            is_authoritative=bool(row.get("is_authoritative_for_covenants", False)),
            reason=row.get("reason", ""),
        )


class DocumentIndex:
    def __init__(self, rows: list[Classification]):
        self._rows = rows

    @classmethod
    def load(cls, path: Path) -> DocumentIndex:
        if not path.exists():
            return cls([])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return cls([Classification.from_row(r) for r in rows])

    def all(self) -> list[Classification]:
        return list(self._rows)

    def for_scenario(self, scenario_id: str) -> list[Classification]:
        return [r for r in self._rows if scenario_id in r.scenario_ids]

    def authoritative(self, scenario_id: str, doc_type: str) -> list[Classification]:
        """EVERY authoritative document of this type, never just the first.

        A borrower's disclosures are not guaranteed to sit in one file: an audit report may state
        that its conclusion "is set out in the agreed-upon-procedures report and is not repeated
        here", leaving the actual reclassification in a second document. Picking one meant the
        right document won only when it happened to sort first by filename.
        """
        return [
            r
            for r in self.for_scenario(scenario_id)
            if r.doc_type == doc_type and r.is_authoritative
        ]

    def borrower_names(self) -> dict[str, str]:
        """scenario_id -> borrower name, taken from its own authoritative credit agreement."""
        names: dict[str, str] = {}
        for row in self._rows:
            if row.doc_type == "credit_agreement" and row.is_authoritative:
                for sid in row.scenario_ids:
                    names.setdefault(sid, row.company_name)
        return names
