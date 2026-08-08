"""Stage 9 -- dev-only: score submission.json against ground_truth.json using the exact
CASE.ru.md Section 4 rubric. Never imported by the real pipeline; run.py must not accept a
ground-truth path at all.

Per cell: status is 0.50 (exact "COMPLIANT"/"BREACH" match only -- any mismatch, missing cell, or
invalid value zeroes the WHOLE cell, not just this component). actual is 0.30 on a linear decaying
scale, zero at 5% relative error. evidence_txn_id is 0.20 -- if the key names a specific txn_id,
exact match or zero; if the key is null (aggregate/ratio tests), those 0.20 decay together with
`actual` on the same scale, since a null-key cell earns them by being accurate on `actual`, not by
guessing an id (guessing is explicitly not worth attempting per the case rules).

CASE.ru.md also mentions cells are weighted by complexity in the real submission's final score --
that weighting scheme isn't disclosed to contestants, so this dev-only scorer reports a plain
unweighted mean instead; it exists to validate pipeline logic, not to reproduce the leaderboard
score.
"""

from __future__ import annotations

import json
from pathlib import Path

_DECAY_ZERO_AT = 0.05  # relative error at which actual/evidence-via-actual score hits zero


def _decay_fraction(value: float, key_value: float) -> float:
    if key_value == 0:
        return 1.0 if value == 0 else 0.0
    relative_error = abs(value - key_value) / abs(key_value)
    return max(0.0, 1.0 - relative_error / _DECAY_ZERO_AT)


def score_cell(submitted: dict | None, key: dict) -> dict:
    """One cell's score breakdown: {"score", "status_score", "actual_score", "evidence_score", "notes"}."""
    if not isinstance(submitted, dict):
        return {
            "score": 0.0,
            "status_score": 0.0,
            "actual_score": 0.0,
            "evidence_score": 0.0,
            "notes": ["cell missing or not an object"],
        }

    status = submitted.get("status")
    if status not in ("COMPLIANT", "BREACH"):
        return {
            "score": 0.0,
            "status_score": 0.0,
            "actual_score": 0.0,
            "evidence_score": 0.0,
            "notes": [f"status invalid: {status!r} (must be exactly COMPLIANT or BREACH)"],
        }
    if status != key["status"]:
        return {
            "score": 0.0,
            "status_score": 0.0,
            "actual_score": 0.0,
            "evidence_score": 0.0,
            "notes": [
                f"status wrong: got {status!r}, expected {key['status']!r} -- whole cell zeroed"
            ],
        }

    notes: list[str] = []
    status_score = 0.50

    actual = submitted.get("actual")
    is_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if not is_number or actual < 0:
        actual_frac = 0.0
        notes.append(f"actual missing, non-numeric, or negative: {actual!r}")
    else:
        actual_frac = _decay_fraction(float(actual), key["actual"])
        if actual_frac < 1.0:
            notes.append(
                f"actual off by {abs(float(actual) - key['actual']):,.2f} ({(1 - actual_frac) * 100:.1f}% of tolerance used)"
            )
    actual_score = 0.30 * actual_frac

    key_evidence = key.get("evidence_txn_id")
    submitted_evidence = submitted.get("evidence_txn_id")
    if key_evidence is None:
        evidence_score = 0.20 * actual_frac  # earned via actual's accuracy, not a guessed id
    elif submitted_evidence == key_evidence:
        evidence_score = 0.20
    else:
        evidence_score = 0.0
        notes.append(
            f"evidence_txn_id wrong: got {submitted_evidence!r}, expected {key_evidence!r}"
        )

    return {
        "score": status_score + actual_score + evidence_score,
        "status_score": status_score,
        "actual_score": actual_score,
        "evidence_score": evidence_score,
        "notes": notes,
    }


def score_submission(submission_path: str, ground_truth_path: str) -> dict:
    try:
        submission = json.loads(Path(submission_path).read_text())
    except json.JSONDecodeError as exc:
        return {
            "error": f"submission.json is not valid JSON: {exc}",
            "cells": {},
            "total": 0.0,
            "max_total": 0.0,
        }
    ground_truth = json.loads(Path(ground_truth_path).read_text())

    answers = submission.get("answers", {})
    cells: dict[str, dict] = {}
    for sid, scenario in ground_truth["scenarios"].items():
        for key, key_cell in scenario["covenants"].items():
            submitted_cell = answers.get(sid, {}).get(key)
            cells[f"{sid}.{key}"] = score_cell(submitted_cell, key_cell)

    total = sum(c["score"] for c in cells.values())
    max_total = float(len(cells))
    return {
        "cells": cells,
        "total": total,
        "max_total": max_total,
        "team": submission.get("team", ""),
    }


def format_report(result: dict) -> str:
    if "error" in result:
        return f"SCORING FAILED: {result['error']}"

    lines = [f"{'cell':<10} {'score':>6}   status  actual  evidence   notes"]
    for cell_id, c in sorted(result["cells"].items()):
        marks = f"{c['status_score']:.2f}    {c['actual_score']:.2f}    {c['evidence_score']:.2f}"
        note = "; ".join(c["notes"])
        lines.append(f"{cell_id:<10} {c['score']:.3f}   {marks}   {note}")

    pct = 100 * result["total"] / result["max_total"] if result["max_total"] else 0.0
    lines.append("")
    lines.append(f"TOTAL: {result['total']:.3f} / {result['max_total']:.0f}  ({pct:.1f}%)")
    lines.append(
        "(unweighted mean -- the real submission's final score weights cells by complexity, undisclosed)"
    )
    return "\n".join(lines)
