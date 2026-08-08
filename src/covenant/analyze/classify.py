from dataclasses import dataclass

from covenant.ingest.documents import Doc
from covenant.llm.client import Client

_SUPERSESSION_MARKERS = [
    "НЕДЕЙСТВУЮЩАЯ",
    "НЕ ПРИМЕНЯЕТСЯ",
    "УТРАТИЛ СИЛУ",
    "ЗАМЕНЕНА",
    "ЗАМЕНЕН",
    "ОКОНЧАТЕЛЬНЫМ ОТЧЁТОМ",
    "ОКОНЧАТЕЛЬНЫМ ОТЧЕТОМ",
]

# a second recurring decoy convention: a generic internal procedure/methodology document that
# talks about KYC (or audit, etc.) review in the abstract, explicitly disclaiming that it covers
# any specific client -- distinct from an actual filled-in dossier with real client data.
_GENERIC_PROCEDURE_MARKERS = [
    "Процедура комплаенса",
    "не содержит заключений о каком-либо конкретном клиенте",
    "авторитетные клиентские досье хранятся отдельно",
    "методическое руководство",
]

_ALL_MARKERS = _SUPERSESSION_MARKERS + _GENERIC_PROCEDURE_MARKERS


def _detected_markers(text: str) -> list[str]:
    return [m for m in _ALL_MARKERS if m in text]


# unlike the hint markers above, this one is a hard override, not a nudge: every superseded
# credit agreement in this dataset (12/12 confirmed) opens with this exact phrase at char 0, with
# zero false-positive contexts -- unlike "ПРОЕКТ", which genuinely means something else elsewhere.
# A small model shown this phrase as a hint still sometimes rules the doc authoritative anyway
# (nondeterministic across reruns), so this one specific, unambiguous, position-anchored signal is
# trusted directly rather than left to the model -- same category of "safe to hardcode" as the
# `ACC-` id or `Пункт X.X` heading conventions.
_HARD_SUPERSEDED_PREFIX = "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ"

CLASSIFY_HEAD_CHARS = 3000  # see classify_document: type/authority are decided on the first page

SYSTEM_PROMPT = """You are reviewing internal banking documents ahead of a corporate credit \
covenant compliance check. Given one document's full text, determine:

- company_name: the borrower/company this document is primarily about.
- doc_type: one of "credit_agreement", "kyc_dossier", "audit_report", or "other" (use "other" \
for anything that is not primarily a credit agreement, a KYC/related-party ownership dossier, \
or an audit/financial-statement report -- e.g. internal memos, marketing, IT, HR, insurance, \
vehicle-log documents).
- is_authoritative_for_covenants: true only if the document is ALL of: (a) current -- not a \
draft, and not superseded/replaced by a later version; (b) its subject matter is the kind \
covenant definitions actually reference: ownership/related-party structure, audited financial \
figures, or the credit agreement's own terms; AND (c) it contains concrete, client-specific \
content (actual names, percentages, dates, figures for this borrower) -- NOT a generic internal \
policy/procedure/methodology document that describes how such reviews are conducted in the \
abstract without stating any conclusion about this specific client. A document can look \
KYC-shaped (mention ownership review, compliance, "KYC") and still fail (c) if it's really a \
procedure manual, not a filled-in dossier. Do NOT require the document to literally contain \
covenant numbers or dollar figures -- e.g. a KYC ownership dossier has none but is exactly what a \
related-party covenant points to. A document merely using a word like "project" in an unrelated \
sense (e.g. the name of an internal IT initiative) is not evidence of draft status.
- reason: one or two sentences citing the specific textual evidence for your judgment (e.g. a \
supersession marker, a draft stamp, or the document's stated purpose).

Respond with strict JSON only, no markdown code fences, no commentary:
{"company_name": "...", "doc_type": "...", "is_authoritative_for_covenants": true, "reason": "..."}
"""


@dataclass(frozen=True)
class Classification:
    company_name: str
    doc_type: str
    is_authoritative_for_covenants: bool
    reason: str


def classify_document(doc: Doc, client: Client) -> Classification:
    markers = _detected_markers(doc.text)
    hint = ""
    if markers:
        hint = (
            f"\n\nAutomated scan found these literal phrase(s) somewhere in the text: {markers}. "
            "This is a hint only, not a verdict -- go read their actual context yourself before "
            "deciding; a phrase can appear in an unrelated sense."
        )
    # Only the head is sent: every signal this stage needs (title block, doc kind, supersession
    # plash, draft stamp, "procedure not a dossier" disclaimer) sits on the first page, while a
    # credit agreement runs 46KB of boilerplate that just dilutes the decision. Marker detection
    # above still scans the FULL text, so a signal buried deeper still reaches the model as a hint.
    head = doc.text[:CLASSIFY_HEAD_CHARS]
    user = f"Document text:\n\n{head}{hint}"
    data = client.complete_json(SYSTEM_PROMPT, user)
    is_authoritative = bool(data.get("is_authoritative_for_covenants", False))
    reason = str(data.get("reason", ""))
    if is_authoritative and doc.text.strip().startswith(_HARD_SUPERSEDED_PREFIX):
        is_authoritative = False
        reason = f"[overridden: document opens with '{_HARD_SUPERSEDED_PREFIX}'] {reason}"
    return Classification(
        company_name=str(data.get("company_name", "")),
        doc_type=str(data.get("doc_type", "other")),
        is_authoritative_for_covenants=is_authoritative,
        reason=reason,
    )
