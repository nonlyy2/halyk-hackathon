"""Slicing the pieces of a document that later stages read: covenant clauses, the audit report's
covenant addendum, the KYC ownership table.

Every slicer here works off a heading string, because that is how these documents are laid out --
and that is also the one dataset-specific assumption in the pipeline. The private set shares the
public set's formats, but a heading it happens to phrase differently would silently take out a
whole class of answers: an unfound ownership section means no counterparty is ever tagged as a
related party, and every related-party covenant then computes zero without erroring anywhere. So
each slicer has a general fallback below its format-specific one, and reports which of the two
fired, so `covenant doctor` can say so before the run is submitted rather than after.
"""

from __future__ import annotations

import json
import re

from covenant.llm.client import Client

_PUNKT_RE = re.compile(r"Пункт\s+(\d+\.\d+)")
# The same numbering without the Russian caption: a numbered heading at the start of a line, or one
# of the usual captions in either language. Deliberately anchored, so a bare "6.1" mentioned inside
# a sentence (a cross-reference) is not mistaken for the clause itself.
_GENERIC_KEY_RE = re.compile(
    r"(?m)^\s*(?:п\.?\s*|пункт\s+|clause\s+|section\s+|article\s+)?(\d+\.\d+)[.\s)]"
    r"|(?:^|\s)(?:п\.\s?|Clause\s|Section\s)(\d+\.\d+)\b",
    re.IGNORECASE,
)


def _article_section(text: str, article: str) -> str:
    """The slice of the document covering one article's body.

    The second occurrence of the article heading is used where there is one: these agreements print
    a table of contents first, and the first hit is that listing, not the article.
    """
    article_heading = f"Статья {article}"
    first = text.find(article_heading)
    second = text.find(article_heading, first + 1) if first >= 0 else -1
    start = second if second >= 0 else 0
    next_article = text.find(f"Статья {int(article) + 1}", start if start > 0 else 0)
    return text[start : next_article if next_article > start else len(text)]


def find_covenant_keys(text: str, article: str = "6") -> list[str]:
    section = _article_section(text, article)
    keys = sorted(set(_PUNKT_RE.findall(section)))
    if keys:
        return keys
    # No captioned headings found -- either the document numbers its clauses differently or the
    # article heading itself was not located, in which case `section` is the whole text. Fall back
    # to any numbered heading under this article number.
    found = {a or b for a, b in _GENERIC_KEY_RE.findall(section)}
    return sorted(k for k in found if k and k.split(".")[0] == article)


def _heading_patterns(key: str) -> list[tuple[str, re.Pattern[str]]]:
    escaped = re.escape(key)
    return [
        ("caption", re.compile(rf"Пункт\s+{escaped}")),
        ("abbrev", re.compile(rf"(?m)^\s*п\.?\s*{escaped}[.\s)]")),
        ("numbered", re.compile(rf"(?m)^\s*{escaped}[.\s)]")),
        ("english", re.compile(rf"(?:Clause|Section|Article)\s+{escaped}\b", re.IGNORECASE)),
    ]


def find_clause_heading(text: str, key: str, search_from: int = 0) -> tuple[int, str] | None:
    """(offset, which pattern matched) for this clause's heading, or None.

    Exposed separately from extract_clause_text so `covenant doctor` can report *how* a clause was
    located without slicing it again -- a key found only by the generic pattern is a signal that the
    document's format has drifted from the one the captioned pattern expects.
    """
    for method, pattern in _heading_patterns(key):
        match = pattern.search(text, search_from)
        if match:
            return match.start(), method
        if search_from:
            match = pattern.search(text)
            if match:
                return match.start(), method
    return None


def extract_clause_text(text: str, key: str, keys: list[str]) -> str:
    article = key.split(".")[0]

    article_heading = f"Статья {article}"
    first = text.find(article_heading)
    second = text.find(article_heading, first + 1) if first >= 0 else -1
    search_from = second if second >= 0 else 0

    located = find_clause_heading(text, key, search_from)
    if located is None:
        raise ValueError(f"clause {key!r} heading not found")
    start, _method = located

    candidates: list[int] = []
    next_article = text.find(f"Статья {int(article) + 1}", start + len(key))
    if next_article > start:
        candidates.append(next_article)
    for other_key in keys:
        if other_key == key:
            continue
        other = find_clause_heading(text, other_key, start + len(key))
        if other and other[0] > start:
            candidates.append(other[0])

    end = min(candidates) if candidates else len(text)
    return text[start:end].strip()


CLAUSE_SLICE_PROMPT = """You are given the full text of a credit agreement and a list of clause \
numbers. For each clause number, return the clause's own text, verbatim, from its heading up to \
the start of the next clause -- exactly as printed, with nothing added, summarised or translated.

If a clause number does not appear in this document at all, map it to an empty string rather than \
guessing or returning a neighbouring clause.

Respond with strict JSON only, no markdown code fences, no commentary:
{"<clause number>": "<verbatim clause text>"}
"""


def llm_clause_texts(agreement_text: str, keys: list[str], client: Client) -> dict[str, str]:
    """Last-resort clause slicing when no heading pattern matched.

    A clause the slicers cannot find is a covenant with no text, which is a lost cell -- and the
    text is plainly there, just captioned in a way the patterns do not anticipate. One call is far
    cheaper than the cells, and it only ever runs for keys the deterministic path already failed on.
    """
    payload = json.dumps({"clause_numbers": keys}, ensure_ascii=False)
    data = client.complete_json(
        CLAUSE_SLICE_PROMPT, f"{payload}\n\nAgreement:\n\n{agreement_text}", max_tokens=8192
    )
    return {k: v for k, v in data.items() if k in set(keys) and isinstance(v, str) and v.strip()}


# audit reports carry their covenant-relevant disclosures (reclassifications, cut-offs, FX
# equivalents) under one of two headings -- the second is what B1's Agreed-Upon-Procedures
# report (a different document shape from a standard audit report) uses instead.
_ADDENDUM_HEADINGS = ["ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ", "Выводы по классификации операций"]
_ADDENDUM_END_MARKER = "За аудитора"


def extract_covenant_addendum(text: str) -> str | None:
    """Slice an audit/AUP report from its covenant-disclosures heading to the auditor's
    signature block (or end of text if no signature block follows). None if neither heading
    is present -- callers must handle that, not assume every audit doc has this section."""
    start = -1
    for heading in _ADDENDUM_HEADINGS:
        pos = text.find(heading)
        if pos >= 0:
            start = pos
            break
    if start < 0:
        return None

    end = text.find(_ADDENDUM_END_MARKER, start)
    return text[start : end if end > start else len(text)].strip()


_OWNERSHIP_START = "Бенефициарное владение и контроль"
_OWNERSHIP_END = "Идентификация и проверка сведений"
# Any of these appearing in a KYC document means the ownership table is somewhere in it, even if the
# section heading is phrased differently -- enough to justify handing the whole document over.
_OWNERSHIP_MARKERS = ("голосующих прав", "связанными сторонами", "связанной стороной")


def ownership_section_found(text: str) -> bool:
    """Whether the ownership section was located by its own heading, for the doctor report."""
    return _OWNERSHIP_START in text


def extract_ownership_section(text: str) -> str | None:
    """The KYC ownership/related-party section, or the whole dossier when the heading is missing.

    Returning None here used to be silent and total: without this text no counterparty is ever
    tagged as a related party, so every related-party covenant computes zero and reports COMPLIANT
    with no error anywhere in the run. Handing the extractor the whole dossier costs a longer prompt
    and nothing else -- it transcribes an ownership table wherever it sits.
    """
    start = text.find(_OWNERSHIP_START)
    if start >= 0:
        end = text.find(_OWNERSHIP_END, start)
        return text[start : end if end > start else len(text)].strip()
    if any(marker in text for marker in _OWNERSHIP_MARKERS):
        return text.strip() or None
    return None
