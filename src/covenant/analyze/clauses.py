import re

_PUNKT_RE = re.compile(r"Пункт\s+(\d+\.\d+)")


def find_covenant_keys(text: str, article: str = "6") -> list[str]:
    article_heading = f"Статья {article}"
    first = text.find(article_heading)
    second = text.find(article_heading, first + 1) if first >= 0 else -1
    start = second if second >= 0 else 0

    next_article = text.find(f"Статья {int(article) + 1}", start if start > 0 else 0)
    section = text[start : next_article if next_article > start else len(text)]
    return sorted(set(_PUNKT_RE.findall(section)))


def extract_clause_text(text: str, key: str, keys: list[str]) -> str:
    article = key.split(".")[0]
    heading = f"Пункт {key}"

    article_heading = f"Статья {article}"
    first = text.find(article_heading)
    second = text.find(article_heading, first + 1) if first >= 0 else -1
    search_from = second if second >= 0 else 0

    start = text.find(heading, search_from)
    if start < 0:
        start = text.find(heading)
    if start < 0:
        raise ValueError(f"heading {heading!r} not found")

    candidates: list[int] = []
    next_article = text.find(f"Статья {int(article) + 1}", start + len(heading))
    if next_article > start:
        candidates.append(next_article)
    for other_key in keys:
        if other_key == key:
            continue
        pos = text.find(f"Пункт {other_key}", start + len(heading))
        if pos > start:
            candidates.append(pos)

    end = min(candidates) if candidates else len(text)
    return text[start:end].strip()


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


def extract_ownership_section(text: str) -> str | None:
    start = text.find(_OWNERSHIP_START)
    if start < 0:
        return None
    end = text.find(_OWNERSHIP_END, start)
    return text[start : end if end > start else len(text)].strip()
