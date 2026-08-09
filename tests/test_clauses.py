from __future__ import annotations

import pytest

from covenant.analyze.clauses import (
    extract_clause_text,
    extract_ownership_section,
    find_clause_heading,
    find_covenant_keys,
    ownership_section_found,
)

CAPTIONED = """Статья 6 Финансовые ковенанты

Статья 6 — Финансовые ковенанты
Пункт 6.1  Минимальная выручка. Не менее $1,000,000.00.
Пункт 6.2  Максимальные расходы. Не более $2,000,000.00.
Статья 7 Ограничительные обязательства
"""

# same agreement, numbered the way a differently-templated set might print it
GENERIC = """Раздел 6. Финансовые ковенанты

6.1 Минимальная выручка. Не менее $1,000,000.00.
6.2 Максимальные расходы. Не более $2,000,000.00.
7. Ограничительные обязательства
"""


def test_captioned_headings_are_found_as_before():
    assert find_covenant_keys(CAPTIONED, "6") == ["6.1", "6.2"]
    assert find_clause_heading(CAPTIONED, "6.1")[1] == "caption"


def test_keys_are_still_found_when_the_caption_is_absent():
    assert find_covenant_keys(GENERIC, "6") == ["6.1", "6.2"]


def test_a_bare_numbered_heading_is_located_by_the_fallback():
    located = find_clause_heading(GENERIC, "6.2")
    assert located is not None and located[1] == "numbered"


def test_clause_text_stops_at_the_next_clause_however_it_is_numbered():
    text = extract_clause_text(GENERIC, "6.1", ["6.1", "6.2"])
    assert "Минимальная выручка" in text
    assert "Максимальные расходы" not in text


def test_a_cross_reference_mid_sentence_is_not_mistaken_for_a_heading():
    text = "Статья 6\nПункт 6.1  Как указано в 6.2, выручка не менее $1.00.\nПункт 6.2  Прочее."
    assert "Как указано в 6.2" in extract_clause_text(text, "6.1", ["6.1", "6.2"])


def test_a_missing_clause_still_raises_so_the_caller_can_recover_it():
    with pytest.raises(ValueError, match="6.9"):
        extract_clause_text(CAPTIONED, "6.9", ["6.1", "6.2", "6.9"])


OWNERSHIP = """Бенефициарное владение и контроль
Организация Доля голосующих прав
Taraz Holding Group LLP 46.8%
Идентификация и проверка сведений
прочее
"""


def test_the_ownership_section_is_sliced_at_its_own_headings():
    section = extract_ownership_section(OWNERSHIP)
    assert "46.8%" in section
    assert "прочее" not in section
    assert ownership_section_found(OWNERSHIP)


def test_a_dossier_with_a_different_heading_falls_back_to_its_whole_text():
    dossier = "Структура владения\nTaraz Holding Group LLP 46.8% голосующих прав\n"
    assert not ownership_section_found(dossier)
    # returning None here would silently untag every related party in the scenario
    assert extract_ownership_section(dossier) == dossier.strip()


def test_a_document_with_no_ownership_content_at_all_still_returns_none():
    assert (
        extract_ownership_section("Проверка по санкционным перечням. Совпадений не выявлено.")
        is None
    )
