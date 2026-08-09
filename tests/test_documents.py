from __future__ import annotations

import json

from covenant.ingest import documents


def _fake_pdf(tmp_path, name="a.pdf"):
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    return path


def test_a_truncated_cache_entry_is_re_extracted_not_fatal(tmp_path, monkeypatch):
    pdf = _fake_pdf(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    documents._cache_path(pdf, cache).write_text("")  # a run killed mid-write

    monkeypatch.setattr(documents, "_render_pages", lambda p: (["recovered"], False), raising=False)
    monkeypatch.setattr(
        documents.pdfium, "PdfDocument", lambda _p: [], raising=False
    )  # no pages -> empty text
    doc = documents._extract_one(pdf, cache)
    assert doc.doc_id == "a"
    # and the entry is now valid, so the next run reads it back
    assert json.loads(documents._cache_path(pdf, cache).read_text())["method"] == "raw"


def test_a_cache_entry_missing_its_fields_is_re_extracted(tmp_path, monkeypatch):
    pdf = _fake_pdf(tmp_path, "b.pdf")
    cache = tmp_path / "cache"
    cache.mkdir()
    documents._cache_path(pdf, cache).write_text('{"text": "x"}')  # no "method"

    monkeypatch.setattr(documents.pdfium, "PdfDocument", lambda _p: [], raising=False)
    assert documents._extract_one(pdf, cache).method == "raw"


def test_a_good_cache_entry_is_used_without_touching_the_pdf(tmp_path, monkeypatch):
    pdf = _fake_pdf(tmp_path, "c.pdf")
    cache = tmp_path / "cache"
    cache.mkdir()
    documents._cache_path(pdf, cache).write_text(
        json.dumps({"text": "cached", "method": "raw+ocr"})
    )

    def explode(_p):
        raise AssertionError("the pdf must not be opened when the cache is readable")

    monkeypatch.setattr(documents.pdfium, "PdfDocument", explode, raising=False)
    doc = documents._extract_one(pdf, cache)
    assert (doc.text, doc.method) == ("cached", "raw+ocr")


def test_no_partial_files_are_left_behind_on_a_successful_write(tmp_path, monkeypatch):
    pdf = _fake_pdf(tmp_path, "d.pdf")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(documents.pdfium, "PdfDocument", lambda _p: [], raising=False)
    documents._extract_one(pdf, cache)
    assert list(cache.glob("*.part")) == []
