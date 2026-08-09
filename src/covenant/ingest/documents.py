import hashlib
import json
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

_PDFIUM_LOCK = threading.Lock()

PAGE_MIN_CHARS = 20  # below this, treat the page as scanned/image-only -> OCR fallback
OCR_LANG = "rus+eng"
OCR_RENDER_SCALE = 3


@dataclass(frozen=True)
class Doc:
    doc_id: str  # filename stem, e.g. "2d44bdf2437c"
    text: str
    method: str  # "raw" | "raw+ocr"


def _cache_path(path: Path, cache_dir: Path) -> Path:
    stat = path.stat()
    key = hashlib.sha256(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return cache_dir / f"{key}.json"


def _ocr_page(page: pdfium.PdfPage) -> str:
    with _PDFIUM_LOCK:  # rendering is pdfium too -- see the note in _extract_one
        bitmap = page.render(scale=OCR_RENDER_SCALE)
        image = bitmap.to_pil()
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        image.save(f.name)
        result = subprocess.run(
            ["tesseract", f.name, "stdout", "-l", OCR_LANG],
            capture_output=True,
            text=True,
            timeout=60,
        )
    return result.stdout


def _extract_one(path: Path, cache_dir: Path) -> Doc:
    cache_path = _cache_path(path, cache_dir)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return Doc(doc_id=path.stem, text=cached["text"], method=cached["method"])
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            # An unreadable entry is a cache miss, not a failure. A run killed mid-write leaves a
            # truncated file behind, and treating that as fatal turned one interrupted extraction
            # into every later run crashing with "Expecting value: line 1 column 1" -- an error
            # naming neither the document nor the cache. Re-extracting costs one document.
            print(f"  cached text for {path.name} unreadable, re-extracting", flush=True)

    # pdfium is NOT thread-safe, whatever the GIL does. Calling it from several threads deadlocks
    # inside its font mapper (CFX_FontMapper::FindSubstFace) on documents that need a substituted
    # face -- the process spins at full CPU and never returns, and which documents trigger it
    # depends entirely on the fonts they embed. One dataset extracted fine in parallel and the next
    # hung on its 48th file. Every pdfium call is serialised here; the pages are fast (milliseconds)
    # and the slow part, OCR, is a subprocess that still runs concurrently.
    parts: list[str] = []
    ocr_used = False
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        pages = [(page, page.get_textpage().get_text_range()) for page in pdf]
    for page, raw in pages:
        if len(raw.strip()) < PAGE_MIN_CHARS:
            ocr_text = _ocr_page(page)
            if len(ocr_text.strip()) > len(raw.strip()):
                parts.append(ocr_text)
                ocr_used = True
                continue
        parts.append(raw)
    text = "\n".join(parts)
    method = "raw+ocr" if ocr_used else "raw"

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename: on POSIX the rename is atomic, so a reader either sees
    # the previous state or the complete new one, and a process killed mid-write leaves the
    # temporary behind rather than a half-written cache entry.
    payload = json.dumps({"text": text, "method": method}, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=cache_dir, prefix=cache_path.stem, suffix=".part", delete=False
    ) as handle:
        handle.write(payload)
        partial = Path(handle.name)
    partial.replace(cache_path)
    return Doc(doc_id=path.stem, text=text, method=method)


def extract_documents(
    documents_dir: str = "documents", cache_dir: str = ".cache/docling"
) -> list[Doc]:
    """Every PDF in documents_dir, extracted and cached.

    Extraction is done in parallel because a cold run has a couple of hundred files to get through
    and the scanned ones each shell out to tesseract page by page -- minutes of the submission
    window, spent waiting on a subprocess. pdfium itself is serialised by a lock (see _extract_one)
    -- what these threads overlap is the OCR subprocess and the file I/O, not the parsing.
    """
    documents_path = Path(documents_dir)
    cache_path = Path(cache_dir)
    paths = sorted(documents_path.glob("*.pdf"))
    workers = min(concurrency(), len(paths)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: _extract_one(p, cache_path), paths))


UNBOUNDED_CONCURRENCY = 64  # a ceiling, not a target: callers cap this at the number of items


def concurrency() -> int:
    """How many documents / scenarios to work on at once.

    0 means "as many as there are", for a paid endpoint whose rate limit is far above anything this
    pipeline produces -- the whole run is then only as slow as its slowest single call. On a metered
    free tier the limit that matters is COVENANT_MIN_INTERVAL, and raising concurrency past it only
    parallelises the waiting.
    """
    raw = os.environ.get("COVENANT_CONCURRENCY", "4")
    try:
        value = int(raw)
    except ValueError:
        return 4
    return UNBOUNDED_CONCURRENCY if value <= 0 else value
