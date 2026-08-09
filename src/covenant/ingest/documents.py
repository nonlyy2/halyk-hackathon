import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

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
        cached = json.loads(cache_path.read_text())
        return Doc(doc_id=path.stem, text=cached["text"], method=cached["method"])

    pdf = pdfium.PdfDocument(str(path))
    parts: list[str] = []
    ocr_used = False
    for page in pdf:
        raw = page.get_textpage().get_text_range()
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
    cache_path.write_text(json.dumps({"text": text, "method": method}, ensure_ascii=False))
    return Doc(doc_id=path.stem, text=text, method=method)


def extract_documents(
    documents_dir: str = "documents", cache_dir: str = ".cache/docling"
) -> list[Doc]:
    """Every PDF in documents_dir, extracted and cached.

    Extraction is done in parallel because a cold run has a couple of hundred files to get through
    and the scanned ones each shell out to tesseract page by page -- minutes of the submission
    window, spent waiting on a subprocess. Both pypdfium2 and the tesseract call release the GIL,
    so threads are enough; the cache write is per-file and atomic enough at this size.
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
