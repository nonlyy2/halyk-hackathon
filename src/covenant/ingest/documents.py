import hashlib
import json
import subprocess
import tempfile
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
    """Every PDF in documents_dir, extracted and cached."""
    documents_path = Path(documents_dir)
    cache_path = Path(cache_dir)
    return [_extract_one(path, cache_path) for path in sorted(documents_path.glob("*.pdf"))]
