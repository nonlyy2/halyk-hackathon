"""Where the dataset lives and where intermediate artefacts go.

Every path is resolved once, here, from COVENANT_DATA (or an explicit --data flag) rather than
from the process's working directory. The previous layout defaulted each path to a bare filename,
so the whole pipeline only ran when invoked from inside the dataset folder and silently found
nothing otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data: Path

    @property
    def ledger(self) -> Path:
        return self.data / "master_ledger_2025.csv"

    @property
    def documents(self) -> Path:
        return self.data / "documents"

    @property
    def template(self) -> Path:
        return self.data / "submission_template.json"

    @property
    def ground_truth(self) -> Path:
        return self.data / "ground_truth.json"

    @property
    def cache(self) -> Path:
        return self.data / ".cache"

    @property
    def text_cache(self) -> Path:
        return self.cache / "text"

    @property
    def classifications(self) -> Path:
        return self.cache / "classifications.jsonl"

    @property
    def enriched(self) -> Path:
        return self.cache / "enriched"

    @property
    def specs(self) -> Path:
        return self.cache / "specs"


def resolve(data_dir: str | os.PathLike[str] | None = None) -> Paths:
    root = Path(data_dir or os.environ.get("COVENANT_DATA", "data")).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"dataset directory not found: {root}\n"
            "pass --data /path/to/dataset or set COVENANT_DATA"
        )
    return Paths(data=root)
