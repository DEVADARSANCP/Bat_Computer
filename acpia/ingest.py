"""
Evidence ingestion — files on disk become Evidence rows.

Deliberately dumb: hash it, classify it by extension, record it. No analysis
happens here. Deciding what to *do* with a file is the planner's job, and
keeping that separation is what lets the same corpus be re-investigated with a
different agent lineup later.

Hashing on ingest is not decoration. In a real forensic context the hash is the
chain-of-custody anchor, and it is also how we deduplicate: ingesting the same
file twice must not double-count it as two pieces of evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .schema import Evidence, EvidenceKind
from .store import Store

#: Extension → evidence kind. Anything unlisted lands as UNKNOWN and the planner
#: is told about it rather than silently skipping it.
EXTENSION_KINDS: dict[str, EvidenceKind] = {
    # images
    ".jpg": EvidenceKind.IMAGE,
    ".jpeg": EvidenceKind.IMAGE,
    ".png": EvidenceKind.IMAGE,
    ".webp": EvidenceKind.IMAGE,
    ".bmp": EvidenceKind.IMAGE,
    ".tiff": EvidenceKind.IMAGE,
    ".tif": EvidenceKind.IMAGE,
    ".heic": EvidenceKind.IMAGE,
    # video
    ".mp4": EvidenceKind.VIDEO,
    ".avi": EvidenceKind.VIDEO,
    ".mov": EvidenceKind.VIDEO,
    ".mkv": EvidenceKind.VIDEO,
    ".webm": EvidenceKind.VIDEO,
    # audio
    ".mp3": EvidenceKind.AUDIO,
    ".wav": EvidenceKind.AUDIO,
    ".m4a": EvidenceKind.AUDIO,
    ".ogg": EvidenceKind.AUDIO,
    ".flac": EvidenceKind.AUDIO,
    # documents
    ".pdf": EvidenceKind.DOCUMENT,
    ".txt": EvidenceKind.DOCUMENT,
    ".md": EvidenceKind.DOCUMENT,
    ".docx": EvidenceKind.DOCUMENT,
    ".rtf": EvidenceKind.DOCUMENT,
    # structured exports
    ".csv": EvidenceKind.STRUCTURED,
    ".json": EvidenceKind.STRUCTURED,
    ".xml": EvidenceKind.STRUCTURED,
    ".xlsx": EvidenceKind.STRUCTURED,
    ".db": EvidenceKind.STRUCTURED,
    ".sqlite": EvidenceKind.STRUCTURED,
}

#: Skipped outright — noise that would otherwise clutter the case.
IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def classify(path: Path) -> EvidenceKind:
    return EXTENSION_KINDS.get(path.suffix.lower(), EvidenceKind.UNKNOWN)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streamed hash so a multi-GB video does not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_file(store: Store, case_id: str, path: str | Path) -> Evidence:
    """Ingest one file into a case and return its Evidence record."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    evidence = Evidence(
        case_id=case_id,
        path=str(path),
        filename=path.name,
        kind=classify(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )
    return store.add_evidence(evidence)


def ingest_directory(
    store: Store, case_id: str, directory: str | Path, *, recursive: bool = True
) -> list[Evidence]:
    """Ingest every usable file in a directory.

    This is the demo's front door: the investigator points at a case folder and
    walks away.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    ingested: list[Evidence] = []

    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        if path.name in IGNORED_NAMES or path.name.startswith("~$"):
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        ingested.append(ingest_file(store, case_id, path))

    store.audit(
        case_id,
        actor="system",
        action="directory_ingested",
        detail={"directory": str(directory), "files": len(ingested)},
    )
    return ingested
