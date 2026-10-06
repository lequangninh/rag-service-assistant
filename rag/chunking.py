"""Split Markdown documents into section-sized chunks.

Each chunk is one `##` section (or the intro above the first `##`), so a
citation always points to a section a human can find in the source file.
Long sections are split further on paragraph boundaries with a small overlap.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1200
OVERLAP_PARAGRAPHS = 1


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name, e.g. "pricing.md"
    doc_title: str  # the document's "# " heading
    section: str  # the "## " heading, or "Overview" for the intro
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} › {self.section}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if current and len("\n\n".join(current + [para])) > MAX_CHARS:
            parts.append("\n\n".join(current))
            current = current[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS else []
        current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    title_match = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else source

    # Split on level-2 headings, keeping the heading with its body.
    pieces = re.split(r"^(?=## )", markdown, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("## "):
            heading, _, body = piece.partition("\n")
            section = heading[3:].strip()
        else:
            section = "Overview"
            body = re.sub(r"^# .+$", "", piece, count=1, flags=re.MULTILINE)
        body = body.strip()
        if not body:
            continue
        parts = _split_long(body) if len(body) > MAX_CHARS else [body]
        for i, part in enumerate(parts):
            suffix = f"-{i + 1}" if len(parts) > 1 else ""
            slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
            chunks.append(
                Chunk(
                    id=f"{Path(source).stem}:{slug}{suffix}",
                    source=source,
                    doc_title=doc_title,
                    section=section,
                    text=part,
                )
            )
    return chunks


def load_corpus(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks
