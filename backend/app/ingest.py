"""Knowledge-base ingestion: parse markdown files with YAML front matter and
split them into ``##``-heading chunks. No retrieval, indexing, or agent logic.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, ConfigDict

_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge-base"


class Document(BaseModel):
    """One knowledge-base markdown file.

    Known front-matter keys are declared explicitly; any other key found in a
    file's front matter is preserved automatically via ``extra="allow"``
    (accessible as attributes and included in ``model_dump()``).
    """

    model_config = ConfigDict(extra="allow")

    filename: str
    body: str
    document_id: str | None = None
    title: str | None = None
    status: str | None = None
    effective_date: str | date | None = None
    last_reviewed: str | date | None = None
    audience: str | None = None
    policy_authority: str | None = None


class Chunk(BaseModel):
    """A single section of a document plus all parent front-matter metadata."""

    chunk_id: str
    filename: str
    heading: str
    body: str
    metadata: dict[str, Any]


_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def load_documents(kb_dir: str | Path) -> list[Document]:
    """Load every top-level ``.md`` file in *kb_dir* into Documents."""
    kb_path = Path(kb_dir)
    documents: list[Document] = []
    for path in sorted(kb_path.glob("*.md")):
        post = frontmatter.load(path)
        documents.append(
            Document(filename=path.name, body=post.content or "", **post.metadata)
        )
    return documents


def _split_body(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into ``(preamble, [(heading, section_body), ...])``.

    Only ``##`` level headings create sections; ``#`` lines are left alone.
    """
    lines = body.splitlines()
    header_idx = [i for i, line in enumerate(lines) if _H2_RE.match(line)]
    if not header_idx:
        return body.strip(), []
    preamble = "\n".join(lines[: header_idx[0]])
    sections: list[tuple[str, str]] = []
    for pos, start in enumerate(header_idx):
        end = header_idx[pos + 1] if pos + 1 < len(header_idx) else len(lines)
        heading = _H2_RE.match(lines[start]).group(1).strip()
        section_body = "\n".join(lines[start + 1 : end]).strip()
        sections.append((heading, section_body))
    return preamble.strip(), sections


def _meaningful_intro(preamble: str) -> str:
    """Prose before the first ``##``, ignoring blank lines and ``#`` titles."""
    kept = [
        line.strip()
        for line in preamble.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n".join(kept)


def _slugify(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return slug or "section"


def chunk_document(doc: Document) -> list[Chunk]:
    """Split one document into chunks, one per ``##`` section.

    Falls back to a single "(full document)" chunk when no ``##`` headings
    exist, so a short product card can never be silently dropped.
    """
    metadata = doc.model_dump(exclude={"filename", "body"})
    preamble, sections = _split_body(doc.body)

    if not sections:
        return [
            Chunk(
                chunk_id=f"{doc.filename}#full-document",
                filename=doc.filename,
                heading="(full document)",
                body=doc.body.strip(),
                metadata=metadata,
            )
        ]

    chunks: list[Chunk] = []
    intro_text = _meaningful_intro(preamble)
    if intro_text:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.filename}#intro",
                filename=doc.filename,
                heading="(intro)",
                body=intro_text,
                metadata=metadata,
            )
        )

    used_slugs: dict[str, int] = {}
    for heading, section_body in sections:
        slug = _slugify(heading)
        seen = used_slugs.get(slug, 0)
        used_slugs[slug] = seen + 1
        if seen:
            slug = f"{slug}-{seen + 1}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc.filename}#{slug}",
                filename=doc.filename,
                heading=heading,
                body=section_body,
                metadata=metadata,
            )
        )
    return chunks


def load_and_chunk_all(kb_dir: str | Path) -> list[Chunk]:
    """Run ingestion across every file and return the flat chunk list."""
    chunks: list[Chunk] = []
    for doc in load_documents(kb_dir):
        chunks.extend(chunk_document(doc))
    return chunks


def _preview(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return f"{flat[:limit]}..." if len(flat) > limit else flat


def main() -> None:
    documents = load_documents(_KB_DIR)
    print(f"Loaded {len(documents)} documents from {_KB_DIR}")

    print("\n--- Intro/preamble audit ---")
    for doc in documents:
        preamble, sections = _split_body(doc.body)
        if not sections:
            print(f"[NO ## HEADINGS: {doc.filename}] body kept as one full-document chunk")
            continue
        intro_text = _meaningful_intro(preamble)
        if intro_text:
            print(f"[INTRO KEPT: {doc.filename}] {_preview(intro_text)}")
        elif preamble:
            print(f"[INTRO DROPPED: {doc.filename}] {_preview(preamble)}")

    chunks = load_and_chunk_all(_KB_DIR)
    print(f"\nTotal chunks produced: {len(chunks)}\n")
    for chunk in chunks:
        status = chunk.metadata.get("status", "?")
        authority = chunk.metadata.get("policy_authority", "?")
        audience = chunk.metadata.get("audience", "?")
        print(f"{chunk.chunk_id} | {status} | {authority} | {audience}")
        print(f"    {_preview(chunk.body)}")


if __name__ == "__main__":
    main()
