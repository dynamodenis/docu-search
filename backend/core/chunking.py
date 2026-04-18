"""Markdown chunker.

Layer 1 — split on headings (# / ## / ### ...). Each section becomes a
          chunk, with the heading chain prepended for context.
Layer 2 — if a section is too long, split on sentences with a small
          overlap so we do not lose context across boundaries.

The original project also had a semantic Layer 3 (MiniLM-driven topic
splitting). We intentionally skip that here to keep ingestion fast and
dependency-light. Bring it back if evaluation shows it matters for
user-submitted content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

MAX_WORDS_PER_CHUNK = 220
MIN_WORDS_PER_CHUNK = 40
SENTENCE_OVERLAP = 2


@dataclass
class Chunk:
    text: str
    page_title: str
    section_title: str
    section_url: str
    chunk_index: int


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-")


def _word_count(s: str) -> int:
    return len(s.split())


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_long_section(text: str) -> List[str]:
    """Sentence-window split with overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    out: List[str] = []
    buf: List[str] = []
    buf_words = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        w = _word_count(s)
        if buf and buf_words + w > MAX_WORDS_PER_CHUNK:
            out.append(" ".join(buf).strip())
            # start next window with overlap
            buf = buf[-SENTENCE_OVERLAP:] if len(buf) >= SENTENCE_OVERLAP else buf[:]
            buf_words = sum(_word_count(x) for x in buf)
        buf.append(s)
        buf_words += w
        i += 1

    if buf:
        out.append(" ".join(buf).strip())
    return out


def chunk_markdown(
    markdown: str,
    page_title: str,
    page_url: str,
) -> List[Chunk]:
    """Split a markdown document into Chunks ready for embedding."""
    lines = markdown.split("\n")
    sections: List[tuple[str, str]] = []  # (section_title, section_body)
    current_title = page_title
    current_body: List[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # flush previous section
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_title, body))
            current_title = m.group(2).strip() or page_title
            current_body = []
        else:
            current_body.append(line)

    body = "\n".join(current_body).strip()
    if body:
        sections.append((current_title, body))

    chunks: List[Chunk] = []
    idx = 0
    for section_title, section_body in sections:
        pieces = (
            [section_body]
            if _word_count(section_body) <= MAX_WORDS_PER_CHUNK
            else _split_long_section(section_body)
        )
        for piece in pieces:
            if _word_count(piece) < MIN_WORDS_PER_CHUNK and chunks:
                # merge tiny tail into the previous chunk
                chunks[-1].text = f"{chunks[-1].text}\n\n{piece}".strip()
                continue
            section_url = f"{page_url}#{_slugify(section_title)}" if section_title != page_title else page_url
            chunks.append(
                Chunk(
                    text=f"{page_title} — {section_title}\n\n{piece}".strip(),
                    page_title=page_title,
                    section_title=section_title,
                    section_url=section_url,
                    chunk_index=idx,
                )
            )
            idx += 1

    return chunks
