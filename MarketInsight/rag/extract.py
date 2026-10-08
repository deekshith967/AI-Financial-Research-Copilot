"""Text extraction and normalization for local documents.

Stdlib only: ``.txt``/``.md`` are decoded directly, ``.html``/``.htm`` go
through a small ``html.parser``-based extractor that drops script/style
content, keeps block structure as paragraph breaks, and records heading
positions so the chunker can label each chunk with its section.

Heading offsets always refer to the *normalized* text returned in
``ExtractedDocument.text`` so downstream stages can map offsets directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}


class ExtractionError(Exception):
    """Raised when a file cannot be decoded or yields no usable text."""


@dataclass(frozen=True)
class Heading:
    offset: int  # char offset into the normalized text
    level: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    title: str
    headings: Tuple[Heading, ...]


# --------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------

def normalize_text(raw: str) -> str:
    """CRLF→LF, strip NULs, collapse spaces/tabs, collapse blank-line runs."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.split("\n")]
    out: List[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "template"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "thead", "tbody",
    "section", "article", "header", "footer", "blockquote", "pre", "hr",
    "dl", "dt", "dd", "figure", "figcaption",
}


class _HTMLTextExtractor(HTMLParser):
    """Emits ("text" | "heading", level, raw_text) segments in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.segments: List[Tuple[str, int, str]] = []
        self.title_text = ""
        self._buf: List[str] = []
        self._title_buf: List[str] = []
        self._skip = 0
        self._in_title = False
        self._heading_level: Optional[int] = None

    def _flush_text(self) -> None:
        if self._buf:
            self.segments.append(("text", 0, "".join(self._buf)))
            self._buf = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip or self._in_title:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADING_TAGS:
            self._flush_text()
            self._heading_level = int(tag[1])
        elif tag in _BLOCK_TAGS:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
            self.title_text = " ".join("".join(self._title_buf).split())
        elif self._in_title:
            return
        elif tag in _HEADING_TAGS and self._heading_level is not None:
            self.segments.append(("heading", self._heading_level, "".join(self._buf)))
            self._buf = []
            self._heading_level = None
        elif tag in _BLOCK_TAGS:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self._title_buf.append(data)
        else:
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        if self._heading_level is not None:
            self.segments.append(("heading", self._heading_level, "".join(self._buf)))
            self._buf = []
            self._heading_level = None
        self._flush_text()


def _join_segments(
    segments: Sequence[Tuple[str, int, str]],
    *,
    title_text: str,
    fallback_title: str,
) -> ExtractedDocument:
    """Join normalized blocks with blank lines, tracking heading offsets."""
    blocks: List[str] = []
    headings: List[Heading] = []
    offset = 0
    for kind, level, raw in segments:
        norm = " ".join(raw.split())
        if not norm:
            continue
        if blocks:
            offset += 2  # the "\n\n" separator
        if kind == "heading":
            headings.append(Heading(offset=offset, level=level, text=norm))
        blocks.append(norm)
        offset += len(norm)

    title = (
        title_text
        or (headings[0].text if headings and headings[0].level == 1 else "")
        or fallback_title
    )
    return ExtractedDocument(text="\n\n".join(blocks), title=title, headings=tuple(headings))


def _extract_html(raw: str, fallback_title: str) -> ExtractedDocument:
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    return _join_segments(parser.segments, title_text=parser.title_text, fallback_title=fallback_title)


# --------------------------------------------------------------------------------
# Markdown / plain text
# --------------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def _extract_markdown(text: str, fallback_title: str) -> ExtractedDocument:
    headings: List[Heading] = []
    title = ""
    offset = 0
    for line in text.split("\n"):
        match = _MD_HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            headings.append(Heading(offset=offset, level=level, text=match.group(2).strip()))
            if not title and level == 1:
                title = match.group(2).strip()
        offset += len(line) + 1
    return ExtractedDocument(text=text, title=title or fallback_title, headings=tuple(headings))


# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------

def extract_text(raw: str, *, kind: str, fallback_title: str) -> ExtractedDocument:
    """Extract from a raw string. ``kind`` is 'txt', 'md' or 'html'."""
    if kind == "html":
        doc = _extract_html(raw, fallback_title)
    else:
        normalized = normalize_text(raw)
        doc = _extract_markdown(normalized, fallback_title) if kind == "md" else ExtractedDocument(
            text=normalized, title=fallback_title, headings=()
        )
    if not doc.text:
        raise ExtractionError("Document contains no usable text.")
    return doc


def extract_file(path: Union[str, Path]) -> ExtractedDocument:
    """Extract from a local file, validating type and decodability."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes[:8192]:
        raise ExtractionError("File appears to be binary, not text.")
    raw = raw_bytes.decode("utf-8", errors="replace")

    fallback_title = " ".join(path.stem.replace("_", " ").replace("-", " ").split())
    kind = "html" if suffix in {".html", ".htm"} else ("md" if suffix in {".md", ".markdown"} else "txt")
    return extract_text(raw, kind=kind, fallback_title=fallback_title)
