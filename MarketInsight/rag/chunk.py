"""Deterministic token-aware chunking.

Pure functions: no I/O, no randomness, same input always yields the same
chunks. Paragraphs are packed up to ``max_tokens``; oversized paragraphs are
split into sentences and, when a single sentence is still too long, hard-split
by words. Consecutive chunks share a trailing overlap. An undersized trailing
chunk is merged into its predecessor (never silently dropped, so no document
content is lost).

The token counter is injected, which keeps unit tests fast and offline; the
production default is tiktoken's cl100k_base encoding.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from MarketInsight.rag.extract import Heading

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    token_count: int
    section: Optional[str]
    start_offset: int
    end_offset: int


_TOKENIZER: Optional[TokenCounter] = None


def tiktoken_counter() -> TokenCounter:
    """cl100k_base token counter (fetches the small BPE table once, then cached)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER = lambda s: len(encoding.encode(s))
    return _TOKENIZER


_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n[^\n]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

# A unit is (text, char offset in the source text, token count).
_Unit = Tuple[str, int, int]


def _split_paragraph(paragraph: str, start: int, max_tokens: int, count: TokenCounter) -> List[_Unit]:
    if count(paragraph) <= max_tokens:
        return [(paragraph, start, count(paragraph))]

    units: List[_Unit] = []
    cursor = start
    for sentence in _SENTENCE_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        idx = paragraph.find(sentence, max(0, cursor - start))
        s_off = start + idx if idx >= 0 else cursor
        if count(sentence) <= max_tokens:
            units.append((sentence, s_off, count(sentence)))
        else:
            words = sentence.split(" ")
            buf: List[str] = []
            for word in words:
                trial = " ".join([*buf, word])
                if buf and count(trial) > max_tokens:
                    piece = " ".join(buf)
                    units.append((piece, s_off, count(piece)))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                piece = " ".join(buf)
                units.append((piece, s_off, count(piece)))
        cursor = s_off + len(sentence)
    return units


def chunk_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 77,
    min_tokens: int = 40,
    count_tokens: Optional[TokenCounter] = None,
    headings: Sequence[Heading] = (),
) -> List[ChunkDraft]:
    if max_tokens < 8:
        raise ValueError("max_tokens must be at least 8.")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < max_tokens.")
    count = count_tokens or tiktoken_counter()
    if not text or not text.strip():
        return []

    units: List[_Unit] = []
    for match in _PARAGRAPH_RE.finditer(text):
        units.extend(_split_paragraph(match.group(0), match.start(), max_tokens, count))
    if not units:
        return []

    # Greedy packing with a trailing overlap carried into the next chunk.
    packs: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    for i, (_, _, u_tokens) in enumerate(units):
        if cur and cur_tokens + u_tokens > max_tokens:
            packs.append(cur)
            tail: List[int] = []
            tail_tokens = 0
            for j in reversed(cur):
                if tail_tokens + units[j][2] > overlap_tokens:
                    break
                tail.insert(0, j)
                tail_tokens += units[j][2]
            cur, cur_tokens = tail, tail_tokens
        cur.append(i)
        cur_tokens += u_tokens
    if cur:
        packs.append(cur)

    # Merge an undersized trailing pack into its predecessor (dropping units
    # that are already part of the overlap); a fully duplicated tail is dropped.
    if len(packs) > 1:
        last_tokens = sum(units[i][2] for i in packs[-1])
        prev_tokens = sum(units[i][2] for i in packs[-2])
        if last_tokens < min_tokens and prev_tokens + last_tokens <= max_tokens + overlap_tokens:
            prev_set = set(packs[-2])
            merged = packs[-2] + [i for i in packs[-1] if i not in prev_set]
            packs[-2] = merged
            packs.pop()

    heading_offsets = [h.offset for h in headings]
    drafts: List[ChunkDraft] = []
    for pack in packs:
        first = units[pack[0]]
        last = units[pack[-1]]
        section: Optional[str] = None
        if headings:
            pos = bisect_right(heading_offsets, first[1]) - 1
            if pos >= 0:
                section = headings[pos].text
        drafts.append(
            ChunkDraft(
                text="\n\n".join(units[i][0] for i in pack),
                token_count=sum(units[i][2] for i in pack),
                section=section,
                start_offset=first[1],
                end_offset=last[1] + len(last[0]),
            )
        )
    return drafts
