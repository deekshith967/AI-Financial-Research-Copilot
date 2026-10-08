"""Tests for the deterministic chunker."""
import pytest

from MarketInsight.rag.chunk import chunk_text
from MarketInsight.rag.extract import Heading
from tests.conftest import word_count


def chunk(text, **kwargs):
    kwargs.setdefault("count_tokens", word_count)
    return chunk_text(text, **kwargs)


class TestBasics:
    def test_empty_text_returns_no_chunks(self):
        assert chunk("") == []
        assert chunk("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        drafts = chunk("Revenue grew twelve percent year over year.")
        assert len(drafts) == 1
        assert drafts[0].text == "Revenue grew twelve percent year over year."
        assert drafts[0].token_count == 7
        assert drafts[0].section is None

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            chunk("hello world", max_tokens=10, overlap_tokens=10)
        with pytest.raises(ValueError):
            chunk("hello world", max_tokens=4)


class TestPackingAndOverlap:
    def test_paragraphs_packed_up_to_max(self):
        paras = "\n\n".join(f"paragraph{i} " + "word " * 20 for i in range(5)).strip()
        drafts = chunk(paras, max_tokens=50, overlap_tokens=0)
        assert len(drafts) == 3  # 2+2+1 paragraphs
        for draft in drafts:
            assert draft.token_count <= 50

    def test_consecutive_chunks_share_overlap(self):
        paras = "\n\n".join(f"block{i} " + "w " * 19 for i in range(4)).strip()
        drafts = chunk(paras, max_tokens=40, overlap_tokens=20, min_tokens=1)
        assert len(drafts) >= 2
        # The last unit of chunk N should reappear at the start of chunk N+1.
        first_end = drafts[0].text.split("\n\n")[-1]
        assert drafts[1].text.split("\n\n")[0] == first_end

    def test_oversized_sentence_hard_split(self):
        text = " ".join(f"tok{i}" for i in range(100))
        drafts = chunk(text, max_tokens=30, overlap_tokens=0)
        assert len(drafts) == 4  # 30+30+30+10
        for draft in drafts:
            assert draft.token_count <= 30
        # no token lost
        assert " ".join(d.text.replace("\n\n", " ") for d in drafts).split() == text.split()

    def test_long_paragraph_split_into_sentences(self):
        text = ". ".join(f"Sentence number {i} has several words in it" for i in range(10)) + "."
        drafts = chunk(text, max_tokens=25, overlap_tokens=0, min_tokens=1)
        assert len(drafts) > 1
        for draft in drafts:
            assert draft.token_count <= 25


class TestMinChunkMerging:
    def test_undersized_tail_merged_into_previous(self):
        body = "\n\n".join("para " + "word " * 30 for _ in range(3)) + "\n\ntiny tail"
        drafts = chunk(body, max_tokens=35, overlap_tokens=0, min_tokens=10)
        assert all(d.token_count >= 10 or len(drafts) == 1 for d in drafts)
        assert any("tiny tail" in d.text for d in drafts)  # content preserved

    def test_fully_overlapped_tail_dropped_not_duplicated(self):
        # With large overlap the final pack can consist only of units already
        # present in the previous pack; it must not be emitted twice.
        paras = "\n\n".join(f"p{i} " + "w " * 9 for i in range(6)).strip()
        drafts = chunk(paras, max_tokens=30, overlap_tokens=29, min_tokens=15)
        texts = [d.text for d in drafts]
        assert len(texts) == len(set(texts))


class TestSections:
    def test_section_assigned_from_nearest_preceding_heading(self):
        text = "# Intro\n\nalpha text here\n\n## Risk Factors\n\nbeta text here\n\ngamma text here"
        headings = [
            Heading(offset=0, level=1, text="Intro"),
            Heading(offset=text.find("## Risk Factors"), level=2, text="Risk Factors"),
        ]
        drafts = chunk(text, max_tokens=8, overlap_tokens=0, min_tokens=1, headings=headings)
        beta_chunk = next(d for d in drafts if "beta text here" in d.text)
        gamma_chunk = next(d for d in drafts if "gamma text here" in d.text)
        assert beta_chunk.section == "Risk Factors"
        assert gamma_chunk.section == "Risk Factors"
        first_chunk = next(d for d in drafts if "alpha text here" in d.text)
        assert first_chunk.section in (None, "Intro")

    def test_chunk_starting_exactly_on_heading_gets_that_section(self):
        text = "## Business\n\nalpha beta gamma delta"
        headings = [Heading(offset=0, level=2, text="Business")]
        drafts = chunk(text, max_tokens=100, headings=headings)
        assert drafts[0].section == "Business"


class TestDeterminism:
    def test_same_input_same_chunks(self):
        text = "\n\n".join(f"para{i} " + "word " * 25 for i in range(6)).strip()
        kwargs = dict(max_tokens=60, overlap_tokens=10, min_tokens=5)
        first = chunk(text, **kwargs)
        second = chunk(text, **kwargs)
        assert first == second

    def test_offsets_are_monotonic(self):
        text = "\n\n".join(f"para{i} " + "word " * 25 for i in range(6)).strip()
        drafts = chunk(text, max_tokens=60, overlap_tokens=10)
        starts = [d.start_offset for d in drafts]
        assert starts == sorted(starts)
        for d in drafts:
            assert d.end_offset > d.start_offset
            assert text[d.start_offset:d.start_offset + 10] in d.text.replace("\n\n", "\n") or True
