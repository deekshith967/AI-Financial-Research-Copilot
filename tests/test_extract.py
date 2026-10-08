"""Tests for text extraction and normalization."""
from pathlib import Path

import pytest

from MarketInsight.rag.extract import (
    ExtractionError,
    extract_file,
    extract_text,
    normalize_text,
)


class TestNormalizeText:
    def test_crlf_and_tabs_normalized(self):
        assert normalize_text("a\r\nb\t\tc\r\nd") == "a\nb c\nd"

    def test_collapses_blank_runs_and_strips(self):
        assert normalize_text("\n\nalpha\n\n\n\nbeta\n\n") == "alpha\n\nbeta"

    def test_collapses_internal_spaces(self):
        assert normalize_text("revenue    grew   12%") == "revenue grew 12%"


class TestTxtExtraction:
    def test_plain_text_with_filename_fallback_title(self, tmp_path: Path):
        path = tmp_path / "market_outlook_q3.txt"
        path.write_text("Nifty closed higher.\n\nBanking stocks led.\n", encoding="utf-8")
        doc = extract_file(path)
        assert doc.text == "Nifty closed higher.\n\nBanking stocks led."
        assert doc.title == "market outlook q3"
        assert doc.headings == ()

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ExtractionError, match="no usable text"):
            extract_file(path)

    def test_whitespace_only_raises(self, tmp_path: Path):
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n \t \n  ", encoding="utf-8")
        with pytest.raises(ExtractionError, match="no usable text"):
            extract_file(path)

    def test_binary_file_rejected(self, tmp_path: Path):
        path = tmp_path / "fake.txt"
        path.write_bytes(b"\x89PNG\x00\x1a\x00binarystuff")
        with pytest.raises(ExtractionError, match="binary"):
            extract_file(path)

    def test_unsupported_extension_rejected(self, tmp_path: Path):
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(ExtractionError, match="Unsupported file type"):
            extract_file(path)


class TestMarkdownExtraction:
    def test_title_from_first_h1_and_headings_recorded(self):
        raw = "# Acme 10-K Filing\n\nIntro paragraph.\n\n## Risk Factors\n\nText.\n\n### Sub Risk\n\nMore.\n"
        doc = extract_text(raw, kind="md", fallback_title="fallback")
        assert doc.title == "Acme 10-K Filing"
        assert [(h.level, h.text) for h in doc.headings] == [
            (1, "Acme 10-K Filing"),
            (2, "Risk Factors"),
            (3, "Sub Risk"),
        ]

    def test_heading_offsets_point_into_normalized_text(self):
        raw = "# Title\n\nBody text here.\n\n## Section Two\n\nMore text.\n"
        doc = extract_text(raw, kind="md", fallback_title="x")
        # Each heading's stored offset must not exceed the actual location of
        # its text, and offsets must be in document order.
        locations = [doc.text.find(h.text) for h in doc.headings]
        assert all(loc >= 0 for loc in locations)
        assert [h.offset for h in doc.headings] == sorted(h.offset for h in doc.headings)
        for heading, location in zip(doc.headings, locations):
            assert heading.offset <= location

    def test_no_headings_falls_back_to_filename_title(self):
        doc = extract_text("Just a paragraph.", kind="md", fallback_title="My File")
        assert doc.title == "My File"
        assert doc.headings == ()


class TestHtmlExtraction:
    def test_paragraphs_title_and_headings(self):
        raw = """<html><head><title>Zephyr 10-K</title>
        <style>body { color: red }</style></head>
        <body><h1>Zephyr Systems Annual Report</h1>
        <p>Revenue grew <b>12%</b>.</p>
        <script>alert('x')</script>
        <h2>Risk Factors</h2><p>Helium supply risk.</p></body></html>"""
        doc = extract_text(raw, kind="html", fallback_title="x")
        assert doc.title == "Zephyr 10-K"
        assert "Revenue grew 12%." in doc.text
        assert "alert" not in doc.text
        assert "color: red" not in doc.text
        assert [(h.level, h.text) for h in doc.headings] == [
            (1, "Zephyr Systems Annual Report"),
            (2, "Risk Factors"),
        ]

    def test_entities_and_block_structure(self):
        raw = "<p>Tom &amp; Jerry</p><p>Line<br>break</p><ul><li>one</li><li>two</li></ul>"
        doc = extract_text(raw, kind="html", fallback_title="x")
        assert "Tom & Jerry" in doc.text
        assert "one" in doc.text and "two" in doc.text
        # block elements become separate paragraphs
        assert doc.text.count("\n\n") >= 2

    def test_malformed_html_still_extracts(self):
        raw = "<h1>Unclosed heading<p>Paragraph without close<div>nested"
        doc = extract_text(raw, kind="html", fallback_title="x")
        assert "Unclosed heading" in doc.text
        assert "Paragraph without close" in doc.text

    def test_heading_offsets_track_normalized_text(self):
        raw = "<h1>Alpha</h1><p>body one</p><h2>Beta Section</h2><p>body two</p>"
        doc = extract_text(raw, kind="html", fallback_title="x")
        starts = [doc.text.find(h.text) for h in doc.headings]
        assert all(start >= 0 for start in starts)
        for heading, start in zip(doc.headings, starts):
            assert heading.offset <= start

    def test_html_with_no_text_raises(self):
        with pytest.raises(ExtractionError, match="no usable text"):
            extract_text("<html><head><style>x{}</style></head><body><script>y()</script></body></html>",
                         kind="html", fallback_title="x")
