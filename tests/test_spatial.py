from __future__ import annotations

import tempfile
from pathlib import Path

import fitz

from src.ingestion.spatial import detect_co_located_blocks


def _make_page_pdf(path: Path, blocks: list[dict]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for b in blocks:
        rect = fitz.Rect(b["x0"], b["y0"], b["x1"], b["y1"])
        if b["type"] == "image":
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, int(rect.width), int(rect.height)))
            page.insert_image(rect, pixmap=pix)
            del pix
        else:
            page.insert_textbox(rect, b.get("text", "text"), fontsize=10)
    doc.save(str(path))
    doc.close()


def test_nearby_text_and_figure_creates_edge():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_page_pdf(pdf_path, [
            {"type": "image", "x0": 50, "y0": 50, "x1": 200, "y1": 200},
            {"type": "text", "x0": 50, "y0": 220, "x1": 400, "y1": 250, "text": "nearby paragraph"},
        ])
        doc = fitz.open(pdf_path)
        result = detect_co_located_blocks(doc, threshold=50)
        doc.close()
    assert len(result) >= 1
    edge = result[0]
    assert edge.distance <= 50


def test_distant_text_and_figure_no_edge():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_page_pdf(pdf_path, [
            {"type": "image", "x0": 50, "y0": 50, "x1": 200, "y1": 200},
            {"type": "text", "x0": 50, "y0": 500, "x1": 400, "y1": 530, "text": "far paragraph"},
        ])
        doc = fitz.open(pdf_path)
        result = detect_co_located_blocks(doc, threshold=50)
        doc.close()
    assert len(result) == 0


def test_configurable_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        blocks = [
            {"type": "image", "x0": 50, "y0": 50, "x1": 200, "y1": 200},
            {"type": "text", "x0": 50, "y0": 260, "x1": 400, "y1": 290, "text": "at 60px"},
        ]
        _make_page_pdf(pdf_path, blocks)
        doc = fitz.open(pdf_path)
        strict = detect_co_located_blocks(doc, threshold=30)
        relaxed = detect_co_located_blocks(doc, threshold=100)
        doc.close()
    assert len(strict) == 0
    assert len(relaxed) >= 1


def test_multiple_figures_on_page():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_page_pdf(pdf_path, [
            {"type": "image", "x0": 50, "y0": 50, "x1": 200, "y1": 200},
            {"type": "image", "x0": 300, "y0": 50, "x1": 450, "y1": 200},
            {"type": "text", "x0": 50, "y0": 220, "x1": 200, "y1": 250, "text": "near fig 1"},
        ])
        doc = fitz.open(pdf_path)
        result = detect_co_located_blocks(doc, threshold=50)
        doc.close()
    assert len(result) == 1


def test_empty_pdf_no_edges():
    doc = fitz.open()
    doc.new_page()
    result = detect_co_located_blocks(doc, threshold=50)
    assert result == []


def test_text_only_no_edges():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_page_pdf(pdf_path, [
            {"type": "text", "x0": 50, "y0": 50, "x1": 400, "y1": 100, "text": "only text"},
        ])
        doc = fitz.open(pdf_path)
        result = detect_co_located_blocks(doc, threshold=50)
        doc.close()
    assert result == []
