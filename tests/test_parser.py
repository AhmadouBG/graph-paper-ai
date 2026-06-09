from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import pytest

from src.exceptions import IngestionError
from src.ingestion.parser import ImageInfo, ProcessingResult, parse_paper


def _make_minimal_pdf(path: Path, with_image: bool = False) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(72, 72),
        "This is a test paper.\n\n"
        "As shown in Figure 1, the results demonstrate significant improvement.\n\n"
        "Table 1 summarizes the key metrics.",
    )
    if with_image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 100))
        page.insert_image(fitz.Rect(72, 200, 272, 300), pixmap=pix)
    doc.save(str(path))
    doc.close()


def test_parse_paper_returns_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_minimal_pdf(pdf_path)
        result = parse_paper(pdf_path, output_dir=tmp)
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.markdown, str)
        assert len(result.markdown) > 0
        assert "test paper" in result.markdown.lower()


def test_parse_paper_raises_on_missing_file():
    with pytest.raises(IngestionError, match="not found"):
        parse_paper(Path("nonexistent.pdf"))


def test_parse_paper_raises_on_non_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        txt_path = Path(tmp) / "test.txt"
        txt_path.write_text("not a pdf")
        with pytest.raises(IngestionError, match="not a PDF"):
            parse_paper(txt_path)


def test_parse_paper_extracts_images():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "cdl.pdf"
        _make_minimal_pdf(pdf_path, with_image=True)
        result = parse_paper(pdf_path, output_dir=tmp)
        assert len(result.images) >= 1
        img = result.images[0]
        assert isinstance(img, ImageInfo)
        assert img.node_id.startswith("fig_") or img.node_id.startswith("table_")
        assert img.path.exists()


def test_parse_paper_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        _make_minimal_pdf(pdf_path)
        result = parse_paper(pdf_path, output_dir=tmp)
        assert "source" in result.metadata
        assert "page_count" in result.metadata
        assert result.metadata["page_count"] == 1
        assert isinstance(result.edges, list)


def test_process_math_formulas():
    from src.ingestion.parser import _process_math_formulas

    # Test block equations
    input_text = "Standard equation: \\[ E = mc^2 \\]"
    expected_text = "Standard equation: $$ E = mc^2 $$"
    assert _process_math_formulas(input_text) == expected_text

    # Test inline equations
    input_inline = "This is an inline equation \\( x + y = z \\) in the text."
    expected_inline = "This is an inline equation $ x + y = z $ in the text."
    assert _process_math_formulas(input_inline) == expected_inline

    # Test mixed equations
    input_mixed = "Here is an inline: \\( a \\), and a block:\n\\[\nb = c\n\\]"
    expected_mixed = "Here is an inline: $ a $, and a block:\n$$\nb = c\n$$"
    assert _process_math_formulas(input_mixed) == expected_mixed

