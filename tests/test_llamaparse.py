from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import networkx as nx
import pytest

from src.exceptions import IngestionError
from src.ingestion.graph import build_adjacency_map
from src.ingestion.parser import (
    _build_complete_hierarchical_tree,
    _extract_images_from_llamaparse_tree,
    _llamaparse_tree_to_markdown,
    parse_paper,
)
from src.ingestion.utils_class import CrossReference, ProcessingResult


class MockBbox:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MockItem:
    def __init__(self, item_type, value="", md="", level=None, name=None, path=None, bbox=None):
        self.type = item_type
        self.value = value
        self.md = md
        if level is not None:
            self.level = level
        if name is not None:
            self.name = name
        if path is not None:
            self.path = path
        if bbox is not None:
            self.bbox = bbox


class MockPage:
    def __init__(self, items):
        self.items = items


class MockPagesList:
    def __init__(self, pages):
        self.pages = pages


class MockParsingResult:
    def __init__(self, pages):
        self.items = MockPagesList(pages)


def _make_minimal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "LlamaParse test PDF.")
    doc.save(str(path))
    doc.close()


def test_build_complete_hierarchical_tree():
    bbox = MockBbox(x0=10.0, y0=20.0, x1=110.0, y1=120.0)
    items = [
        MockItem("heading", value="Introduction", level=1),
        MockItem("text", value="This is an introduction paragraph."),
        MockItem("heading", value="Methodology", level=2),
        MockItem("text", value="This describes the method."),
        MockItem("image", name="fig_method", path="images/fig_method.png", bbox=[bbox]),
        MockItem("table", md="| col1 | col2 |\n|---|---|\n| a | b |", bbox=[bbox]),
    ]
    pages = [MockPage(items)]
    raw_result = MockParsingResult(pages)

    tree = _build_complete_hierarchical_tree(raw_result)
    assert len(tree) == 1
    intro_node = tree[0]
    assert intro_node["title"] == "Introduction"
    assert "introduction paragraph" in intro_node["summary"]

    assert len(intro_node["nodes"]) == 1
    method_node = intro_node["nodes"][0]
    assert method_node["title"] == "Methodology"
    assert "describes the method" in method_node["summary"]

    assert len(method_node["visuals"]) == 1
    assert method_node["visuals"][0]["image_id"] == "fig_method"

    assert len(method_node["tables"]) == 1
    assert "col1" in method_node["tables"][0]["markdown_content"]


def test_llamaparse_tree_to_markdown():
    tree = [
        {
            "node_id": "0000",
            "title": "Intro Section",
            "summary": "This is introductory text.",
            "visuals": [],
            "tables": [],
            "nodes": [
                {
                    "node_id": "0001",
                    "title": "Sub Section",
                    "summary": "This is subsection text.",
                    "visuals": [{"image_id": "fig_1", "path": "images/fig_1.png"}],
                    "tables": [{"markdown_content": "| val1 | val2 |"}],
                    "nodes": [],
                }
            ],
        }
    ]
    markdown = _llamaparse_tree_to_markdown(tree)
    assert "# Intro Section" in markdown
    assert "This is introductory text." in markdown
    assert "## Sub Section" in markdown
    assert "This is subsection text." in markdown
    assert "![fig_1](images/fig_1.png)" in markdown
    assert "| val1 | val2 |" in markdown


def test_extract_images_from_llamaparse_tree():
    bbox_dict = {"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 120.0}
    tree = [
        {
            "node_id": "0000",
            "title": "Intro Section",
            "summary": "text",
            "visuals": [
                {
                    "image_id": "fig_1",
                    "page": 1,
                    "path": "images/fig_1.png",
                    "bbox": [bbox_dict],
                }
            ],
            "tables": [],
            "nodes": [],
        }
    ]
    images_dir = Path("dummy_images")
    images = _extract_images_from_llamaparse_tree(tree, images_dir)
    assert len(images) == 1
    img = images[0]
    assert img.node_id == "fig_1"
    assert img.path == Path("images/fig_1.png")
    assert img.page == 1
    assert img.bbox == (10.0, 20.0, 110.0, 120.0)


@patch.dict(os.environ, {"LLAMACLOUD_API_KEY": "fake_key"})
@patch("llama_cloud.AsyncLlamaCloud")
def test_parse_paper_with_llamaparse(mock_async_client_cls):
    # Setup mocks
    mock_client = MagicMock()
    mock_async_client_cls.return_value = mock_client
    
    mock_file_obj = MagicMock()
    mock_file_obj.id = "file_123"
    mock_client.files.create = AsyncMock(return_value=mock_file_obj)

    bbox = MockBbox(x0=10.0, y0=20.0, x1=110.0, y1=120.0)
    items = [
        MockItem("heading", value="System Overview", level=1),
        MockItem("text", value="Overview text detailing equation $$E = mc^2$$."),
    ]
    pages = [MockPage(items)]
    mock_parsing_result = MockParsingResult(pages)
    mock_client.parsing.parse = AsyncMock(return_value=mock_parsing_result)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "test.pdf"
        _make_minimal_pdf(pdf_path)

        result = parse_paper(pdf_path, output_dir=tmp_dir, use_llamaparse=True)
        assert isinstance(result, ProcessingResult)
        assert result.metadata.get("parser") == "llamaparse"
        assert "# System Overview" in result.markdown
        assert "System Overview" in result.metadata["llamaparse_tree"][0]["title"]


def test_parse_paper_llamaparse_raises_when_no_api_key():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "test.pdf"
        _make_minimal_pdf(pdf_path)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(IngestionError, match="LLAMACLOUD_API_KEY environment variable is not set"):
                parse_paper(pdf_path, output_dir=tmp_dir, use_llamaparse=True)


def test_build_llamaparse_graph():
    tree = [
        {
            "node_id": "0000",
            "title": "Introduction",
            "page_index": 1,
            "summary": "Introduction text with an equation $$y = mx + c$$.",
            "visuals": [
                {
                    "image_id": "fig_intro",
                    "page": 1,
                    "path": "images/fig_intro.png",
                    "bbox": [{"x0": 0.0, "y0": 0.0, "x1": 50.0, "y1": 50.0}],
                }
            ],
            "tables": [
                {
                    "page": 1,
                    "markdown_content": "| a | b |",
                }
            ],
            "nodes": [],
        }
    ]

    result = ProcessingResult(
        markdown="# Introduction\n\nIntroduction text with an equation $$y = mx + c$$.\n\n| a | b |",
        images=[],
        edges=[],
        metadata={
            "parser": "llamaparse",
            "llamaparse_tree": tree,
        },
    )

    refs = [
        CrossReference(
            target_node_id="fig_intro",
            reference_type="figure",
            context="Introduction text with an equation",
            page=1,
        )
    ]

    graph = build_adjacency_map(result, refs=refs)

    # Verify nodes
    assert graph.has_node("tree_0000")
    assert graph.nodes["tree_0000"]["node_type"] == "section"
    assert graph.nodes["tree_0000"]["title"] == "Introduction"

    # Verify visuals
    assert graph.has_node("fig_intro")
    assert graph.nodes["fig_intro"]["node_type"] == "figure"
    assert graph.has_edge("tree_0000", "fig_intro")

    # Verify tables
    assert graph.has_node("tbl_0000_0")
    assert graph.nodes["tbl_0000_0"]["node_type"] == "text_block"
    assert graph.has_edge("tree_0000", "tbl_0000_0")

    # Verify formulas
    assert any(
        graph.nodes[n]["node_type"] == "formula" and "y = mx + c" in graph.nodes[n]["content"]
        for n in graph.nodes
    )

    # Verify references edge
    assert any(
        graph.edges[u, v].get("edge_type") == "references" and v == "fig_intro"
        for u, v in graph.edges
    )
