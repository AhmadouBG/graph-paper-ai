from __future__ import annotations

from pathlib import Path

from src.ingestion.graph import _parse_sections, _section_node_id, build_adjacency_map
from src.ingestion.utils_class import CoLocatedEdge, CrossReference, ImageInfo, ProcessingResult


def _sample_result() -> ProcessingResult:
    md = (
        "# Introduction\n\nThis is the intro."
        "\n\n## Method\n\nWe propose a new method."
        "\n\n## Results\n\nSee Figure 1."
    )
    return ProcessingResult(
        markdown=md,
        images=[
            ImageInfo(
                node_id="fig_1", path=Path("images/fig_1.png"),
                page=1, bbox=(0, 0, 100, 100),
            ),
        ],
        edges=[
            CoLocatedEdge(
                source_id="fig_1", target_id="text_1_1",
                edge_type="co-located", distance=10.0, page=1,
            ),
        ],
    )


def test_build_adjacency_map_returns_digraph():
    result = _sample_result()
    graph = build_adjacency_map(result)
    assert graph.is_directed()


def test_node_count_matches_content_blocks():
    result = _sample_result()
    graph = build_adjacency_map(result)
    assert graph.number_of_nodes() >= 3


def test_typed_nodes():
    result = _sample_result()
    graph = build_adjacency_map(result)
    types = {graph.nodes[n]["node_type"] for n in graph.nodes}
    assert "section" in types
    assert "figure" in types


def test_contains_edges_from_sections():
    result = _sample_result()
    graph = build_adjacency_map(result)
    edge_types = {e for _, _, e in graph.edges(data="edge_type")}
    assert "contains" in edge_types


def test_cross_references_add_references_edges():
    result = _sample_result()
    refs = [
        CrossReference(
            target_node_id="fig_1", reference_type="figure",
            context="See Figure 1 for results", page=1,
        ),
    ]
    graph = build_adjacency_map(result, refs=refs)
    edge_types = {e for _, _, e in graph.edges(data="edge_type")}
    assert "references" in edge_types


def test_co_located_edges_included():
    result = _sample_result()
    graph = build_adjacency_map(result)
    edge_types = {e for _, _, e in graph.edges(data="edge_type")}
    assert "co-located" in edge_types


def test_empty_result():
    result = ProcessingResult(markdown="", images=[], edges=[])
    graph = build_adjacency_map(result)
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0


def test_section_hierarchy_nesting():
    md = (
        "# Title\n\nIntro\n\n## Section 1\n\nContent"
        "\n\n### Subsection\n\nDetail\n\n## Section 2\n\nMore"
    )
    result = ProcessingResult(
        markdown=md,
        images=[],
        edges=[],
    )
    graph = build_adjacency_map(result)
    section_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "section"]
    assert len(section_nodes) == 4


def test_node_attributes():
    result = _sample_result()
    graph = build_adjacency_map(result)
    for n in graph.nodes:
        attrs = graph.nodes[n]
        assert "node_type" in attrs
        assert "content" in attrs


def test_parse_sections():
    sections = _parse_sections("# A\n\n## B\n\n### C\n\nText\n\n## D")
    assert len(sections) == 4
    assert sections[0]["level"] == 1
    assert sections[0]["title"] == "A"
    assert sections[2]["level"] == 3
    assert sections[2]["title"] == "C"


def test_section_node_id():
    used: set[str] = set()
    assert _section_node_id("Introduction", 1, used) == "section_introduction"
    assert _section_node_id("## Method", 2, used) == "section_method"


def test_section_node_id_dedup():
    used: set[str] = set()
    a = _section_node_id("Introduction", 1, used)
    used.add(a)
    b = _section_node_id("Introduction", 1, used)
    assert b == "section_introduction_1"


def test_formula_nodes_added():
    markdown = "# Section A\n\nEquation: $$E = mc^2$$\n\nInline $a^2 + b^2 = c^2$"
    from src.ingestion.graph import _find_formulas
    formulas = _find_formulas(markdown)
    assert len(formulas) > 0
    assert any("E = mc^2" in f["content"] for f in formulas)
    assert any("a^2 + b^2 = c^2" in f["content"] for f in formulas)
    assert all(f["node_id"].startswith("formula_") for f in formulas)
