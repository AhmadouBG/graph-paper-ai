from __future__ import annotations

import networkx as nx

from src.ingestion.tree import (
    SectionNode,
    _find_section_page,
    _parse_sections,
    build_node_id_map,
    build_page_index,
    build_section_tree,
    print_tree,
)


class TestBuildPageIndex:
    def test_sections_with_page_markers(self):
        md = (
            "## Page 1\n\n"
            "# Introduction\n\n"
            "Some text.\n\n"
            "## Page 2\n\n"
            "# Methods\n\n"
            "Method details.\n\n"
            "## Page 3\n\n"
            "## Results\n\n"
            "Results here."
        )
        result = build_page_index(md).split("\n")
        assert len(result) == 3
        assert "[0000] Introduction (p.1)" in result[0]
        assert "[0001] Methods (p.2)" in result[1]
        assert "[0002] Results (p.3)" in result[2]

    def test_sequential_ids_are_contiguous(self):
        md = (
            "## Page 1\n\n"
            "# A\n\n" * 5
        )
        result = build_page_index(md).split("\n")
        for i, line in enumerate(result):
            assert line.startswith(f"[{i:04d}]")

    def test_no_heading_falls_back_to_document(self):
        md = "## Page 1\n\nJust a paragraph without any headings."
        result = build_page_index(md)
        assert "[0000] Document (p.1)" in result

    def test_page_markers_filtered_out(self):
        md = (
            "## Page 1\n\n"
            "# Introduction\n\n"
            "Content.\n\n"
            "## Page 2\n\n"
        )
        result = build_page_index(md)
        assert "Page 1" not in result
        assert "Page 2" not in result

    def test_empty_markdown(self):
        md = ""
        result = build_page_index(md)
        assert "[0000] Document (p.1)" in result

    def test_no_page_markers_defaults_page_1(self):
        md = "# Title\n\nContent without page markers."
        result = build_page_index(md)
        assert "[0000] Title (p.1)" in result


class TestBuildNodeIdMap:
    def test_maps_section_idx_to_graph_node_id(self):
        g = nx.DiGraph()
        g.add_node(
            "section_intro", node_type="section", title="Intro",
            content="Hi", page=1, _section_idx=0,
        )
        g.add_node(
            "section_methods", node_type="section", title="Methods",
            content="Details", page=2, _section_idx=1,
        )
        mapping = build_node_id_map(g)
        assert mapping["0000"] == "section_intro"
        assert mapping["0001"] == "section_methods"

    def test_non_section_nodes_excluded(self):
        g = nx.DiGraph()
        g.add_node("section_a", node_type="section", title="A", content="", page=1, _section_idx=0)
        g.add_node("fig_1", node_type="figure", content="img.png", page=1)
        mapping = build_node_id_map(g)
        assert "0000" in mapping
        assert len(mapping) == 1


class TestBuildSectionTree:
    def test_assigns_sequential_ids(self):
        g = nx.DiGraph()
        g.add_node(
            "section_intro", node_type="section", title="Introduction",
            content="Hi", page=1, _section_idx=0,
        )
        g.add_node(
            "section_methods", node_type="section", title="Methods",
            content="Details", page=2, _section_idx=1,
        )
        g.add_edge("section_intro", "section_methods", edge_type="contains")

        trees = build_section_tree(g)
        assert trees[0].node_id == "0000"
        assert trees[0].nodes[0].node_id == "0001"

    def test_populates_text_from_content(self):
        g = nx.DiGraph()
        g.add_node(
            "section_a", node_type="section", title="A",
            content="Section body text.", page=1, _section_idx=0,
        )
        trees = build_section_tree(g)
        assert trees[0].text == "Section body text."

    def test_multiple_roots(self):
        g = nx.DiGraph()
        g.add_node("section_a", node_type="section", title="A", content="", page=1, _section_idx=0)
        g.add_node("section_b", node_type="section", title="B", content="", page=2, _section_idx=1)

        trees = build_section_tree(g)
        assert len(trees) == 2
        assert trees[0].node_id == "0000"
        assert trees[1].node_id == "0001"

    def test_page_marker_sections_filtered_out(self):
        g = nx.DiGraph()
        g.add_node(
            "sec_page_1", node_type="section", title="Page 1",
            content="# Intro\n", page=1, _section_idx=0,
        )
        g.add_node(
            "section_intro", node_type="section", title="Introduction",
            content="Hi", page=1, _section_idx=1,
        )
        g.add_edge("sec_page_1", "section_intro", edge_type="contains")

        trees = build_section_tree(g)
        assert len(trees) == 1
        assert trees[0].title == "Introduction"

    def test_non_section_nodes_excluded(self):
        g = nx.DiGraph()
        g.add_node("section_a", node_type="section", title="A", content="", page=1, _section_idx=0)
        g.add_node("fig_1", node_type="figure", content="img.png", page=1)
        g.add_node("text_1", node_type="text_block", content="caption", page=1)
        g.add_edge("section_a", "fig_1", edge_type="contains")
        g.add_edge("fig_1", "text_1", edge_type="captions")

        trees = build_section_tree(g)
        assert len(trees) == 1
        assert len(trees[0].nodes) == 0

    def test_nested_hierarchy(self):
        g = nx.DiGraph()
        g.add_node("section_a", node_type="section", title="A", content="", page=1, _section_idx=0)
        g.add_node("section_b", node_type="section", title="B", content="", page=1, _section_idx=1)
        g.add_node("section_c", node_type="section", title="C", content="", page=2, _section_idx=2)
        g.add_edge("section_a", "section_b", edge_type="contains")
        g.add_edge("section_b", "section_c", edge_type="contains")

        trees = build_section_tree(g)
        assert trees[0].nodes[0].nodes[0].node_id == "0002"

    def test_empty_graph(self):
        g = nx.DiGraph()
        trees = build_section_tree(g)
        assert trees == []

    def test_page_index_from_graph(self):
        g = nx.DiGraph()
        g.add_node("section_a", node_type="section", title="A", content="", page=5, _section_idx=0)
        trees = build_section_tree(g)
        assert trees[0].page_index == 5


class TestPrintTree:
    def test_single_node(self):
        nodes = [SectionNode(node_id="0000", title="A", page_index=3, text="")]
        result = print_tree(nodes)
        assert "└── A (page 3) [0000]" in result

    def test_multiple_roots(self):
        nodes = [
            SectionNode(node_id="0000", title="A", page_index=1, text=""),
            SectionNode(node_id="0001", title="B", page_index=2, text=""),
        ]
        result = print_tree(nodes)
        assert "├── A (page 1) [0000]" in result
        assert "└── B (page 2) [0001]" in result

    def test_shows_hierarchy_with_ids(self):
        nodes = [
            SectionNode(
                node_id="0000", title="A", page_index=1, text="",
                nodes=[
                    SectionNode(node_id="0001", title="B", page_index=2, text=""),
                    SectionNode(node_id="0002", title="C", page_index=3, text=""),
                ],
            ),
        ]
        result = print_tree(nodes)
        assert "    ├── B (page 2) [0001]" in result
        assert "    └── C (page 3) [0002]" in result

    def test_empty_list(self):
        assert print_tree([]) == ""

    def test_no_page_omits_suffix(self):
        nodes = [SectionNode(node_id="0000", title="A", page_index=None, text="")]
        result = print_tree(nodes)
        assert "(page" not in result
        assert "└── A [0000]" in result


class TestFindSectionPage:
    def test_finds_last_page_before_position(self):
        md = "## Page 3\n\nsome text\n\n## Page 5\n\n# Heading"
        assert _find_section_page(md, md.index("# Heading")) == 5

    def test_defaults_to_1_no_page_marker(self):
        md = "# Heading\n\ncontent"
        assert _find_section_page(md, 0) == 1

    def test_ignores_non_page_headings(self):
        md = "## Page 1\n\n# Intro\n\n## Background\n\n## Page 2\n\n# Methods"
        assert _find_section_page(md, md.index("# Methods")) == 2


class TestParseSections:
    def test_parses_sections(self):
        md = "# Title\n\nContent.\n\n## Sub\n\nSub content."
        sections = _parse_sections(md)
        assert len(sections) == 2
        assert sections[0]["title"] == "Title"
        assert sections[1]["title"] == "Sub"
