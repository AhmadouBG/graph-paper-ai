from __future__ import annotations

import networkx as nx

from src.retrieval import ContextResult, bfs_traverse


def _sample_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("section_1", node_type="section", content="Introduction.", page=1)
    g.add_node("fig_1", node_type="figure", content="figure1.png", page=2)
    g.add_node("text_1", node_type="text_block", content="Some text about the figure.", page=2)
    g.add_node("table_1", node_type="table", content="table_data.csv", page=3)
    g.add_node("section_2", node_type="section", content="Related work.", page=4)
    g.add_edge("section_1", "fig_1", edge_type="contains")
    g.add_edge("fig_1", "text_1", edge_type="captions")
    g.add_edge("section_1", "table_1", edge_type="contains")
    g.add_edge("section_1", "section_2", edge_type="contains")
    return g


class TestBfsTraverse:
    def test_returns_contextresult(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"])
        assert isinstance(result, ContextResult)

    def test_direct_neighbor_collection_depth_1(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=1)
        node_ids = {n.node_id for n in result.nodes}
        assert "section_1" in node_ids
        assert "fig_1" in node_ids
        assert "table_1" in node_ids
        assert "section_2" in node_ids
        assert "text_1" not in node_ids

    def test_depth_0_only_start_node(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=0)
        assert len(result.nodes) == 1
        assert result.nodes[0].node_id == "section_1"

    def test_depth_2_includes_two_hops(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=2)
        node_ids = {n.node_id for n in result.nodes}
        assert "text_1" in node_ids

    def test_node_content_and_type_included(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["fig_1"], depth=1)
        fig = next(n for n in result.nodes if n.node_id == "fig_1")
        assert fig.node_type == "figure"
        assert fig.content == "figure1.png"
        assert fig.page == 2

    def test_page_number_included(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=1)
        sec = next(n for n in result.nodes if n.node_id == "section_1")
        assert sec.page == 1

    def test_multiple_start_nodes(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["fig_1", "table_1"], depth=1)
        node_ids = {n.node_id for n in result.nodes}
        assert "fig_1" in node_ids
        assert "table_1" in node_ids

    def test_invalid_start_node_skipped(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["nonexistent"], depth=1)
        assert len(result.nodes) == 0

    def test_empty_graph(self):
        g = nx.DiGraph()
        result = bfs_traverse(g, ["a"], depth=1)
        assert len(result.nodes) == 0

    def test_deduplication(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1", "section_1"], depth=1)
        section_count = sum(1 for n in result.nodes if n.node_id == "section_1")
        assert section_count == 1

    def test_token_truncation(self):
        g = nx.DiGraph()
        g.add_node("a", node_type="text", content="word " * 100, page=1)
        g.add_node("b", node_type="text", content="word " * 200, page=1)
        g.add_node("c", node_type="text", content="word " * 300, page=1)
        g.add_edge("a", "b", edge_type="contains")
        g.add_edge("b", "c", edge_type="contains")
        result = bfs_traverse(g, ["a"], depth=2, max_tokens=200)
        assert result.truncated
        assert result.total_tokens <= 200

    def test_total_tokens_counted(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=1)
        assert result.total_tokens > 0

    def test_context_node_metadata(self):
        g = nx.DiGraph()
        g.add_node("fig_1", node_type="figure", content="img.png",
                    page=2, extra="test")
        result = bfs_traverse(g, ["fig_1"], depth=0)
        assert result.nodes[0].metadata.get("extra") == "test"

    def test_not_truncated_when_under_limit(self):
        g = _sample_graph()
        result = bfs_traverse(g, ["section_1"], depth=2)
        assert not result.truncated
