from __future__ import annotations

from unittest.mock import MagicMock

import networkx as nx

from src.retrieval.tree_search import (
    TreeSearchResult,
    _fetch_section_content,
    _parse_selected_ids,
    answer_query,
    tree_search,
)


def _sample_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(
        "section_introduction", node_type="section", title="Introduction",
        content="Intro text.", page=1, _section_idx=0,
    )
    g.add_node(
        "section_methods", node_type="section", title="Methods",
        content="Method details.", page=2, _section_idx=1,
    )
    g.add_edge(
        "section_introduction", "section_methods", edge_type="contains",
    )
    return g


class TestParseSelectedIds:
    def test_extracts_section_ids(self):
        result = _parse_selected_ids("section_introduction,section_methods")
        assert result == ["section_introduction", "section_methods"]

    def test_handles_varying_whitespace(self):
        result = _parse_selected_ids("section_introduction, section_methods")
        assert result == ["section_introduction", "section_methods"]

    def test_deduplicates(self):
        result = _parse_selected_ids(
            "section_introduction,section_introduction,section_methods",
        )
        assert result == ["section_introduction", "section_methods"]

    def test_respects_max_ids(self):
        result = _parse_selected_ids(
            "section_a,section_b,section_c", max_ids=2,
        )
        assert result == ["section_a", "section_b"]

    def test_no_section_ids_returns_empty(self):
        result = _parse_selected_ids("no relevant sections here")
        assert result == []

    def test_empty_string(self):
        result = _parse_selected_ids("")
        assert result == []

    def test_filters_non_section_ids(self):
        result = _parse_selected_ids("fig_1,section_introduction,table_2")
        assert result == ["section_introduction"]

    def test_extracts_from_prose_response(self):
        result = _parse_selected_ids(
            "The relevant sections are section_introduction and section_methods.",
        )
        assert result == ["section_introduction", "section_methods"]

    def test_handles_newlines(self):
        result = _parse_selected_ids("section_a\nsection_b")
        assert result == ["section_a", "section_b"]


class TestFetchSectionContent:
    def test_returns_matching_sections(self):
        g = _sample_graph()
        result = _fetch_section_content(
            g, ["section_introduction", "section_methods"],
        )
        assert "Intro text" in result
        assert "Method details" in result

    def test_unknown_ids_skipped(self):
        g = _sample_graph()
        result = _fetch_section_content(
            g, ["section_introduction", "nonexistent"],
        )
        assert "Intro text" in result
        assert "nonexistent" not in result

    def test_empty_ids_returns_empty(self):
        g = _sample_graph()
        assert _fetch_section_content(g, []) == ""

    def test_formats_section_with_id_and_content(self):
        g = _sample_graph()
        result = _fetch_section_content(g, ["section_introduction"])
        assert "[section_introduction] Introduction" in result
        assert "Intro text" in result


class TestTreeSearch:
    def test_returns_treesearchresult(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "section_introduction"
        result = tree_search("test query", g, mock_client)
        assert isinstance(result, TreeSearchResult)

    def test_uses_provided_tree(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "section_introduction"
        custom_tree = "custom tree representation"
        tree_search("q", g, mock_client, tree=custom_tree)
        assert custom_tree in mock_client.chat.call_args[0][0][0].content

    def test_chat_called_with_search_prompt(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "section_introduction"
        tree_search("hello", g, mock_client)
        sent = mock_client.chat.call_args[0][0][0].content
        assert "hello" in sent
        assert "Introduction" in sent

    def test_includes_fetched_content_in_result(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "section_introduction"
        result = tree_search("q", g, mock_client)
        assert "Intro text" in result.context

    def test_empty_llm_response_returns_empty_context(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = ""
        result = tree_search("q", g, mock_client)
        assert result.context == ""
        assert result.selected_ids == []


class TestAnswerQuery:
    def test_uses_answer_prompt(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "42"
        result = answer_query("what is x?", "x = 42", mock_client)
        assert result == "42"

    def test_sends_context_and_query(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "answer"
        answer_query("q", "ctx", mock_client)
        sent = mock_client.chat.call_args[0][0][0].content
        assert "q" in sent
        assert "ctx" in sent

    def test_empty_context(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "no context"
        result = answer_query("q", "", mock_client)
        assert result == "no context"
