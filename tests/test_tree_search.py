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


def _sample_id_map() -> dict:
    return {"0000": "section_introduction", "0001": "section_methods"}


class TestParseSelectedIds:
    def test_extracts_four_digit_ids(self):
        result = _parse_selected_ids("0000,0001,0005")
        assert result == ["0000", "0001", "0005"]

    def test_handles_varying_whitespace(self):
        result = _parse_selected_ids("0000, 0001")
        assert result == ["0000", "0001"]

    def test_deduplicates(self):
        result = _parse_selected_ids("0000,0000,0001")
        assert result == ["0000", "0001"]

    def test_respects_max_ids(self):
        result = _parse_selected_ids("0000,0001,0002", max_ids=2)
        assert result == ["0000", "0001"]

    def test_no_ids_returns_empty(self):
        result = _parse_selected_ids("none found")
        assert result == []

    def test_empty_string(self):
        result = _parse_selected_ids("")
        assert result == []

    def test_filters_non_digit_ids(self):
        result = _parse_selected_ids("fig_1,section_intro,0000")
        assert result == ["0000"]

    def test_extracts_from_prose(self):
        result = _parse_selected_ids(
            "Relevant sections are 0000 and 0001.",
        )
        assert result == ["0000", "0001"]

    def test_handles_newlines(self):
        result = _parse_selected_ids("0000\n0001")
        assert result == ["0000", "0001"]


class TestFetchSectionContent:
    def test_returns_matching_sections(self):
        g = _sample_graph()
        id_map = _sample_id_map()
        result = _fetch_section_content(g, ["0000", "0001"], id_map)
        assert "Intro text" in result
        assert "Method details" in result

    def test_unknown_ids_skipped(self):
        g = _sample_graph()
        result = _fetch_section_content(g, ["0000", "9999"], _sample_id_map())
        assert "Intro text" in result
        assert "9999" not in result

    def test_empty_ids_returns_empty(self):
        g = _sample_graph()
        assert _fetch_section_content(g, [], _sample_id_map()) == ""

    def test_formats_section_with_id_and_content(self):
        g = _sample_graph()
        result = _fetch_section_content(g, ["0000"], _sample_id_map())
        assert "[0000] Introduction" in result
        assert "Intro text" in result

    def test_unknown_id_in_map_not_in_graph(self):
        g = _sample_graph()
        id_map = {"0000": "section_introduction", "0002": "nonexistent"}
        result = _fetch_section_content(g, ["0000", "0002"], id_map)
        assert "Intro text" in result
        assert "0002" not in result


class TestTreeSearch:
    def test_returns_treesearchresult(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "0000"
        result = tree_search("test query", g, mock_client)
        assert isinstance(result, TreeSearchResult)

    def test_uses_provided_tree(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "0000"
        custom_tree = "custom tree"
        tree_search("q", g, mock_client, tree=custom_tree)
        assert custom_tree in mock_client.chat.call_args[0][0][0].content

    def test_chat_called_with_search_prompt(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "0000"
        tree_search("hello", g, mock_client)
        sent = mock_client.chat.call_args[0][0][0].content
        assert "hello" in sent
        assert "Introduction" in sent

    def test_includes_fetched_content_in_result(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = "0000"
        result = tree_search("q", g, mock_client, node_id_map=_sample_id_map())
        assert "Intro text" in result.context

    def test_empty_llm_response_returns_empty_context(self):
        g = _sample_graph()
        mock_client = MagicMock()
        mock_client.chat.return_value = ""
        result = tree_search("q", g, mock_client, node_id_map=_sample_id_map())
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
