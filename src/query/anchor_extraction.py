from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class AnchorType(str, Enum):
    FIGURE = "figure"
    TABLE = "table"
    SECTION = "section"
    EQUATION = "equation"


class QueryType(str, Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass
class Anchor:
    node_id: str
    anchor_type: AnchorType
    label: str
    requested_children: bool = False


@dataclass
class QueryParseResult:
    anchors: List[Anchor] = field(default_factory=list)
    has_anchors: bool = False
    query_type: QueryType = QueryType.SEMANTIC


PATTERNS = {
    AnchorType.FIGURE: [
        re.compile(
            r"\b(?:Fig(?:ure)?s?\.?\s*(\d+(?:\.\d+)*(?:[a-zA-Z])?))\b",
            re.IGNORECASE,
        ),
    ],
    AnchorType.TABLE: [
        re.compile(
            r"\b(?:Tables?\.?\s*(\d+(?:\.\d+)*(?:[a-zA-Z])?))\b",
            re.IGNORECASE,
        ),
    ],
    AnchorType.SECTION: [
        re.compile(
            r"\b(?:Section|Sec(?:t)?\.?)\s+(\d+(?:\.\d+)*)\b",
            re.IGNORECASE,
        ),
    ],
    AnchorType.EQUATION: [
        re.compile(
            r"\b(?:Equation(?:s)?\.?\s*(\d+(?:\.\d+)*(?:[a-zA-Z])?))\b",
            re.IGNORECASE,
        ),
    ],
}

NODE_ID_PREFIX = {
    AnchorType.FIGURE: "fig",
    AnchorType.TABLE: "table",
    AnchorType.SECTION: "section",
    AnchorType.EQUATION: "equation",
}


def _to_node_id(anchor_type: AnchorType, label: str) -> str:
    prefix = NODE_ID_PREFIX[anchor_type]
    safe_label = label.replace(".", "_")
    return f"{prefix}_{safe_label}"


def extract_query_anchors(query: str) -> QueryParseResult:
    if not query:
        return QueryParseResult()

    anchors: List[Anchor] = []
    seen: set[str] = set()

    for anchor_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(query):
                label = match.group(1)
                node_id = _to_node_id(anchor_type, label)
                if node_id not in seen:
                    seen.add(node_id)
                    requested_children = anchor_type == AnchorType.SECTION
                    anchors.append(
                        Anchor(
                            node_id=node_id,
                            anchor_type=anchor_type,
                            label=label,
                            requested_children=requested_children,
                        )
                    )

    has_anchors = len(anchors) > 0
    query_type = QueryType.STRUCTURAL if has_anchors else QueryType.SEMANTIC
    return QueryParseResult(
        anchors=anchors,
        has_anchors=has_anchors,
        query_type=query_type,
    )
