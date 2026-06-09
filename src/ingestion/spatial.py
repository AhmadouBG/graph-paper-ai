from __future__ import annotations

from typing import List

import fitz

from src.ingestion.utils_class import CoLocatedEdge


def _bbox_distance(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> float:
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return dx + dy


def detect_co_located_blocks(
    doc: fitz.Document,
    threshold: float = 50.0,
) -> List[CoLocatedEdge]:
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    edges: List[CoLocatedEdge] = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])

        image_blocks = [(idx, b) for idx, b in enumerate(blocks) if b.get("type") == 1]
        text_blocks = [(idx, b) for idx, b in enumerate(blocks) if b.get("type") == 0]

        for img_idx, ib in image_blocks:
            ibox = ib.get("bbox")
            if ibox is None:
                continue
            for txt_idx, tb in text_blocks:
                tbox = tb.get("bbox")
                if tbox is None:
                    continue
                dist = _bbox_distance(*ibox, *tbox)
                if dist <= threshold:
                    source_id = f"fig_{page_num + 1}_{img_idx + 1}"
                    target_id = f"text_{page_num + 1}_{txt_idx + 1}"
                    edges.append(CoLocatedEdge(
                        source_id=source_id,
                        target_id=target_id,
                        distance=round(dist, 2),
                        page=page_num + 1,
                    ))

    return edges
