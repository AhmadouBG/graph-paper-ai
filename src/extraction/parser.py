from __future__ import annotations


import logging
from pathlib import Path
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

import hashlib
from llama_cloud import LlamaCloud

def _calculate_file_hash(file_path: str) -> str:
    """Calcule l'empreinte unique (SHA-256) du fichier temporaire."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _parse_with_llamacloud(pdf_path: str, api_key: str, project_id: str = None) -> list[dict]:
    # 1. Initialisation du client
    client = LlamaCloud(api_key=api_key)
    
    # 2. Récupération du hash unique du fichier temporaire
    file_hash = _calculate_file_hash(pdf_path)
    uploaded_file_id = None

    print(f"Recherche du fichier dans le projet LlamaCloud (Project ID: {project_id or 'Défaut'})...")
    
    # 3. Lister les fichiers uniquement au sein du projet ciblé
    # Note: On passe project_id dans les filtres si l'API du SDK le permet, 
    # ou on filtre manuellement les attributs de file_info
    files_list = client.files.list(project_id=project_id) if project_id else client.files.list()
    
    for file_info in files_list:
        # 1. Vérification primaire via le hash stocké dans external_file_id
        if getattr(file_info, "external_file_id", None) == file_hash:
            uploaded_file_id = file_info.id
            print(f"✨ Fichier identique trouvé dans le projet via hash (ID: {uploaded_file_id}). Cache activé !")
            break
        # 2. Vérification secondaire (fallback) via le nom exact du fichier original
        elif getattr(file_info, "name", None) == Path(pdf_path).name:
            uploaded_file_id = file_info.id
            print(f"✨ Fichier identique trouvé dans le projet via le nom (ID: {uploaded_file_id}). Cache activé !")
            break

    # 4. Si absent du projet, on le téléverse au bon endroit
    if not uploaded_file_id:
        print(f"Téléversement du fichier temporaire vers le projet...")
        uploaded_file = client.files.create(
            file=Path(pdf_path), 
            purpose="parse",
            project_id=project_id, # Associe explicitement le fichier au projet
            external_file_id=file_hash # Associe le hash unique du fichier
        )
        uploaded_file_id = uploaded_file.id

    # 5. Extraction via l'Agentic Tier (Bénéficie du cache si l'ID existait déjà)
    print("Démarrage du parsing agentique...")
    result = client.parsing.parse(
        file_id=uploaded_file_id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    # 6. Extraction des pages
    pages_data = []
    if hasattr(result, "markdown") and result.markdown and hasattr(result.markdown, "pages"):
        pages = result.markdown.pages
    else:
        result_dict = result.dict() if hasattr(result, "dict") else vars(result)
        pages = result_dict.get("markdown", {}).get("pages", [])

    for p in pages:
        p_status = getattr(p, "success", True) if not isinstance(p, dict) else p.get("success", True)
        if p_status:
            page_num = getattr(p, "page_number", None) if not isinstance(p, dict) else p.get("page_number")
            md_content = getattr(p, "markdown", "") if not isinstance(p, dict) else p.get("markdown", "")
            pages_data.append({"page": page_num, "md": md_content or ""})

    print(f"Parsing terminé. {len(pages_data)} pages extraites.")
    return pages_data

import re

# ── Granularity constants ──────────────────────────────────────────
MAX_TOKENS_PER_NODE = 300          # ~1200 chars; tune to your retriever
TABLE_CAPTION_RE   = re.compile(r'^(TABLE\s+[IVXLCDM\d]+\..*)', re.IGNORECASE)
FIGURE_CAPTION_RE  = re.compile(r'^(Fig\.?\s*\d+[\s\.\:].*)',   re.IGNORECASE)

def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)

def _split_long_content(text: str, max_tokens: int = MAX_TOKENS_PER_NODE) -> list[str]:
    """Split oversized text into paragraph-level sub-chunks."""
    if _count_tokens(text) <= max_tokens:
        return [text]
    paragraphs = text.split("\n\n")
    sub_chunks, current, current_tokens = [], [], 0
    for para in paragraphs:
        pt = _count_tokens(para)
        if current_tokens + pt > max_tokens and current:
            sub_chunks.append("\n\n".join(current))
            current, current_tokens = [para], pt
        else:
            current.append(para)
            current_tokens += pt
    if current:
        sub_chunks.append("\n\n".join(current))
    return sub_chunks


def _make_sub_nodes(parent_node: dict, node_counter: int) -> tuple[list[dict], int]:
    """
    Post-process a finalized node's content:
    Split it into atomic sub-nodes at TABLE / Figure / paragraph boundaries.
    Returns (list_of_sub_nodes, updated_counter).
    """
    content = parent_node.get("content", "")
    if not content:
        return [], node_counter

    # ── Step 1: Split content at TABLE and Figure caption boundaries ──
    raw_segments = []
    current_segment_lines = []

    for line in content.splitlines():
        stripped = line.strip()
        is_boundary = (
            TABLE_CAPTION_RE.match(stripped) or
            FIGURE_CAPTION_RE.match(stripped)
        )
        if is_boundary and current_segment_lines:
            raw_segments.append("\n".join(current_segment_lines).strip())
            current_segment_lines = [line]
        else:
            current_segment_lines.append(line)

    if current_segment_lines:
        raw_segments.append("\n".join(current_segment_lines).strip())

    # ── Step 2: Further split oversized segments by paragraph ──
    fine_segments = []
    for seg in raw_segments:
        fine_segments.extend(_split_long_content(seg, MAX_TOKENS_PER_NODE))

    # ── Step 3: Only create sub-nodes if we actually split ──
    if len(fine_segments) <= 1:
        return [], node_counter   # no split needed — keep parent as-is

    sub_nodes = []
    for seg in fine_segments:
        if not seg.strip():
            continue
        first_line = seg.strip().splitlines()[0].strip()
        title = first_line[:80] if first_line else f"sub_{node_counter:04d}"
        sub_nodes.append({
            "node_id": f"{node_counter:04d}",
            "title": title,
            "page_start": parent_node["page_start"],
            "page_end":   parent_node["page_end"],
            "content":    seg.strip(),
            "nodes":      [],
        })
        node_counter += 1

    return sub_nodes, node_counter


def _build_pure_text_tree(markdown_text: str) -> list[dict]:
    """
    Construit un arbre de documents sémantique à partir du Markdown de LlamaCloud.
    Optimisé pour les articles scientifiques (gestion des titres répétés et hiérarchie).
    Now with fine-grained sub-node splitting at TABLE / Figure / paragraph boundaries.
    """
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")

    root_nodes   = []
    stack        = []
    current_page = 1
    node_counter = 0
    seen_titles  = set()

    def make_node(node_id: str, title: str, page: int, level: int) -> dict:
        return {
            "node_id":       node_id,
            "title":         title,
            "level":         level,
            "page_start":    page,
            "page_end":      page,
            "content_lines": [],
            "nodes":         [],
        }

    intro_node = make_node(f"{node_counter:04d}", "Document Header / Abstract", 1, 0)
    node_counter += 1
    root_nodes.append(intro_node)
    stack.append({"level": 0, "node": intro_node})

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            if stack and any("</table" in l for l in stack[-1]["node"]["content_lines"][-5:]):
                stack[-1]["node"]["content_lines"].append("")
            continue

        page_match = re.search(r'\[\[PAGE_(\d+)\]\]', cleaned_line)
        if page_match:
            current_page = int(page_match.group(1))
            for item in stack:
                item["node"]["page_end"] = max(item["node"]["page_end"], current_page)
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.*)$', cleaned_line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if level == 1 and title in seen_titles:
                continue
            if level == 1:
                seen_titles.add(title)

            new_node = make_node(f"{node_counter:04d}", title, current_page, level)
            node_counter += 1

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            if not stack:
                root_nodes.append(new_node)
                stack.append({"level": level, "node": new_node})
            else:
                stack[-1]["node"]["nodes"].append(new_node)
                stack.append({"level": level, "node": new_node})
        else:
            if stack:
                stack[-1]["node"]["content_lines"].append(line)

    # ── Finalize: clean content, then inject sub-nodes ──────────────
    def finalize_tree(nodes: list[dict], next_start: int | None = None) -> None:
        nonlocal node_counter
        for i, n in enumerate(nodes):
            n["content"] = "\n".join(n["content_lines"]).strip()
            del n["content_lines"]
            del n["level"]

            if i + 1 < len(nodes):
                n["page_end"] = max(n["page_end"], nodes[i + 1]["page_start"])
            elif next_start:
                n["page_end"] = max(n["page_end"], next_start)

            # ── NEW: inject fine-grained sub-nodes into content ──
            if not n["nodes"]:
                # Leaf node — safe to split its content into sub-nodes
                sub_nodes, node_counter = _make_sub_nodes(n, node_counter)
                if sub_nodes:
                    # Replace content with sub-nodes; keep a summary title
                    n["nodes"]   = sub_nodes
                    n["content"] = ""   # content now lives in sub-nodes
            else:
                # Non-leaf — recurse into existing children first
                finalize_tree(n["nodes"], n["page_end"])

    finalize_tree(root_nodes)
    return root_nodes
