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

def _build_pure_text_tree(markdown_text: str) -> list[dict]:
    """
    Construit un arbre de documents sémantique à partir du Markdown de LlamaCloud.
    Optimisé pour les articles scientifiques (gestion des titres répétés et hiérarchie).
    """
    # 1. Normalisation des marqueurs de page (LlamaCloud utilise parfois des syntaxes variées)
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")

    root_nodes = []
    stack = []
    current_page = 1
    node_counter = 0
    seen_titles = set() # Pour éviter de dupliquer les titres de garde répétés en haut de page

    def make_node(node_id: str, title: str, page: int, level: int) -> dict:
        return {
            "node_id": node_id,
            "title": title,
            "level": level,
            "page_start": page,
            "page_end": page,
            "content_lines": [],
            "nodes": [],
        }

    # Nœud racine initial pour capturer les métadonnées (auteurs, abstract si hors titre)
    intro_node = make_node(f"{node_counter:04d}", "Document Header / Abstract", 1, 0)
    node_counter += 1
    root_nodes.append(intro_node)
    stack.append({"level": 0, "node": intro_node})

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            # Conserver les lignes vides uniquement si on est dans un tableau HTML pour garder la structure
            if stack and any("</table" in l for l in stack[-1]["node"]["content_lines"][-5:]):
                stack[-1]["node"]["content_lines"].append("")
            continue

        # Détection du changement de page
        page_match = re.search(r'\[\[PAGE_(\d+)\]\]', cleaned_line)
        if page_match:
            current_page = int(page_match.group(1))
            # Met à jour la page de fin de tous les nœuds actuellement ouverts dans la pile
            for item in stack:
                item["node"]["page_end"] = max(item["node"]["page_end"], current_page)
            continue

        # Détection d'un titre Markdown (# Titre)
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', cleaned_line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Anti-bug : Si le titre exact de niveau 1 a déjà été vu (ex: titre du papier répété en p.2)
            # On ignore la création d'un nouveau nœud et on traite la ligne comme du texte ou on passe.
            if level == 1 and title in seen_titles:
                continue
            if level == 1:
                seen_titles.add(title)

            new_node = make_node(f"{node_counter:04d}", title, current_page, level)
            node_counter += 1

            # Dépiler jusqu'à trouver le parent légitime (un niveau strictement inférieur)
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            if not stack:
                # Sécurité si la pile est vide (ne devrait pas arriver avec le nœud 0)
                root_nodes.append(new_node)
                stack.append({"level": level, "node": new_node})
            else:
                stack[-1]["node"]["nodes"].append(new_node)
                stack.append({"level": level, "node": new_node})
        else:
            # Ajout du texte au nœud actif le plus profond
            if stack:
                stack[-1]["node"]["content_lines"].append(line)

    def finalize_tree(nodes: list[dict], next_start: int | None = None) -> None:
        """Nettoie les lignes, fusionne le texte et ajuste les pages de fin."""
        for i, n in enumerate(nodes):
            n["content"] = "\n".join(n["content_lines"]).strip()
            del n["content_lines"]
            
            # Gestion des niveaux pour l'affichage/RAG
            del n["level"] 

            if i + 1 < len(nodes):
                n["page_end"] = max(n["page_end"], nodes[i + 1]["page_start"])
            elif next_start:
                n["page_end"] = max(n["page_end"], next_start)
            
            if n["nodes"]:
                finalize_tree(n["nodes"], n["page_end"])

    finalize_tree(root_nodes)
    return root_nodes
