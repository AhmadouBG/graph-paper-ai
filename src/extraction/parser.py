from __future__ import annotations

from pathlib import Path
from llama_parse import LlamaParse 
import logging
import re
from dotenv import load_dotenv
import base64

logger = logging.getLogger(__name__)
load_dotenv()


async def _parse_with_llamaparse(pdf_path: Path, api_key: str):

    client = AsyncLlamaCloud(api_key=api_key)
    file_obj = await client.files.create(file=pdf_path, purpose="parse")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        expand=["items"],
        version="latest",
    )
    return result 

async def run_pipeline_orchestration(pdf_path: Path, api_key: str) -> list[dict]:
    print(f"Parsing: {pdf_path} with LlamaParse SDK...")
    
    # Étape 1 : Initialisation officielle recommandée
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # Conserve le formatage des titres pour l'arbre
        verbose=True
    )
    
    # Étape 2 : Récupération du résultat JSON complet
    json_objs = parser.get_json_result(str(pdf_path))
    json_list = json_objs[0]["pages"]
    
    # Étape 3 : Extraction des images directement associés à chaque page en RAM
    # LlamaParse stocke les métadonnées d'images par page dans json_objs.
    # Pour éviter le téléchargement physique, nous créons un dictionnaire {page_num: [base64_strings]}
    page_image_map = {}
    
    print("🖼️ Mapping images to pages and converting to Base64 in RAM...")
    for page_data in json_list:
        page_num = page_data["page"]
        page_image_map[page_num] = []
        
        # Récupération des images de la page courante via le SDK
        # Si get_images demande un chemin, on peut utiliser les données brutes "images" du JSON de la page si présentes
        if "images" in page_data:
            for img_info in page_data["images"]:
                # Si l'image contient déjà les bytes ou si on passe par une conversion directe :
                if "base64" in img_info:
                    page_image_map[page_num].append(img_info["base64"])
                elif "bytes" in img_info:
                    import base64
                    b64_str = base64.b64encode(img_info["bytes"]).decode('utf-8')
                    page_image_map[page_num].append(b64_str)
                    
        # Alternative officielle si vous préférez utiliser la méthode get_images du SDK sans stockage persistant :
        # On utilise un dossier temporaire en RAM (BytesIO / Tempfile virtuel) ou le dossier local standard :
        # images_dicts = parser.get_images(json_objs, download_path="tmp_imgs")
        
    # Étape 4 : Extraction du texte brut markdown complet
    # On reconstruit la chaîne markdown globale intégrant les balises de pages indispensables à votre arbre
    markdown_chunks = []
    for page_data in json_list:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", page_data.get("text", "")))
        
    markdown_content = "\n".join(markdown_chunks)
    
    # Étape 5 : Construction de votre arbre textuel connecté aux images en mémoire
    tree = _build_pure_text_tree(markdown_content, page_image_map)
    
    return tree



def _build_pure_text_tree(markdown_text: str, page_image_map: dict[int, list[str]]) -> list[dict]:
    text_with_page_tags = re.sub(r'---\s*Page\s*(\d+)\s*---', r'[[PAGE_\1]]', markdown_text)
    lines = text_with_page_tags.split("\n")
    
    root_nodes = []
    stack = []
    current_page = 1
    node_counter = 0
    
    intro_node = {
        "node_id": f"{node_counter:04d}",
        "title": "Document Header / Introduction",
        "page_start": 1,
        "page_end": 1,
        "content_lines": [],
        "base64_images": page_image_map.get(1, []), # ✨ In-memory Base64 strings
        "nodes": []
    }
    node_counter += 1
    root_nodes.append(intro_node)
    stack.append({"level": 0, "node": intro_node})
    
    for line in lines:
        page_match = re.search(r'\[\[PAGE_(\d+)\]\]', line)
        if page_match:
            current_page = int(page_match.group(1))
            if stack:
                stack[-1]["node"]["page_end"] = current_page
            continue
            
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            
            new_node = {
                "node_id": f"{node_counter:04d}",
                "title": title,
                "page_start": current_page,
                "page_end": current_page,
                "content_lines": [],
                "base64_images": page_image_map.get(current_page, []), # ✨ In-memory Base64 strings
                "nodes": []
            }
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
            if stack and line.strip():
                stack[-1]["node"]["content_lines"].append(line)

    def finalize_tree(nodes, next_start=None):
        for i, n in enumerate(nodes):
            n["content"] = "\n".join(n["content_lines"])
            del n["content_lines"]
            if i + 1 < len(nodes):
                n["page_end"] = max(n["page_start"], nodes[i+1]["page_start"])
            elif next_start:
                n["page_end"] = max(n["page_start"], next_start)
            if n["nodes"]:
                finalize_tree(n["nodes"], n["page_end"])
                
    finalize_tree(root_nodes)
    return root_nodes

 


