import argparse
import asyncio  # ✨ Requis pour exécuter votre fonction asynchrone
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import ollama
from llama_parse import LlamaParse
from src.extraction import extract_images_with_captions 
from src.pipeline import vectorless_rag_no_loss
from src.extraction.parser import _build_pure_text_tree

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


async def run_pipeline_orchestration(pdf_path: Path, api_key: str) -> list[dict]:
    # 1. Extraction locale et ultra-rapide des images + légendes avec PyMuPDF
    print(f"🖼️ Extracting figures and captions locally with PyMuPDF...")
    liste_images_locales = extract_images_with_captions(str(pdf_path))
    
    # Construction des dictionnaires de correspondance pour l'arbre
    page_image_map = {}
    page_captions_text = {} # Permet d'injecter la légende dans le texte pour la recherche
    
    for img in liste_images_locales:
        p_num = img["numero_page"]
        if p_num not in page_image_map:
            page_image_map[p_num] = []
            page_captions_text[p_num] = []
            
        page_image_map[p_num].append(img["base64"])
        if img["legende_detectee"] != "Aucune légende trouvée":
            page_captions_text[p_num].append(f"[Visual Component] Caption: {img['legende_detectee']}")

    # 2. LlamaParse standard (plus besoin d'activer l'option payante 'use_vendor_multimodal_model')
    print(f"Parsing text: {pdf_path} with LlamaParse Standard...")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )
    
    json_objs = parser.get_json_result(str(pdf_path))
    json_list = json_objs[0]["pages"] if isinstance(json_objs, list) else json_objs["pages"]
    
    # 3. Reconstruction du flux Markdown
    markdown_chunks = []
    for page_data in json_list:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        
        # Injection des légendes locales directement dans le texte brut de la page correspondante
        # Cela garantit que la recherche par mot-clé (ou LLM) trouvera "Fig 4" à la bonne page
        if page_num in page_captions_text and page_captions_text[page_num]:
            captions_block = "\n".join(page_captions_text[page_num])
            markdown_chunks.append(captions_block)
            
        markdown_chunks.append(page_data.get("md", page_data.get("text", "")))
        
    markdown_content = "\n".join(markdown_chunks)
    
    # 4. Construction de l'arbre final associant le texte, les légendes et le Base64
    tree = _build_pure_text_tree(markdown_content, page_image_map)
    
    return tree



def main():
    parser = argparse.ArgumentParser(
        description="Parse a PDF with LlamaParse and answer questions using vectorless Graph-RAG"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b",
                        help="Ollama model (default: qwen2.5vl:3b)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", "-q", type=str, help="Single question to answer")
    group.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive Q&A session")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    api_key = os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        print("Error: LLAMACLOUD_API_KEY environment variable is not set.")
        sys.exit(1)

    os.environ["OLLAMA_HOST"] = _url_to_host(args.ollama_url)

    # ── Check Ollama availability ─────────────────────────────
    try:
        ollama.list()
    except Exception:
        print(f"\nError: Ollama not available at {args.ollama_url}")
        print("  Start the server:  ollama serve")
        print(f"  Then pull the model:  ollama pull {args.model}")
        sys.exit(1)
    print(f"  Ollama: {args.model} connected")

    # ── Orchestration Loop (Async to Sync) ─────────────────────
    # On exécute la fonction de chaînage asynchrone de manière sécurisée
    tree = asyncio.run(run_pipeline_orchestration(pdf_path, api_key))
    print(f"  Tree successfully built: {len(tree)} root sections identified with embedded visuals.")

    # ── Answer ─────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        result_data = vectorless_rag_no_loss(args.query, tree, args.model)
        
        print(f"📝 Answer:\n{result_data['answer']}\n")
        print("📍 Sources used:")
        for citation in result_data["sources"]:
            print(f"👉 {citation}")

    else:
        print("\nInteractive mode. Type questions (Ctrl+C to quit).\n")
        try:
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "/q"):
                    break
                
                result_data = vectorless_rag_no_loss(q, tree, args.model)
                
                print(f"📝 Answer:\n{result_data['answer']}\n")
                print("📍 Sources used:")
                for citation in result_data["sources"]:
                    print(f"👉 {citation}")
                print()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()