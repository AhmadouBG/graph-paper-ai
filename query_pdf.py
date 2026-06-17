import argparse
import logging
import sys
from pathlib import Path

from src.extraction.graph import _parse_sections
from src.extraction.parser import parse_paper
from src.llm.ollama_client import OllamaClient
from src.pipeline import vectorless_rag

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _build_tree_from_markdown(markdown: str) -> list[dict]:
    sections = _parse_sections(markdown)
    if not sections:
        return [{
            "node_id": "0000",
            "title": "Document",
            "page_index": 1,
            "summary": markdown.strip(),
            "text": markdown.strip(),
            "visuals": [],
            "tables": [],
            "nodes": [],
        }]
    root: dict = {
        "node_id": "0000",
        "title": "Document",
        "page_index": 1,
        "summary": "",
        "text": "",
        "visuals": [],
        "tables": [],
        "nodes": [],
    }
    for i, sec in enumerate(sections):
        root["nodes"].append({
            "node_id": f"{i + 1:04d}",
            "title": sec["title"],
            "page_index": 1,
            "summary": sec["content"][:2000],
            "text": sec["content"][:2000],
            "visuals": [],
            "tables": [],
            "nodes": [],
        })
    return [root]


def main():
    parser = argparse.ArgumentParser(
        description="Parse a PDF and answer questions using vectorless RAG"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b",
                        help="Ollama model (default: qwen2.5vl:3b)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--llamaparse", action="store_true",
                        help="Use LlamaParse for structured parsing (recommended)")
    parser.add_argument("--marker", action="store_true",
                        help="Use Marker parser")
    parser.add_argument("--marker-timeout", type=int, default=120,
                        help="Timeout in seconds for Marker (default: 120)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", "-q", type=str, help="Single query to answer")
    group.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive Q&A session")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    # ── Parse PDF ──────────────────────────────────────────────
    method = "LlamaParse" if args.llamaparse else ("Marker" if args.marker else "PyMuPDF")
    print(f"Parsing PDF: {pdf_path} (method: {method}) ...")
    if args.marker:
        print("  (Marker model download may take several minutes on first run)")

    result = parse_paper(
        pdf_path,
        use_llamaparse=args.llamaparse,
        use_marker=args.marker,
        marker_timeout=args.marker_timeout,
    )

    # ── Build tree ─────────────────────────────────────────────
    tree = result.metadata.get("llamaparse_tree")
    if tree:
        print(f"  LlamaParse tree: {len(tree)} root nodes")
    else:
        print("  Building section tree from markdown...")
        tree = _build_tree_from_markdown(result.markdown)

    print(f"  Pages: {result.metadata.get('page_count', '?')}")
    print(f"  Images: {result.metadata.get('image_count', 0)}")

    # ── Connect to Ollama ──────────────────────────────────────
    llm = OllamaClient(base_url=args.ollama_url, model=args.model)
    if not llm.is_available():
        print(f"\nError: Ollama is not available at {args.ollama_url}")
        print("  Start the server:  ollama serve")
        print(f"  Then pull the model:  ollama pull {args.model}")
        sys.exit(1)
    print(f"  Ollama: connected ({args.model})")

    # ── Answer ─────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        vectorless_rag(args.query, tree, llm)

    elif args.interactive or not args.query:
        print("\nInteractive mode. Type your questions (Ctrl+C to quit).\n")
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
                print()
                vectorless_rag(q, tree, llm)
                print()
        except KeyboardInterrupt:
            print()

    llm.close()


if __name__ == "__main__":
    main()
