import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import ollama

from src.extraction.parser import parse_paper
from src.pipeline import vectorless_rag

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


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

    os.environ["OLLAMA_HOST"] = _url_to_host(args.ollama_url)

    # ── Parse PDF with LlamaParse ──────────────────────────────
    print(f"Parsing: {pdf_path} with LlamaParse ...")
    result = parse_paper(pdf_path)
    tree = result["metadata"]["llamaparse_tree"]
    print(f"  Pages: {result['metadata'].get('page_count', '?')}")
    print(f"  Tree: {len(tree)} root nodes")

    # ── Check Ollama availability ─────────────────────────────
    try:
        ollama.list()
    except Exception:
        print(f"\nError: Ollama not available at {args.ollama_url}")
        print("  Start the server:  ollama serve")
        print(f"  Then pull the model:  ollama pull {args.model}")
        sys.exit(1)
    print(f"  Ollama: {args.model} connected")

    # ── Answer ─────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        vectorless_rag(args.query, tree, args.model)

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
                print()
                vectorless_rag(q, tree, args.model)
                print()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
