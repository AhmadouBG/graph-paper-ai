import logging
import os
import tempfile
from urllib.parse import urlparse

import ollama
import streamlit as st
from dotenv import load_dotenv
# Importations sécurisées de votre framework local
from src.extraction import extract_images_with_captions, build_page_image_map
from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud
from src.pipeline import vectorless_rag_no_loss, print_tree


load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(page_title="Graph Paper AI", page_icon="📄", layout="wide")

DEFAULT_MODEL = "qwen2.5vl:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


def run_pipeline(pdf_path: str) -> list[dict]:
    # ✨ FIX 1 : Correction du nom de la clé d'environnement standard de LlamaIndex
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        st.error("LLAMA_CLOUD_API_KEY not set in your .env file.")
        st.stop()

    with st.spinner("Extracting figures and captions locally with PyMuPDF..."):
        liste_images_locales = extract_images_with_captions(pdf_path)

    page_image_map = build_page_image_map(liste_images_locales)
    # Fix: extract caption string from each dict
    page_captions_text = {
        p: [f"[Visual Component] Caption: {img['caption']}" 
            for img in imgs if img["caption"] != "Aucune légende trouvée"]
        for p, imgs in page_image_map.items()
    }

    with st.spinner("Parsing text structure with LlamaParse..."):
        json_list = _parse_with_llamacloud(tmp_path, api_key)

    markdown_chunks = []
    for page_data in json_list:
        page_num = page_data["page"]  # dict access, not attribute
        markdown_chunks.append(f"--- Page {page_num} ---")
        
        if page_num in page_captions_text and page_captions_text[page_num]:
            markdown_chunks.append("\n".join(page_captions_text[page_num]))
    
        markdown_chunks.append(page_data.get("md", ""))

    markdown_content = "\n".join(markdown_chunks)

    with st.spinner("Assembling Context-Aware Document Tree..."):
        # ✨ Assurez-vous que votre parser de fichiers accepte bien page_image_map en 2e argument
        tree = _build_pure_text_tree(markdown_content, page_image_map)

    return tree


def check_ollama(model: str, url: str) -> bool:
    os.environ["OLLAMA_HOST"] = _url_to_host(url)
    try:
        ollama.list()
        return True
    except Exception:
        return False


def count_nodes(nodes):
    c = len(nodes)
    for n in nodes:
        if n.get("nodes"):
            c += count_nodes(n["nodes"])
    return c


def collect_images(nodes):
    seen = set()
    imgs = []
    def walk(ns):
        for n in ns:
            for img in n.get("base64_images", []):
                # img is now a dict with "base64", "caption", "extension"
                b64 = img["base64"] if isinstance(img, dict) else img
                if b64 not in seen:
                    seen.add(b64)
                    caption = img.get("caption", "") if isinstance(img, dict) else ""
                    imgs.append((n["title"], n.get("page_start", "?"), b64, caption))
            if n.get("nodes"):
                walk(n["nodes"])
    walk(nodes)
    return imgs


# ── INITIALISATION DES ÉTATS STREAMLIT ───────────────────────────────────────
if "tree" not in st.session_state:
    st.session_state.tree = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ── SIDEBAR CONFIGURATION ───────────────────────────────────────────────────
with st.sidebar:
    model = st.text_input("Ollama Model", value=DEFAULT_MODEL)
    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)

    ollama_ok = check_ollama(model, ollama_url)
    if ollama_ok:
        st.success(f"Connected ({model})")
    else:
        st.error(f"Ollama not reachable at {ollama_url}")

    # Vérification double de la clé API
    api_key_check = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if api_key_check:
        st.success("LlamaCloud key configured")
    else:
        st.error("LLAMA_CLOUD_API_KEY missing in .env")

    if st.session_state.tree:
        st.divider()
        st.subheader("📚 Document Tree")

        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_tree(st.session_state.tree)
        sys.stdout = old_stdout
        st.code(buf.getvalue(), language="text")

        total = count_nodes(st.session_state.tree)
        st.metric("Sections identified", total)
        
        # Récupération sécurisée de la dernière page
        last_page = st.session_state.tree[-1].get("page_end", "?") if st.session_state.tree else "?"
        st.metric("Total Pages", last_page)

        st.divider()
        if st.button("New Document", use_container_width=True):
            st.session_state.tree = None
            st.session_state.messages = []
            st.session_state.pdf_name = ""
            st.rerun()

# ── INTERFACE GRAPHIQUE PRINCIPALE ──────────────────────────────────────────
if not st.session_state.tree:
    st.markdown(
        "<h1 style='text-align: center; margin-top: 2rem;'>📄 Graph Paper AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #888; margin-bottom: 2rem;'>"
        "Vectorless Graph-RAG for academic papers (In-Memory Processing)</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a PDF", type=["pdf"], label_visibility="collapsed"
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                st.session_state.tree = run_pipeline(tmp_path)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        st.markdown(
            "<p style='text-align: center; color: #888; font-size: 0.9rem;'>"
            "Drop a PDF to start asking questions about it</p>",
            unsafe_allow_html=True,
        )
else:
    st.title(f"📄 {st.session_state.pdf_name}")
    chat_tab, preview_tab = st.tabs(["💬 Chat Session", "🖼️ Extracted Figures (RAM Cache)"])

    with chat_tab:
        # Affichage de l'historique des discussions de la session
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📍 Sources Used"):
                        for s in msg["sources"]:
                            st.markdown(f"- {s}")

        if prompt := st.chat_input("Ask a question about text, tables or figures..."):
            if not ollama_ok:
                st.error("Ollama is not connected. Check server connection status.")
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Executing Context-Aware Routing & Visual Analysis..."):
                        # ✨ Exécute le pipeline hybride durci et fluide sur CPU
                        result = vectorless_rag_no_loss(
                            prompt, st.session_state.tree, model
                        )
                    
                    st.markdown(result["answer"])
                    if result.get("sources"):
                        with st.expander("📍 Sources Used"):
                            for s in result["sources"]:
                                st.markdown(f"- {s}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )

    with preview_tab:
        all_imgs = collect_images(st.session_state.tree)
        if all_imgs:
            cols = st.columns(2)
            for i, (title, page, b64, caption) in enumerate(all_imgs):
                with cols[i % 2]:
                    st.image(
                    f"data:image/png;base64,{b64}",
                    caption=f"[Page {page}] {caption or title}",
                    width="stretch",
                )
        else:
            st.info("No visual figure components extracted from this document structure.")
