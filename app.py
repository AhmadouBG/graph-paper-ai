import logging
import os
import tempfile
from urllib.parse import urlparse

import ollama
import streamlit as st
from dotenv import load_dotenv

from src.extraction import extract_images_with_captions, build_page_image_map
from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud
from src.pipeline import vectorless_rag_no_loss, print_tree, get_total_pages
from src.extraction.extract_caption import build_caption_map_from_markdown, match_images_to_captions

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(
    page_title="Graph Paper AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }

/* Hide default header */
[data-testid="stHeader"] { background: transparent; }

/* Upload zone */
.upload-zone {
    border: 2px dashed #2a3347;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    background: #161b27;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #4f8ef7; }

/* Hero title */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #6b7a99;
    font-size: 1rem;
    margin-bottom: 2.5rem;
}

/* Source expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}

/* Status pills */
.status-ok {
    display: inline-block;
    background: #0d2b1e;
    color: #34d399;
    border: 1px solid #065f46;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 500;
}
.status-err {
    display: inline-block;
    background: #2b0d0d;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Tree code block */
.tree-block {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    max-height: 300px;
    overflow-y: auto;
}

/* Figure card */
.fig-caption {
    font-size: 0.78rem;
    color: #6b7a99;
    margin-top: 0.4rem;
    line-height: 1.4;
}


/* Spinner */
[data-testid="stSpinner"] { color: #4f8ef7; }

/* Divider */
hr { border-color: #e2e8f0; }
[data-testid="stSidebar"] hr {
    margin-top: -0.5rem !important;
    margin-bottom: -0.5rem !important;
}
[data-testid="stSidebar"] h3 {
    margin-top: 0.5rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 2px; }
::-webkit-scrollbar-thumb { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = "qwen2.5vl:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _url_to_host(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


def run_pipeline(pdf_path: str) -> tuple[list[dict], dict]:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        st.error("LLAMA_CLOUD_API_KEY not set in your .env file.")
        st.stop()

    with st.spinner("🔬 Extracting figures with PyMuPDF..."):
        liste_images_locales = extract_images_with_captions(pdf_path)

    raw_page_image_map = build_page_image_map(liste_images_locales)

    with st.spinner("📖 Parsing document structure with LlamaCloud..."):
        json_list = _parse_with_llamacloud(pdf_path, api_key)

    markdown_chunks = []
    for page_data in json_list:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)

    caption_map = build_caption_map_from_markdown(markdown_content)
    page_image_map = match_images_to_captions(raw_page_image_map, caption_map)
    print("\n🔍 DEBUG page_image_map after match:")
    for page, imgs in page_image_map.items():
        for img in imgs:
            print(f"  p.{page} label='{img.get('label','')}' cap='{img.get('caption','')[:50]}'")
    page_captions_text = {}
    for page_num, imgs in page_image_map.items():
        caps = [
            f"[Visual Component] Caption: {img['caption']}"
            for img in imgs
            if img["caption"] != "Aucune légende trouvée"
        ]
        if caps:
            page_captions_text[page_num] = caps

    markdown_chunks = []
    for page_data in json_list:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        if page_num in page_captions_text:
            markdown_chunks.append("\n".join(page_captions_text[page_num]))
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)

    with st.spinner("🌲 Building document tree..."):
        tree = _build_pure_text_tree(markdown_content, page_image_map)

    return tree, page_image_map


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
                b64 = img["base64"] if isinstance(img, dict) else img
                if b64 not in seen:
                    seen.add(b64)
                    caption = img.get("caption", "") if isinstance(img, dict) else ""
                    label = img.get("label", "") if isinstance(img, dict) else ""
                    imgs.append((n["title"], n.get("page_start", "?"), b64, caption, label))
            if n.get("nodes"):
                walk(n["nodes"])
    walk(nodes)
    return imgs


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("tree", None),
    ("messages", []),
    ("pdf_name", ""),
    ("page_image_map", {}),
    ("processing", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="Ollama model name")
    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)

    ollama_ok = check_ollama(model, ollama_url)
    api_key_ok = bool(
        os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if ollama_ok:
            st.markdown('<span class="status-ok">● Ollama</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✕ Ollama</span>', unsafe_allow_html=True)
    with col_b:
        if api_key_ok:
            st.markdown('<span class="status-ok">● API Key</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✕ API Key</span>', unsafe_allow_html=True)

    if st.session_state.tree:
        st.divider()
        if st.button("📂 Load new document", use_container_width=True):
            for key in ["tree", "messages", "pdf_name", "page_image_map"]:
                st.session_state[key] = None if key == "tree" else ([] if key == "messages" else {} if key == "page_image_map" else "")
            st.rerun()
        st.divider()
        
        st.markdown("### 📚 Document")
        st.caption(f"**{st.session_state.pdf_name}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Sections", count_nodes(st.session_state.tree))
        with c2:
            total_pages = get_total_pages(st.session_state.tree) if st.session_state.tree else "?"
            st.metric("Pages", total_pages)

        with st.expander("🌲 Document tree", expanded=False):
            import io, sys
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            print_tree(st.session_state.tree)
            sys.stdout = old_stdout
            st.code(buf.getvalue(), language="text")

        

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.tree:
    # ── Landing / Upload ──────────────────────────────────────────────────
    st.markdown("<div style='height: 6vh'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<p class="hero-title">Graph Paper AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Ask anything about your PDF — text, tables, and figures.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload a PDF to get started",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if not api_key_ok:
                st.error("Set LLAMA_CLOUD_API_KEY in your .env file before uploading.")
            elif not ollama_ok:
                st.error(f"Ollama is not reachable at {ollama_url}. Start Ollama and retry.")
            else:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    progress = st.progress(0, text="Starting…")
                    progress.progress(10, text="Extracting figures…")
                    tree, page_image_map = run_pipeline(tmp_path)
                    progress.progress(100, text="Ready!")
                    st.session_state.tree = tree
                    st.session_state.page_image_map = page_image_map
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

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Feature hints
        cols = st.columns(3)
        hints = [
            ("📝", "Text questions", "Definitions, methods, results"),
            ("📊", "Tables & data", "\"Show me Table 3\""),
            ("🖼️", "Figures", "\"Explain Figure 4\""),
        ]
        for col, (icon, label, example) in zip(cols, hints):
            with col:
                st.markdown(
                    f"""<div style='border:1px solid #e2e8f0;
                    border-radius:12px;padding:1rem;text-align:center'>
                    <div style='font-size:1.5rem'>{icon}</div>
                    <div style='color:#6b777bb;font-weight:600;font-size:0.85rem;
                    margin:0.4rem 0 0.2rem'>{label}</div>
                    <div style='color:#6b7a99;font-size:0.75rem'>{example}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

else:
    # ── Chat interface ────────────────────────────────────────────────────
    # chat_tab, figures_tab = st.tabs(["💬 Chat", "🖼️ Figures"])

    # Welcome message if no history
    if not st.session_state.messages:
        fig_count = len(collect_images(st.session_state.tree))
        node_count = count_nodes(st.session_state.tree)
        st.markdown(
            f"""<div style='border:1px solid #e2e8f0;
            border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem'>
            <div style='color:#4f8ef7;font-weight:600;margin-bottom:0.4rem'>
            ✅ {st.session_state.pdf_name} is ready</div>
            <div style='color:#6b7a99;font-size:0.875rem'>
            Found <b style='color:#272829'>{node_count} sections</b> and
            <b style='color:#272829'>{fig_count} figures</b>.
            Ask me anything about this document.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📍 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # Input
    if prompt := st.chat_input(
        "Ask about text, a table, or a figure…",
        disabled=not ollama_ok,
    ):
        if not ollama_ok:
            st.error("Ollama is not connected.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = vectorless_rag_no_loss(
                        prompt,
                        st.session_state.tree,
                        model,
                        page_image_map=st.session_state.page_image_map,
                    )
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander("📍 Sources", expanded=False):
                        for s in result["sources"]:
                            st.markdown(f"- {s}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })

    # with figures_tab:
    #     all_imgs = collect_images(st.session_state.tree)
    #     if all_imgs:
    #         cols = st.columns(2)
    #         for i, (title, page, b64, caption, label) in enumerate(all_imgs):
    #                 with cols[i % 2]:
    #                     st.image(
    #                         f"data:image/png;base64,{b64}",
    #                         use_container_width=True,
    #                     )
    #                     display_caption = caption if caption and caption != "Aucune légende trouvée" else title
    #                     st.markdown(
    #                         f'<p class="fig-caption">📄 p.{page} &nbsp;·&nbsp; {display_caption}</p>',
    #                         unsafe_allow_html=True,
    #                     )
    #                     st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    #     else:
    #         st.markdown(
    #             """<div style='text-align:center;padding:3rem;color:#6b7a99'>
    #             <div style='font-size:2rem;margin-bottom:0.5rem'>🖼️</div>
    #             No figures were extracted from this document.
    #             </div>""",
    #             unsafe_allow_html=True,
    #         )