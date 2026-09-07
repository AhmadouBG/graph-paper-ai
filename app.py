import io
import logging
import os
import sys
import tempfile
import ollama
import streamlit as st
from dotenv import load_dotenv
from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud
from src.pipeline import vectorless_rag_no_loss, print_tree, get_total_pages
from src.utils.dictionary import DICTIONARY

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
# Les variables par défaut de l'application
DEFAULT_MODEL = "qwen2.5:3b"

# ── Custom CSS (Nettoyé des styles inutilisés) ─────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    line-height: 1.7;
    word-break: break-word;
}
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    padding-left: 1.5rem;
    margin: 0.5rem 0;
}
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
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
[data-testid="stMetric"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
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
[data-testid="stSpinner"] { color: #4f8ef7; }
hr { border-color: #e2e8f0; }
::-webkit-scrollbar { width: 2px; }
::-webkit-scrollbar-thumb { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = "qwen2.5:3b"

# ── Core Pipeline (Version purement textuelle) ────────────────────────────────
def run_pipeline(pdf_path: str) -> list[dict]:
    """Exécute le parsing LlamaCloud et construit l'arbre sémantique pur."""
    t = DICTIONARY[st.session_state.lang] # Accès à la langue courante
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        st.error(t["err_missing_api_env"])
        st.stop()

    with st.spinner("📖 Extraction du PDF avec LlamaCloud (Agentic)..."):
        page_dicts = _parse_with_llamacloud(pdf_path, api_key)

    # Re build Markdown with page markers for tree builder
    markdown_chunks = []
    for page_data in page_dicts:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)
    with st.spinner("🌲 Structuration de l'arbre documentaire..."):
        tree = _build_pure_text_tree(markdown_content)
    return tree

def check_ollama(model: str) -> bool:
    """Vérifie si l'instance locale Ollama est accessible."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

def count_nodes(nodes: list[dict]) -> int:
    """Compte récursivement le nombre total de nœuds dans l'arbre."""
    c = len(nodes)
    for n in nodes:
        if n.get("nodes"):
            c += count_nodes(n["nodes"])
    return c

# ── Initialize session state ───────────────────────────────────────────
for key, default in [
    ("tree", None),
    ("messages", []),
    ("pdf_name", ""),
    ("processing", False),
    ("lang", "fr"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# short cut for writing translations
t = DICTIONARY[st.session_state.lang]

def toggle_language():
    """Bascule d'une langue à l'autre."""
    st.session_state.lang = "en" if st.session_state.lang == "fr" else "fr"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.button(t["switch_btn"], on_click=toggle_language, use_container_width=True)
    st.divider()

    st.markdown(t["cfg_title"])
    model = st.text_input(t["lbl_model"], value=DEFAULT_MODEL, help=t["help_model"])

    ollama_ok = check_ollama(model)
    api_key_ok = bool(
        os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if ollama_ok:
            st.markdown(f'<span class="status-ok">{t["status_ollama_ok"]}</span>', 
                                    unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-err">{t["status_ollama_err"]}</span>', 
                                    unsafe_allow_html=True)
    with col_b:
        if api_key_ok:
            st.markdown(f'<span class="status-ok">{t["status_api_ok"]}</span>', 
                                    unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-err">{t["status_api_err"]}</span>', 
                                    unsafe_allow_html=True)

    if st.session_state.tree:
        st.divider()
        if st.button(t["btn_new_doc"], use_container_width=True):
            for key in ["tree", "messages", "pdf_name"]:
                st.session_state[key] = (
                    None if key == "tree"
                    else [] if key == "messages"
                    else ""
                )
            st.rerun()
        st.divider()
        st.markdown(t["active_doc"])
        st.caption(f"**{st.session_state.pdf_name}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric(t["metric_sections"], count_nodes(st.session_state.tree))
        with c2:
            total_pages = get_total_pages(st.session_state.tree) if st.session_state.tree else "?"
            st.metric(t["metric_pages"], total_pages)

        with st.expander(t["tree_expander"], expanded=False):
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            print_tree(st.session_state.tree)
            sys.stdout = old_stdout
            st.code(buf.getvalue(), language="text")

# ── 5. Main area ───────────────────────────────────────────────────────────────
if not st.session_state.tree:
    st.markdown("<div style='height: 6vh'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(f'<p class="hero-title">{t["hero_title"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="hero-sub">{t["hero_sub"]}</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            t["uploader_label"],
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if not api_key_ok:
                st.error(t["err_api"])
            elif not ollama_ok:
                st.error(t["err_ollama"].format(model=model))
            else:
                tmpdir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmpdir, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    progress = st.progress(0, text=t["p_init"])
                    progress.progress(20, text=t["p_parse"])                   
                    tree = run_pipeline(tmp_path)                   
                    progress.progress(100, text=t["p_ready"])
                    st.session_state.tree = tree
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(t["err_failed"].format(e))
                finally:
                    try:
                        import shutil
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # grid of usage examples
        cols = st.columns(3)
        hints = [
            ("📝", t["hint_1_title"], t["hint_1_desc"]),
            ("📊", t["hint_2_title"], t["hint_2_desc"]),
            ("🔍", t["hint_3_title"], t["hint_3_desc"]),
        ]
        for col, (icon, label, example) in zip(cols, hints):
            with col:
                st.markdown(
                    f"""<div style='border:1px solid #e2e8f0;
                    border-radius:12px;padding:1rem;text-align:center'>
                    <div style='font-size:1.5rem'>{icon}</div>
                    <div style='color:#4f8ef7;font-weight:600;font-size:0.85rem;
                    margin:0.4rem 0 0.2rem'>{label}</div>
                    <div style='color:#6b7a99;font-size:0.75rem'>{example}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

else:
    # ── chat Area ────────────────────────────────────────────────────────────
    if not st.session_state.messages:
        node_count = count_nodes(st.session_state.tree)
        st.markdown(
            f"""<div style='border:1px solid #e2e8f0;
            border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem'>
            <div style='color:#34d399;font-weight:600;margin-bottom:0.4rem'>
            {t["chat_success"].format(st.session_state.pdf_name)}</div>
            <div style='color:#6b7a99;font-size:0.875rem'>
            {t["chat_sub"].format(node_count)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # history of messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("thinking"):
                with st.expander(t["expander_thinking"], expanded=False):
                    st.info(msg["thinking"])
            if msg.get("sources"):
                with st.expander(t["expander_sources"], expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # user input area
    if prompt := st.chat_input(
        t["chat_input_placeholder"],
        disabled=not ollama_ok,
    ):
        if not ollama_ok:
            st.error(t["err_ollama_disconnected"])
        else:
            # Adding and displaying user question
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(t["spinner_analyzing"]):
                    # Execution of the new corrected pipeline
                    result = vectorless_rag_no_loss(
                        query=prompt,
                        tree=st.session_state.tree,
                        model=model
                    )  
                # Displaying the final answer
                st.markdown(result["answer"])
                # Displaying the routing of the Vectorless RAG
                with st.expander(t["expander_thinking"], expanded=False):
                    st.info(result["thinking"])                  
                if result.get("retrieved_sections"):
                    with st.expander(t["expander_sources"], expanded=False):
                        for s in result["retrieved_sections"]:
                            st.markdown(f"- {s}")
                if result.get("retrieved_texts"):
                    with st.expander("Context Sources", expanded=False):
                        for idx, chunk in enumerate(result["retrieved_texts"]):
                            st.markdown(f"**Context Chunk {idx+1}:**")
                            st.markdown(f"```text\n{chunk}\n```")
                            st.markdown("---")
            # Persistence in the session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "thinking": result["thinking"],
                "sources": result.get("retrieved_sections", []),
                "retrieved_texts": result.get("retrieved_texts", []),
            })

