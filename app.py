import logging
import os
import tempfile
import ollama
import streamlit as st
from dotenv import load_dotenv

# Importations de vos modules corrigés et validés
from src.extraction.parser import _build_pure_text_tree, _parse_with_llamacloud
from src.pipeline import vectorless_rag_no_loss, print_tree, get_total_pages

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
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMACLOUD_API_KEY")
    if not api_key:
        st.error("Clé LLAMA_CLOUD_API_KEY manquante dans le fichier .env.")
        st.stop()

    with st.spinner("📖 Extraction du PDF avec LlamaCloud (Agentic)..."):
        page_dicts = _parse_with_llamacloud(pdf_path, api_key)

    # Reconstruction linéaire propre avec balises de pages pour le constructeur d'arbre
    markdown_chunks = []
    for page_data in page_dicts:
        page_num = page_data["page"]
        markdown_chunks.append(f"--- Page {page_num} ---")
        markdown_chunks.append(page_data.get("md", ""))
    markdown_content = "\n".join(markdown_chunks)
    
    with st.spinner("🌲 Structuration de l'arbre documentaire..."):
        tree = _build_pure_text_tree(markdown_content)
    print("="*60 + "\n" + str(tree) + "\n" + "="*60 + "\n")
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

# ── Initialisation du Session State ───────────────────────────────────────────
for key, default in [
    ("tree", None),
    ("messages", []),
    ("pdf_name", ""),
    ("processing", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Donnez-moi la suite de votre app.py (la mise en page UI, la sidebar et la chat zone) 
# et je l'adapterai directement à cette structure propre !
import io
import sys
import os
import tempfile
import streamlit as st

# Les variables par défaut de l'application
DEFAULT_MODEL = "qwen2.5:3b"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model = st.text_input("Modèle", value=DEFAULT_MODEL, help="Nom du modèle local dans Ollama")

    # Vérification de l'état des dépendances de manière simplifiée
    ollama_ok = check_ollama(model)
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
        # ── Sidebar: Charger un nouveau document ──────────────────────────────────────
        if st.button("📂 Charger un nouveau document", use_container_width=True):
            for key in ["tree", "messages", "pdf_name"]:
                st.session_state[key] = (
                    None if key == "tree"
                    else [] if key == "messages"
                    else ""
                )
            st.rerun()
        st.divider()
        
        st.markdown("### 📚 Document Actif")
        st.caption(f"**{st.session_state.pdf_name}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Sections", count_nodes(st.session_state.tree))
        with c2:
            total_pages = get_total_pages(st.session_state.tree) if st.session_state.tree else "?"
            st.metric("Pages", total_pages)

        with st.expander("🌲 Arbre hiérarchique", expanded=False):
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            print_tree(st.session_state.tree)
            sys.stdout = old_stdout
            st.code(buf.getvalue(), language="text")

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.tree:
    # ── Landing / Zone de Téléversement ──────────────────────────────────────
    st.markdown("<div style='height: 6vh'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<p class="hero-title">Graph Paper AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Recherche et analyse de papiers scientifiques sans vecteurs.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Glissez un fichier PDF ici pour commencer",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if not api_key_ok:
                st.error("Veuillez configurer la clé LLAMA_CLOUD_API_KEY dans votre fichier .env.")
            elif not ollama_ok:
                st.error(f"Impossible de joindre Ollama. Assurez-vous que le modèle '{model}' est lancé.")
            else:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # ── Handler de téléchargement et traitement ────────────────────────────
                try:
                    progress = st.progress(0, text="Initialisation…")
                    progress.progress(20, text="Parsing du document avec LlamaCloud (Agentic)...")
                    
                    # Appel simplifié sans l'ancien traitement de captures d'images
                    tree = run_pipeline(tmp_path)
                    
                    progress.progress(100, text="Prêt !")
                    st.session_state.tree = tree
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Échec du traitement du document : {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Grille d'exemples d'utilisation
        cols = st.columns(3)
        hints = [
            ("📝", "Questions Textuelles", "Définitions, méthodologie, conclusions sémantiques"),
            ("📊", "Tableaux & Données", "Analyse des données converties en Markdown natif"),
            ("🔍", "Zéro Vecteur", "Aiguillage déterministe via la structure du document"),
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
    # ── Interface de Discussion (Chat Area) ────────────────────────────────────
    if not st.session_state.messages:
        node_count = count_nodes(st.session_state.tree)
        st.markdown(
            f"""<div style='border:1px solid #e2e8f0;
            border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem'>
            <div style='color:#34d399;font-weight:600;margin-bottom:0.4rem'>
            ✅ {st.session_state.pdf_name} est chargé avec succès</div>
            <div style='color:#6b7a99;font-size:0.875rem'>
            L'arbre sémantique contient <b style='color:#4f8ef7'>{node_count} sections</b> exploitables. Posez votre question.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Affichage de l'historique des messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("thinking"):
                with st.expander("💭 Processus de réflexion (Routage)", expanded=False):
                    st.info(msg["thinking"])
            if msg.get("sources"):
                with st.expander("📍 Sections consultées", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # Zone de saisie utilisateur
    if prompt := st.chat_input(
        "Posez une question sur le texte ou les tableaux du document...",
        disabled=not ollama_ok,
    ):
        if not ollama_ok:
            st.error("Ollama n'est pas connecté.")
        else:
            # Enregistrement et affichage immédiat de la question
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyse de la structure et génération de la réponse..."):
                    # Exécution de votre nouveau pipeline corrigé
                    result = vectorless_rag_no_loss(
                        query=prompt,
                        tree=st.session_state.tree,
                        model=model
                    )
                
                # Affichage de la réponse finale
                st.markdown(result["answer"])
                
                # Affichage transparent du routage du Vectorless RAG
                with st.expander("💭 Processus de réflexion (Routage)", expanded=False):
                    st.info(result["thinking"])
                    
                if result.get("retrieved_sections"):
                    with st.expander("📍 Sections consultées", expanded=False):
                        for s in result["retrieved_sections"]:
                            st.markdown(f"- {s}")

            # Persistance dans l'historique de session
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "thinking": result["thinking"],
                "sources": result.get("retrieved_sections", []),
            })
