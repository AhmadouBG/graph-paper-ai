import json
import re

import ollama
from rank_bm25 import BM25Okapi


class FastTreeRetriever:
    """
    3-pass hybrid router for Vectorless RAG:
      - Pass A (Regex): Instant detection (<1ms) for Tables / Figures.
      - Pass B (BM25): Fast pre-filtering of the top 6-8 candidate nodes.
      - Pass C (SLM): Semantic re-ranking by `qwen2.5:1.5b-instruct-q4_K_M`
                         for intelligent final selection.
    """

    def __init__(self, tree: list, slm_model: str = "qwen2.5:1.5b-instruct-q4_K_M"):
        self.slm_model = slm_model
        self.flattened_nodes = []
        self._flatten(tree)

        # Pre-compilation of the global regex for Pass A
        self.visual_routing_pattern = re.compile(
            r'(?:(table)|(fig\.?|figure))\s+([a-zA-Z0-9_]+)', re.IGNORECASE
        )

        # Unique initialization of the BM25 engine (Pass B)
        self.corpus_tokens = []
        if self.flattened_nodes:
            for n in self.flattened_nodes:
                # Title weighting integrated from indexing (title duplication)
                node_text = f"{n['title']} {n['title']} {n.get('content', '')}"
                self.corpus_tokens.append(self.tokenize(node_text))
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

        # Mots-clés pour la détection d'intention comparative
        self.COMPARATIVE_KEYWORDS = {
            "best", "worst", "compare", "comparison", "overall", "performance",
            "conclusion", "versus", "vs", "summary", "highest", "lowest", "rank",
            "ranking", "evaluate", "evaluation", "which", "aim", "goal", "purpose",
            "objective"
        }
        self.SYNTHESIS_SECTION_KEYWORDS = {
            "performance", "introduction", "analysis", "conclusion", "discussion", "results",
            "result", "comparison", "summary", "evaluation", "overview"
        }

        self.COMPARATIVE_PHRASE_MIN_HITS = 2

    def _flatten(self, nodes: list[dict]) -> None:
        """Flattens the node tree recursively."""
        for n in nodes:
            self.flattened_nodes.append(n)
            if n.get("nodes"):
                self._flatten(n["nodes"])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Cleans and tokenizes text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def find_visual_element_by_regex(self, query: str, top_k: int = 3) -> list[str]:
        """Optimized Pass A: Deterministic detection by Regex."""
        cleaned_query = re.sub(r'\s+', ' ', query.lower().strip())
        match = self.visual_routing_pattern.search(cleaned_query)

        if not match:
            return []

        is_table = bool(match.group(1))
        is_figure = bool(match.group(2))
        target_number = match.group(3)

        matched_indices = []
        visual_pattern = re.compile(
            rf'(?:table|fig\.?|figure)\s+{re.escape(target_number)}(?:\b|[^a-z0-9_]|<)',
            re.IGNORECASE,
        )

        for idx, n in enumerate(self.flattened_nodes):
            content_to_scan = f"{n['title']} {n.get('content', '')}".lower()
            if visual_pattern.search(content_to_scan):
                matched_indices.append(idx)

        if not matched_indices:
            return []

        final_node_ids = []
        seen_ids = set()

        for idx in matched_indices:
            current_node = self.flattened_nodes[idx]

            if is_figure:
                if current_node["node_id"] not in seen_ids:
                    seen_ids.add(current_node["node_id"])
                    final_node_ids.append(current_node["node_id"])
                if idx > 0:
                    prev_node = self.flattened_nodes[idx - 1]
                    if prev_node["node_id"] not in seen_ids:
                        seen_ids.add(prev_node["node_id"])
                        final_node_ids.append(prev_node["node_id"])

            elif is_table:
                if current_node["node_id"] not in seen_ids:
                    seen_ids.add(current_node["node_id"])
                    final_node_ids.append(current_node["node_id"])

                if idx < len(self.flattened_nodes) - 1:
                    next_node = self.flattened_nodes[idx + 1]
                    if next_node["node_id"] not in seen_ids:
                        seen_ids.add(next_node["node_id"])
                        final_node_ids.append(next_node["node_id"])

        return final_node_ids[:top_k]

    def _bm25_candidate_filter(self, query: str, max_candidates: int = 5) -> list[dict]:
        """Passe B : BM25 filtrers nodes pour créer une sélection restreinte."""
        if not self.bm25:
            return self.flattened_nodes[:max_candidates]

        query_tokens = self.tokenize(query)
        scores = list(self.bm25.get_scores(query_tokens))
        comparative_hits = sum(1 for kw in query_tokens if kw in self.COMPARATIVE_KEYWORDS)
        is_comparative = comparative_hits >= self.COMPARATIVE_PHRASE_MIN_HITS

        node_scores = []
        for idx, score in enumerate(scores):
            n = self.flattened_nodes[idx]
            title_lower = n["title"].lower()

            if is_comparative:
                if any(s_kw in title_lower for s_kw in self.SYNTHESIS_SECTION_KEYWORDS):
                    score *= 2.0 if score > 0 else 3.0

            node_scores.append({"node": n, "score": score})

        node_scores.sort(key=lambda x: x["score"], reverse=True)

        candidates = [item["node"] for item in node_scores[:max_candidates] if item["score"] > 0]
        return candidates if candidates else self.flattened_nodes[:max_candidates]

    def _slm_rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> dict:
        """
        Passe C : Use qwen2.5:1.5b-instruct-q4_K_M to select the best
        nodes among the filtered candidates.
        """
        candidate_catalog = []
        valid_ids = set()

        for n in candidates:
            nid = n["node_id"]
            valid_ids.add(nid)
            preview = n.get("content", "").strip()[:400]
            candidate_catalog.append({
                "node_id": nid,
                "title": n["title"],
                "content_preview": preview if preview else "(no text content)"
            })

        prompt = (
            f"You are a precise RAG routing agent. Select at most {top_k} node IDs.\n"
            f'User Question: "{query}"\n'
            f"Available Candidate Nodes:\n"
            f"{json.dumps(candidate_catalog, ensure_ascii=False, indent=2)}\n"
            f"Instructions:\n"
            f"1. Order node_list from MOST to LEAST relevant.\n"
            f"2. Return ONLY a single valid JSON object.\n"
            f'JSON Format: {{"thinking": "Reasoning", "node_list": ["id1", "id2"]}}'
        )

        try:
            response = ollama.chat(
                model=self.slm_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.0,
                    "num_ctx": 1524,
                    "num_predict": 128,
                    "keep_alive": "10m",
                }
            )

            raw_content = response["message"]["content"]
            match = re.search(r'\{.*\}', raw_content, re.DOTALL)

            if match:
                parsed = json.loads(match.group(0))
                raw_nodes = parsed.get("node_list", [])

                # Validation anti-hallucination : ne conserver que les IDs valides
                filtered_ids = [nid for nid in raw_nodes if nid in valid_ids]

                if filtered_ids:
                    thinking = parsed.get("thinking", "SLM semantic routing.")
                    return {
                        "thinking": f"🤖 [{self.slm_model}] {thinking}",
                        "node_list": filtered_ids[:top_k]
                    }

        except Exception as e:
            print(f"⚠️ Warning SLM Reranker ({self.slm_model}): {e}")

        # Fallback aux candidats BM25
        fallback_ids = [n["node_id"] for n in candidates[:top_k]]
        fallback_titles = [n["title"] for n in candidates[:top_k]]
        return {
            "thinking": f"🔍 [BM25 Fallback] Selected: {', '.join(fallback_titles)}",
            "node_list": fallback_ids
        }

    def search(self, query: str, top_k: int = 3) -> dict:
        """Point d'entrée principal du routeur hybride 3-pass."""
        if not self.flattened_nodes:
            return {"thinking": "L'arbre documentaire est vide.", "node_list": ["0000"]}

        # 1. Passe A : Détection déterministe Regex (<1ms)
        regex_matched_ids = self.find_visual_element_by_regex(query, top_k)
        if regex_matched_ids:
            titles = [n["title"] for n in self.flattened_nodes if n["node_id"] in regex_matched_ids]
            return {
                "thinking": f"🎯 [Regex] Composant visuel trouvé dans : {', '.join(titles)}",
                "node_list": regex_matched_ids
            }

        # 2. Passe B : Pré-filtrage rapide des candidats avec BM25
        candidates = self._bm25_candidate_filter(query, max_candidates=6)

        # 3. Passe C : Sélection/Re-ranking sémantique par le SLM Qwen 1.5B
        return self._slm_rerank(query, candidates, top_k=top_k)


# --- FONCTION DE COMPATIBILITÉ POUR LE PIPELINE ET DEEPEVAL ---
def llm_tree_search_ollama(
    query: str,
    tree: list,
    model: str = "qwen2.5:1.5b-instruct-q4_K_M",
    top_k: int = 3
) -> dict:
    """
    Fonction wrapper standard du pipeline RAG.
    """
    retriever = FastTreeRetriever(tree, slm_model=model)
    return retriever.search(query, top_k=top_k)
