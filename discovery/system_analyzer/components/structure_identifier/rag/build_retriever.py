import os
import re
from typing import Dict, List

try:  # non-deprecated path (langchain >= 0.2 split); falls back if pkg not installed
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from .build_index import _safe_int_env


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercase identifier-like tokens (letters/digits/underscore)."""
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower()))


def _doc_key(doc) -> tuple:
    """Build a stable dedup key for a document from its source, line range, and text prefix."""
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("source"),
        meta.get("line_start"),
        meta.get("line_end"),
        (getattr(doc, "page_content", "") or "")[:240],
    )


class SimpleHybridRetriever:
    """
    Hybrid retriever with:
    - deeper candidate pools from BM25 + vector
    - reciprocal-rank-fusion (RRF) instead of naive concatenation
    - lightweight lexical rerank tie-breaker for code terms
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vec,
        k: int = 8,
        candidate_k: int = 24,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vec_weight: float = 1.0,
    ):
        """Store the BM25 and vector retrievers plus fusion settings (final k, candidate pool size, RRF constant, weights)."""
        self.bm25 = bm25
        self.vec = vec
        self.k = max(1, int(k))
        self.candidate_k = max(self.k, int(candidate_k))
        self.rrf_k = max(1, int(rrf_k))
        self.bm25_weight = float(bm25_weight)
        self.vec_weight = float(vec_weight)

    def _lexical_overlap(self, query: str, doc_text: str) -> float:
        """Return the fraction of the query's tokens that also appear in the document text (0..1)."""
        q_terms = _tokenize(query)
        if not q_terms:
            return 0.0
        d_terms = _tokenize(doc_text)
        if not d_terms:
            return 0.0
        return len(q_terms & d_terms) / max(1.0, float(len(q_terms)))

    def _get_ranked_lists(self, query: str):
        """Query BM25 and the vector store for their candidate document lists; each side falls back to [] on error."""
        # Keep deeper pools for better fusion quality.
        try:
            self.bm25.k = self.candidate_k
        except Exception:
            pass
        try:
            bm_docs = self.bm25.invoke(query) or []
        except Exception:
            bm_docs = []

        try:
            vec_docs = self.vec.invoke(query) or []
        except Exception:
            try:
                vec_docs = self.vec.get_relevant_documents(query) or []
            except Exception:
                vec_docs = []

        return list(bm_docs), list(vec_docs)

    def get_relevant_documents(self, query: str):
        """Fuse the BM25 and vector candidate lists via RRF (lexical-overlap tie-break) and return the top-k documents."""
        bm_docs, vec_docs = self._get_ranked_lists(query)

        by_key: Dict[tuple, object] = {}
        fused_scores: Dict[tuple, float] = {}

        for rank, doc in enumerate(bm_docs, start=1):
            key = _doc_key(doc)
            by_key.setdefault(key, doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )
        for rank, doc in enumerate(vec_docs, start=1):
            key = _doc_key(doc)
            by_key.setdefault(key, doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (
                self.vec_weight / (self.rrf_k + rank)
            )

        ranked = sorted(
            by_key.items(),
            key=lambda kv: (
                fused_scores.get(kv[0], 0.0),
                self._lexical_overlap(query, getattr(kv[1], "page_content", "") or ""),
            ),
            reverse=True,
        )

        out: List[object] = []
        for key, doc in ranked[: self.k]:
            try:
                doc.metadata = dict(getattr(doc, "metadata", {}) or {})
                doc.metadata["_hybrid_score"] = round(float(fused_scores.get(key, 0.0)), 6)
            except Exception:
                pass
            out.append(doc)
        return out

    def invoke(self, query: str):
        """Alias for get_relevant_documents, matching the LangChain retriever interface used by call-sites."""
        return self.get_relevant_documents(query)


def build_bm25(chunks, k: int = 8):
    """Build a BM25 retriever over the given chunks, returning at most k results per query."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = max(1, int(k))
    return bm25


def load_retriever(persist_dir: str, chunks_for_bm25=None):
    """Load the persisted FAISS index and return a hybrid BM25+vector retriever (or vector-only if BM25 can't be built)."""
    final_k = _safe_int_env("RAG_TOP_K", 8)
    candidate_k = _safe_int_env("RAG_CANDIDATE_K", 24, minimum=final_k)
    rrf_k = _safe_int_env("RAG_RRF_K", 60)
    model_name = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    vec = vs.as_retriever(search_kwargs={"k": candidate_k})

    if chunks_for_bm25 is None:
        # Reuse persisted FAISS documents so call-sites remain unchanged.
        try:
            docstore = getattr(vs, "docstore", None)
            docs_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
            chunks_for_bm25 = list(docs_dict.values()) if isinstance(docs_dict, dict) else []
        except Exception:
            chunks_for_bm25 = []

    if not chunks_for_bm25:
        return vec  # fallback when BM25 cannot be prepared

    bm25 = build_bm25(chunks_for_bm25, k=candidate_k)
    return SimpleHybridRetriever(
        bm25=bm25,
        vec=vec,
        k=final_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "1.0")),
        vec_weight=float(os.getenv("RAG_VEC_WEIGHT", "1.0")),
    )
