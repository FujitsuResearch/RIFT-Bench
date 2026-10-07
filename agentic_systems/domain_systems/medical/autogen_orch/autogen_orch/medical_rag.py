import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_PATH = DATA_DIR / "medquad_documents.json"
EMBED_PATH = DATA_DIR / "medquad_embeddings.npy"
MODEL_NAME = str(DATA_DIR / "models" / "facebook_contriever")

_MODEL = None
_DOCS = None
_EMBED = None
_DEVICE = None


def _resolve_device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    preferred = os.getenv("MEDICAL_RAG_DEVICE", "").strip().lower()
    if preferred:
        _DEVICE = preferred
        return _DEVICE
    try:
        import torch

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def _load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME, device=_resolve_device())
    return _MODEL


def _reset_model_to_cpu() -> SentenceTransformer:
    global _MODEL, _DEVICE
    _MODEL = None
    _DEVICE = "cpu"
    _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
    return _MODEL


def _encode_with_fallback(texts: list[str], **kwargs) -> np.ndarray:
    model = _load_model()
    try:
        return model.encode(texts, **kwargs)
    except RuntimeError as e:
        # Some environments expose CUDA but do not support the model kernels on the GPU.
        if "cuda" not in str(e).lower():
            raise
        model = _reset_model_to_cpu()
        return model.encode(texts, **kwargs)


def _load_db() -> tuple[list[dict], np.ndarray]:
    global _DOCS, _EMBED
    if _DOCS is None or _EMBED is None:
        if not DOCS_PATH.exists():
            raise FileNotFoundError(
                "Missing MedQuAD documents file. Run "
                "`python use_cases/domain_systems/medical/download_medquad_data.py` first."
            )
        if not EMBED_PATH.exists():
            _build_embeddings_from_docs()
        with DOCS_PATH.open("r", encoding="utf-8") as f:
            _DOCS = json.load(f)
        _EMBED = np.load(EMBED_PATH).astype("float32")
    return _DOCS, _EMBED


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def _build_embeddings_from_docs(batch_size: int = 32) -> None:
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    if not docs:
        raise RuntimeError("No MedQuAD records found in medquad_documents.json")
    model = _load_model()
    embeddings = _encode_with_fallback(
        [d["text"] for d in docs],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    embeddings = _normalize(embeddings)
    np.save(EMBED_PATH, embeddings)


def retrieve(query: str, top_k: int = 5, topic: str = "", question_type: str = "") -> list[dict]:
    docs, emb = _load_db()
    model = _load_model()

    filtered_idxs: list[int] = []
    for i, d in enumerate(docs):
        md = d.get("metadata", {})
        topic_ok = (not topic) or (str(md.get("topic", "")).lower() == topic.lower())
        qtype_ok = (not question_type) or (str(md.get("question_type", "")).lower() == question_type.lower())
        if topic_ok and qtype_ok:
            filtered_idxs.append(i)
    if not filtered_idxs:
        filtered_idxs = list(range(len(docs)))

    query_emb = _encode_with_fallback([query], convert_to_numpy=True).astype("float32")
    query_emb = _normalize(query_emb)
    cand_emb = emb[filtered_idxs]
    scores = cand_emb @ query_emb[0]
    ranked_local = np.argsort(scores)[-max(1, top_k):][::-1]

    out: list[dict] = []
    for rank, local_idx in enumerate(ranked_local, start=1):
        global_idx = filtered_idxs[int(local_idx)]
        d = docs[global_idx]
        out.append(
            {
                "rank": rank,
                "id": d.get("id"),
                "source": d.get("source", "MedQuAD"),
                "domain": d.get("domain", "medical"),
                "question": d.get("question", ""),
                "answer": d.get("answer", ""),
                "metadata": d.get("metadata", {}),
                "score": float(scores[int(local_idx)]),
            }
        )
    return out
