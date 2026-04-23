"""
Embeddings — Influencer Shortlist Agent
Centralised dense + sparse embedding interface.

Dense:  sentence-transformers/all-MiniLM-L6-v2 (384-d, free, local)
Sparse: FastEmbed Qdrant/bm25 (token-id sparse vectors with IDF computed by Qdrant)

Both models lazy-load on first call. The first load downloads weights into
~/.cache/huggingface/ (dense) and ~/.cache/fastembed/ (sparse). Subsequent
calls hit the on-disk cache. Run scripts/warm_cache.py to trigger the
downloads with visible progress before any latency-sensitive work runs.

The dense model is the same one referenced in CLAUDE.md as the project default —
do not change it without updating the Qdrant collection's vector size config.
"""

import logging
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"

_dense_model = None
_sparse_model = None


def get_dense_model():
    """Return a singleton SentenceTransformer for dense embeddings."""
    global _dense_model
    if _dense_model is not None:
        return _dense_model

    from sentence_transformers import SentenceTransformer
    logger.info("Loading dense model: %s", DENSE_MODEL_NAME)
    _dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    return _dense_model


def get_sparse_model():
    """Return a singleton FastEmbed BM25 sparse embedder."""
    global _sparse_model
    if _sparse_model is not None:
        return _sparse_model

    from fastembed import SparseTextEmbedding
    logger.info("Loading sparse model: %s", SPARSE_MODEL_NAME)
    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


def embed_dense_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts to 384-d float vectors. Returns plain Python lists."""
    if not texts:
        return []
    model = get_dense_model()
    arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return arr.tolist()


def embed_sparse_batch(texts: list[str]) -> list[dict]:
    """
    Encode a batch of texts to BM25 sparse vectors.
    Returns: [{"indices": [int...], "values": [float...]}, ...]
    """
    if not texts:
        return []
    model = get_sparse_model()
    out = []
    for emb in model.embed(texts):
        out.append({
            "indices": emb.indices.tolist(),
            "values": emb.values.tolist(),
        })
    return out


def embed_dense(text: str) -> list[float]:
    """Single-text dense embedding."""
    return embed_dense_batch([text])[0]


def embed_sparse(text: str) -> dict:
    """Single-text sparse embedding."""
    return embed_sparse_batch([text])[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embeddings smoke test")
    parser.add_argument("--warm", action="store_true", help="Pre-download both models.")
    parser.add_argument("--demo", action="store_true", help="Embed a sample sentence and print shape info.")
    args = parser.parse_args()

    if args.warm:
        print(f"Pre-downloading dense model ({DENSE_MODEL_NAME}) ~90MB...")
        get_dense_model()
        print("  Dense model ready.\n")
        print(f"Pre-downloading sparse model ({SPARSE_MODEL_NAME}) ~small...")
        get_sparse_model()
        print("  Sparse model ready.")
    elif args.demo:
        text = "Clean skincare creator with ingredient-conscious voice"
        dense = embed_dense(text)
        sparse = embed_sparse(text)
        print(f"Text: {text!r}")
        print(f"Dense:  dim={len(dense)}  first 3 values={dense[:3]}")
        print(f"Sparse: nnz={len(sparse['indices'])}  first 3 indices={sparse['indices'][:3]}")
    else:
        print("Use --warm or --demo.")
