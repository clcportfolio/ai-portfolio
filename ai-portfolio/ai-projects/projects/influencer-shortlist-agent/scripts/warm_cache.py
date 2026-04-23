"""
warm_cache.py — Pre-download embedding model weights.

Runs once before any latency-sensitive work (seed script, first pipeline run,
Streamlit cold boot). Surfaces the ~90MB dense model download as a visible
status message instead of a mysterious 60-second pause mid-run, and validates
the models are fetchable before any LLM tokens are spent.

What this caches
----------------
1. sentence-transformers/all-MiniLM-L6-v2  (~90MB, dense embeddings)
   → ~/.cache/huggingface/hub/
2. Qdrant/bm25                              (~few MB, sparse BM25)
   → ~/.cache/fastembed/

Subsequent uses load from disk — no network call.

Usage:
    python scripts/warm_cache.py
"""

import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


def main():
    from storage.embeddings import (
        DENSE_MODEL_NAME,
        SPARSE_MODEL_NAME,
        get_dense_model,
        get_sparse_model,
        embed_dense,
        embed_sparse,
    )

    print("=" * 60)
    print("Warming embedding model caches")
    print("=" * 60)

    print(f"\n[1/2] Dense model: {DENSE_MODEL_NAME}")
    print("      First-time download is ~90MB. Subsequent loads are instant.")
    t0 = time.time()
    get_dense_model()
    sample = embed_dense("warmup")
    print(f"      Loaded in {time.time() - t0:.1f}s. Vector dim={len(sample)}.")

    print(f"\n[2/2] Sparse model: {SPARSE_MODEL_NAME}")
    print("      Small model — usually quick to fetch.")
    t0 = time.time()
    get_sparse_model()
    sample = embed_sparse("warmup")
    print(f"      Loaded in {time.time() - t0:.1f}s. Sparse nnz={len(sample['indices'])}.")

    print("\nCaches warm. Future embedding calls hit the disk cache.")


if __name__ == "__main__":
    main()
