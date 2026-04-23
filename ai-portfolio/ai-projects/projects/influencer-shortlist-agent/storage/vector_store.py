"""
Qdrant Client — Influencer Shortlist Agent
Hybrid (dense + sparse) vector store for post chunk embeddings.

Why Qdrant (not Chroma)
------------------------
- Native hybrid search via prefetch + RRF/DBSF fusion (one query, both signals)
- Sparse vector support (BM25 via FastEmbed) without a second service
- Filtered search at the index level — `creator_id IN [...]` runs inside Qdrant,
  not after retrieval, so we can cap candidates without losing recall

Mode switch
-----------
QDRANT_MODE=embedded (default) → QdrantClient(path=QDRANT_PATH); in-process,
                                 persists to disk, no daemon. Single-process lock.
QDRANT_MODE=local              → connects to QDRANT_URL (http://localhost:6333)
                                 for `docker run -p 6333:6333 qdrant/qdrant`
QDRANT_MODE=cloud              → connects to QDRANT_URL with QDRANT_API_KEY
                                 (Qdrant Cloud free tier endpoint)

Collection layout
-----------------
Named vectors per point:
  "dense"  — 384-d float32 from sentence-transformers/all-MiniLM-L6-v2
  "sparse" — BM25 via FastEmbed Qdrant/bm25

Each point is one chunk = 5 consecutive posts from one creator.
Payload:
  creator_id      int    — required for hard-filter intersection at query time
  post_ids        list   — referenced by rationale agent to cite source posts
  platform        str
  posted_at_min   ISO    — earliest post timestamp in the chunk
  posted_at_max   ISO    — latest post timestamp in the chunk
  categories      list   — primary_categories of the creator (denormalised)
  text            str    — concatenated chunk text (used by reranker)
"""

import atexit
import logging
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# Qdrant SDK imports — kept inside functions where possible so importing this
# module doesn't force the (large) qdrant-client import on consumers that only
# need the constants.
from qdrant_client import QdrantClient, models

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim

_COLLECTION = os.getenv("QDRANT_COLLECTION", "influencer_posts")
_MODE = os.getenv("QDRANT_MODE", "embedded").lower()
_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_API_KEY = os.getenv("QDRANT_API_KEY") or None
_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Return a process-singleton QdrantClient. Honours QDRANT_MODE."""
    global _client
    if _client is not None:
        return _client

    if _MODE == "cloud":
        if not _API_KEY:
            raise ValueError("QDRANT_MODE=cloud requires QDRANT_API_KEY in env.")
        logger.info("Connecting to Qdrant Cloud at %s", _URL)
        _client = QdrantClient(url=_URL, api_key=_API_KEY)
    elif _MODE == "local":
        logger.info("Connecting to local Qdrant server at %s", _URL)
        _client = QdrantClient(url=_URL)
    else:
        # embedded — in-process, persisted to disk. Resolves QDRANT_PATH relative
        # to the current working dir; the seed script runs from project root,
        # so by default the data dir sits alongside pipeline.py.
        os.makedirs(_PATH, exist_ok=True)
        logger.info("Using embedded Qdrant at %s", os.path.abspath(_PATH))
        _client = QdrantClient(path=_PATH)

    # Close the client cleanly at interpreter exit. Without this, QdrantClient's
    # __del__ runs late in shutdown and trips an ImportError on sys.meta_path —
    # cosmetic, but noisy on every script invocation.
    atexit.register(_close_client)
    return _client


def _close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def collection_name() -> str:
    return _COLLECTION


def ensure_collection() -> None:
    """
    Create the influencer_posts collection if absent, with named dense + sparse vectors.
    Idempotent — safe to call on every startup.
    """
    client = get_client()
    if client.collection_exists(_COLLECTION):
        return

    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # IDF computed by Qdrant — required for BM25
            ),
        },
    )

    # Payload indexes are server-only — embedded Qdrant ignores them and emits
    # a warning. With 100 creators × ~5 chunks = 500 points the linear scan is
    # instant anyway, so we only bother on remote modes.
    if _MODE in ("local", "cloud"):
        client.create_payload_index(_COLLECTION, "creator_id", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(_COLLECTION, "platform", models.PayloadSchemaType.KEYWORD)

    logger.info("Created Qdrant collection: %s", _COLLECTION)


def upsert_chunks(points: list[dict]) -> int:
    """
    Upsert chunk points. Each point dict must contain:
      id              int — globally unique chunk id
      dense_vector    list[float] of length 384
      sparse_indices  list[int]
      sparse_values   list[float]
      payload         dict (must include creator_id, post_ids, platform, ...)
    Returns number of points upserted.
    """
    if not points:
        return 0

    client = get_client()
    point_structs = [
        models.PointStruct(
            id=p["id"],
            vector={
                DENSE_VECTOR_NAME: p["dense_vector"],
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=p["sparse_indices"],
                    values=p["sparse_values"],
                ),
            },
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(_COLLECTION, points=point_structs, wait=True)
    return len(point_structs)


def hybrid_search(
    dense_query: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    creator_id_filter: Optional[list[int]] = None,
    limit: int = 200,
    prefetch_limit: int = 400,
) -> list[dict]:
    """
    Run a single hybrid query: dense + sparse prefetch, fused by Reciprocal Rank Fusion.

    creator_id_filter is applied on BOTH prefetch branches AND the final fusion query
    so candidates outside the hard-filter pool can never enter the result set, even
    via one branch that didn't see them.

    Returns a list of dicts: {id, score, payload}.
    """
    client = get_client()

    qdrant_filter = None
    if creator_id_filter:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="creator_id",
                    match=models.MatchAny(any=creator_id_filter),
                ),
            ],
        )

    response = client.query_points(
        collection_name=_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                filter=qdrant_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VECTOR_NAME,
                filter=qdrant_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    )

    return [
        {"id": pt.id, "score": pt.score, "payload": pt.payload}
        for pt in response.points
    ]


def reset_collection() -> None:
    """Drop and recreate the collection. Used by seed scripts with --reset."""
    client = get_client()
    if client.collection_exists(_COLLECTION):
        client.delete_collection(_COLLECTION)
        logger.info("Dropped Qdrant collection: %s", _COLLECTION)
    ensure_collection()


def sample_points(limit: int = 10, with_text_chars: int = 200) -> list[dict]:
    """
    Scroll a few points from the collection for inspection. Returns lightweight
    dicts (id, creator_id, platform, post_ids, snippet of text). Vector data
    is omitted — too noisy to print.
    """
    client = get_client()
    if not client.collection_exists(_COLLECTION):
        return []

    points, _ = client.scroll(
        collection_name=_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    out = []
    for p in points:
        payload = p.payload or {}
        text = (payload.get("text") or "")
        out.append({
            "id": p.id,
            "creator_id": payload.get("creator_id"),
            "platform": payload.get("platform"),
            "post_ids": payload.get("post_ids"),
            "categories": payload.get("categories"),
            "text_snippet": text[:with_text_chars] + ("…" if len(text) > with_text_chars else ""),
        })
    return out


def query_text(query: str, limit: int = 5) -> list[dict]:
    """
    Run a hybrid search with the given text and print matched chunks.
    Imports embeddings lazily so vector_store.py stays usable when only metadata
    inspection is needed.
    """
    from storage.embeddings import embed_dense, embed_sparse

    dense = embed_dense(query)
    sparse = embed_sparse(query)
    hits = hybrid_search(
        dense_query=dense,
        sparse_indices=sparse["indices"],
        sparse_values=sparse["values"],
        creator_id_filter=None,
        limit=limit,
        prefetch_limit=limit * 4,
    )
    out = []
    for h in hits:
        payload = h.get("payload") or {}
        text = (payload.get("text") or "")
        out.append({
            "id": h["id"],
            "score": round(h["score"], 4),
            "creator_id": payload.get("creator_id"),
            "platform": payload.get("platform"),
            "categories": payload.get("categories"),
            "text_snippet": text[:240] + ("…" if len(text) > 240 else ""),
        })
    return out


def collection_info() -> dict:
    """Light status snapshot for the admin UI / smoke test."""
    client = get_client()
    if not client.collection_exists(_COLLECTION):
        return {"exists": False, "name": _COLLECTION, "mode": _MODE, "url": _URL}
    info = client.get_collection(_COLLECTION)
    return {
        "exists": True,
        "name": _COLLECTION,
        "mode": _MODE,
        "url": _URL if _MODE != "embedded" else None,
        "path": os.path.abspath(_PATH) if _MODE == "embedded" else None,
        "points_count": info.points_count,
        "status": str(info.status),
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys as _sys

    # When run as a script, sys.path[0] is storage/, not the project root, so
    # `from storage.embeddings import ...` (used by --query) can't resolve.
    # Add the project root before any sibling-package imports happen.
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_here)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    parser = argparse.ArgumentParser(description="Qdrant client smoke test + inspection")
    parser.add_argument("--info", action="store_true", help="Print collection info (count, status).")
    parser.add_argument("--init", action="store_true", help="Ensure collection exists.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection. Destructive.")
    parser.add_argument("--sample", type=int, metavar="N", help="Print N random points with payload snippets.")
    parser.add_argument("--query", type=str, metavar="TEXT",
                        help="Run a hybrid search with this query string and print matches.")
    parser.add_argument("--limit", type=int, default=5, help="Result count for --query (default 5).")
    args = parser.parse_args()

    if args.reset:
        confirm = input(f"This will DROP collection '{_COLLECTION}'. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            reset_collection()
            print("Collection reset.")
        else:
            print("Aborted.")
    elif args.init:
        ensure_collection()
        print("Collection ready.")
    elif args.info:
        print(json.dumps(collection_info(), indent=2, default=str))
    elif args.sample is not None:
        rows = sample_points(limit=args.sample)
        print(f"Sampled {len(rows)} points from '{_COLLECTION}':\n")
        for r in rows:
            print(f"  id={r['id']:>10}  creator_id={r['creator_id']}  "
                  f"platform={r['platform']}  posts={r['post_ids']}")
            print(f"    categories: {r['categories']}")
            print(f"    text:       {r['text_snippet']}\n")
    elif args.query:
        rows = query_text(args.query, limit=args.limit)
        print(f"Top {len(rows)} hybrid matches for: {args.query!r}\n")
        for r in rows:
            print(f"  score={r['score']:.4f}  id={r['id']}  creator_id={r['creator_id']}  "
                  f"platform={r['platform']}")
            print(f"    categories: {r['categories']}")
            print(f"    text:       {r['text_snippet']}\n")
    else:
        print("Use --info, --init, --reset, --sample N, or --query TEXT.")
