"""
reranker.py — Influencer Shortlist Agent

Reranks the top 200 retrieval candidates down to 50 using a cross-encoder
that scores (query, document) pairs jointly — much sharper than the
hybrid-retrieval bi-encoder, which scores them independently.

Provider switch
---------------
Cohere `rerank-english-v3.0` (preferred) — fast, accurate, $1/1k requests.
BAAI `bge-reranker-base` via local CrossEncoder (fallback) — free, runs
on CPU, ~80MB model, ~3-5x slower per pair.

The choice is automatic: if COHERE_API_KEY is set, we use Cohere; otherwise
we fall back. Both produce a relevance score in roughly the same range, so
the rest of the pipeline doesn't care which one ran.

Why rerank at all
-----------------
Bi-encoders (the dense + sparse retrievers) embed the query and the document
SEPARATELY and compare vectors. They're fast enough to run over millions of
documents but they miss subtleties — "warm voice for Gen-Z parents" might
match a chunk about parenting tone that has nothing to do with warmth, simply
because both vectors live in similar regions of embedding space.

Cross-encoders feed (query, document) into a transformer TOGETHER and read
out a single relevance score. They catch alignments bi-encoders miss but
can't be precomputed — we only run them on the top 200 the bi-encoder pre-filtered.

Query construction
------------------
The brief has multiple soft constraints — voice, topic, audience. We
concatenate them into a single representative query for the reranker rather
than reranking once per constraint, because cross-encoder calls are the
expensive step and a single fused query produces stable rankings.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

AGENT_NAME = "reranker"

TOP_N_AFTER_RERANK = 50
COHERE_MODEL = "rerank-english-v3.0"
LOCAL_MODEL = "BAAI/bge-reranker-base"

_local_model = None  # singleton CrossEncoder, lazy-loaded


def _build_query(parser_output: dict, brief_text: str) -> str:
    """Concatenate soft-constraint descriptions; fall back to the raw brief if absent."""
    soft = (parser_output or {}).get("soft_constraints") or []
    parts = [s.get("description", "") for s in soft if s.get("description")]
    if parts:
        return " | ".join(parts)
    return brief_text


def _build_document(creator: dict) -> str:
    """A creator's 'document' is the concatenated text of their top-matching chunks."""
    chunks = creator.get("top_chunks") or []
    if not chunks:
        return ""
    return "\n\n".join(c.get("text", "") for c in chunks)


def _rerank_cohere(query: str, documents: list[str]) -> list[float]:
    """Return relevance scores in the same order as `documents`. Raises if Cohere unreachable."""
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set")

    client = cohere.ClientV2(api_key=api_key)
    response = client.rerank(
        model=COHERE_MODEL,
        query=query,
        documents=documents,
        top_n=len(documents),  # we want every score, not Cohere's top_n filter
    )

    # response.results carries .index and .relevance_score; we need to
    # re-order back to the input order.
    scores = [0.0] * len(documents)
    for r in response.results:
        scores[r.index] = float(r.relevance_score)
    return scores


def _rerank_local(query: str, documents: list[str]) -> list[float]:
    """Local CrossEncoder fallback. Loads the model once and caches."""
    global _local_model
    from sentence_transformers import CrossEncoder

    if _local_model is None:
        logger.info("Loading local cross-encoder %s (one-time download)...", LOCAL_MODEL)
        _local_model = CrossEncoder(LOCAL_MODEL)

    pairs = [(query, doc) for doc in documents]
    scores = _local_model.predict(pairs, show_progress_bar=False)
    return [float(s) for s in scores]


async def run(state: dict) -> dict:
    """
    Read retrieval_output.creators, rerank, slice to TOP_N_AFTER_RERANK,
    write state["reranker_output"].
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    retrieval = state.get("retrieval_output") or {}
    creators = retrieval.get("creators") or []
    if not creators:
        state["reranker_output"] = {"creators": [], "total_creators_returned": 0, "reranker_used": None}
        return state

    parser_out = state.get("parser_output") or {}
    brief_text = (state.get("input") or {}).get("brief_text", "")
    query = _build_query(parser_out, brief_text)

    # Filter out creators with no chunks — there's nothing for the reranker
    # to score them on. They keep retrieval_score=0 from the no-soft-constraint
    # edge case in retrieval_agent. We pass them through unchanged at the bottom.
    rankable = [c for c in creators if c.get("top_chunks")]
    unrankable = [c for c in creators if not c.get("top_chunks")]
    documents = [_build_document(c) for c in rankable]

    provider = "cohere" if os.getenv("COHERE_API_KEY") else "local"
    scores: list[float]

    if rankable:
        try:
            if provider == "cohere":
                scores = _rerank_cohere(query, documents)
            else:
                scores = _rerank_local(query, documents)
        except Exception as e:
            logger.warning("Primary reranker (%s) failed: %s. Trying fallback.", provider, e)
            try:
                scores = _rerank_local(query, documents)
                provider = "local"
            except Exception as e2:
                state["errors"].append(f"{AGENT_NAME}: both rerankers failed: {e2}")
                state["reranker_output"] = None
                return state

        for c, s in zip(rankable, scores):
            c["rerank_score"] = float(s)

        rankable.sort(key=lambda c: c["rerank_score"], reverse=True)
    else:
        rankable = []

    # Unrankable creators (no chunks at all — possible only when there were 0 soft
    # constraints) get rerank_score=0 and tail the list. The deterministic ranking
    # step will still apply its own scoring on top.
    for c in unrankable:
        c["rerank_score"] = 0.0

    combined = (rankable + unrankable)[:TOP_N_AFTER_RERANK]
    state["reranker_output"] = {
        "creators": combined,
        "total_creators_returned": len(combined),
        "reranker_used": provider,
    }
    logger.info("reranker: %d → %d (provider: %s)", len(creators), len(combined), provider)
    return state


if __name__ == "__main__":
    import asyncio

    test_state = {
        "input": {"brief_text": "Clean skincare with ingredient-conscious voice for women 25-40."},
        "parser_output": {
            "soft_constraints": [
                {"description": "ingredient-conscious clean beauty voice", "weight": 1.0},
                {"description": "warm authentic tone", "weight": 0.6},
            ],
        },
        "retrieval_output": {
            "creators": [
                {
                    "creator_id": 1, "retrieval_score": 0.5,
                    "top_chunks": [{"text": "I always check the squalane percentage on labels.",
                                    "post_ids": [1], "chunk_id": 100}],
                },
                {
                    "creator_id": 2, "retrieval_score": 0.45,
                    "top_chunks": [{"text": "10 makeup hacks that will blow your mind!!!",
                                    "post_ids": [2], "chunk_id": 200}],
                },
                {
                    "creator_id": 3, "retrieval_score": 0.42,
                    "top_chunks": [{"text": "Slow beauty routine, focus on what your skin actually needs.",
                                    "post_ids": [3], "chunk_id": 300}],
                },
            ],
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== reranker smoke test ===")
    print(f"COHERE_API_KEY set: {bool(os.getenv('COHERE_API_KEY'))}")
    result = asyncio.run(run(test_state))
    out = result.get("reranker_output") or {}
    print(f"Provider used: {out.get('reranker_used')}")
    for c in out.get("creators", []):
        print(f"  creator_id={c['creator_id']} rerank={c['rerank_score']:.4f} "
              f"retrieval={c['retrieval_score']:.4f}")
    print("errors:", result["errors"])
