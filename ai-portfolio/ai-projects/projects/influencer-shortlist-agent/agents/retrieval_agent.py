"""
retrieval_agent.py — Influencer Shortlist Agent

Hybrid (dense + sparse) retrieval against Qdrant, restricted to the candidate
creator pool from hard_filter. Aggregates chunk-level scores up to the creator
level and returns the top 200 creators.

Why hybrid retrieval
--------------------
- Dense embeddings (MiniLM) capture *semantic* similarity — "ingredient-curious"
  matches "I always read the label" even with no shared words.
- Sparse BM25 captures *keyword* matches — "squalane" should only match "squalane".
Brand briefs blend both kinds of intent. RRF fusion (handled inside vector_store)
combines the rankings without us having to tune a manual blend.

Why we filter at index time
---------------------------
The hard_filter stage produces the legal pool of creators. We pass that as a
filter to Qdrant via creator_id IN [...]. Filtering AFTER retrieval would miss
candidates whose chunks didn't make it into the top-K; filtering INSIDE the
prefetch keeps the recall budget focused on the eligible pool.

Aggregation: max + mean(top-3)
-------------------------------
A creator may have many matching chunks. We don't want one strong chunk to
overshadow a creator with consistently strong-but-not-peak content, nor a
creator with a single accidental match to outrank one with broad-but-modest
relevance. The aggregation `0.5 * max + 0.5 * mean(top-3)` blends peak signal
and breadth signal; tested empirically as a sane default in hybrid systems.

Multiple soft constraints
-------------------------
Each soft_constraint in parsed_brief becomes one query. Per-query scores are
multiplied by the constraint's `weight` and summed across queries before the
chunk → creator aggregation. A weight=0.5 constraint contributes half as much
as a weight=1.0 one to the final retrieval score.

Output
------
state["retrieval_output"] = {
    "creators": [
        {
            "creator_id": int,
            "retrieval_score": float,
            "top_chunks": [
                {"chunk_id": int, "score": float, "text": str, "post_ids": [int],
                 "matching_constraint": str},
                ...up to 3 best...
            ],
        },
        ...
    ],
    "total_creators_returned": int,
    "queries_run": int,
}
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

AGENT_NAME = "retrieval_agent"

TOP_CREATORS = 200
PREFETCH_PER_QUERY = 400      # chunks fetched from Qdrant per soft-constraint query
TOP_CHUNKS_PER_CREATOR = 3    # used for both aggregation and the rationale stage


async def run(state: dict) -> dict:
    """
    Read parser_output.soft_constraints + hard_filter_output.candidate_ids,
    run one hybrid query per soft constraint, aggregate to creator level,
    write top 200 creators to state["retrieval_output"].
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    parsed = state.get("parser_output") or {}
    soft = parsed.get("soft_constraints") or []
    hard = state.get("hard_filter_output") or {}
    candidate_ids = hard.get("candidate_ids") or []

    if not candidate_ids:
        state["retrieval_output"] = {"creators": [], "total_creators_returned": 0, "queries_run": 0}
        state["errors"].append(f"{AGENT_NAME}: hard filter returned 0 candidates")
        return state

    # Edge case: no soft constraints. All hard-filter candidates pass through
    # with retrieval_score=0 and no chunks. The reranker stage will fall back
    # to its own ranking using creator metadata only.
    if not soft:
        creators = [
            {"creator_id": cid, "retrieval_score": 0.0, "top_chunks": []}
            for cid in candidate_ids[:TOP_CREATORS]
        ]
        state["retrieval_output"] = {
            "creators": creators,
            "total_creators_returned": len(creators),
            "queries_run": 0,
        }
        return state

    try:
        from storage.embeddings import embed_dense, embed_sparse
        from storage.vector_store import hybrid_search
    except ImportError as e:
        state["errors"].append(f"{AGENT_NAME}: storage import failed: {e}")
        state["retrieval_output"] = None
        return state

    # ── Per-creator chunk score table ─────────────────────────────────────────
    # creator_chunks[creator_id] = list of (chunk_id, weighted_score, payload, source_constraint)
    # We accumulate across all soft-constraint queries.
    creator_chunks: dict[int, list[tuple]] = defaultdict(list)
    queries_run = 0

    for sc in soft:
        description = sc.get("description", "").strip()
        if not description:
            continue
        weight = float(sc.get("weight", 1.0))
        if weight <= 0:
            continue

        try:
            dense = embed_dense(description)
            sparse = embed_sparse(description)
            hits = hybrid_search(
                dense_query=dense,
                sparse_indices=sparse["indices"],
                sparse_values=sparse["values"],
                creator_id_filter=candidate_ids,
                limit=PREFETCH_PER_QUERY,
                prefetch_limit=PREFETCH_PER_QUERY,
            )
        except Exception as e:
            state["errors"].append(f"{AGENT_NAME}: query '{description[:30]}...' failed: {e}")
            continue

        queries_run += 1

        for hit in hits:
            cid = hit["payload"].get("creator_id")
            if cid is None:
                continue
            weighted = hit["score"] * weight
            creator_chunks[cid].append((hit["id"], weighted, hit["payload"], description))

    # ── Aggregate chunk → creator ────────────────────────────────────────────
    # Sort each creator's chunks by score desc, take top-3 for the mean.
    # Score = 0.5 * max + 0.5 * mean(top-3). With a single chunk, max == mean.
    aggregated = []
    for cid, chunks in creator_chunks.items():
        chunks.sort(key=lambda c: c[1], reverse=True)
        scores = [c[1] for c in chunks]
        top3 = scores[:TOP_CHUNKS_PER_CREATOR]
        max_s = top3[0]
        mean_s = sum(top3) / len(top3)
        retrieval_score = 0.5 * max_s + 0.5 * mean_s

        # Keep the top-3 chunks (with their texts and post_ids) for downstream.
        # Deduplicate by chunk_id in case the same chunk hit on multiple queries.
        seen_chunk_ids: set = set()
        top_chunks = []
        for chunk_id, score, payload, source_constraint in chunks:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            top_chunks.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": payload.get("text", ""),
                "post_ids": payload.get("post_ids", []),
                "matching_constraint": source_constraint,
            })
            if len(top_chunks) >= TOP_CHUNKS_PER_CREATOR:
                break

        aggregated.append({
            "creator_id": cid,
            "retrieval_score": retrieval_score,
            "top_chunks": top_chunks,
        })

    aggregated.sort(key=lambda c: c["retrieval_score"], reverse=True)
    top = aggregated[:TOP_CREATORS]

    state["retrieval_output"] = {
        "creators": top,
        "total_creators_returned": len(top),
        "queries_run": queries_run,
    }
    logger.info(
        "retrieval: %d candidates → %d returned (queries: %d)",
        len(candidate_ids), len(top), queries_run,
    )
    return state


if __name__ == "__main__":
    import asyncio
    import json

    test_state = {
        "parser_output": {
            "soft_constraints": [
                {"description": "ingredient-conscious clean beauty voice", "weight": 1.0},
                {"description": "warm authentic tone, no performative marketing", "weight": 0.6},
            ],
        },
        "hard_filter_output": {
            "candidate_ids": list(range(1, 51)),  # any 50 ids in DB
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== retrieval_agent smoke test ===")
    result = asyncio.run(run(test_state))
    out = result.get("retrieval_output") or {}
    print(f"Returned: {out.get('total_creators_returned')} creators across "
          f"{out.get('queries_run')} queries")
    for c in out.get("creators", [])[:5]:
        print(f"  creator_id={c['creator_id']:>4} score={c['retrieval_score']:.4f} "
              f"top_chunks={len(c['top_chunks'])}")
    print("errors:", result["errors"])
