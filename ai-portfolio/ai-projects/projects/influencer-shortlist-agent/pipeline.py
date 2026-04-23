"""
pipeline.py — Influencer Shortlist Agent

Orchestrates the 7-step pipeline that turns a campaign brief into a ranked
creator shortlist.

Stages
------
  0. validate_input + rate_limit_check                           (guardrails)
  1. parser_agent          natural language → ParsedBrief        (Sonnet, T=0)
  2. hard_filter           ParsedBrief.hard_constraints → ids    (SQL, no LLM)
  3. retrieval_agent       hybrid search per soft constraint     (Qdrant)
  4. reranker              top 200 → top 50 cross-encoder        (Cohere or local)
  5. scorer_agent          score 50 on 4 dimensions              (Sonnet, T=0)
  6. deterministic_rank    weighted sum + sort + slice to N      (Python)
  7. rationale_agent       prose for the final N                 (Sonnet, T=0.3)
  8. sanitize_output                                             (guardrails)

Deterministic ranking
---------------------
The final score is a weighted sum where risk_penalty is inverted to "safety":

  final = w_topic*topic_fit + w_voice*voice_fit
        + w_audience*audience_fit + w_risk*(10 - risk_penalty)

This keeps the weighted-sum interpretation honest while preserving the
"risk_penalty: higher = worse" semantics the scorer uses.

Caching
-------
  Layer 2 (final output): keyed on full brief, hits return immediately.
  Layer 1 (scored pool):  keyed on brief with `count` stripped, hits skip
                          stages 1-5 and re-run only the deterministic slice
                          + rationale. Guarantees that re-running with a
                          different count returns a strict subset.

Both layers fail open — if Redis is unavailable, the pipeline still runs.

Observability
-------------
@observe(name="influencer_shortlist") creates the root trace.
propagate_attributes pins trace_name + user_id on every child span,
including the per-agent CallbackHandler spans, so no agent's run_name
overwrites the trace display name.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from agents.parser_agent import run as parser_run
from agents.hard_filter import run as hard_filter_run
from agents.retrieval_agent import run as retrieval_run
from agents.reranker import run as reranker_run
from agents.scorer_agent import run as scorer_run
from agents.rationale_agent import run as rationale_run
from guardrails import validate_input, sanitize_output, rate_limit_check
from storage import cache_client

logger = logging.getLogger(__name__)


def build_initial_state(validated_input: dict, user_id: str) -> dict:
    return {
        "input": validated_input,
        "user_id": user_id,
        "pipeline_step": 0,
        "max_pipeline_steps": 12,
        "errors": [],
        "langfuse_handler": None,
        "cache_hit": None,                # "layer_1" | "layer_2" | None
        "parser_output": None,
        "hard_filter_output": None,
        "retrieval_output": None,
        "reranker_output": None,
        "scorer_output": None,
        "scored_pool": None,              # 50 candidates with metadata + final_score
        "final_top_n": None,              # the sliced N
        "rationale_output": None,
        "output": None,
        "latency_ms": None,
    }


# ── Deterministic ranking + slicing ──────────────────────────────────────────

def _compute_final_score(score: dict, weights: dict) -> float:
    """
    weighted sum with risk_penalty inverted to 'safety' (10 - risk_penalty).
    Defaults if any weight is missing — keeps the formula well-defined.
    """
    w_t = weights.get("topic_fit", 0.30)
    w_v = weights.get("voice_fit", 0.30)
    w_a = weights.get("audience_fit", 0.30)
    w_r = weights.get("risk_penalty", 0.10)
    return (
        w_t * score.get("topic_fit", 0)
        + w_v * score.get("voice_fit", 0)
        + w_a * score.get("audience_fit", 0)
        + w_r * (10.0 - score.get("risk_penalty", 0))
    )


def _build_scored_pool(state: dict) -> list[dict]:
    """
    Join scorer_output (LLM scores) with reranker_output (chunks) and DB
    (creator metadata) to produce a self-contained pool of 50 dicts.
    Each pool entry has everything downstream needs — the pool is what gets
    cached at Layer 1.
    """
    from storage.db_client import get_all_creators

    scorer_out = state.get("scorer_output") or {}
    rerank_out = state.get("reranker_output") or {}
    parser_out = state.get("parser_output") or {}

    weights = (parser_out.get("scoring_weights") or {})
    chunks_by_creator = {c["creator_id"]: c.get("top_chunks", []) for c in rerank_out.get("creators", [])}

    creator_meta = {c["id"]: c for c in get_all_creators()}

    pool: list[dict] = []
    for s in scorer_out.get("scores", []):
        cid = s["creator_id"]
        meta = creator_meta.get(cid, {})
        final = _compute_final_score(s, weights)
        pool.append({
            "creator_id": cid,
            "name": meta.get("name"),
            "platform": meta.get("platform"),
            "tier": meta.get("tier"),
            "country": meta.get("country"),
            "follower_count": meta.get("follower_count"),
            "primary_categories": meta.get("primary_categories"),
            "voice_descriptor": meta.get("voice_descriptor"),
            "scores": {
                "topic_fit": s["topic_fit"],
                "voice_fit": s["voice_fit"],
                "audience_fit": s["audience_fit"],
                "risk_penalty": s["risk_penalty"],
            },
            "final_score": final,
            "scorer_justification": s.get("justification"),
            "scorer_cited_post_ids": s.get("cited_post_ids", []),
            "top_chunks": chunks_by_creator.get(cid, []),
        })
    return pool


def _slice_top_n(scored_pool: list[dict], n: int) -> list[dict]:
    """Sort by final_score desc, return top N."""
    s = sorted(scored_pool, key=lambda c: c["final_score"], reverse=True)
    return s[:n]


# ── Validation: every output creator_id must come from the pool ──────────────

def _validate_creator_ids(rationale_output: list[dict], pool_ids: set[int]) -> list[dict]:
    """
    Drop any output entry whose creator_id isn't in the candidate pool.
    Defence against LLM hallucination — fits the CLAUDE.md hard rule.
    """
    cleaned = []
    dropped = 0
    for r in rationale_output:
        if r.get("creator_id") in pool_ids:
            cleaned.append(r)
        else:
            dropped += 1
    if dropped:
        logger.warning("validation: dropped %d output entries with invalid creator_ids", dropped)
    return cleaned


# ── Cache snapshots ──────────────────────────────────────────────────────────
# Cache values carry every intermediate output so that on a cache hit, the UI
# expanders and Langfuse trace see the same data they'd see on a cold run.
# Without this, cache hits leave parser_output / hard_filter_output / etc. as
# None, and the trace shows no agent spans (because nothing actually ran).

# Stages 1-5 outputs — cached at Layer 1 (count-agnostic). Parser is excluded
# because count/required_fields can vary across briefs that share the L1 key,
# so parser is always re-run on L1 hit (single Sonnet call at T=0).
_POOL_SNAPSHOT_KEYS = (
    "hard_filter_output", "retrieval_output", "reranker_output",
    "scorer_output", "scored_pool",
)

# L2 = L1 + parser_output + rationale_output + final_top_n. parser_output is
# safe to include because L2 hits are exact-brief matches.
_FINAL_SNAPSHOT_KEYS = (*_POOL_SNAPSHOT_KEYS, "parser_output", "rationale_output", "final_top_n")


def _build_pool_snapshot(state: dict) -> dict:
    return {k: state.get(k) for k in _POOL_SNAPSHOT_KEYS}


def _build_final_snapshot(state: dict) -> dict:
    return {k: state.get(k) for k in _FINAL_SNAPSHOT_KEYS}


def _restore_snapshot(state: dict, snapshot) -> None:
    """
    Copy every cached key into state. Skips None values.

    Backwards-compatible: pre-snapshot caches stored a bare list (the
    rationale_output) instead of a dict. If we encounter that shape, treat
    it as {"rationale_output": <list>} so the cache hit path still works
    on data written before the snapshot refactor.
    """
    if isinstance(snapshot, list):
        snapshot = {"rationale_output": snapshot}
    for k, v in snapshot.items():
        if v is not None:
            state[k] = v


# ── Persistence ──────────────────────────────────────────────────────────────

def _persist_run(state: dict, brief_text: str) -> None:
    """Insert a shortlist_runs row. Non-fatal on failure."""
    try:
        from storage.db_client import insert_shortlist_run
        insert_shortlist_run(
            brief_text=brief_text,
            parsed_brief=state.get("parser_output") or {},
            scored_pool=state.get("scored_pool") or [],
            final_output=state.get("rationale_output") or [],
            latency_ms=state.get("latency_ms") or 0,
            cost_usd=0.0,  # Langfuse tracks actuals; we don't double-bookkeep here
        )
    except Exception as e:
        logger.warning("shortlist_runs insert failed (non-fatal): %s", e)


# ── Trace I/O helpers ────────────────────────────────────────────────────────
# Single source of truth for trace input + output shape so cold runs and cache
# hits show identical fields at the trace root in Langfuse. Asymmetric trace
# metadata is the kind of thing that quietly trains engineers to look for the
# answer in different places per code path.

_BRIEF_PREVIEW_CHARS = 240


def _set_trace_input(lf, brief_text: str, cache_hit: str) -> None:
    """Always-the-same input shape, set as soon as we know cache_hit."""
    try:
        lf.set_current_trace_io(input={
            "brief_chars": len(brief_text),
            "brief_preview": brief_text[:_BRIEF_PREVIEW_CHARS]
                + ("…" if len(brief_text) > _BRIEF_PREVIEW_CHARS else ""),
            "cache_hit": cache_hit,                # "miss" | "layer_1" | "layer_2"
        })
    except Exception:
        pass


def _set_trace_output(lf, state: dict) -> None:
    """Always-the-same output shape, called once near return."""
    output = state.get("rationale_output") or []
    top_ids = [r.get("creator_id") for r in (output[:3] if isinstance(output, list) else [])]
    try:
        lf.set_current_trace_io(output={
            "n_returned": len(output) if isinstance(output, list) else 0,
            "latency_ms": state.get("latency_ms"),
            "cache_hit": state.get("cache_hit"),
            "top_creator_ids": top_ids,
        })
    except Exception:
        pass


# ── Main entry ───────────────────────────────────────────────────────────────

@observe(name="influencer_shortlist")
async def run(input_data: dict, user_id: str = "anonymous") -> dict:
    """
    Full pipeline. Returns the final state dict; user-facing list is in
    state["output"] (alias of state["rationale_output"]).
    """
    t_start = time.time()

    if not rate_limit_check(user_id):
        return {
            "input": input_data, "user_id": user_id,
            "pipeline_step": 0, "max_pipeline_steps": 12,
            "errors": ["Rate limit exceeded. Please try again later."],
            "output": None, "latency_ms": int((time.time() - t_start) * 1000),
        }

    validated = validate_input(input_data)
    state = build_initial_state(validated, user_id)
    brief_text = validated["brief_text"]

    # propagate_attributes wraps the WHOLE function (cache hits included) so the
    # Langfuse trace is named/tagged consistently regardless of cache state.
    # Without this, L2 hits would return before tracing fires and the trace
    # would appear unnamed in the Langfuse UI.
    with propagate_attributes(trace_name="influencer-shortlist-agent", user_id=user_id):
        lf = get_client()
        # Initial input — cache_hit is updated below if we end up taking a hit.
        _set_trace_input(lf, brief_text, cache_hit="miss")

        # ── Cache Layer 2: full brief → full pipeline-state snapshot ─────────
        cached_final = cache_client.get_final_output(brief_text)
        if cached_final is not None:
            _restore_snapshot(state, cached_final)
            state["cache_hit"] = "layer_2"
            state["output"] = state.get("rationale_output")
            state["latency_ms"] = int((time.time() - t_start) * 1000)
            logger.info("cache: layer_2 HIT (%dms)", state["latency_ms"])
            _set_trace_input(lf, brief_text, cache_hit="layer_2")
            _set_trace_output(lf, state)
            return sanitize_output(state)

        # Build a CallbackHandler scoped to this trace, shared across all agents.
        state["langfuse_handler"] = CallbackHandler(
            trace_context=TraceContext(
                trace_id=lf.get_current_trace_id(),
                parent_span_id=lf.get_current_observation_id(),
            )
        )

        # ── Cache Layer 1: count-stripped brief → stages-1-5 snapshot ────────
        cached_pool = cache_client.get_scored_pool(brief_text)
        if cached_pool is not None:
            _restore_snapshot(state, cached_pool)
            state["cache_hit"] = "layer_1"
            _set_trace_input(lf, brief_text, cache_hit="layer_1")
            logger.info("cache: layer_1 HIT — skipping stages 2-5")

            # Parser still re-runs because output_spec.count or required_fields
            # may differ between briefs that share the L1 key. Single Sonnet
            # call at T=0 — by far the cheapest way to get the new count.
            state = await parser_run(state)
            if state.get("parser_output") is None:
                state["errors"].append("pipeline: parser failed on cache-hit path")
                return sanitize_output(state)
        else:
            # Cold path — stages 1-5
            state = await parser_run(state)
            if not state.get("parser_output"):
                return sanitize_output(state)
            if state["parser_output"].get("ambiguities"):
                state["output"] = {"ambiguities": state["parser_output"]["ambiguities"]}
                state["latency_ms"] = int((time.time() - t_start) * 1000)
                return sanitize_output(state)

            state = await hard_filter_run(state)
            if not (state.get("hard_filter_output") or {}).get("candidate_count"):
                state["output"] = {"error": "Hard filter returned 0 creators. Loosen constraints."}
                state["latency_ms"] = int((time.time() - t_start) * 1000)
                return sanitize_output(state)

            state = await retrieval_run(state)
            state = await reranker_run(state)
            state = await scorer_run(state)

            if not (state.get("scorer_output") or {}).get("scores"):
                state["errors"].append("pipeline: scorer produced 0 entries")
                return sanitize_output(state)

            # Build the scored pool, then snapshot all stages-1-5 outputs to L1.
            state["scored_pool"] = _build_scored_pool(state)
            cache_client.set_scored_pool(brief_text, _build_pool_snapshot(state))

        # ── Deterministic ranking + slice (always runs) ──────────────────────
        count = ((state.get("parser_output") or {}).get("output_spec") or {}).get("count", 20)
        state["final_top_n"] = _slice_top_n(state["scored_pool"], count)

        # ── Rationale (always runs) ──────────────────────────────────────────
        state = await rationale_run(state)

        pool_ids = {c["creator_id"] for c in state["scored_pool"]}
        state["rationale_output"] = _validate_creator_ids(state["rationale_output"] or [], pool_ids)
        state["output"] = state["rationale_output"]

        # ── Persist + L2 snapshot write + trace IO ───────────────────────────
        cache_client.set_final_output(brief_text, _build_final_snapshot(state))
        state["latency_ms"] = int((time.time() - t_start) * 1000)
        _persist_run(state, brief_text)

        # state["cache_hit"] was already set to "layer_1" if applicable; if we
        # came through the cold path it's still None — surface that as "miss"
        # in the trace so the field is always present.
        if state["cache_hit"] is None:
            state["cache_hit"] = "miss"
        _set_trace_output(lf, state)

    return sanitize_output(state)


# ── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Influencer Shortlist pipeline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; validate guardrails wiring only.")
    parser.add_argument("--brief", type=str, default=None,
                        help="Brief text. If omitted, uses a built-in sample.")
    args = parser.parse_args()

    sample_brief = args.brief or (
        "Clean-ingredient skincare brand launching for women 25-40 in the US and Canada. "
        "Voice should be ingredient-conscious and authentic — no performative marketing. "
        "Mix of mid and macro creators. Skip anyone who collabed with PureSkin Co or "
        "Verde Beauty in the last 6 months. Want 20 creators."
    )

    if args.dry_run:
        validated = validate_input({"brief_text": sample_brief})
        state = build_initial_state(validated, "dry-run")
        state["output"] = "dry-run placeholder"
        state = sanitize_output(state)
        print("Dry run passed. State keys:", list(state.keys()))
        print("pipeline_step:", state["pipeline_step"])
    else:
        print("Running full pipeline (LLM calls will be made)...")
        result = asyncio.run(run({"brief_text": sample_brief}, user_id="cli-test"))
        print("\n--- RESULT ---")
        print(f"Latency:    {result.get('latency_ms')} ms")
        print(f"Cache hit:  {result.get('cache_hit')}")
        print(f"Errors:     {result.get('errors')}")
        out = result.get("output") or []
        print(f"Returned:   {len(out) if isinstance(out, list) else 1}")
        if isinstance(out, list):
            for i, r in enumerate(out[:3], 1):
                print(f"\n#{i} {r.get('name')} ({r.get('tier')}/{r.get('country')}) "
                      f"score={r.get('score_breakdown', {}).get('final_score', 0):.2f}")
                print(f"  {r.get('rationale')}")
        else:
            print(json.dumps(out, indent=2, default=str))
