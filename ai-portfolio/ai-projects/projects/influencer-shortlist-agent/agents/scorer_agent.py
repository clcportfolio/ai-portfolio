"""
scorer_agent.py — Influencer Shortlist Agent

Scores all 50 reranker survivors on four fixed dimensions in a single Sonnet
call with structured output.

Why count-agnostic
------------------
The scorer is COUNT-AGNOSTIC: it always scores all 50 candidates regardless of
how many the user asked for in the brief. The deterministic ranking step
applies the requested slice. This guarantees that re-running with count=10
returns a strict subset of the count=20 result — no jitter from re-running
the scorer with a different prompt. It's also why Layer 1 of the cache stores
the scored pool (key strips count from the brief).

Why a single call, not 50
-------------------------
Per-candidate calls are 50x more API calls but produce inconsistent calibration —
one creator's "8" depends on the model's recent history of what 8 means. Scoring
all 50 in one call gives the model the whole pool as context, so its rubric
stays calibrated across candidates. Token cost is modest: ~17k input + ~5k
output for the full pool, well under Sonnet's window.

Dimension semantics (CRITICAL)
------------------------------
All four dimensions are 0-10. Three are "more is better"; risk_penalty is "more
is worse". The deterministic ranking step inverts risk via (10 - risk_penalty)
before applying the weight, so the formula is:

  final = w_topic*topic_fit + w_voice*voice_fit + w_audience*audience_fit
          + w_risk*(10 - risk_penalty)

This preserves the weighted-sum interpretation while keeping the variable name
honest about what HIGH risk_penalty means.

Output contract
---------------
state["scorer_output"] = {
    "scores": [
        {
            "creator_id": int,
            "topic_fit": float,        # 0-10, higher is better
            "voice_fit": float,        # 0-10, higher is better
            "audience_fit": float,     # 0-10, higher is better
            "risk_penalty": float,     # 0-10, higher is WORSE
            "justification": str,
            "cited_post_ids": [int],   # subset of the creator's chunk post_ids
        },
        ...50 entries always...
    ]
}
"""

from __future__ import annotations

import logging
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field, field_validator

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

AGENT_NAME = "scorer_agent"


# ── Schema ────────────────────────────────────────────────────────────────────

class CreatorScore(BaseModel):
    creator_id: int = Field(description="The creator id from the input pool — must match exactly.")
    topic_fit: float = Field(description="0-10. How well their content matches the campaign topic.")
    voice_fit: float = Field(description="0-10. How well their voice matches what the brand wants.")
    audience_fit: float = Field(description="0-10. How well their audience matches the target.")
    risk_penalty: float = Field(
        description="0-10. HIGHER means MORE brand-safety risk (controversy, low quality, off-brand).",
    )
    justification: str = Field(
        description="One-sentence rationale for the scores, citing specific post_ids in brackets.",
    )
    cited_post_ids: list[int] = Field(
        description="post_ids referenced in the justification. Must be from the creator's own posts.",
        default_factory=list,
    )

    @field_validator("topic_fit", "voice_fit", "audience_fit", "risk_penalty")
    @classmethod
    def _bound_0_10(cls, v: float) -> float:
        return max(0.0, min(10.0, v))


class ScoringResult(BaseModel):
    scores: List[CreatorScore] = Field(
        description="One entry per candidate in the input pool. Must include every candidate.",
    )


# ── Prompt ───────────────────────────────────────────────────────────────────

SCORER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a brand marketing analyst scoring creator candidates against a campaign brief.

You will be shown a brief and a pool of 50 creator candidates, each with metadata
and their top-matching post snippets. Score every candidate on four dimensions,
all 0-10:

  topic_fit     — How well do their posts match the campaign's topical area?
                  10: posts directly cover the brand's category and product type.
                  5:  adjacent or partial fit.
                  0:  off-topic.
  voice_fit     — Does the creator's voice/tone match what the brand wants?
                  Read the cited posts; what's the tone? Match it against the brief's
                  voice description.
  audience_fit  — Do their declared audience and follower tier align with the brief?
                  Use country, tier, and follower_count as proxies for audience.
  risk_penalty  — How risky would this creator be for the brand?
                  HIGHER is WORSE. Look for: off-brand sponsored content, low post
                  quality, misalignment with the brand's positioning, or red flags
                  in the post text. A safe creator scores ~0-2 here, not high.

In your justification, name 1-3 specific post_ids in brackets (e.g. [42, 87])
that drove the score — reviewers need to verify your reasoning by reading those
posts.

Score every single creator in the pool. Do not skip any. Do not return more
than the input count. Be honest — calibrate so that average creators land
around 5; reserve 9-10 for genuinely outstanding fit and 0-2 for clear misfits."""),
    ("human",
     """Campaign brief:
{brief_text}

Brand emphasis (parsed):
{brief_summary}

Candidates ({n_candidates} total):
{candidate_block}

Score every candidate on topic_fit, voice_fit, audience_fit, and risk_penalty (all 0-10).
Return one entry per candidate, citing 1-3 post_ids in the justification."""),
])


# ── Chain ────────────────────────────────────────────────────────────────────

_SCORER_LLM = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=8192,
    temperature=0,
)
_chain = SCORER_PROMPT | _SCORER_LLM.with_structured_output(ScoringResult)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_brief_summary(parser_output: dict) -> str:
    """Compact one-line summary of hard constraints + scoring emphasis for the prompt."""
    if not parser_output:
        return "(no parsed brief)"
    hc = parser_output.get("hard_constraints") or {}
    weights = parser_output.get("scoring_weights") or {}
    bits = []
    if hc.get("countries"):    bits.append(f"countries={hc['countries']}")
    if hc.get("tiers"):        bits.append(f"tiers={hc['tiers']}")
    if hc.get("platforms"):    bits.append(f"platforms={hc['platforms']}")
    if hc.get("exclude_brand_collabs"):
        bits.append(f"excluded_brands={hc['exclude_brand_collabs']}")
    weight_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
    return " | ".join(bits) + f"\n  scoring weights: {weight_str}"


def _format_candidate_block(creators: list[dict], creator_meta: dict[int, dict]) -> str:
    """
    Compose the per-creator block fed to the scorer prompt.
    creator_meta is the lookup of full creator rows from DB, keyed by creator_id.
    """
    blocks = []
    for c in creators:
        cid = c["creator_id"]
        meta = creator_meta.get(cid, {})
        name = meta.get("name", "(unknown)")
        platform = meta.get("platform", "?")
        country = meta.get("country", "?")
        tier = meta.get("tier", "?")
        followers = meta.get("follower_count", 0)
        cats = meta.get("primary_categories") or []
        voice = meta.get("voice_descriptor", "")

        chunks = c.get("top_chunks") or []
        chunk_text_parts = []
        for ch in chunks:
            post_ids = ch.get("post_ids") or []
            chunk_text_parts.append(f"  [post_ids: {post_ids}]\n  {ch.get('text', '')}")
        chunk_block = "\n\n".join(chunk_text_parts) if chunk_text_parts else "  (no matching chunks)"

        blocks.append(
            f"--- creator_id={cid} ---\n"
            f"name: {name}\n"
            f"platform: {platform}, country: {country}, tier: {tier}, followers: {followers:,}\n"
            f"categories: {cats}\n"
            f"voice: {voice}\n"
            f"top matching content:\n{chunk_block}"
        )
    return "\n\n".join(blocks)


def _get_handler(state: dict) -> CallbackHandler:
    return state.get("langfuse_handler") or CallbackHandler()


# ── Run ──────────────────────────────────────────────────────────────────────

async def run(state: dict) -> dict:
    """
    Score all reranker survivors. Reads reranker_output.creators; writes
    state["scorer_output"]. Always scores the full pool (count-agnostic).
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    rerank = state.get("reranker_output") or {}
    creators = rerank.get("creators") or []
    if not creators:
        state["scorer_output"] = {"scores": []}
        return state

    parser_out = state.get("parser_output") or {}
    brief_text = (state.get("input") or {}).get("brief_text", "")

    # Fetch creator metadata for the prompt
    try:
        from storage.db_client import get_all_creators
        all_creators = await _to_thread(get_all_creators)
        creator_meta = {c["id"]: c for c in all_creators}
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: DB read failed: {e}")
        state["scorer_output"] = None
        return state

    # Defensive: only score creators we have metadata for. An id in reranker_output
    # but missing from DB shouldn't happen, but if it does, drop with a warning.
    scorable = [c for c in creators if c["creator_id"] in creator_meta]
    if len(scorable) < len(creators):
        logger.warning("scorer: dropped %d creators with no DB metadata",
                       len(creators) - len(scorable))

    candidate_block = _format_candidate_block(scorable, creator_meta)
    brief_summary = _format_brief_summary(parser_out)
    handler = _get_handler(state)

    try:
        result: ScoringResult = await _chain.ainvoke(
            {
                "brief_text": brief_text,
                "brief_summary": brief_summary,
                "n_candidates": len(scorable),
                "candidate_block": candidate_block,
            },
            config={"callbacks": [handler], "run_name": AGENT_NAME},
        )
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: LLM call failed: {e}")
        state["scorer_output"] = None
        return state

    # Defensive validation: drop hallucinated creator_ids and clamp post_ids to
    # the candidate's actual posts. The LLM occasionally fabricates ids — we
    # trust nothing.
    valid_ids = {c["creator_id"] for c in scorable}
    chunks_by_creator = {c["creator_id"]: c.get("top_chunks", []) for c in scorable}

    cleaned: list[dict] = []
    dropped = 0
    for s in result.scores:
        if s.creator_id not in valid_ids:
            dropped += 1
            continue
        valid_post_ids = set()
        for ch in chunks_by_creator.get(s.creator_id, []):
            valid_post_ids.update(ch.get("post_ids", []))
        s.cited_post_ids = [pid for pid in s.cited_post_ids if pid in valid_post_ids]
        cleaned.append(s.model_dump())

    if dropped:
        state["errors"].append(f"{AGENT_NAME}: dropped {dropped} hallucinated creator_ids")

    state["scorer_output"] = {"scores": cleaned}
    logger.info("scorer: scored %d candidates (%d dropped as hallucinated)", len(cleaned), dropped)
    return state


async def _to_thread(fn, *args, **kwargs):
    """Tiny helper so the sync DB call doesn't block the event loop."""
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)


if __name__ == "__main__":
    import asyncio
    import json

    test_state = {
        "input": {"brief_text": "Clean skincare for women 25-40, ingredient-conscious voice."},
        "parser_output": {
            "hard_constraints": {"countries": ["US"], "tiers": ["mid"]},
            "scoring_weights": {"topic_fit": 0.3, "voice_fit": 0.4, "audience_fit": 0.2, "risk_penalty": 0.1},
        },
        "reranker_output": {
            "creators": [
                {
                    "creator_id": 1, "rerank_score": 0.9,
                    "top_chunks": [{"text": "Always check the squalane percentage on your serums.",
                                    "post_ids": [1], "chunk_id": 100}],
                },
                {
                    "creator_id": 2, "rerank_score": 0.4,
                    "top_chunks": [{"text": "10 wild makeup hacks!!!! NUMBER 4 WILL SHOCK YOU",
                                    "post_ids": [2], "chunk_id": 200}],
                },
            ],
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== scorer smoke test (requires real creator rows in DB) ===")
    result = asyncio.run(run(test_state))
    print(json.dumps(result.get("scorer_output"), indent=2, default=str))
    print("errors:", result["errors"])
