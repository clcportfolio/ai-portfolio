"""
rationale_agent.py — Influencer Shortlist Agent

Generates prose rationale for the already-selected top N creators. Runs AFTER
the deterministic ranking + slice — never against the full scored pool — so
that the user-facing rationale matches the final list exactly.

Why a separate agent (not folded into the scorer)
-------------------------------------------------
- Scorer is count-agnostic and produces 50 entries; writing prose for 50 is
  wasted work when we'll only show 20.
- Scorer runs at temperature 0 for stable scoring; rationale benefits from
  modest temperature (0.3) for varied, readable prose without inventing facts.
- Cache architecture: scorer output is cached at Layer 1 (count-agnostic);
  rationale output is cached at Layer 2 (per-final-list). Splitting them
  cleanly maps to which cache layer each call belongs in.

Output contract
---------------
state["rationale_output"] = [
    {
        "creator_id": int,
        "name": str,
        "tier": str,
        "country": str,
        "platform": str,
        "follower_count": int,
        "rationale": str,        # 2-3 sentences of prose
        "risk": str,             # one-line risk summary, or "None identified."
        "cited_post_ids": [int],
        "score_breakdown": {     # passed through from scorer for UI display
            "topic_fit": float, "voice_fit": float,
            "audience_fit": float, "risk_penalty": float,
            "final_score": float,
        },
    },
    ...
]
"""

from __future__ import annotations

import logging
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

AGENT_NAME = "rationale_agent"


# ── Schema ────────────────────────────────────────────────────────────────────

class CreatorRationale(BaseModel):
    creator_id: int = Field(description="Must match an id from the input top-N.")
    rationale: str = Field(description="2-3 sentences explaining why this creator fits the brief.")
    risk: str = Field(
        description="One-line brand-safety summary. Use 'None identified.' when nothing concerning.",
    )
    cited_post_ids: list[int] = Field(
        description="post_ids from the creator's chunks that the rationale references.",
        default_factory=list,
    )


class RationaleResult(BaseModel):
    rationales: List[CreatorRationale] = Field(
        description="One entry per creator in the input top-N. Same length as the input.",
    )


# ── Prompt ───────────────────────────────────────────────────────────────────

RATIONALE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You write concise, defensible rationales for why each creator was shortlisted
for a brand campaign. The list has already been ranked and sliced — you are not
re-evaluating fit, you are EXPLAINING the ranking to a brand marketing manager.

For each creator, write 2-3 sentences that:
  1. Name the topical and voice fit specifically — what's IN their content that
     aligns with the brief? Cite post_ids in brackets like [42, 87].
  2. Note the audience fit briefly — tier, country, why it lines up.
  3. Flag any brand-safety concerns honestly. If the risk_penalty score is low,
     write "None identified." Don't manufacture risks that aren't there.

Don't repeat the brief back at the user. Don't praise generically — be specific
about what THIS creator brings. Vary your sentence openings across the list so
the output doesn't read as templated."""),
    ("human",
     """Brief:
{brief_text}

The ranking is final — write rationale for each creator in the order given.

{candidate_block}

Return exactly {n} entries — one per creator, in the same order."""),
])


# ── Chain ────────────────────────────────────────────────────────────────────

_RATIONALE_LLM = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    temperature=0.3,
)
_chain = RATIONALE_PROMPT | _RATIONALE_LLM.with_structured_output(RationaleResult)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_candidates(top_n: list[dict]) -> str:
    blocks = []
    for rank, c in enumerate(top_n, 1):
        chunks = c.get("top_chunks") or []
        chunk_text = "\n  ".join(
            f"[post_ids={ch.get('post_ids')}] {ch.get('text', '')[:200]}"
            for ch in chunks[:3]
        ) or "(no chunks available)"

        scores = c.get("scores") or {}
        blocks.append(
            f"Rank #{rank} — creator_id={c['creator_id']}\n"
            f"  name: {c.get('name')}\n"
            f"  platform/tier/country: {c.get('platform')}/{c.get('tier')}/{c.get('country')}\n"
            f"  followers: {c.get('follower_count', 0):,}\n"
            f"  scores: topic={scores.get('topic_fit'):.1f} voice={scores.get('voice_fit'):.1f} "
            f"audience={scores.get('audience_fit'):.1f} risk_penalty={scores.get('risk_penalty'):.1f}\n"
            f"  matching content:\n  {chunk_text}"
        )
    return "\n\n".join(blocks)


def _get_handler(state: dict) -> CallbackHandler:
    return state.get("langfuse_handler") or CallbackHandler()


# ── Run ──────────────────────────────────────────────────────────────────────

async def run(state: dict) -> dict:
    """
    Read state["final_top_n"] (set by pipeline's deterministic slice) and
    produce prose rationale. Writes state["rationale_output"] = list of dicts.
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    top_n = state.get("final_top_n") or []
    if not top_n:
        state["rationale_output"] = []
        state["errors"].append(f"{AGENT_NAME}: no creators in final_top_n")
        return state

    brief_text = (state.get("input") or {}).get("brief_text", "")
    candidate_block = _format_candidates(top_n)
    handler = _get_handler(state)

    try:
        result: RationaleResult = await _chain.ainvoke(
            {
                "brief_text": brief_text,
                "candidate_block": candidate_block,
                "n": len(top_n),
            },
            config={"callbacks": [handler], "run_name": AGENT_NAME},
        )
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: LLM call failed: {e}")
        state["rationale_output"] = None
        return state

    # Build the final output by joining rationales (LLM-generated) with the
    # per-creator metadata (passed through from earlier stages).
    rationale_by_id = {r.creator_id: r for r in result.rationales}
    output: list[dict] = []
    dropped = 0

    for c in top_n:
        cid = c["creator_id"]
        r = rationale_by_id.get(cid)
        if r is None:
            # LLM skipped a creator — produce a placeholder rather than dropping
            # silently, so the slice count still matches what the user requested.
            dropped += 1
            r = CreatorRationale(
                creator_id=cid,
                rationale="(rationale generation failed for this creator — manual review recommended)",
                risk="Unknown (rationale failure).",
                cited_post_ids=[],
            )

        # Validate cited_post_ids against the creator's actual posts (defence
        # against fabricated ids). The pipeline already validated creator_ids;
        # post_ids deserve the same treatment.
        valid_post_ids = set()
        for ch in c.get("top_chunks") or []:
            valid_post_ids.update(ch.get("post_ids", []))
        cited = [pid for pid in r.cited_post_ids if pid in valid_post_ids]

        scores = c.get("scores") or {}
        output.append({
            "creator_id": cid,
            "name": c.get("name"),
            "platform": c.get("platform"),
            "tier": c.get("tier"),
            "country": c.get("country"),
            "follower_count": c.get("follower_count"),
            "rationale": r.rationale,
            "risk": r.risk,
            "cited_post_ids": cited,
            "score_breakdown": {
                "topic_fit": scores.get("topic_fit"),
                "voice_fit": scores.get("voice_fit"),
                "audience_fit": scores.get("audience_fit"),
                "risk_penalty": scores.get("risk_penalty"),
                "final_score": c.get("final_score"),
            },
        })

    if dropped:
        state["errors"].append(
            f"{AGENT_NAME}: {dropped} creators received placeholder rationale (LLM skipped them)"
        )

    state["rationale_output"] = output
    logger.info("rationale: produced %d entries (%d placeholders)", len(output), dropped)
    return state


if __name__ == "__main__":
    import asyncio
    import json

    test_state = {
        "input": {"brief_text": "Clean skincare for women 25-40, ingredient-conscious voice."},
        "final_top_n": [
            {
                "creator_id": 1,
                "name": "Elena R",
                "platform": "IG",
                "tier": "mid",
                "country": "US",
                "follower_count": 220_000,
                "final_score": 8.4,
                "scores": {"topic_fit": 9.0, "voice_fit": 8.5, "audience_fit": 8.0, "risk_penalty": 1.0},
                "top_chunks": [
                    {"post_ids": [1, 2], "text": "Always check the squalane percentage on your serums."},
                ],
            },
        ],
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== rationale smoke test ===")
    result = asyncio.run(run(test_state))
    print(json.dumps(result.get("rationale_output"), indent=2, default=str))
    print("errors:", result["errors"])
