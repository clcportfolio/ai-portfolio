"""
hard_filter.py — Influencer Shortlist Agent

Pure-Python deterministic filter against the Supabase creators table. Reads the
hard_constraints from parser_output and produces a list of candidate creator_ids
that satisfy every constraint exactly.

Why this is NOT an LLM
----------------------
Hard constraints are inviolable: country, tier, follower range, exclusion lists.
If the brief says "no creators who collabed with Brand X in the last 6 months",
that's a guarantee, not a suggestion. LLMs are notoriously poor at honouring
exclusion lists in long context windows — they'll occasionally surface the very
candidate the user said to exclude. We sidestep the problem by enforcing all
hard constraints in SQL BEFORE the retrieval, reranker, or scorer ever see the
candidate pool.

Output contract
---------------
state["hard_filter_output"] = {
    "candidate_ids": [int, ...],
    "applied_filters": dict,        # echoes the constraints that were active
    "candidate_count": int,
}

If candidate_count == 0, the pipeline can short-circuit and tell the user the
filter was too restrictive — no point spending tokens on retrieval or scoring.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

AGENT_NAME = "hard_filter"


async def run(state: dict) -> dict:
    """
    Read state["parser_output"]["hard_constraints"] and apply them via
    storage.db_client.filter_creators(). Async signature is for symmetry with the
    other agents — the SQL call itself is sync, run inline (sub-millisecond).
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    parsed = state.get("parser_output") or {}
    hc = parsed.get("hard_constraints") or {}

    try:
        from storage.db_client import filter_creators

        candidate_ids = filter_creators(
            countries=hc.get("countries"),
            tiers=hc.get("tiers"),
            platforms=hc.get("platforms"),
            min_followers=hc.get("follower_min"),
            max_followers=hc.get("follower_max"),
            exclude_creator_ids=hc.get("exclude_creator_ids"),
            exclude_brand_collabs=hc.get("exclude_brand_collabs"),
            exclude_collab_window_days=hc.get("exclude_collab_window_days"),
        )
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: {e}")
        state["hard_filter_output"] = None
        return state

    state["hard_filter_output"] = {
        "candidate_ids": candidate_ids,
        "applied_filters": hc,
        "candidate_count": len(candidate_ids),
    }
    logger.info("hard_filter: %d candidates after constraints", len(candidate_ids))
    return state


if __name__ == "__main__":
    import asyncio
    import json

    test_state = {
        "input": {"brief_text": "[not used in this agent]"},
        "parser_output": {
            "hard_constraints": {
                "countries": ["US", "CA"],
                "tiers": ["mid", "macro"],
                "platforms": None,
                "follower_min": None,
                "follower_max": None,
                "exclude_creator_ids": None,
                "exclude_brand_collabs": ["PureSkin Co"],
                "exclude_collab_window_days": 180,
            },
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== hard_filter smoke test ===")
    result = asyncio.run(run(test_state))
    out = result.get("hard_filter_output") or {}
    print(f"Candidates: {out.get('candidate_count')} ids")
    if out.get("candidate_ids"):
        print(f"First 10:   {out['candidate_ids'][:10]}")
    print("errors:", result["errors"])
