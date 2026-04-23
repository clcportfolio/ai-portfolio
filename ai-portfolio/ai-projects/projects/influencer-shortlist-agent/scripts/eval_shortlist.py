"""
eval_shortlist.py — Heuristic precision/recall evaluation.

For each hand-written test brief, an `expected_archetypes` list defines the
traits a "good" recommendation should have. Creators matching any archetype
are counted as relevant, and we compute precision@N and recall@N over the
top-N pipeline output.

This is NOT a ground-truth eval — synthetic data has no labels. The metric
is a sanity floor: did the pipeline return creators in the right neighborhood?
A regression here means hard_filter or retrieval has drifted in a way that
warrants investigation.

Output
------
- Always writes docs/eval_results.json (a per-brief breakdown + overall).
- If `mlflow` is installed, also logs each brief as a separate MLflow run.

Usage
-----
    python scripts/eval_shortlist.py
    python scripts/eval_shortlist.py --brief-id 2     # run just one
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)


# ── Test set ─────────────────────────────────────────────────────────────────
# Each archetype is a dict of trait predicates joined by AND. A creator matches
# if every predicate is satisfied. A test brief is "satisfied" by a creator
# if it matches ANY archetype (OR across the list).

TEST_BRIEFS = [
    {
        "id": 1,
        "name": "Clean skincare — mid+macro US/CA",
        "brief": (
            "Clean-ingredient skincare brand for women 25-40 in the US and Canada. "
            "Voice: ingredient-conscious, authentic. Mid and macro creators. Want 20."
        ),
        "expected_archetypes": [
            {"primary_categories_any": ["beauty"], "tier_in": ["mid", "macro"], "country_in": ["US", "CA"]},
            {"primary_categories_any": ["lifestyle", "beauty"], "tier_in": ["mid", "macro"], "country_in": ["US", "CA"]},
        ],
    },
    {
        "id": 2,
        "name": "Strength fitness app — micro US",
        "brief": (
            "Strength-training app for women 25-45. Micro creators (10k-100k) on IG and "
            "TikTok in the US. No-nonsense form-first voice. Want 15."
        ),
        "expected_archetypes": [
            {"primary_categories_any": ["fitness"], "tier_in": ["micro"],
             "country_in": ["US"], "platform_in": ["IG", "TikTok"]},
        ],
    },
    {
        "id": 3,
        "name": "Sustainable Gen-Z fashion",
        "brief": (
            "Sustainable Gen-Z fashion. 10 creators on TikTok or IG, US/UK/CA, mid tier. "
            "Audience cares about sustainability."
        ),
        "expected_archetypes": [
            {"primary_categories_any": ["fashion", "lifestyle"], "tier_in": ["mid"],
             "country_in": ["US", "UK", "CA"], "platform_in": ["IG", "TikTok"]},
        ],
    },
    {
        "id": 4,
        "name": "Kitchen tech — food + tech audiences",
        "brief": (
            "Smart kitchen device launch. Want 10 creators across food and tech, US, mid or macro tier. "
            "Voice should be practical and curious about new gear."
        ),
        "expected_archetypes": [
            {"primary_categories_any": ["food", "tech"], "tier_in": ["mid", "macro"], "country_in": ["US"]},
        ],
    },
    {
        "id": 5,
        "name": "Parenting brand — gentle voice",
        "brief": (
            "Gentle baby-care brand. 12 creators in the parenting space, US/UK, micro or mid tier. "
            "Voice should be warm and reassuring — no scary-headlines."
        ),
        "expected_archetypes": [
            {"primary_categories_any": ["parenting", "lifestyle"], "tier_in": ["micro", "mid"],
             "country_in": ["US", "UK"]},
        ],
    },
]


# ── Matching ─────────────────────────────────────────────────────────────────

def _matches_archetype(creator: dict, archetype: dict) -> bool:
    cats = set(creator.get("primary_categories") or [])
    if "primary_categories_any" in archetype:
        if not (cats & set(archetype["primary_categories_any"])):
            return False
    if "tier_in" in archetype:
        if creator.get("tier") not in archetype["tier_in"]:
            return False
    if "country_in" in archetype:
        if creator.get("country") not in archetype["country_in"]:
            return False
    if "platform_in" in archetype:
        if creator.get("platform") not in archetype["platform_in"]:
            return False
    return True


def _matches_any(creator: dict, archetypes: list[dict]) -> bool:
    return any(_matches_archetype(creator, a) for a in archetypes)


def _count_total_relevant(all_creators: list[dict], archetypes: list[dict]) -> int:
    return sum(1 for c in all_creators if _matches_any(c, archetypes))


# ── Per-brief evaluation ─────────────────────────────────────────────────────

async def evaluate_one(case: dict, all_creators_by_id: dict[int, dict]) -> dict:
    import pipeline as pl

    print(f"\n=== Brief #{case['id']}: {case['name']} ===")
    result = await pl.run({"brief_text": case["brief"]}, user_id="eval-script")

    output = result.get("output") or []
    if isinstance(output, dict):  # ambiguity / error path
        return {**case, "status": "error", "reason": str(output), "metrics": None}

    n = len(output)
    if n == 0:
        return {**case, "status": "empty", "metrics": None}

    # Reconstruct full creator dicts (output entries don't carry primary_categories)
    full_outputs = []
    for o in output:
        cid = o.get("creator_id")
        if cid in all_creators_by_id:
            full_outputs.append(all_creators_by_id[cid])

    relevant_in_top_n = sum(1 for c in full_outputs if _matches_any(c, case["expected_archetypes"]))
    total_relevant = _count_total_relevant(list(all_creators_by_id.values()), case["expected_archetypes"])

    precision_at_n = relevant_in_top_n / n if n else 0.0
    recall_at_n = relevant_in_top_n / total_relevant if total_relevant else 0.0

    print(f"  n={n}  relevant_in_top_n={relevant_in_top_n}  total_relevant={total_relevant}")
    print(f"  precision@{n} = {precision_at_n:.2%}  recall@{n} = {recall_at_n:.2%}")
    print(f"  latency = {result.get('latency_ms')}ms  cache_hit = {result.get('cache_hit')}")

    return {
        **case,
        "status": "ok",
        "n": n,
        "relevant_in_top_n": relevant_in_top_n,
        "total_relevant": total_relevant,
        "precision_at_n": precision_at_n,
        "recall_at_n": recall_at_n,
        "latency_ms": result.get("latency_ms"),
        "cache_hit": result.get("cache_hit"),
    }


# ── MLflow (optional) ────────────────────────────────────────────────────────

def _try_mlflow_log(per_brief_results: list[dict]) -> bool:
    try:
        import mlflow
    except ImportError:
        return False

    mlflow.set_experiment("influencer_shortlist_eval")
    for r in per_brief_results:
        if r.get("status") != "ok":
            continue
        with mlflow.start_run(run_name=f"brief_{r['id']}_{r['name']}"):
            mlflow.log_param("brief_id", r["id"])
            mlflow.log_param("brief_name", r["name"])
            mlflow.log_metric("precision_at_n", r["precision_at_n"])
            mlflow.log_metric("recall_at_n", r["recall_at_n"])
            mlflow.log_metric("latency_ms", r["latency_ms"] or 0)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

async def main_async(brief_id: int = None) -> None:
    from storage.db_client import get_all_creators

    creators = get_all_creators()
    if not creators:
        print("No creators in DB. Run seed_synthetic_data.py first.")
        sys.exit(1)
    by_id = {c["id"]: c for c in creators}

    cases = TEST_BRIEFS if brief_id is None else [b for b in TEST_BRIEFS if b["id"] == brief_id]
    if not cases:
        print(f"No brief with id={brief_id}.")
        sys.exit(1)

    results = []
    for case in cases:
        try:
            results.append(await evaluate_one(case, by_id))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({**case, "status": "exception", "error": str(e)})

    # Aggregate
    ok = [r for r in results if r.get("status") == "ok"]
    overall = {}
    if ok:
        overall = {
            "mean_precision_at_n": sum(r["precision_at_n"] for r in ok) / len(ok),
            "mean_recall_at_n":    sum(r["recall_at_n"]    for r in ok) / len(ok),
            "mean_latency_ms":     sum(r["latency_ms"] or 0 for r in ok) / len(ok),
            "n_briefs":            len(ok),
        }
        print(f"\n=== Overall ({len(ok)} briefs) ===")
        print(f"  mean precision: {overall['mean_precision_at_n']:.2%}")
        print(f"  mean recall:    {overall['mean_recall_at_n']:.2%}")
        print(f"  mean latency:   {overall['mean_latency_ms']:.0f}ms")

    # Persist
    out = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "n_creators_in_db": len(creators),
        "per_brief": results,
        "overall": overall,
    }
    docs_path = os.path.join(_PROJECT_ROOT, "docs", "eval_results.json")
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    with open(docs_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {docs_path}")

    if _try_mlflow_log(results):
        print("Also logged to MLflow.")
    else:
        print("MLflow not installed; skipped MLflow logging.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief-id", type=int, default=None,
                        help="Run only the brief with this id (1-5). Omit to run all.")
    args = parser.parse_args()
    asyncio.run(main_async(args.brief_id))


if __name__ == "__main__":
    main()
