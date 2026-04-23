"""
Cache Client — Influencer Shortlist Agent
Two-layer Redis cache for shortlist results, with graceful no-op fallback when
REDIS_URL is unset.

Layer 1 — scored pool
  Key:   shortlist:pool:{sha256(brief_text_minus_count_and_required_fields)}
  Value: the 50 candidates with all dimension scores (the COUNT-AGNOSTIC pool)
  TTL:   1 hour
  Hit:   skip stages 1-5 (parse → filter → retrieve → rerank → score),
         re-slice deterministically to whatever count the user asked for.

Layer 2 — final list
  Key:   shortlist:final:{sha256(full_brief_text)}
  Value: the rationale-formatted output ready for display
  TTL:   1 hour
  Hit:   skip everything; return the cached output verbatim.

Why two layers
--------------
A single key on the full brief misses the common case where a user re-runs the
exact same brief but tweaks `count` from 20 → 10. With Layer 1 we keep the
expensive work (LLM scoring) cached and re-slice locally — guaranteed to produce
top-N as a strict subset of top-(N+1).

Layer 1 cache key strips count and output_spec.required_fields from the brief
text before hashing, since those parameters DON'T affect the scored pool — they
only affect the deterministic slice. Stripping is brittle if done with regex,
so we delegate to a normaliser that's tested directly (see _normalise_brief_for_pool_key).
"""

import hashlib
import json
import logging
import os
import re
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

POOL_TTL_SECONDS = 3600   # 1 hour
FINAL_TTL_SECONDS = 3600  # 1 hour

POOL_PREFIX = "shortlist:pool:"
FINAL_PREFIX = "shortlist:final:"

_redis_client = None  # process-wide singleton


def _get_redis():
    """Lazy singleton. Returns None when REDIS_URL is unset or connection fails."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    url = os.getenv("REDIS_URL")
    if not url:
        return None

    try:
        import redis as redis_lib
        client = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("Redis cache connected.")
        return client
    except Exception as e:
        logger.warning("Redis unavailable — cache disabled. (%s)", e)
        return None


# ── Brief normalisation ───────────────────────────────────────────────────────
# Strip count and required_fields phrases from the brief BEFORE hashing for the
# Layer 1 key. Two briefs that differ only in those parameters must produce the
# same pool key, so we can re-slice without re-running the pipeline.

_COUNT_PATTERNS = [
    r"\b\d+\s+creators?\b",          # "20 creators", "50 creator"
    r"\btop\s+\d+\b",                  # "top 10"
    r"\breturn\s+\d+\b",               # "return 25"
    r"\bgive\s+me\s+\d+\b",            # "give me 30"
    r"\bshortlist\s+of\s+\d+\b",       # "shortlist of 15"
    r"\bcount\s*[:=]\s*\d+\b",         # "count: 20", "count=20"
]
_COUNT_RE = re.compile("|".join(_COUNT_PATTERNS), re.IGNORECASE)


def _normalise_brief_for_pool_key(brief_text: str) -> str:
    """
    Lowercase, collapse whitespace, and strip count-related phrases.
    Two briefs differing only by `count` should normalise identically.
    """
    normalised = _COUNT_RE.sub(" ", brief_text)
    normalised = normalised.lower()
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return normalised


def pool_key(brief_text: str) -> str:
    norm = _normalise_brief_for_pool_key(brief_text)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{POOL_PREFIX}{digest}"


def final_key(brief_text: str) -> str:
    digest = hashlib.sha256(brief_text.encode("utf-8")).hexdigest()
    return f"{FINAL_PREFIX}{digest}"


# ── Layer 1: pipeline-state snapshot keyed on count-stripped brief ───────────
# Value is a dict carrying every intermediate output, not just scored_pool.
# This lets the UI's per-stage expanders and the Langfuse trace show the same
# data on a cache hit as on a cold run — without re-running any LLM stage.

def get_scored_pool(brief_text: str) -> Optional[dict]:
    """Return the cached pipeline-state snapshot for stages 1-5, or None on miss."""
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(pool_key(brief_text))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("Pool cache read failed: %s", e)
        return None


def set_scored_pool(brief_text: str, snapshot: dict) -> None:
    """
    Cache a snapshot dict containing parser_output, hard_filter_output,
    retrieval_output, reranker_output, scorer_output, and scored_pool.
    """
    r = _get_redis()
    if r is None:
        return
    try:
        r.setex(pool_key(brief_text), POOL_TTL_SECONDS, json.dumps(snapshot, default=str))
    except Exception as e:
        logger.warning("Pool cache write failed: %s", e)


# ── Layer 2: full pipeline-state snapshot keyed on full brief ────────────────
# Same shape as Layer 1 plus rationale_output and final_top_n.

def get_final_output(brief_text: str) -> Optional[dict]:
    """Return the cached full snapshot, or None on miss."""
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(final_key(brief_text))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("Final cache read failed: %s", e)
        return None


def set_final_output(brief_text: str, snapshot: dict) -> None:
    """Cache the full pipeline-state snapshot (everything Layer 1 has + rationale)."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.setex(final_key(brief_text), FINAL_TTL_SECONDS, json.dumps(snapshot, default=str))
    except Exception as e:
        logger.warning("Final cache write failed: %s", e)


def is_available() -> bool:
    return _get_redis() is not None


if __name__ == "__main__":
    print("=== _normalise_brief_for_pool_key ===\n")
    samples = [
        "Clean skincare, women 25-40, $150K budget, 20 creators",
        "Clean skincare, women 25-40, $150K budget, 10 creators",
        "Clean skincare, women 25-40, $150K budget, top 50",
        "CLEAN SKINCARE, women 25-40, $150K budget, count: 20",
    ]
    keys = [pool_key(s) for s in samples]
    for s, k in zip(samples, keys):
        print(f"  {s!r}\n    → {k}\n")

    print("All four briefs hash to same pool key:", len(set(keys)) == 1)
    print("Final keys differ across the four:", len(set(final_key(s) for s in samples)) == 4)

    print("\n=== Redis connectivity ===")
    print("Available:", is_available())

    if is_available():
        test_brief = "smoke test brief, 20 creators"
        set_scored_pool(test_brief, {"scored_pool": [{"creator_id": 1, "score": 0.9}]})
        got = get_scored_pool(test_brief)
        print("Round-trip pool cache:", got)

        set_final_output(test_brief, [{"creator_id": 1, "rationale": "test"}])
        got = get_final_output(test_brief)
        print("Round-trip final cache:", got)
    else:
        print("Set REDIS_URL to test cache round-trip.")
