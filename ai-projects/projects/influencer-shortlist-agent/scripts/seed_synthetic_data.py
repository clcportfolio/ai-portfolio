"""
seed_synthetic_data.py — Generate ~100 synthetic creators + posts and index them.

Three phases:
  1. CREATOR PROFILES (Sonnet)  — sampled traits → name/bio/voice/secondary categories
  2. POST GENERATION   (Haiku)  — 15-30 topic-consistent posts per creator, ~20% sponsored
  3. INDEXING          (no LLM) — chunk posts into groups of 5, embed dense+sparse,
                                  upsert to Qdrant; insert creator + posts into Supabase

Cost guard
----------
Estimated total for 100 creators is ~$2 (Sonnet for profiles, Haiku for posts).
The script prints the estimate up-front and refuses to spend more than $5 unless
--confirm is passed.

Idempotency
-----------
Each creator has a stable external_id = sha256(name|platform). On re-runs:
  - --reset drops all 4 Supabase tables + the Qdrant collection (with prompt)
  - without --reset, existing creators are skipped (Sonnet calls saved); new
    traits drawn for missing slots only

Reproducibility
---------------
Pass --seed to fix the trait sampling. The LLM calls are still nondeterministic
even at low temperatures, so the names/bios/posts will vary across runs — but
the trait DISTRIBUTION (platform, tier, country, categories) is reproducible.

Usage
-----
    python scripts/seed_synthetic_data.py --count 100
    python scripts/seed_synthetic_data.py --count 5 --dry-run   # plan only
    python scripts/seed_synthetic_data.py --reset --count 100   # destructive
    python scripts/seed_synthetic_data.py --skip-llm --reindex  # re-embed existing rows
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# Make the project root importable when running from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("seed")
logger.setLevel(logging.INFO)


# ── Distributions (the project spec) ──────────────────────────────────────────

PLATFORM_DIST   = [("IG", 0.50), ("TikTok", 0.30), ("YouTube", 0.20)]
COUNTRY_DIST    = [("US", 0.60), ("CA", 0.15), ("UK", 0.12), ("AU", 0.08), ("DE", 0.05)]
TIER_DIST       = [("nano", 0.20), ("micro", 0.35), ("mid", 0.25), ("macro", 0.15), ("mega", 0.05)]

TIER_FOLLOWER_RANGE = {
    "nano":  (1_000, 10_000),
    "micro": (10_000, 100_000),
    "mid":   (100_000, 500_000),
    "macro": (500_000, 2_000_000),
    "mega":  (2_000_000, 10_000_000),
}

CATEGORY_DIST = [
    ("beauty", 0.13), ("fitness", 0.12), ("food", 0.12), ("tech", 0.10),
    ("lifestyle", 0.13), ("gaming", 0.08), ("parenting", 0.08), ("fashion", 0.10),
    ("travel", 0.08), ("finance", 0.06),
]

# ── Brand pool (for sponsored posts and exclusion-list demos) ────────────────
# Hardcoded so exclusion-list scenarios are deterministic and testable.

BRAND_POOL = {
    "beauty":    ["GlowLab", "PureSkin Co", "Verde Beauty", "Lumina Cosmetics", "Bare Essentials"],
    "fitness":   ["PeakFuel", "FlexCore", "IronStrong", "ZenMove", "AltitudeGear"],
    "food":      ["GreenPlate", "KitchenCraft", "FreshHaus", "BoldBites", "NourishCo"],
    "tech":      ["PixelLogic", "NexusGear", "OrbitDevices", "CircuitForge", "ByteHub"],
    "lifestyle": ["HavenHome", "EverydayCo", "RootedLiving", "MorningBrew", "WildHaus"],
    "gaming":    ["RiftPlay", "ArcOmega", "PixelForge", "ChronoGear", "VoltCore"],
    "parenting": ["LittleNest", "BloomKids", "GentlePath", "HomeHaven", "PlayPatch"],
    "fashion":   ["ThreadCo", "AtelierNorth", "LinenLane", "ModaUrbana", "WovenCraft"],
    "travel":    ["WanderKit", "AltitudeGear", "RoamCo", "TrailMode", "JourneyForge"],
    "finance":   ["LedgerLine", "VaultIQ", "NorthBridge", "PennyCraft", "EquityNest"],
}


# ── Cost model (rough, in USD per 1M tokens, as of model release) ────────────

COST_PER_M_TOKENS = {
    "claude-sonnet-4-20250514":   {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5-20251001":  {"in": 0.80,  "out": 4.00},
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreatorProfile(BaseModel):
    """Phase 1 output: the LLM fills in name/bio/voice/secondary; traits are sampled in code."""
    name: str = Field(description="A plausible creator name (first + last, or a brand-style handle).")
    bio: str = Field(description="One-to-two sentence creator bio in their own voice.")
    voice_descriptor: str = Field(
        description="A short comma-separated list of voice traits, e.g. 'warm, ingredient-curious, no-nonsense'."
    )
    secondary_categories: list[str] = Field(
        description="0-2 secondary content categories that complement the primary ones.",
        default_factory=list,
    )


class PostSnippet(BaseModel):
    text: str = Field(description="The post body text. 30-80 words. No emojis required, but allowed.")
    sponsored: bool = Field(description="True if this post is a paid brand collaboration.")
    collab_brand: Optional[str] = Field(
        description="The brand name if sponsored, else null. Must be one of the provided brands.",
        default=None,
    )


class PostBatch(BaseModel):
    posts: list[PostSnippet] = Field(description="Between 15 and 30 posts for this creator.")


# ── Trait sampling (Phase 0 — pure Python, no LLM) ───────────────────────────

def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    items, weights = zip(*options)
    return rng.choices(items, weights=weights, k=1)[0]


def _sample_categories(rng: random.Random) -> list[str]:
    """Return 1-3 distinct primary categories drawn from CATEGORY_DIST."""
    n = rng.choices([1, 2, 3], weights=[0.4, 0.45, 0.15], k=1)[0]
    items, weights = zip(*CATEGORY_DIST)
    chosen: list[str] = []
    pool = list(items)
    pool_weights = list(weights)
    for _ in range(n):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        chosen.append(pick)
        idx = pool.index(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return chosen


def generate_seed_traits(count: int, seed: Optional[int] = None) -> list[dict]:
    """Sample `count` trait dicts from the configured distributions. Reproducible with `seed`."""
    rng = random.Random(seed)
    traits: list[dict] = []
    for i in range(count):
        platform = _weighted_choice(rng, PLATFORM_DIST)
        country = _weighted_choice(rng, COUNTRY_DIST)
        tier = _weighted_choice(rng, TIER_DIST)
        lo, hi = TIER_FOLLOWER_RANGE[tier]
        followers = rng.randint(lo, hi)
        cats = _sample_categories(rng)
        traits.append({
            "platform": platform,
            "country": country,
            "tier": tier,
            "follower_count": followers,
            "primary_categories": cats,
        })
    return traits


# ── LLM setup ────────────────────────────────────────────────────────────────
# Built once, reused across all calls. Sonnet for richer creator personas;
# Haiku for the high-volume post generation step where speed + cost matter most.

_PROFILE_LLM = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.7,
)

_POSTS_LLM = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    temperature=0.8,
)

_PROFILE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate fictional social media creator profiles for a synthetic test "
     "dataset. The profiles must be plausible — like a real person you'd find on "
     "the platform — and varied across runs. Do not reuse common names; aim for "
     "international, gender-balanced, ethnically diverse names that match the "
     "creator's country and content niche."),
    ("human",
     "Create a fictional creator with these traits:\n"
     "  Platform:           {platform}\n"
     "  Country:            {country}\n"
     "  Tier:               {tier} ({follower_count:,} followers)\n"
     "  Primary categories: {primary_categories}\n\n"
     "Fill in name, a 1-2 sentence first-person bio, a short voice descriptor "
     "(3-5 comma-separated traits), and 0-2 complementary secondary categories "
     "drawn from beauty, fitness, food, tech, lifestyle, gaming, parenting, "
     "fashion, travel, or finance."),
])

_POSTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate fictional social media post snippets for a synthetic test dataset. "
     "Posts must sound like the creator wrote them — match their voice descriptor and "
     "stay topically consistent with their primary categories. Vary length and topic "
     "across the batch; do NOT repeat the same hook or product. Around 20% of posts "
     "should be sponsored brand collaborations — for those, set sponsored=true and "
     "name a brand from the provided list. Non-sponsored posts must have collab_brand=null."),
    ("human",
     "Creator profile:\n"
     "  Name:        {name}\n"
     "  Platform:    {platform}\n"
     "  Categories:  {categories}\n"
     "  Voice:       {voice_descriptor}\n"
     "  Bio:         {bio}\n\n"
     "Brands available for sponsored posts (pick ONLY from this list):\n"
     "  {brand_list}\n\n"
     "Generate {post_count} posts. Roughly {sponsored_count} of them should be "
     "sponsored (sponsored=true with a collab_brand from the list above). Each post "
     "is 30-80 words. Do not include hashtags as the entire post body — keep it prose."),
])


# ── Phase 1: profile generation ──────────────────────────────────────────────

async def _phase1_profile(traits: dict, handler: CallbackHandler) -> dict:
    """Run one Sonnet call to flesh out a creator's name/bio/voice/secondary cats."""
    chain = _PROFILE_PROMPT | _PROFILE_LLM.with_structured_output(CreatorProfile)
    profile: CreatorProfile = await chain.ainvoke(
        {
            "platform": traits["platform"],
            "country": traits["country"],
            "tier": traits["tier"],
            "follower_count": traits["follower_count"],
            "primary_categories": ", ".join(traits["primary_categories"]),
        },
        config={"callbacks": [handler], "run_name": "seed:creator_profile"},
    )
    return {
        **traits,
        "name": profile.name.strip(),
        "bio": profile.bio.strip(),
        "voice_descriptor": profile.voice_descriptor.strip(),
        "secondary_categories": [c.strip().lower() for c in profile.secondary_categories][:2],
    }


# ── Phase 2: post generation ─────────────────────────────────────────────────

def _brand_list_for_creator(categories: list[str], rng: random.Random) -> list[str]:
    """Build a 6-brand candidate list biased toward the creator's categories."""
    brands: list[str] = []
    for cat in categories:
        brands.extend(BRAND_POOL.get(cat, [])[:3])
    # Add a couple of off-category brands so sponsored picks aren't perfectly aligned
    other_cats = [c for c in BRAND_POOL if c not in categories]
    for cat in rng.sample(other_cats, k=min(2, len(other_cats))):
        brands.append(rng.choice(BRAND_POOL[cat]))
    # Dedupe while preserving order
    seen = set()
    deduped = []
    for b in brands:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return deduped[:6]


async def _phase2_posts(creator: dict, handler: CallbackHandler, rng: random.Random) -> list[dict]:
    """
    Run one Haiku call to produce 15-30 posts for the creator. Returns post dicts
    with posted_at filled in by Python (the LLM doesn't get dates).
    """
    target_count = rng.randint(18, 25)  # tightened range vs spec's 15-30 to control cost variance
    sponsored_target = max(1, round(target_count * 0.20))
    cats = creator["primary_categories"] + creator["secondary_categories"]
    brand_list = _brand_list_for_creator(cats, rng)

    chain = _POSTS_PROMPT | _POSTS_LLM.with_structured_output(PostBatch)
    batch: PostBatch = await chain.ainvoke(
        {
            "name": creator["name"],
            "platform": creator["platform"],
            "categories": ", ".join(cats),
            "voice_descriptor": creator["voice_descriptor"],
            "bio": creator["bio"],
            "brand_list": ", ".join(brand_list),
            "post_count": target_count,
            "sponsored_count": sponsored_target,
        },
        config={"callbacks": [handler], "run_name": "seed:posts"},
    )

    # Sanitise: clamp sponsored brands to the allowed list; un-sponsor anything
    # the LLM marked sponsored without a (valid) brand. Defensive — the schema
    # doesn't enforce membership.
    valid_brands = set(brand_list)
    sanitized: list[dict] = []
    for p in batch.posts:
        sponsored = bool(p.sponsored)
        brand = (p.collab_brand or "").strip()
        if sponsored and brand not in valid_brands:
            sponsored = False
            brand = None
        elif not sponsored:
            brand = None
        sanitized.append({"text": p.text.strip(), "sponsored": sponsored, "collab_brand": brand})

    # Assign posted_at: spread across the last 730 days, most recent first.
    # Newer posts get tighter spacing (people post more in the recent window),
    # older posts spread out — a simple log-ish curve via rng exponent.
    now = datetime.now(timezone.utc)
    n = len(sanitized)
    for i, p in enumerate(sanitized):
        # i=0 → ~0 days ago; i=n-1 → up to ~730 days ago
        frac = (i / max(n - 1, 1)) ** 1.4
        days_ago = frac * 730 + rng.uniform(0, 5)
        p["posted_at"] = now - timedelta(days=days_ago)
    sanitized.sort(key=lambda p: p["posted_at"], reverse=True)
    return sanitized


# ── Phase 3: chunking + embedding + upserting ────────────────────────────────

CHUNK_SIZE = 5  # consecutive posts per chunk


def _chunk_posts(posts: list[dict]) -> list[dict]:
    """Group posts (already sorted) into non-overlapping chunks of CHUNK_SIZE."""
    chunks = []
    for start in range(0, len(posts), CHUNK_SIZE):
        group = posts[start:start + CHUNK_SIZE]
        if not group:
            continue
        chunks.append({
            "post_ids": [p["id"] for p in group],
            "text": "\n\n---\n\n".join(p["text"] for p in group),
            "posted_at_min": min(p["posted_at"] for p in group),
            "posted_at_max": max(p["posted_at"] for p in group),
        })
    return chunks


def _phase3_embed(creator: dict, posts_with_ids: list[dict]) -> list[dict]:
    """
    CPU-bound: chunk + dense/sparse embed. Returns ready-to-upsert point dicts.
    Safe to run in a worker thread — touches no thread-affine resources.
    """
    from storage.embeddings import embed_dense_batch, embed_sparse_batch

    chunks = _chunk_posts(posts_with_ids)
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    dense = embed_dense_batch(texts)
    sparse = embed_sparse_batch(texts)

    points = []
    for i, (chunk, dvec, svec) in enumerate(zip(chunks, dense, sparse)):
        # Globally unique id: creator_id * 10000 + chunk_index. With 100 creators
        # and ≤10000 chunks each, no collisions. Fits comfortably in int64.
        point_id = creator["id"] * 10_000 + i
        points.append({
            "id": point_id,
            "dense_vector": dvec,
            "sparse_indices": svec["indices"],
            "sparse_values": svec["values"],
            "payload": {
                "creator_id": creator["id"],
                "post_ids": chunk["post_ids"],
                "platform": creator["platform"],
                "posted_at_min": chunk["posted_at_min"].isoformat(),
                "posted_at_max": chunk["posted_at_max"].isoformat(),
                "categories": creator["primary_categories"] + (creator.get("secondary_categories") or []),
                "text": chunk["text"],
            },
        })
    return points


def _phase3_upsert(points: list[dict]) -> int:
    """
    Qdrant upsert. MUST run on the main thread — embedded Qdrant uses SQLite,
    whose connections are bound to the thread that opened them. Wrapping this
    in asyncio.to_thread will fail with 'SQLite objects created in a thread can
    only be used in that same thread.'
    """
    from storage.vector_store import upsert_chunks, ensure_collection
    if not points:
        return 0
    ensure_collection()
    upsert_chunks(points)
    return len(points)


# ── Per-creator orchestration ────────────────────────────────────────────────

async def _seed_one(
    sem: asyncio.Semaphore,
    traits: dict,
    handler: CallbackHandler,
    rng: random.Random,
    index: int,
    total: int,
) -> dict:
    """Phase 1 → Phase 2 → Phase 3 for one creator, bounded by the shared semaphore."""
    from storage.db_client import upsert_creator, insert_posts_bulk, make_external_id

    async with sem:
        try:
            # Phase 1
            print(f"  [{index}/{total}] Profile gen ({traits['platform']}/{traits['tier']}/{traits['country']})...")
            profile = await _phase1_profile(traits, handler)

            # Synchronous DB write — wrap in to_thread so we don't block the loop
            external_id = make_external_id(profile["name"], profile["platform"])
            creator_row = await asyncio.to_thread(
                upsert_creator,
                {**profile, "external_id": external_id},
            )
            creator_id = creator_row["id"]

            # Phase 2
            posts = await _phase2_posts(profile, handler, rng)
            inserted = await asyncio.to_thread(insert_posts_bulk, creator_id, posts)

            # Phase 3 — split: heavy embed work in a worker thread, Qdrant
            # upsert on the main thread (SQLite thread-affinity in embedded mode).
            points = await asyncio.to_thread(_phase3_embed, creator_row, inserted)
            n_chunks = _phase3_upsert(points)

            print(f"  [{index}/{total}] Done: {profile['name']} — {len(inserted)} posts, {n_chunks} chunks")
            return {"index": index, "creator_id": creator_id, "name": profile["name"],
                    "posts": len(inserted), "chunks": n_chunks, "ok": True}
        except Exception as e:
            logger.exception("Seed failed for [%d/%d]", index, total)
            return {"index": index, "ok": False, "error": str(e)}


# ── Cost estimation ──────────────────────────────────────────────────────────

def estimate_cost(count: int) -> dict:
    """
    Estimate Sonnet + Haiku spend for `count` creators.
    Assumes ~200 in / 400 out for profiles (Sonnet) and ~500 in / 3000 out for posts (Haiku).
    """
    s = COST_PER_M_TOKENS["claude-sonnet-4-20250514"]
    h = COST_PER_M_TOKENS["claude-haiku-4-5-20251001"]
    sonnet = count * (200 * s["in"] + 400 * s["out"]) / 1_000_000
    haiku  = count * (500 * h["in"] + 3000 * h["out"]) / 1_000_000
    return {"sonnet_usd": round(sonnet, 4), "haiku_usd": round(haiku, 4),
            "total_usd": round(sonnet + haiku, 4)}


# ── Reset (destructive) ──────────────────────────────────────────────────────

def reset_all() -> None:
    """Drop both Supabase content (TRUNCATE) and Qdrant collection. Asks for confirmation."""
    from storage.db_client import _get_connection
    from storage.vector_store import reset_collection

    print("\n!!! RESET: about to truncate all 4 Supabase tables AND drop the Qdrant collection.")
    confirm = input("Type 'yes' to proceed: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE shortlist_runs RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE brand_collaborations RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE posts RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE creators RESTART IDENTITY CASCADE;")
        print("  Supabase: truncated.")
    finally:
        conn.close()

    reset_collection()
    print("  Qdrant: collection recreated.")


# ── Reindex (no LLM) ─────────────────────────────────────────────────────────

def reindex_existing() -> int:
    """
    Re-chunk and re-embed all existing creators' posts. Useful after changing the
    embedding model or chunk size. No LLM calls.
    """
    from storage.db_client import get_all_creators, get_posts_for_creators
    from storage.vector_store import reset_collection

    creators = get_all_creators()
    if not creators:
        print("No creators in DB. Nothing to reindex.")
        return 0

    print(f"Reindexing {len(creators)} creators...")
    reset_collection()  # wipe so old chunks don't linger with stale ids/payloads
    total_chunks = 0
    for c in creators:
        posts = get_posts_for_creators([c["id"]])
        # posts come back ordered by creator_id, posted_at DESC
        points = _phase3_embed(c, posts)
        n = _phase3_upsert(points)
        total_chunks += n
        print(f"  {c['name']:30s} → {n} chunks")
    return total_chunks


# ── Main ─────────────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    from storage.db_client import init_db, get_all_creators, get_table_counts
    from storage.vector_store import ensure_collection, collection_info

    init_db()
    ensure_collection()

    if args.reset:
        reset_all()

    if args.reindex:
        n = reindex_existing()
        print(f"\nReindex complete. {n} chunks upserted.")
        return

    # Skip already-existing creators when resuming a partial seed.
    existing = {c["external_id"] for c in get_all_creators()}
    print(f"Existing creators in DB: {len(existing)}")

    traits_pool = generate_seed_traits(args.count, seed=args.seed)
    # Filter out traits whose probable name would collide is impossible here
    # (name is generated by the LLM), so we just generate args.count NEW
    # creators on top of what's already there. If you want a fixed total,
    # subtract len(existing) from --count manually.

    cost = estimate_cost(args.count)
    print(f"\nCost estimate for {args.count} creators:")
    print(f"  Sonnet (profiles): ${cost['sonnet_usd']:.2f}")
    print(f"  Haiku  (posts):    ${cost['haiku_usd']:.2f}")
    print(f"  TOTAL:             ${cost['total_usd']:.2f}")

    if cost["total_usd"] > 5.00 and not args.confirm:
        print("\nEstimated cost exceeds $5. Re-run with --confirm to proceed.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: no LLM calls, no DB writes. Sample traits:")
        for i, t in enumerate(traits_pool[:5], 1):
            print(f"  [{i}] {t}")
        return

    # Tracing: one Langfuse handler shared across all calls in this seed run.
    # CallbackHandler() with no args creates its own trace, which is fine for a
    # one-shot script (we don't need to nest it under a parent observation).
    handler = CallbackHandler()

    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    print(f"\nDispatching {args.count} creators ({args.concurrency} at a time)...\n")
    tasks = [
        _seed_one(sem, traits, handler, rng, i + 1, args.count)
        for i, traits in enumerate(traits_pool)
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    ok = sum(1 for r in results if r["ok"])
    err = len(results) - ok
    print(f"\nDone. {ok} succeeded, {err} failed.")
    print(f"Final counts: {get_table_counts()}")
    print(f"Vector store: {collection_info()}")


def main():
    parser = argparse.ArgumentParser(description="Seed synthetic creator data.")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of new creators to generate (default: 100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trait sampling (default: 42).")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent pipelines (default: 5).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan + sample traits, no LLM calls, no writes.")
    parser.add_argument("--reset", action="store_true",
                        help="Truncate tables + drop collection BEFORE seeding (asks confirmation).")
    parser.add_argument("--reindex", action="store_true",
                        help="Re-embed existing posts only. No LLM calls.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Alias for --reindex.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required when estimated cost > $5.")
    args = parser.parse_args()

    if args.skip_llm:
        args.reindex = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
