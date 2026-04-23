"""
Database Client — Influencer Shortlist Agent
Supabase / PostgreSQL persistence via psycopg2.

Schema
------
Table: creators
  id                    SERIAL PRIMARY KEY
  external_id           TEXT NOT NULL UNIQUE   -- SHA-256 of name+platform; stable dedup key
  name                  TEXT NOT NULL
  platform              TEXT NOT NULL          -- IG | TikTok | YouTube
  country               TEXT NOT NULL          -- ISO-2: US, CA, UK, AU, ...
  tier                  TEXT NOT NULL          -- nano | micro | mid | macro | mega
  follower_count        INTEGER NOT NULL
  primary_categories    TEXT[] NOT NULL        -- e.g. {beauty, lifestyle}
  secondary_categories  TEXT[] NOT NULL DEFAULT '{}'
  bio                   TEXT
  voice_descriptor      TEXT
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()

Table: posts
  id            SERIAL PRIMARY KEY
  creator_id    INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE
  text          TEXT NOT NULL
  posted_at     TIMESTAMPTZ NOT NULL
  sponsored     BOOLEAN NOT NULL DEFAULT FALSE
  collab_brand  TEXT                            -- only set when sponsored=TRUE

Table: brand_collaborations
  -- Denormalised one-row-per-creator-per-brand for fast exclusion-window filtering
  -- in the hard_filter stage. Updated on insert of every sponsored post.
  creator_id      INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE
  brand           TEXT NOT NULL
  last_collab_at  TIMESTAMPTZ NOT NULL
  PRIMARY KEY (creator_id, brand)

Table: shortlist_runs
  -- Every pipeline run is logged here for replay, audit, and analytics.
  id             SERIAL PRIMARY KEY
  brief_text     TEXT NOT NULL
  parsed_brief   JSONB                          -- ParsedBrief from parser_agent
  scored_pool    JSONB                          -- the 50 scored candidates
  final_output   JSONB                          -- the sliced + rationalised list
  latency_ms     INTEGER
  cost_usd       NUMERIC(8,4)
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

_DB_URI = os.getenv("SUPABASE_DB_URI", "")


CREATE_CREATORS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS creators (
    id                    SERIAL PRIMARY KEY,
    external_id           TEXT NOT NULL UNIQUE,
    name                  TEXT NOT NULL,
    platform              TEXT NOT NULL,
    country               TEXT NOT NULL,
    tier                  TEXT NOT NULL,
    follower_count        INTEGER NOT NULL,
    primary_categories    TEXT[] NOT NULL,
    secondary_categories  TEXT[] NOT NULL DEFAULT '{}',
    bio                   TEXT,
    voice_descriptor      TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_POSTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS posts (
    id            SERIAL PRIMARY KEY,
    creator_id    INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    posted_at     TIMESTAMPTZ NOT NULL,
    sponsored     BOOLEAN NOT NULL DEFAULT FALSE,
    collab_brand  TEXT
);
"""

CREATE_BRAND_COLLABS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS brand_collaborations (
    creator_id      INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE,
    brand           TEXT NOT NULL,
    last_collab_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (creator_id, brand)
);
"""

CREATE_SHORTLIST_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS shortlist_runs (
    id             SERIAL PRIMARY KEY,
    brief_text     TEXT NOT NULL,
    parsed_brief   JSONB,
    scored_pool    JSONB,
    final_output   JSONB,
    latency_ms     INTEGER,
    cost_usd       NUMERIC(8,4),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_posts_creator   ON posts(creator_id);
CREATE INDEX IF NOT EXISTS idx_posts_sponsored ON posts(sponsored) WHERE sponsored = TRUE;
CREATE INDEX IF NOT EXISTS idx_brand_collabs_brand ON brand_collaborations(brand);
"""


def _get_connection():
    if not _DB_URI:
        raise ValueError("SUPABASE_DB_URI environment variable is not set.")
    try:
        return psycopg2.connect(_DB_URI)
    except Exception:
        raise RuntimeError(
            "Database connection failed. Check that SUPABASE_DB_URI is correct "
            "and the Supabase project is reachable."
        ) from None


def make_external_id(name: str, platform: str) -> str:
    """Stable dedup key for a creator across reseeds. SHA-256 keeps it short and collision-free."""
    return hashlib.sha256(f"{name.lower().strip()}|{platform.lower().strip()}".encode()).hexdigest()[:32]


def init_db() -> None:
    """Create all tables + indexes if they don't exist. Idempotent — safe to call on every startup."""
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_CREATORS_TABLE_SQL)
                cur.execute(CREATE_POSTS_TABLE_SQL)
                cur.execute(CREATE_BRAND_COLLABS_TABLE_SQL)
                cur.execute(CREATE_SHORTLIST_RUNS_TABLE_SQL)
                cur.execute(CREATE_INDEXES_SQL)
        logger.info("DB initialised: creators, posts, brand_collaborations, shortlist_runs.")
    finally:
        conn.close()


# ── Creators ──────────────────────────────────────────────────────────────────

def upsert_creator(creator: dict) -> dict:
    """
    Insert a creator or return the existing row matched by external_id.
    Returns the row including its DB-assigned `id`.
    """
    external_id = creator.get("external_id") or make_external_id(creator["name"], creator["platform"])
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO creators (
                        external_id, name, platform, country, tier, follower_count,
                        primary_categories, secondary_categories, bio, voice_descriptor
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (external_id) DO UPDATE SET
                        name = EXCLUDED.name  -- no-op rewrite to force RETURNING on conflict
                    RETURNING *;
                    """,
                    (
                        external_id,
                        creator["name"],
                        creator["platform"],
                        creator["country"],
                        creator["tier"],
                        creator["follower_count"],
                        creator["primary_categories"],
                        creator.get("secondary_categories", []),
                        creator.get("bio"),
                        creator.get("voice_descriptor"),
                    ),
                )
                return dict(cur.fetchone())
    finally:
        conn.close()


def get_creator(creator_id: int) -> Optional[dict]:
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM creators WHERE id = %s;", (creator_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_all_creators() -> list[dict]:
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM creators ORDER BY id ASC;")
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def filter_creators(
    countries: Optional[list[str]] = None,
    tiers: Optional[list[str]] = None,
    platforms: Optional[list[str]] = None,
    min_followers: Optional[int] = None,
    max_followers: Optional[int] = None,
    exclude_creator_ids: Optional[list[int]] = None,
    exclude_brand_collabs: Optional[list[str]] = None,
    exclude_collab_window_days: Optional[int] = None,
) -> list[int]:
    """
    Hard-filter creators against deterministic constraints. Returns matching creator ids.

    Hard filtering happens BEFORE the LLM ever sees a candidate — never trust the LLM
    to honour an exclusion list. exclude_brand_collabs uses the brand_collaborations
    table; if exclude_collab_window_days is set, only collabs newer than that window
    disqualify (so an old collab from 5 years ago is fine).
    """
    where_clauses: list[str] = []
    params: list[Any] = []

    if countries:
        where_clauses.append("country = ANY(%s)")
        params.append(countries)
    if tiers:
        where_clauses.append("tier = ANY(%s)")
        params.append(tiers)
    if platforms:
        where_clauses.append("platform = ANY(%s)")
        params.append(platforms)
    if min_followers is not None:
        where_clauses.append("follower_count >= %s")
        params.append(min_followers)
    if max_followers is not None:
        where_clauses.append("follower_count <= %s")
        params.append(max_followers)
    if exclude_creator_ids:
        where_clauses.append("id <> ALL(%s)")
        params.append(exclude_creator_ids)

    if exclude_brand_collabs:
        if exclude_collab_window_days is not None:
            where_clauses.append(
                "id NOT IN ("
                "  SELECT creator_id FROM brand_collaborations"
                "  WHERE brand = ANY(%s)"
                "  AND last_collab_at >= NOW() - (%s || ' days')::INTERVAL"
                ")"
            )
            params.extend([exclude_brand_collabs, str(exclude_collab_window_days)])
        else:
            where_clauses.append(
                "id NOT IN ("
                "  SELECT creator_id FROM brand_collaborations WHERE brand = ANY(%s)"
                ")"
            )
            params.append(exclude_brand_collabs)

    where = " AND ".join(where_clauses) if where_clauses else "TRUE"
    sql = f"SELECT id FROM creators WHERE {where} ORDER BY follower_count DESC;"

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


# ── Posts ─────────────────────────────────────────────────────────────────────

def insert_posts_bulk(creator_id: int, posts: list[dict]) -> list[dict]:
    """
    Insert a batch of posts for one creator and update the brand_collaborations
    denormalised table for every sponsored post (in a single transaction).
    Returns the inserted rows including DB-assigned ids and posted_at.
    """
    if not posts:
        return []

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                values = [
                    (creator_id, p["text"], p["posted_at"], bool(p.get("sponsored")), p.get("collab_brand"))
                    for p in posts
                ]
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO posts (creator_id, text, posted_at, sponsored, collab_brand) "
                    "VALUES %s RETURNING id, creator_id, text, posted_at, sponsored, collab_brand;",
                    values,
                )
                inserted = [dict(r) for r in cur.fetchall()]

                # Denormalise sponsored posts into brand_collaborations.
                # Keep the most recent collab date per (creator, brand).
                sponsored = [(creator_id, p["collab_brand"], p["posted_at"])
                             for p in posts if p.get("sponsored") and p.get("collab_brand")]
                if sponsored:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO brand_collaborations (creator_id, brand, last_collab_at) "
                        "VALUES %s "
                        "ON CONFLICT (creator_id, brand) DO UPDATE SET "
                        "last_collab_at = GREATEST(brand_collaborations.last_collab_at, EXCLUDED.last_collab_at);",
                        sponsored,
                    )

        return inserted
    finally:
        conn.close()


def get_posts_for_creators(creator_ids: list[int]) -> list[dict]:
    """Return all posts belonging to the given creator ids. Used by retrieval + scoring stages."""
    if not creator_ids:
        return []
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, creator_id, text, posted_at, sponsored, collab_brand "
                "FROM posts WHERE creator_id = ANY(%s) ORDER BY creator_id, posted_at DESC;",
                (creator_ids,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_posts_by_ids(post_ids: list[int]) -> list[dict]:
    """Fetch specific posts by id. Used by the rationale agent to cite source content."""
    if not post_ids:
        return []
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, creator_id, text, posted_at, sponsored, collab_brand "
                "FROM posts WHERE id = ANY(%s);",
                (post_ids,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ── Shortlist runs ────────────────────────────────────────────────────────────

def insert_shortlist_run(
    brief_text: str,
    parsed_brief: dict,
    scored_pool: list[dict],
    final_output: list[dict],
    latency_ms: int,
    cost_usd: float,
) -> dict:
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO shortlist_runs (
                        brief_text, parsed_brief, scored_pool, final_output, latency_ms, cost_usd
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (
                        brief_text,
                        json.dumps(parsed_brief, default=str),
                        json.dumps(scored_pool, default=str),
                        json.dumps(final_output, default=str),
                        latency_ms,
                        cost_usd,
                    ),
                )
                return dict(cur.fetchone())
    finally:
        conn.close()


def get_recent_shortlist_runs(limit: int = 25) -> list[dict]:
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, brief_text, latency_ms, cost_usd, created_at "
                "FROM shortlist_runs ORDER BY created_at DESC LIMIT %s;",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_shortlist_run(run_id: int) -> Optional[dict]:
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM shortlist_runs WHERE id = %s;", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


# ── Counts (utility for seed scripts and admin UI) ───────────────────────────

def get_table_counts() -> dict:
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            counts = {}
            for table in ("creators", "posts", "brand_collaborations", "shortlist_runs"):
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                counts[table] = cur.fetchone()[0]
            return counts
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Influencer shortlist DB client smoke test")
    parser.add_argument("--init", action="store_true", help="Create tables and exit.")
    parser.add_argument("--counts", action="store_true", help="Print row counts and exit.")
    parser.add_argument("--smoke", action="store_true", help="Insert one creator + 2 posts, print results.")
    args = parser.parse_args()

    if args.init:
        init_db()
        print("Tables initialised.")
    elif args.counts:
        print(json.dumps(get_table_counts(), indent=2))
    elif args.smoke:
        from datetime import datetime, timezone
        init_db()
        creator = upsert_creator({
            "name": "Test Creator",
            "platform": "IG",
            "country": "US",
            "tier": "micro",
            "follower_count": 50000,
            "primary_categories": ["beauty", "lifestyle"],
            "bio": "Test bio",
            "voice_descriptor": "warm, ingredient-curious",
        })
        print("Upserted creator:", creator["id"], creator["name"])
        posts = insert_posts_bulk(creator["id"], [
            {"text": "Loving my new clean-beauty routine", "posted_at": datetime.now(timezone.utc), "sponsored": False},
            {"text": "Partnering with @cleanbrand on this drop", "posted_at": datetime.now(timezone.utc), "sponsored": True, "collab_brand": "CleanBrand"},
        ])
        print(f"Inserted {len(posts)} posts.")
        print("Counts:", get_table_counts())
        # filter test
        ids = filter_creators(countries=["US"], tiers=["micro"], min_followers=10000)
        print("Filter matched ids:", ids)
    else:
        print("Use --init, --counts, or --smoke.")
