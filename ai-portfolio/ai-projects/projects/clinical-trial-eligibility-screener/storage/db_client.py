"""
Database Client — Clinical Trial Eligibility Screener
Handles all PostgreSQL interactions via psycopg2 against Supabase.

Schema
------
Table: trials
  id                  SERIAL PRIMARY KEY
  name                TEXT NOT NULL UNIQUE    -- display name for the trial dropdown
  criteria_text       TEXT NOT NULL           -- raw inclusion/exclusion criteria text
  structured_criteria JSONB                   -- CriteriaExtractionResult from criteria_agent
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()

Table: screenings
  id                    SERIAL PRIMARY KEY
  trial_id              INTEGER REFERENCES trials(id) ON DELETE CASCADE
  patient_summary_hash  TEXT NOT NULL           -- SHA-256 of patient summary; no raw PII stored
  eligibility_status    TEXT NOT NULL           -- ELIGIBLE | INELIGIBLE | NEEDS_REVIEW
  confidence_score      NUMERIC(4,3)            -- 0.000 – 1.000
  criteria_evaluations  JSONB                   -- full EvaluationResult from evaluation_agent
  verdict_output        JSONB                   -- full VerdictResult from verdict_agent
  is_synthetic          BOOLEAN NOT NULL DEFAULT FALSE
  patient_age           INTEGER                 -- extracted from summary for analytics (not PII alone)
  patient_sex           TEXT                    -- extracted from summary: male | female | other | unknown
  screened_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()

Deduplication strategy
-----------------------
Screenings are deduplicated per (trial_id, patient_summary_hash). Running the same
patient against the same trial twice returns the existing screening rather than
inserting a duplicate. The hash is SHA-256 of the patient summary string — no raw
text is stored.

Production note
---------------
Row-level security (RLS) in Supabase should restrict coordinator reads to their own
trial's screenings. Admins get full access. The current implementation enforces this
at the application layer via auth.py.
"""

import hashlib
import json
import logging
import os
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

_DB_URI = os.getenv("SUPABASE_DB_URI", "")

CREATE_TRIALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trials (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    criteria_text       TEXT NOT NULL,
    structured_criteria JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_SCREENINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS screenings (
    id                    SERIAL PRIMARY KEY,
    trial_id              INTEGER REFERENCES trials(id) ON DELETE CASCADE,
    patient_summary_hash  TEXT NOT NULL,
    eligibility_status    TEXT NOT NULL,
    confidence_score      NUMERIC(4,3),
    criteria_evaluations  JSONB,
    verdict_output        JSONB,
    is_synthetic          BOOLEAN NOT NULL DEFAULT FALSE,
    patient_age           INTEGER,
    patient_sex           TEXT,
    screened_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (trial_id, patient_summary_hash)
);
"""

# Migrate existing tables — safe to run on every startup (IF NOT EXISTS / IF EXISTS)
MIGRATE_SCREENINGS_SQL = """
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS patient_age INTEGER;
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS patient_sex TEXT;
"""


def _get_connection():
    """Return a psycopg2 connection using SUPABASE_DB_URI from environment."""
    if not _DB_URI:
        raise ValueError("SUPABASE_DB_URI environment variable is not set.")
    try:
        return psycopg2.connect(_DB_URI)
    except Exception:
        raise RuntimeError(
            "Database connection failed. Check that SUPABASE_DB_URI is correct "
            "and the Supabase project is reachable."
        ) from None


def hash_patient_summary(patient_summary: str) -> str:
    """Return SHA-256 hex digest of the patient summary string."""
    return hashlib.sha256(patient_summary.encode("utf-8")).hexdigest()


def _extract_age(text: str) -> Optional[int]:
    """Extract approximate patient age from a summary string. Returns None if not found."""
    import re
    m = re.search(r'\b(\d{1,3})[- ]?year[s]?[- ]?old\b', text, re.IGNORECASE)
    if m:
        age = int(m.group(1))
        return age if 0 < age < 120 else None
    m = re.search(r'\bage[d]?\s+(\d{1,3})\b', text, re.IGNORECASE)
    if m:
        age = int(m.group(1))
        return age if 0 < age < 120 else None
    return None


def _extract_sex(text: str) -> str:
    """Extract patient sex from a summary string. Returns 'male', 'female', or 'unknown'."""
    import re
    t = text.lower()
    if re.search(r'\b(female|woman|women|girl|she|her)\b', t):
        return "female"
    if re.search(r'\b(male|man|men|boy|he|his)\b', t):
        return "male"
    return "unknown"


def init_db() -> None:
    """
    Create both tables if they do not already exist, and apply any pending column migrations.
    Safe to call on every app startup — fully idempotent.
    Trials table is created first because screenings has a FK reference to it.
    """
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TRIALS_TABLE_SQL)
                cur.execute(CREATE_SCREENINGS_TABLE_SQL)
                cur.execute(MIGRATE_SCREENINGS_SQL)
        logger.info("Database initialised (tables: trials, screenings).")
    finally:
        conn.close()


# ── Trials ────────────────────────────────────────────────────────────────────

def get_trials() -> list[dict]:
    """
    Return all trials ordered by name.
    Used to populate the trial dropdown in the Streamlit UI.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, criteria_text, structured_criteria, created_at "
                "FROM trials ORDER BY name ASC;"
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_trial_by_id(trial_id: int) -> Optional[dict]:
    """
    Return a single trial by id, or None if not found.
    Includes structured_criteria so the pipeline can skip criteria_agent.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, criteria_text, structured_criteria, created_at "
                "FROM trials WHERE id = %s LIMIT 1;",
                (trial_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def trial_name_exists(name: str) -> bool:
    """Return True if a trial with this name already exists."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM trials WHERE name = %s LIMIT 1;", (name,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def insert_trial(
    name: str,
    criteria_text: str,
    structured_criteria: Optional[dict] = None,
) -> dict:
    """
    Insert a new trial. Returns the inserted row as a dict.

    Args:
        name: Display name shown in the dropdown (must be unique).
        criteria_text: Raw inclusion/exclusion criteria text as pasted by coordinator.
        structured_criteria: Optional CriteriaExtractionResult dict. If provided, the
            pipeline will skip criteria_agent for this trial on future screenings.

    Returns:
        Row dict with id, name, criteria_text, structured_criteria, created_at.

    Raises:
        ValueError: If a trial with this name already exists.
    """
    if trial_name_exists(name):
        raise ValueError(f"A trial named '{name}' already exists.")

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO trials (name, criteria_text, structured_criteria)
                    VALUES (%s, %s, %s)
                    RETURNING id, name, criteria_text, structured_criteria, created_at;
                    """,
                    (
                        name,
                        criteria_text,
                        json.dumps(structured_criteria) if structured_criteria else None,
                    ),
                )
                row = dict(cur.fetchone())
                logger.info("Inserted trial id=%s name='%s'", row["id"], row["name"])
                return row
    finally:
        conn.close()


def update_trial_structured_criteria(trial_id: int, structured_criteria: dict) -> None:
    """
    Store structured_criteria on a trial that was saved without it.
    Called after criteria_agent runs for the first time on a new trial.
    """
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE trials SET structured_criteria = %s WHERE id = %s;",
                    (json.dumps(structured_criteria), trial_id),
                )
        logger.info("Updated structured_criteria for trial_id=%s", trial_id)
    finally:
        conn.close()


# ── Screenings ────────────────────────────────────────────────────────────────

def screening_exists(trial_id: int, patient_summary_hash: str) -> Optional[dict]:
    """
    Check whether a screening for this (trial_id, patient_summary_hash) pair exists.

    Returns:
        The existing row as a dict if found, otherwise None.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM screenings
                WHERE trial_id = %s AND patient_summary_hash = %s
                LIMIT 1;
                """,
                (trial_id, patient_summary_hash),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def insert_screening(
    trial_id: int,
    patient_summary: str,
    pipeline_state: dict,
    is_synthetic: bool = False,
) -> dict:
    """
    Insert a new screening result. Returns the inserted row as a dict.

    Deduplicates on (trial_id, patient_summary_hash). If an identical screening
    already exists, returns the existing row with _duplicate=True.

    Args:
        trial_id: FK to the trials table.
        patient_summary: Raw patient summary text — hashed, not stored.
        pipeline_state: Full state dict returned by pipeline.run().
        is_synthetic: True for seeded synthetic patients.

    Returns:
        Row dict. Includes "_duplicate": True if a prior identical screening existed.
    """
    summary_hash = hash_patient_summary(patient_summary)

    existing = screening_exists(trial_id, summary_hash)
    if existing:
        logger.info(
            "Duplicate screening detected (trial_id=%s, hash prefix %s). Skipping insert.",
            trial_id,
            summary_hash[:12],
        )
        existing["_duplicate"] = True
        return existing

    verdict = pipeline_state.get("verdict_agent_output") or {}
    evaluations = pipeline_state.get("evaluation_agent_output") or {}

    patient_age = _extract_age(patient_summary)
    patient_sex = _extract_sex(patient_summary)

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO screenings (
                        trial_id, patient_summary_hash, eligibility_status,
                        confidence_score, criteria_evaluations, verdict_output,
                        is_synthetic, patient_age, patient_sex
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (
                        trial_id,
                        summary_hash,
                        verdict.get("eligibility_status", "NEEDS_REVIEW"),
                        verdict.get("confidence_score"),
                        json.dumps(evaluations),
                        json.dumps(verdict),
                        is_synthetic,
                        patient_age,
                        patient_sex,
                    ),
                )
                row = dict(cur.fetchone())
                row["_duplicate"] = False
                logger.info(
                    "Inserted screening id=%s for trial_id=%s status=%s",
                    row["id"],
                    trial_id,
                    row["eligibility_status"],
                )
                return row
    finally:
        conn.close()


def get_screenings_for_trial(trial_id: int) -> list[dict]:
    """
    Return all screenings for a given trial, ordered most-recent first.
    Used by the analytics tab to build charts.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, trial_id, patient_summary_hash, eligibility_status,
                       confidence_score, criteria_evaluations, verdict_output,
                       is_synthetic, patient_age, patient_sex, screened_at
                FROM screenings
                WHERE trial_id = %s
                ORDER BY screened_at DESC;
                """,
                (trial_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_screening_stats(trial_id: int) -> dict:
    """
    Return aggregate statistics for a trial's screenings.
    Used to populate the summary metrics at the top of the analytics tab.

    Returns dict with keys:
        total, eligible, ineligible, needs_review, synthetic_count,
        avg_confidence, latest_screened_at
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                        AS total,
                    COUNT(*) FILTER (WHERE eligibility_status = 'ELIGIBLE')      AS eligible,
                    COUNT(*) FILTER (WHERE eligibility_status = 'INELIGIBLE')     AS ineligible,
                    COUNT(*) FILTER (WHERE eligibility_status = 'NEEDS_REVIEW')   AS needs_review,
                    COUNT(*) FILTER (WHERE is_synthetic = TRUE)                   AS synthetic_count,
                    AVG(confidence_score)                                          AS avg_confidence,
                    MAX(screened_at)                                               AS latest_screened_at
                FROM screenings
                WHERE trial_id = %s;
                """,
                (trial_id,),
            )
            row = cur.fetchone()
            stats = dict(row) if row else {}
            # Cast Decimal to float for JSON serialisability
            if stats.get("avg_confidence") is not None:
                stats["avg_confidence"] = float(stats["avg_confidence"])
            return stats
    finally:
        conn.close()


if __name__ == "__main__":
    """
    Smoke test — validates Supabase connectivity, table creation, trial insert,
    screening insert, dedup, and stats query.
    Requires SUPABASE_DB_URI in .env.
    """
    from datetime import datetime

    print("=== DB Client Smoke Test ===\n")

    print("1. Initialising database (CREATE TABLE IF NOT EXISTS)...")
    init_db()
    print("   OK.")

    trial_name = f"Smoke Test Trial {datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    print(f"\n2. Inserting test trial ('{trial_name}')...")
    trial = insert_trial(
        name=trial_name,
        criteria_text="Inclusion: Age 18-65, Type 2 diabetes. Exclusion: Pregnancy.",
        structured_criteria={
            "inclusion_criteria": [
                {"criterion_id": "INC_001", "criterion_text": "Age 18-65", "criterion_type": "inclusion"},
                {"criterion_id": "INC_002", "criterion_text": "Type 2 diabetes", "criterion_type": "inclusion"},
            ],
            "exclusion_criteria": [
                {"criterion_id": "EXC_001", "criterion_text": "Pregnancy", "criterion_type": "exclusion"},
            ],
            "total_criteria_count": 3,
            "extraction_confidence": 0.95,
        },
    )
    print(f"   Inserted trial id={trial['id']}")

    print("\n3. Fetching trial by id...")
    fetched = get_trial_by_id(trial["id"])
    assert fetched is not None and fetched["name"] == trial_name
    print(f"   OK. structured_criteria present: {fetched['structured_criteria'] is not None}")

    print("\n4. Inserting a test screening...")
    fake_state = {
        "verdict_agent_output": {
            "eligibility_status": "ELIGIBLE",
            "confidence_score": 0.88,
            "summary": "Patient meets all criteria.",
            "key_factors": ["Age 45 within 18-65", "Type 2 diabetes confirmed"],
            "next_steps": "Proceed with consent process.",
        },
        "evaluation_agent_output": {
            "evaluations": [],
            "overall_assessment": "All criteria met.",
        },
    }
    screening = insert_screening(
        trial_id=trial["id"],
        patient_summary="45-year-old male, Type 2 diabetes, non-pregnant.",
        pipeline_state=fake_state,
        is_synthetic=True,
    )
    print(f"   Inserted screening id={screening['id']}, duplicate={screening['_duplicate']}")

    print("\n5. Testing dedup...")
    dup = insert_screening(
        trial_id=trial["id"],
        patient_summary="45-year-old male, Type 2 diabetes, non-pregnant.",
        pipeline_state=fake_state,
        is_synthetic=True,
    )
    assert dup["_duplicate"] is True
    print("   Duplicate correctly detected.")

    print("\n6. Fetching screenings for trial...")
    screenings = get_screenings_for_trial(trial["id"])
    print(f"   Found {len(screenings)} screening(s).")

    print("\n7. Fetching screening stats...")
    stats = get_screening_stats(trial["id"])
    print(f"   Stats: {stats}")

    print("\n8. Fetching all trials...")
    all_trials = get_trials()
    print(f"   Found {len(all_trials)} trial(s) total.")

    print("\n=== All checks passed ===")
