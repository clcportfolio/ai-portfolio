"""
Database Client — Clinical Intake Router
Handles all PostgreSQL interactions via psycopg2 against Supabase.

Schema
------
Table: intake_submissions
  id                SERIAL PRIMARY KEY
  file_hash         TEXT NOT NULL UNIQUE   -- SHA-256 of raw file bytes; dedup key
  s3_key            TEXT                   -- S3 object key (null if text-only submission)
  s3_bucket         TEXT
  original_filename TEXT
  file_size_bytes   INTEGER
  patient_name      TEXT
  chief_complaint   TEXT
  urgency_level     TEXT                   -- Routine / Urgent / Emergent
  department        TEXT
  routing_summary   TEXT
  extraction_output JSONB                  -- full extraction_agent output
  classification_output JSONB              -- full classification_agent output
  routing_output    JSONB                  -- full routing_agent output
  submitted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()

Deduplication strategy
-----------------------
On insert, we check whether file_hash already exists. If it does, we return
the existing row rather than inserting a duplicate. This catches byte-for-byte
identical files regardless of filename or upload date.

Text-only submissions (no file upload) are hashed from the text content itself,
so pasting the same form twice also deduplicates correctly.

Production note
---------------
In production, this table should have audit logging enabled and access should
be restricted to a least-privilege DB user. Supabase row-level security (RLS)
policies would add an additional access control layer.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

_DB_URI = os.getenv("SUPABASE_DB_URI", "")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS intake_submissions (
    id                    SERIAL PRIMARY KEY,
    file_hash             TEXT NOT NULL UNIQUE,
    s3_key                TEXT,
    s3_bucket             TEXT,
    original_filename     TEXT,
    file_size_bytes       INTEGER,
    patient_name          TEXT,
    chief_complaint       TEXT,
    urgency_level         TEXT,
    department            TEXT,
    routing_summary       TEXT,
    extraction_output     JSONB,
    classification_output JSONB,
    routing_output        JSONB,
    submitted_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _get_connection():
    """Return a psycopg2 connection using SUPABASE_DB_URI from environment."""
    if not _DB_URI:
        raise ValueError("SUPABASE_DB_URI environment variable is not set.")
    try:
        return psycopg2.connect(_DB_URI)
    except Exception as e:
        # Never surface the raw connection string in error messages —
        # it contains credentials. Raise a sanitized error instead.
        raise RuntimeError(
            "Database connection failed. Check that SUPABASE_DB_URI is correct "
            "and the Supabase project is reachable."
        ) from None


def init_db() -> None:
    """
    Create the intake_submissions table if it does not already exist.
    Safe to call on every app startup — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        logger.info("Database initialised (table: intake_submissions).")
    finally:
        conn.close()


def submission_exists(file_hash: str) -> Optional[dict]:
    """
    Check whether a submission with the given file_hash already exists.

    Args:
        file_hash: SHA-256 hex digest of the file or text content.

    Returns:
        The existing row as a dict if found, otherwise None.
    """
    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM intake_submissions WHERE file_hash = %s LIMIT 1;",
                (file_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def insert_submission(
    file_hash: str,
    pipeline_state: dict,
    s3_result: Optional[dict] = None,
    original_filename: Optional[str] = None,
) -> dict:
    """
    Insert a new submission row. Returns the inserted row as a dict.

    If file_hash already exists (duplicate detected), returns the existing row
    without inserting. Callers should check result["_duplicate"] == True.

    Args:
        file_hash: SHA-256 of the raw content (from s3_client.hash_file).
        pipeline_state: The full state dict returned by pipeline.run().
        s3_result: Optional dict returned by s3_client.upload_file().
        original_filename: Display filename (for text-only submissions, pass None).

    Returns:
        Row dict. Includes "_duplicate": True if this was a pre-existing submission.
    """
    # Dedup check before insert
    existing = submission_exists(file_hash)
    if existing:
        logger.info("Duplicate submission detected (hash: %s). Skipping insert.", file_hash[:12])
        existing["_duplicate"] = True
        return existing

    extraction = pipeline_state.get("extraction_output") or {}
    classification = pipeline_state.get("classification_output") or {}
    routing = pipeline_state.get("routing_output") or {}

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO intake_submissions (
                        file_hash, s3_key, s3_bucket, original_filename, file_size_bytes,
                        patient_name, chief_complaint, urgency_level, department,
                        routing_summary, extraction_output, classification_output,
                        routing_output
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    RETURNING *;
                    """,
                    (
                        file_hash,
                        s3_result.get("s3_key") if s3_result else None,
                        s3_result.get("s3_bucket") if s3_result else None,
                        original_filename,
                        s3_result.get("file_size_bytes") if s3_result else None,
                        extraction.get("patient_name"),
                        extraction.get("chief_complaint"),
                        routing.get("urgency_level") or classification.get("urgency_level"),
                        routing.get("department") or classification.get("department"),
                        routing.get("routing_summary"),
                        json.dumps(extraction),
                        json.dumps(classification),
                        json.dumps(routing),
                    ),
                )
                row = cur.fetchone()
                inserted = dict(row)
                inserted["_duplicate"] = False
                logger.info(
                    "Inserted submission id=%s for patient=%s",
                    inserted.get("id"),
                    inserted.get("patient_name", "unknown"),
                )
                return inserted
    finally:
        conn.close()


def query_submissions(sql: str, params: tuple = ()) -> list[dict]:
    """
    Execute a raw SELECT query against intake_submissions and return rows as dicts.

    Used by the NL2SQL agent to run LLM-generated SQL. Only SELECT statements
    are permitted — any other statement type raises ValueError.

    Args:
        sql: A SELECT SQL string, optionally with %s placeholders.
        params: Tuple of parameters for the placeholders.

    Returns:
        List of row dicts.

    Raises:
        ValueError: If the statement is not a SELECT.
        RuntimeError: On database error.
    """
    normalised = sql.strip().upper()
    if not normalised.startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are permitted. Got: {sql[:60]}")

    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Pass None (not empty tuple) when there are no params so psycopg2
            # does not interpret literal % characters in LLM-generated SQL
            # (e.g. ILIKE '%Smith%') as parameter placeholders.
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except psycopg2.Error as e:
        logger.error("Database query error: %s", e)
        raise RuntimeError(f"Database query failed: {e}") from e
    finally:
        conn.close()


def get_table_schema() -> str:
    """
    Return a plain-text description of the intake_submissions schema.
    Passed to the NL2SQL agent as context so it knows what columns exist.
    """
    return """
Table: intake_submissions

Columns:
  id                    INTEGER       — auto-increment primary key
  file_hash             TEXT          — SHA-256 of file content; unique per document
  s3_key                TEXT          — S3 object key (null for text-only submissions)
  s3_bucket             TEXT          — S3 bucket name
  original_filename     TEXT          — original upload filename
  file_size_bytes       INTEGER       — file size in bytes
  patient_name          TEXT          — extracted patient name
  chief_complaint       TEXT          — extracted chief complaint
  urgency_level         TEXT          — 'Routine', 'Urgent', or 'Emergent'
  department            TEXT          — routed department (free text, e.g. 'Cardiology')
  routing_summary       TEXT          — plain-English routing summary from routing agent
  extraction_output     JSONB         — full extraction agent output as JSON
  classification_output JSONB         — full classification agent output as JSON
  routing_output        JSONB         — full routing agent output as JSON
  submitted_at          TIMESTAMPTZ   — timestamp of submission (UTC)

Example queries:
  SELECT patient_name, urgency_level, department FROM intake_submissions ORDER BY submitted_at DESC LIMIT 10;
  SELECT department, COUNT(*) FROM intake_submissions GROUP BY department ORDER BY COUNT(*) DESC;
  SELECT * FROM intake_submissions WHERE urgency_level = 'Emergent';
""".strip()


def get_existing_s3_keys() -> set:
    """Return the set of s3_keys already present in the database."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT s3_key FROM intake_submissions WHERE s3_key IS NOT NULL;")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def update_submission(file_hash: str, pipeline_state: dict) -> dict:
    """
    Update an existing row with pipeline results.
    Called when a previously upload-only row is routed for the first time.

    Args:
        file_hash: SHA-256 of the file content — used to locate the row.
        pipeline_state: Full state dict returned by pipeline.run().

    Returns:
        Updated row as a dict.
    """
    extraction = pipeline_state.get("extraction_output") or {}
    classification = pipeline_state.get("classification_output") or {}
    routing = pipeline_state.get("routing_output") or {}

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE intake_submissions SET
                        patient_name          = %s,
                        chief_complaint       = %s,
                        urgency_level         = %s,
                        department            = %s,
                        routing_summary       = %s,
                        extraction_output     = %s,
                        classification_output = %s,
                        routing_output        = %s
                    WHERE file_hash = %s
                    RETURNING *;
                    """,
                    (
                        extraction.get("patient_name"),
                        extraction.get("chief_complaint"),
                        routing.get("urgency_level") or classification.get("urgency_level"),
                        routing.get("department") or classification.get("department"),
                        routing.get("routing_summary"),
                        json.dumps(extraction),
                        json.dumps(classification),
                        json.dumps(routing),
                        file_hash,
                    ),
                )
                row = dict(cur.fetchone())
                logger.info("Updated submission id=%s with pipeline results.", row.get("id"))
                return row
    finally:
        conn.close()


def insert_file_only(
    file_hash: str,
    s3_result: dict,
    original_filename: str,
) -> dict:
    """
    Insert a minimal DB row for a file that was uploaded to S3 without being routed.
    No pipeline outputs are stored — patient_name, urgency_level, etc. will be null
    until the file is routed through the pipeline.

    If file_hash already exists, returns the existing row with _duplicate=True.
    """
    existing = submission_exists(file_hash)
    if existing:
        existing["_duplicate"] = True
        return existing

    conn = _get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO intake_submissions (
                        file_hash, s3_key, s3_bucket, original_filename, file_size_bytes
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (
                        file_hash,
                        s3_result.get("s3_key"),
                        s3_result.get("s3_bucket"),
                        original_filename,
                        s3_result.get("file_size_bytes"),
                    ),
                )
                row = dict(cur.fetchone())
                row["_duplicate"] = False
                logger.info("Inserted file-only row id=%s for %s", row.get("id"), original_filename)
                return row
    finally:
        conn.close()


def get_recent_submissions(limit: int = 20) -> list[dict]:
    """
    Return the most recent submissions for the S3 directory navigator.
    Lightweight — only fetches display columns, not full JSONB blobs.
    file_hash is included so the admin delete flow can identify the row.
    """
    return query_submissions(
        """
        SELECT id, file_hash, patient_name, chief_complaint, urgency_level, department,
               original_filename, s3_key, file_size_bytes, submitted_at
        FROM intake_submissions
        ORDER BY submitted_at DESC
        LIMIT %s;
        """,
        (limit,),
    )


def delete_submission(file_hash: str) -> bool:
    """
    Permanently delete the submission row identified by file_hash.

    Args:
        file_hash: SHA-256 hex digest — the natural unique key for a submission.

    Returns:
        True if a row was deleted, False if no matching row was found.

    Raises:
        RuntimeError: On database error.
    """
    conn = _get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM intake_submissions WHERE file_hash = %s;",
                    (file_hash,),
                )
                deleted = cur.rowcount > 0
                if deleted:
                    logger.info("Deleted submission with file_hash prefix %s", file_hash[:12])
                else:
                    logger.warning(
                        "delete_submission: no row found for hash prefix %s", file_hash[:12]
                    )
                return deleted
    except psycopg2.Error as e:
        logger.error("Database delete error: %s", e)
        raise RuntimeError(f"Database delete failed: {e}") from e
    finally:
        conn.close()


if __name__ == "__main__":
    """
    Smoke test — validates Supabase connectivity, table creation, insert, and query.
    Requires SUPABASE_DB_URI in .env.
    """
    import json as _json

    print("=== DB Client Smoke Test ===\n")

    print("1. Initialising database (CREATE TABLE IF NOT EXISTS)...")
    init_db()
    print("   OK.")

    fake_hash = "smoketest_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    fake_state = {
        "extraction_output": {
            "patient_name": "Smoke Test Patient",
            "chief_complaint": "Testing database connectivity",
            "age": 42,
        },
        "classification_output": {
            "urgency_level": "Routine",
            "department": "Primary Care",
            "classification_reasoning": "Smoke test — not a real patient.",
            "confidence": 0.99,
            "red_flags": [],
        },
        "routing_output": {
            "urgency_level": "Routine",
            "department": "Primary Care",
            "routing_summary": "Route to Primary Care for routine follow-up.",
            "recommended_next_steps": ["Schedule appointment"],
            "follow_up_actions": [],
            "estimated_response_time": "Within 1 week",
        },
    }

    print(f"\n2. Inserting test submission (hash prefix: {fake_hash[:20]})...")
    row = insert_submission(
        file_hash=fake_hash,
        pipeline_state=fake_state,
        original_filename="smoke_test.txt",
    )
    print(f"   Inserted id={row['id']}, duplicate={row['_duplicate']}")

    print("\n3. Checking duplicate detection...")
    row2 = insert_submission(
        file_hash=fake_hash,
        pipeline_state=fake_state,
    )
    assert row2["_duplicate"] is True, "Expected duplicate flag"
    print("   Duplicate correctly detected.")

    print("\n4. Querying recent submissions...")
    recent = get_recent_submissions(limit=3)
    print(f"   Found {len(recent)} recent row(s).")
    for r in recent:
        print(f"   - [{r['id']}] {r['patient_name']} | {r['urgency_level']} | {r['department']}")

    print("\n5. Schema string:")
    print(get_table_schema())

    print("\n=== All checks passed ===")
