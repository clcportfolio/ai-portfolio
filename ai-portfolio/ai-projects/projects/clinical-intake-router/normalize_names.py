"""
One-time migration — normalize patient_name values in intake_submissions.

Converts names stored in clinical "Last, First Middle" format to "First Last"
to match the extraction agent's current output convention.

Rules applied:
  "Simmons, Robert Earl"  → "Robert Simmons"   (Last, First Middle → First Last)
  "Marsh, Evelyn Ruth"    → "Evelyn Marsh"
  "Price, Nathaniel Owen" → "Nathaniel Price"
  "Gloria E Reyes"        → "Gloria Reyes"      (First Middle Last → First Last)
  "Marcus Webb"           → "Marcus Webb"        (no change — already correct)

Only rows whose patient_name contains a comma OR has three or more tokens
(indicating a middle name/initial) are updated. Single-token or already-correct
two-token names are left untouched.

Safe to run multiple times — idempotent for already-normalized names.

Usage:
    python normalize_names.py           # preview changes (dry run)
    python normalize_names.py --apply   # write changes to the database
"""

import argparse
import re
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)


def normalize_name(name: str) -> str:
    """
    Normalize a patient name to "First Last" format.

    Handles:
      - "Last, First"         → "First Last"
      - "Last, First Middle"  → "First Last"  (middle dropped)
      - "First Middle Last"   → "First Last"  (middle dropped)
      - "First M. Last"       → "First Last"  (initial dropped)
      - "First Last"          → "First Last"  (unchanged)
    """
    if not name or not name.strip():
        return name

    name = name.strip()

    if "," in name:
        # "Last, First [Middle...]" format
        last, _, given = name.partition(",")
        given_parts = given.strip().split()
        first = given_parts[0] if given_parts else ""
        last = last.strip()
        if first and last:
            return f"{first} {last}"
        return name  # can't parse — leave as-is

    parts = name.split()
    if len(parts) >= 3:
        # "First Middle Last" or "First M. Last" — keep first and last only
        return f"{parts[0]} {parts[-1]}"

    # Already "First Last" or single token — no change
    return name


def main():
    parser = argparse.ArgumentParser(description="Normalize patient names in intake_submissions.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to the database. Without this flag, runs as a dry run (preview only).",
    )
    args = parser.parse_args()

    from storage.db_client import _get_connection

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, patient_name FROM intake_submissions WHERE patient_name IS NOT NULL;")
            rows = cur.fetchall()
    finally:
        # Don't close yet if we're applying — reuse the connection below
        if not args.apply:
            conn.close()

    changes = []
    for row_id, current_name in rows:
        normalized = normalize_name(current_name)
        if normalized != current_name:
            changes.append((row_id, current_name, normalized))

    if not changes:
        print("No names need normalization — all records are already in 'First Last' format.")
        if args.apply:
            conn.close()
        return

    print(f"{'DRY RUN — ' if not args.apply else ''}Found {len(changes)} name(s) to normalize:\n")
    for row_id, old, new in changes:
        print(f"  id={row_id:>4}  '{old}'  →  '{new}'")

    if not args.apply:
        print(
            f"\nNo changes written. Run with --apply to update {len(changes)} record(s) in the database."
        )
        conn.close()
        return

    # Apply updates
    print()
    updated = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for row_id, old, new in changes:
                    cur.execute(
                        "UPDATE intake_submissions SET patient_name = %s WHERE id = %s;",
                        (new, row_id),
                    )
                    print(f"  ✓ Updated id={row_id}: '{old}' → '{new}'")
                    updated += 1
        print(f"\nDone — {updated} record(s) updated.")
    except Exception as e:
        print(f"\nError during update: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
