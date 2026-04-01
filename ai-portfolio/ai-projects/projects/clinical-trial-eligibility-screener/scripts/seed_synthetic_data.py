"""
seed_synthetic_data.py — Seed synthetic patient screenings for a given trial.

Usage:
    python scripts/seed_synthetic_data.py --trial-id 1
    python scripts/seed_synthetic_data.py --trial-id 1 --count 50
    python scripts/seed_synthetic_data.py --trial-id 1 --count 10 --no-llm

By default, patient summaries are generated on-the-fly by Claude Haiku based on the
trial's actual criteria — so any count works and profiles are always relevant to the
selected trial.

Pass --no-llm to use the 15 hardcoded fallback profiles instead (no API call for
generation, useful for offline testing).

Requires:
  - SUPABASE_DB_URI in .env (or environment)
  - ANTHROPIC_API_KEY, LANGFUSE_* in .env
  - The target trial must already exist in the trials table (create it via the app)
"""

import argparse
import asyncio
import json
import sys
import os

# Allow running from any directory by adding the project root to sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# ── Synthetic patient profiles ────────────────────────────────────────────────
# Each profile is a brief summary designed to produce a realistic variety of
# eligibility outcomes when evaluated against a diabetes trial. For trials with
# different criteria, the LLM will still produce meaningful (if sometimes wrong)
# verdicts — the point is analytics volume, not ground-truth accuracy.

_SYNTHETIC_PROFILES = [
    # Strong ELIGIBLE candidates
    "52-year-old female with Type 2 diabetes diagnosed 4 years ago. HbA1c: 7.8%. On metformin 1000mg BID. No history of DKA. Not pregnant. BMI: 29. eGFR: 88 (normal). No current insulin therapy.",
    "38-year-old male with Type 2 diabetes, HbA1c 8.1%. Managed with diet and metformin. No complications. Non-smoker. Labs: normal renal and liver function.",
    "61-year-old male, T2DM for 6 years, HbA1c 7.4%. Stable on metformin. No cardiovascular events. eGFR 75. Blood pressure well controlled.",
    "44-year-old female with recent T2DM diagnosis (8 months ago). HbA1c 7.9%. Not on insulin. No DKA. BMI 31. Normal kidney function.",
    "57-year-old male, T2DM 10 years, HbA1c 8.5%. Metformin 500mg twice daily. No history of ketoacidosis. Non-pregnant. Renal function intact.",

    # Likely INELIGIBLE candidates
    "29-year-old female with T2DM, currently 24 weeks pregnant. HbA1c 7.2%. On insulin pump. OB/GYN monitoring closely. Not eligible per pregnancy exclusion.",
    "67-year-old male, T2DM with severe diabetic nephropathy. eGFR 18 (Stage 5 CKD). On dialysis three times weekly. Current insulin regimen.",
    "41-year-old male, history of two DKA episodes (most recent 6 months ago). Currently on basal-bolus insulin therapy. HbA1c 9.1%.",
    "72-year-old female, T2DM, eGFR 22. Stage 4 CKD with active proteinuria. On insulin glargine. Recent hospitalization for acute kidney injury.",
    "35-year-old female, T2DM, breastfeeding a 4-month-old infant. HbA1c 7.6%. On metformin (held per lactation guidance). Seeking alternative therapy.",

    # NEEDS_REVIEW borderline cases
    "63-year-old male with T2DM. HbA1c 6.8% (borderline — just under 7%). On metformin 500mg daily. Mild CKD stage 2 (eGFR 68). No history of DKA.",
    "48-year-old female, possible T2DM vs. MODY (genetic testing pending). HbA1c 7.1%. Not on insulin. Family history ambiguous. No DKA.",
    "55-year-old male, T2DM with recent cardiac stent placement (3 months ago). HbA1c 8.2%. On metformin. Cardiologist approval pending for trial participation.",
    "39-year-old female, T2DM, age 39 (within range). HbA1c 7.3%. Recently started low-dose insulin (2 weeks ago after metformin failure). Not yet on stable insulin regimen.",
    "66-year-old male, T2DM 15 years. HbA1c 7.0% (marginal). BMI 40. eGFR 55. No DKA. On metformin with dose reduction for renal function — borderline eligibility.",
]


_CONCURRENCY = 5   # max simultaneous pipeline runs; each patient is now 2 LLM calls (evaluation + verdict)


async def _screen_one(
    sem: asyncio.Semaphore,
    pl,
    trial_id: int,
    summary: str,
    index: int,
    total: int,
) -> dict:
    """Run one patient through the pipeline, bounded by the shared semaphore."""
    async with sem:
        print(f"  [{index}/{total}] Starting: {summary[:70]}...")
        try:
            result = await pl.run(
                user_input={
                    "trial_id": trial_id,
                    "patient_summary": summary,
                    "is_synthetic": True,
                },
                user_id="seed-script",
            )
            verdict = (result.get("verdict_agent_output") or {}).get("eligibility_status", "UNKNOWN")
            conf = (result.get("verdict_agent_output") or {}).get("confidence_score", 0.0)
            dup = (result.get("storage") or {}).get("duplicate", False)
            errors = result.get("errors", [])
            status_str = "DUP" if dup else verdict
            print(f"  [{index}/{total}] Done → {status_str} (conf: {conf:.0%}){' errors: ' + str(errors) if errors else ''}")
            return {"index": index, "status": status_str, "conf": conf, "errors": errors}
        except Exception as e:
            print(f"  [{index}/{total}] ERROR: {e}")
            return {"index": index, "status": "ERROR", "conf": 0.0, "errors": [str(e)]}


async def seed_trial(
    trial_id: int,
    count: int,
    dry_run: bool = False,
    use_llm: bool = True,
    on_progress=None,   # optional callable(completed: int, total: int)
) -> None:
    """
    Seed `count` synthetic patient screenings for the given trial.

    Profile generation:
      use_llm=True (default): Claude Haiku generates profiles on-the-fly from the
        trial's criteria text. Supports any count — no hardcoded cap.
      use_llm=False: Uses the 15 hardcoded fallback profiles. Count is capped at 15.

    Screening runs at most _CONCURRENCY pipelines simultaneously via asyncio.Semaphore.
    """
    from storage.db_client import get_trial_by_id, init_db
    import pipeline as pl

    await asyncio.to_thread(init_db)
    trial = await asyncio.to_thread(get_trial_by_id, trial_id)
    if trial is None:
        print(f"ERROR: No trial found with id={trial_id}. Create one via the Streamlit app first.")
        sys.exit(1)

    print(f"Seeding synthetic data for trial: '{trial['name']}' (id={trial_id})")

    if use_llm:
        print(f"Generating {count} patient profile(s) with Claude Sonnet...")
        from agents.synthetic_generator import generate_patient_summaries
        structured = trial.get("structured_criteria")
        if structured:
            print("  Using cached structured criteria for higher eligible accuracy.")
        profiles = await generate_patient_summaries(
            trial["criteria_text"], count, structured_criteria=structured
        )
        print(f"Generated {len(profiles)} profile(s).\n")
    else:
        profiles = _SYNTHETIC_PROFILES[:count]
        if count > len(_SYNTHETIC_PROFILES):
            print(f"WARNING: --no-llm caps count at {len(_SYNTHETIC_PROFILES)} hardcoded profiles.")
        print(f"Using {len(profiles)} hardcoded profile(s).\n")

    print(f"Dispatching {len(profiles)} patient(s) ({_CONCURRENCY} at a time)...\n")

    if dry_run:
        for i, summary in enumerate(profiles, 1):
            print(f"  [{i}/{len(profiles)}] DRY RUN — would screen: {summary[:80]}...")
        return

    sem = asyncio.Semaphore(_CONCURRENCY)
    tasks = [
        _screen_one(sem, pl, trial_id, summary, i, len(profiles))
        for i, summary in enumerate(profiles, 1)
    ]

    results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if on_progress:
            on_progress(completed, len(profiles))

    ok = sum(1 for r in results if r["status"] not in ("ERROR", "DUP"))
    dups = sum(1 for r in results if r["status"] == "DUP")
    errs = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\nDone. {ok} inserted, {dups} duplicate(s), {errs} error(s).")
    return {"ok": ok, "dups": dups, "errors": errs, "generated": len(profiles)}


def main():
    parser = argparse.ArgumentParser(
        description="Seed synthetic patient screenings for a clinical trial."
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        required=True,
        help="ID of the trial in the trials table (create via Streamlit app first).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of synthetic patients to seed (default: 15, no max when using LLM generation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print profiles without running the pipeline or writing to DB.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use hardcoded fallback profiles instead of LLM generation (caps at 15).",
    )
    args = parser.parse_args()

    asyncio.run(seed_trial(
        args.trial_id,
        args.count,
        dry_run=args.dry_run,
        use_llm=not args.no_llm,
    ))


if __name__ == "__main__":
    main()
