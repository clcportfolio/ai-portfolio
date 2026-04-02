"""
pipeline.py — Clinical Trial Eligibility Screener
Orchestrates criteria extraction → per-criterion evaluation → verdict synthesis.

Input modes:
  1. {"trial_id": int, "patient_summary": str}
       — Trial is loaded from DB. If structured_criteria is already cached,
         criteria_agent is skipped (zero LLM cost for criteria extraction).
  2. {"trial_criteria": str, "patient_summary": str}
       — Ad-hoc criteria; criteria_agent always runs.
  3. {"trial_id": int, "trial_criteria": str, "patient_summary": str, "trial_name": str}
       — Save-and-screen: criteria text is inserted as a new trial on first run.

Async design:
  - criteria_agent and verdict_agent are single async ainvoke() calls.
  - evaluation_agent fans out N per-criterion ainvoke() calls via asyncio.gather(),
    so wall-clock time for 10 criteria ≈ wall-clock time for 1.

Observability:
  - @observe(name="eligibility_screen") creates the root Langfuse trace.
  - propagate_attributes(trace_name="clinical-trial-eligibility-screener") pins the
    trace display name on this span and all child spans, including CallbackHandler spans.
  - A single CallbackHandler scoped to this trace is stored in state["langfuse_handler"]
    and shared across all agents.

Storage:
  - After a successful screen, the result is written to the screenings table in Supabase.
  - If the trial was new (trial_id was None), it is inserted into the trials table first
    and its structured_criteria is cached for future screens.
  - Storage failures are non-fatal.
"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

import asyncio
import logging
from typing import Optional

from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from agents.criteria_agent import run as criteria_run
from agents.evaluation_agent import run as evaluation_run
from agents.verdict_agent import run as verdict_run
from guardrails import validate_input, sanitize_output, rate_limit_check

logger = logging.getLogger(__name__)


def build_initial_state(validated_input: dict) -> dict:
    return {
        "input": validated_input,
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
        "langfuse_handler": None,
        "criteria_agent_output": None,
        "evaluation_agent_output": None,
        "verdict_agent_output": None,
        "output": None,
    }


@observe(name="eligibility_screen")
async def run(
    user_input: dict,
    user_id: str = "anonymous",
) -> dict:
    """
    Main pipeline entry point. Called by app.py.

    Flow:
    1. Validate input
    2. Load trial from DB if trial_id provided (skip criteria_agent if cached)
    3. criteria_agent  — extract and structure criteria (may be skipped)
    4. evaluation_agent — evaluate patient per criterion in parallel
    5. verdict_agent   — synthesize final verdict
    6. Store result to DB (non-fatal)

    Returns the full state dict. Check state["output"] for the verdict.
    """
    if not rate_limit_check(user_id):
        return {
            "input": user_input,
            "pipeline_step": 0,
            "max_pipeline_steps": 10,
            "errors": ["Rate limit exceeded. Please try again later."],
            "langfuse_handler": None,
            "output": None,
        }

    validated = validate_input(user_input)
    state = build_initial_state(validated)

    # ── Load stored trial (if trial_id provided) ──────────────────────────────
    trial_id: Optional[int] = validated.get("trial_id")
    trial_row: Optional[dict] = None

    if trial_id is not None:
        try:
            from storage.db_client import get_trial_by_id
            trial_row = await asyncio.to_thread(get_trial_by_id, trial_id)
            if trial_row is None:
                state["errors"].append(f"Trial id={trial_id} not found in database.")
                return state
            # Inject trial_criteria from DB so guardrails / agents can access it
            state["input"]["trial_criteria"] = trial_row["criteria_text"]
            if trial_row.get("structured_criteria"):
                # Cache hit — skip criteria_agent entirely
                state["criteria_agent_output"] = trial_row["structured_criteria"]
                state["pipeline_step"] += 1   # account for the skipped agent step
                logger.info("Skipping criteria_agent — using cached structured_criteria for trial_id=%s", trial_id)
        except ImportError:
            logger.warning("DB storage not configured — proceeding without trial cache.")
        except Exception as e:
            logger.warning("Failed to load trial from DB (non-fatal): %s", e)
            state["errors"].append(f"Could not load trial from DB: {e}")

    with propagate_attributes(trace_name="clinical-trial-eligibility-screener", user_id=user_id):

        lf = get_client()
        lf.set_current_trace_io(
            input={
                "trial_id": trial_id,
                "has_custom_criteria": "trial_criteria" in validated and trial_id is None,
                "patient_summary_length": len(validated.get("patient_summary", "")),
            }
        )

        state["langfuse_handler"] = CallbackHandler(
            trace_context=TraceContext(
                trace_id=lf.get_current_trace_id(),
                parent_span_id=lf.get_current_observation_id(),
            )
        )

        # Step 1: Extract criteria (skip if cached from DB)
        if state["criteria_agent_output"] is None:
            state = await criteria_run(state)
            if state["pipeline_step"] >= state["max_pipeline_steps"]:
                state["errors"].append("pipeline: max_pipeline_steps reached after criteria_agent")
                return sanitize_output(state)

        # Step 2: Evaluate patient against each criterion (parallel LLM calls)
        state = await evaluation_run(state)
        if state["pipeline_step"] >= state["max_pipeline_steps"]:
            state["errors"].append("pipeline: max_pipeline_steps reached after evaluation_agent")
            return sanitize_output(state)

        # Step 3: Synthesize final verdict
        state = await verdict_run(state)

        # Surface verdict as state["output"]
        state["output"] = state.get("verdict_agent_output")
        state = sanitize_output(state)

        # Update trace output — metadata only, never breaks the return
        try:
            verdict = state.get("verdict_agent_output") or {}
            lf.set_current_trace_io(
                output={
                    "eligibility_status": verdict.get("eligibility_status"),
                    "confidence_score": verdict.get("confidence_score"),
                }
            )
        except Exception:
            pass

    # ── Storage: write screening result to DB ─────────────────────────────────
    patient_summary = validated.get("patient_summary", "")
    if patient_summary and state.get("verdict_agent_output"):
        try:
            from storage.db_client import (
                init_db,
                insert_trial,
                update_trial_structured_criteria,
                insert_screening,
            )

            await asyncio.to_thread(init_db)

            # If this was an ad-hoc run with a trial_name provided, save the trial
            if trial_id is None and validated.get("trial_name") and validated.get("trial_criteria"):
                try:
                    trial_row = await asyncio.to_thread(
                        insert_trial,
                        validated["trial_name"],
                        validated["trial_criteria"],
                        state.get("criteria_agent_output"),
                    )
                    trial_id = trial_row["id"]
                    logger.info("Saved new trial id=%s name='%s'", trial_id, validated["trial_name"])
                except ValueError:
                    # Trial name already exists — that's fine, just continue
                    pass
                except Exception as e:
                    logger.warning("Could not save trial (non-fatal): %s", e)

            # If trial was loaded from DB but structured_criteria was missing, cache it now
            if (
                trial_id is not None
                and trial_row is not None
                and not trial_row.get("structured_criteria")
                and state.get("criteria_agent_output")
            ):
                try:
                    await asyncio.to_thread(
                        update_trial_structured_criteria,
                        trial_id,
                        state["criteria_agent_output"],
                    )
                except Exception as e:
                    logger.warning("Could not cache structured_criteria (non-fatal): %s", e)

            # Insert screening row
            if trial_id is not None:
                is_synthetic = bool(validated.get("is_synthetic", False))
                db_row = await asyncio.to_thread(
                    insert_screening,
                    trial_id,
                    patient_summary,
                    state,
                    is_synthetic,
                )
                state["storage"] = {"db": db_row, "duplicate": db_row.get("_duplicate", False)}
                logger.info(
                    "Screening stored: id=%s duplicate=%s",
                    db_row.get("id"),
                    db_row.get("_duplicate"),
                )

        except ImportError:
            logger.warning("DB storage not configured — screening result not persisted.")
        except Exception as e:
            logger.warning("Storage write failed (non-fatal): %s", e)
            state.setdefault("errors", []).append(f"Storage write failed: {e}")

    return state


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Clinical Trial Eligibility Screener pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; validate wiring only")
    args = parser.parse_args()

    test_input = {
        "trial_criteria": "Inclusion: Age 18-65, diagnosed diabetes type 2. Exclusion: pregnancy, severe kidney disease.",
        "patient_summary": "45-year-old male with type 2 diabetes, well-controlled on metformin. No kidney issues.",
    }

    if args.dry_run:
        validated = validate_input(test_input)
        state = build_initial_state(validated)
        state["output"] = "dry-run placeholder"
        state = sanitize_output(state)
        print("Dry run passed. State keys:", list(state.keys()))
        print("pipeline_step:", state["pipeline_step"])
        print("langfuse_handler in state:", "langfuse_handler" in state)
        print("errors:", state["errors"])
    else:
        print("Running full pipeline (LLM calls will be made)...")
        result = asyncio.run(run(test_input))
        print("\n--- PIPELINE RESULT ---")
        print("Final verdict:", result.get("output"))
        print(json.dumps(result, indent=2, default=str))
