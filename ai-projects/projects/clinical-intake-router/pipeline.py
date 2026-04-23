"""
Pipeline — Clinical Intake Router
Orchestrates extraction → classification → routing with guardrails middleware.

Storage layer (added in v2):
- On successful run, the raw file (if provided) is uploaded to S3.
- A structured submission row is written to Supabase via db_client.
- Duplicate detection is done before the pipeline runs — if the file hash
  already exists in the DB, the pipeline is skipped and the existing result
  is returned immediately.

Observability: @observe on run() creates the root Langfuse trace. A single
CallbackHandler built from the current trace/span context is retrieved once and
passed to all three agents via state['langfuse_handler'], so every LLM call
nests under one trace in Langfuse.

Langfuse v4 API notes:
- observe, get_client, propagate_attributes imported from langfuse directly
- propagate_attributes(trace_name=...)  sets langfuse.trace.name on the current
  span AND all child spans — the correct way to pin the trace display name so
  CallbackHandler spans cannot overwrite it
- CallbackHandler  accepts trace_context=TraceContext(trace_id, parent_span_id)
- set_current_trace_io  updates input/output on the current trace
"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

import asyncio
import logging
from typing import Optional

from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from agents.extraction_agent import run as extraction_run
from agents.classification_agent import run as classification_run
from agents.routing_agent import run as routing_run
from guardrails import validate_input, sanitize_output, rate_limit_check

logger = logging.getLogger(__name__)


def build_initial_state(validated_input: dict) -> dict:
    return {
        "input": validated_input,
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
        "langfuse_handler": None,   # set by pipeline before agents run
        "extraction_output": None,
        "classification_output": None,
        "routing_output": None,
        "output": None,
    }


@observe(name="intake_route")
async def run(
    input_data: dict,
    user_id: str = "anonymous",
    file_bytes: Optional[bytes] = None,
    original_filename: Optional[str] = None,
    content_type: str = "text/plain",
) -> dict:
    """
    Full pipeline: validate → (dedup check) → extract → classify → route → sanitize → store.

    The @observe(name="intake_route") decorator creates the root span visible
    in the Langfuse trace hierarchy.

    propagate_attributes(trace_name="clinical-intake-router") pins the trace
    display name on the current span AND all child spans created within that
    context — including the CallbackHandler spans for each agent. This prevents
    any agent's run_name from overwriting the trace name in the Langfuse list view.

    Standalone agent runs fall back to CallbackHandler() and create their own
    trace named by run_name (e.g. "extraction_agent").

    Args:
        input_data:        dict with key 'text' containing raw intake form text
        user_id:           optional identifier for rate limiting and Langfuse trace metadata
        file_bytes:        optional raw bytes of the uploaded file (for S3 storage)
        original_filename: original filename, used as S3 key suffix and DB display name
        content_type:      MIME type of the uploaded file

    Returns:
        Final state dict. Check state['output'] for routing result.
        Check state['errors'] for non-fatal errors.
        Check state['storage'] for S3/DB write results (if storage is configured).
    """
    # Rate limit check (stub — always passes; replace with Redis in prod)
    if not rate_limit_check(user_id):
        return {
            "input": input_data,
            "pipeline_step": 0,
            "max_pipeline_steps": 10,
            "errors": ["Rate limit exceeded. Please try again later."],
            "langfuse_handler": None,
            "output": None,
        }

    # Pre-flight: validate input
    validated = validate_input(input_data)

    # ── Deduplication check ───────────────────────────────────────────────────
    # Hash the content before running any LLM agents. If we've seen this exact
    # file before, return the stored result immediately — no API cost incurred.
    storage_result = {"s3": None, "db": None, "duplicate": False, "storage_errors": []}

    try:
        from storage.s3_client import hash_file
        from storage.db_client import submission_exists, init_db

        content_to_hash = file_bytes if file_bytes else validated.get("text", "").encode("utf-8")
        file_hash = hash_file(content_to_hash)

        existing = submission_exists(file_hash)
        if existing and existing.get("routing_output"):
            # Only short-circuit if the cached row has routing output.
            # Files uploaded without routing (upload-only or synced from S3)
            # have no routing output and must be run through the pipeline.
            logger.info("Duplicate submission detected — returning cached result.")
            routing = existing.get("routing_output") or {}
            return {
                "input": validated,
                "pipeline_step": 0,
                "max_pipeline_steps": 10,
                "errors": [],
                "langfuse_handler": None,
                "extraction_output": existing.get("extraction_output"),
                "classification_output": existing.get("classification_output"),
                "routing_output": routing,
                "output": routing,
                "storage": {
                    "s3": None,
                    "db": existing,
                    "duplicate": True,
                    "storage_errors": [],
                },
            }
    except Exception as e:
        # Storage is not critical — if env vars are missing, log and continue
        logger.warning("Storage dedup check skipped: %s", e)
        file_hash = None
        storage_result["storage_errors"].append(f"dedup check skipped: {e}")

    # Build initial state
    state = build_initial_state(validated)

    # propagate_attributes pins trace_name on this span and ALL child spans,
    # including those created by CallbackHandler inside each agent.
    # user_id is also propagated so every span carries it for Langfuse aggregations.
    with propagate_attributes(trace_name="clinical-intake-router", user_id=user_id):

        lf = get_client()
        lf.set_current_trace_io(
            input={"text_length": len(validated.get("text", ""))},
        )

        # Build a CallbackHandler scoped to this trace and observation.
        # All three agents share this single handler — their LLM calls nest
        # under this trace as child spans.
        state["langfuse_handler"] = CallbackHandler(
            trace_context=TraceContext(
                trace_id=lf.get_current_trace_id(),
                parent_span_id=lf.get_current_observation_id(),
            )
        )

        # Step 1: Extract structured fields
        state = await extraction_run(state)
        if state["pipeline_step"] >= state["max_pipeline_steps"]:
            state["errors"].append("pipeline: max_pipeline_steps reached after extraction")
            return sanitize_output(state)

        # Step 2: Classify urgency and department
        state = await classification_run(state)
        if state["pipeline_step"] >= state["max_pipeline_steps"]:
            state["errors"].append("pipeline: max_pipeline_steps reached after classification")
            return sanitize_output(state)

        # Step 3: Generate routing summary
        state = await routing_run(state)

        # Post-flight: sanitize output
        state = sanitize_output(state)

        # Update trace output — metadata only, never breaks the return
        try:
            routing = state.get("routing_output") or {}
            lf.set_current_trace_io(
                output={
                    "urgency_level": routing.get("urgency_level"),
                    "department": routing.get("department"),
                },
            )
        except Exception:
            pass

    # ── Storage: S3 upload + DB insert ───────────────────────────────────────
    # Runs AFTER the pipeline succeeds. Storage failures are non-fatal.
    # S3 and DB writes are dispatched via asyncio.to_thread() so they don't
    # block the event loop — other concurrent requests can proceed while these
    # synchronous I/O calls (boto3, psycopg2) are in flight.
    # S3 must complete before DB insert because insert_submission needs the s3_key.
    if file_hash:
        try:
            from storage.s3_client import upload_file
            from storage.db_client import insert_submission, init_db

            # Ensure table exists (idempotent) — run in thread, non-blocking
            try:
                await asyncio.to_thread(init_db)
            except Exception as e:
                logger.warning("DB init skipped: %s", e)

            # S3 upload — non-blocking via to_thread
            s3_result = None
            if file_bytes and original_filename:
                try:
                    s3_result = await asyncio.to_thread(
                        upload_file, file_bytes, original_filename, content_type
                    )
                    storage_result["s3"] = s3_result
                    logger.info("File uploaded to S3: %s", s3_result.get("s3_key"))
                except Exception as e:
                    logger.warning("S3 upload failed (non-fatal): %s", e)
                    storage_result["storage_errors"].append(f"S3 upload failed: {e}")

            # DB write — non-blocking via to_thread.
            # If a partial row already exists (upload-only or synced from S3),
            # update it with the pipeline results rather than inserting a duplicate.
            try:
                from storage.db_client import submission_exists, update_submission

                def _db_write():
                    existing = submission_exists(file_hash)
                    if existing and not existing.get("routing_output"):
                        row = update_submission(file_hash, state)
                        row["_duplicate"] = False
                        return row
                    return insert_submission(
                        file_hash=file_hash,
                        pipeline_state=state,
                        s3_result=s3_result,
                        original_filename=original_filename,
                    )

                db_row = await asyncio.to_thread(_db_write)
                storage_result["db"] = db_row
                storage_result["duplicate"] = db_row.get("_duplicate", False)
            except Exception as e:
                logger.warning("DB insert/update failed (non-fatal): %s", e)
                storage_result["storage_errors"].append(f"DB write failed: {e}")

        except ImportError as e:
            logger.warning("Storage modules not available: %s", e)
            storage_result["storage_errors"].append(f"storage import failed: {e}")

    state["storage"] = storage_result
    return state


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Clinical Intake Router pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; validate guardrails wiring only",
    )
    args = parser.parse_args()

    test_input = {
        "text": (
            "Patient: James Whitfield, 67 y/o male\n"
            "Chief Complaint: Sudden onset severe headache, worst of his life. "
            "Associated with neck stiffness and photophobia. Onset 3 hours ago.\n"
            "PMH: Hypertension, previous TIA (2019)\n"
            "Medications: aspirin 81mg daily, amlodipine 5mg\n"
            "Allergies: sulfa drugs\n"
            "Insurance: Medicare\n"
            "Referred by: Primary Care MD"
        )
    }

    if args.dry_run:
        from guardrails import validate_input as vi, sanitize_output as so
        validated = vi(test_input)
        state = build_initial_state(validated)
        state["output"] = "dry-run placeholder"
        state = so(state)
        print("Dry run passed. State keys:", list(state.keys()))
        print("pipeline_step:", state["pipeline_step"])
        print("langfuse_handler in state:", "langfuse_handler" in state)
        print("errors:", state["errors"])
    else:
        print("Running full pipeline (LLM calls will be made)...")
        result = asyncio.run(run(test_input))
        print("\n--- PIPELINE RESULT ---")
        print(json.dumps(result, indent=2, default=str))
