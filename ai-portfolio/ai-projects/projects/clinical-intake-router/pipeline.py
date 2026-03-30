"""
Pipeline — Clinical Intake Router
Orchestrates extraction → classification → routing with guardrails middleware.

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

from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from agents.extraction_agent import run as extraction_run
from agents.classification_agent import run as classification_run
from agents.routing_agent import run as routing_run
from guardrails import validate_input, sanitize_output, rate_limit_check


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
def run(input_data: dict, user_id: str = "anonymous") -> dict:
    """
    Full pipeline: validate → extract → classify → route → sanitize.

    The @observe(name="intake_route") decorator creates the root span visible
    in the Langfuse trace hierarchy.

    propagate_attributes(trace_name="clinical-intake-router") pins the trace
    display name on the current span AND all child spans created within that
    context — including the CallbackHandler spans for each agent. This prevents
    any agent's run_name from overwriting the trace name in the Langfuse list view.

    Standalone agent runs fall back to CallbackHandler() and create their own
    trace named by run_name (e.g. "extraction_agent").

    Args:
        input_data: dict with key 'text' containing raw intake form text
        user_id: optional identifier for rate limiting and Langfuse trace metadata

    Returns:
        Final state dict. Check state['output'] for routing result.
        Check state['errors'] for non-fatal errors.
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
        state = extraction_run(state)
        if state["pipeline_step"] >= state["max_pipeline_steps"]:
            state["errors"].append("pipeline: max_pipeline_steps reached after extraction")
            return sanitize_output(state)

        # Step 2: Classify urgency and department
        state = classification_run(state)
        if state["pipeline_step"] >= state["max_pipeline_steps"]:
            state["errors"].append("pipeline: max_pipeline_steps reached after classification")
            return sanitize_output(state)

        # Step 3: Generate routing summary
        state = routing_run(state)

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
        result = run(test_input)
        print("\n--- PIPELINE RESULT ---")
        print(json.dumps(result, indent=2, default=str))
