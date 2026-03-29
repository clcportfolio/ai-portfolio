"""
pipeline.py — Clinical Trial Eligibility Screener
Automates eligibility screening by evaluating patient summaries against trial criteria.
"""

from guardrails import validate_input, sanitize_output
from agents import criteria_agent, evaluation_agent, verdict_agent


def build_initial_state(user_input: dict) -> dict:
    """Initialize pipeline state with user input and tracking variables."""
    return {
        "input": user_input,
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }


def run(user_input: dict) -> dict:
    """
    Main pipeline entry point. Called by app.py.
    
    Flow:
    1. Trial criteria text → criteria_agent for extraction
    2. Structured criteria + patient summary → evaluation_agent for assessments
    3. All evaluations → verdict_agent for final synthesis
    
    Returns the final state dict with state["output"] populated.
    """
    validated = validate_input(user_input)
    state = build_initial_state(validated)
    
    # Extract and structure trial criteria
    state = criteria_agent.run(state)
    
    # Evaluate patient against each criterion
    state = evaluation_agent.run(state)
    
    # Synthesize final eligibility verdict
    state = verdict_agent.run(state)

    # Surface final verdict as state["output"] for sanitize_output and app.py
    state["output"] = state.get("verdict_agent_output")

    state = sanitize_output(state)
    return state


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; validate wiring only")
    args = parser.parse_args()

    test_input = {
        "trial_criteria": "Inclusion: Age 18-65, diagnosed diabetes type 2. Exclusion: pregnancy, severe kidney disease.",
        "patient_summary": "45-year-old male with type 2 diabetes, well-controlled on metformin. No kidney issues.",
    }

    if args.dry_run:
        from guardrails import validate_input, sanitize_output
        validated = validate_input(test_input)
        state = build_initial_state(validated)
        state["output"] = "dry-run placeholder"
        state = sanitize_output(state)
        print("Dry run passed. State keys:", list(state.keys()))
    else:
        result = run(test_input)
        print("Final verdict:", result.get("output"))
        print(json.dumps(result, indent=2, default=str))