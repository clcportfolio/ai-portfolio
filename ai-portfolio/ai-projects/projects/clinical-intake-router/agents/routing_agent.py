"""
Routing Agent — Clinical Intake Router
Generates a plain-English routing summary with department assignment,
urgency flag, and recommended next steps.
"""

import os
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical care coordinator generating routing instructions for
healthcare staff. Write clear, actionable instructions that a non-clinical staff member
can follow immediately without needing medical training.

Your routing_summary should be 2-3 plain-English sentences: what this patient needs,
where they are going, and why it's at this urgency level.

recommended_next_steps should be 3-5 ordered, concrete actions (e.g., "Call the patient
to confirm appointment", "Notify the on-call cardiologist immediately", "Prepare intake
paperwork for Emergency").

follow_up_actions are secondary tasks that must happen after the initial routing
(e.g., "Update insurance verification", "Send patient intake summary to receiving
department within 30 minutes").

Write for a healthcare front-desk coordinator — professional, concise, no clinical jargon.""",
    ),
    (
        "human",
        """Generate routing instructions for this patient:

Patient name: {patient_name}
Department assigned: {department}
Urgency level: {urgency_level}
Chief complaint: {chief_complaint}
Classification reasoning: {classification_reasoning}
Red flags: {red_flags}""",
    ),
])


class RoutingResult(BaseModel):
    department: str = Field(description="Department the patient is being routed to")
    urgency_level: str = Field(description="Urgency level: Routine, Urgent, or Emergent")
    routing_summary: str = Field(
        description="2-3 plain-English sentences describing the routing decision for non-clinical staff"
    )
    recommended_next_steps: List[str] = Field(
        description="Ordered list of 3-5 concrete actions for the staff member to take now"
    )
    follow_up_actions: List[str] = Field(
        description="Secondary actions required after the initial routing is complete"
    )
    estimated_response_time: Optional[str] = Field(
        description="Expected response or appointment timeline based on urgency (e.g., 'Within 15 minutes', 'Within 48 hours')",
        default=None,
    )


def _get_handler(state: dict) -> CallbackHandler:
    """Return the pipeline-scoped handler if present, else a standalone one."""
    return state.get("langfuse_handler") or CallbackHandler()


# Built once at module load — reused on every request.
_chain = ROUTING_PROMPT | ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.3,
).with_structured_output(RoutingResult)


async def run(state: dict) -> dict:
    """Generate plain-English routing summary. Writes to state['routing_output'] and state['output']."""
    extraction = state.get("extraction_output")
    classification = state.get("classification_output")

    if not classification:
        state["errors"].append("routing_agent: no classification_output to route from")
        state["routing_output"] = None
        state["output"] = None
        state["pipeline_step"] += 1
        return state

    handler = _get_handler(state)
    red_flags = classification.get("red_flags", [])

    try:
        result: RoutingResult = await _chain.ainvoke(
            {
                "patient_name": extraction.get("patient_name", "Patient") if extraction else "Patient",
                "department": classification.get("department", "Unknown"),
                "urgency_level": classification.get("urgency_level", "Routine"),
                "chief_complaint": extraction.get("chief_complaint", "Not provided") if extraction else "Not provided",
                "classification_reasoning": classification.get("classification_reasoning", ""),
                "red_flags": ", ".join(red_flags) if red_flags else "None identified",
            },
            config={"callbacks": [handler], "run_name": "routing_agent"},
        )
        state["routing_output"] = result.model_dump()
        state["output"] = result.model_dump()
    except Exception as e:
        state["errors"].append(f"routing_agent error: {str(e)}")
        state["routing_output"] = None
        state["output"] = None

    state["pipeline_step"] += 1
    return state


if __name__ == "__main__":
    import asyncio
    import json

    test_state = {
        "input": {"text": ""},
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
        "extraction_output": {
            "patient_name": "Maria Gonzalez",
            "age": 52,
            "chief_complaint": "Severe chest pain radiating to left arm, onset ~2 hours ago",
            "symptoms": ["chest tightness", "shortness of breath", "nausea", "diaphoresis"],
            "medical_history": ["hypertension", "hyperlipidemia", "type 2 diabetes"],
            "current_medications": ["metformin 500mg BID", "lisinopril 10mg daily", "atorvastatin 40mg"],
            "allergies": ["penicillin (rash)"],
            "insurance": "Blue Cross Blue Shield PPO",
            "referral_source": "self-referral via ER walk-in",
            "additional_notes": None,
        },
        "classification_output": {
            "urgency_level": "Emergent",
            "department": "Emergency",
            "classification_reasoning": (
                "The patient presents with classic ACS symptoms including chest pain "
                "radiating to the left arm, diaphoresis, and shortness of breath in a "
                "52-year-old with multiple cardiac risk factors. This is a time-sensitive "
                "emergency requiring immediate evaluation."
            ),
            "confidence": 0.97,
            "red_flags": [
                "Chest pain radiating to left arm",
                "Diaphoresis",
                "Multiple cardiac risk factors (HTN, HLD, DM2)",
                "Onset 2 hours ago — within intervention window",
            ],
        },
    }
    result = asyncio.run(run(test_state))
    print("pipeline_step:", result["pipeline_step"])
    print("routing_output present:", "routing_output" in result)
    print("errors:", result["errors"])
    print(json.dumps(result.get("routing_output"), indent=2, default=str))
