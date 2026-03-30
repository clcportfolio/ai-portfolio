"""
Extraction Agent — Clinical Intake Router
Extracts structured fields from raw intake form text.
"""

import os
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical data extraction specialist. Your job is to extract structured
information from raw clinical intake forms submitted by healthcare staff.

Extract every available field from the text. If a field is not present, use null.
Do not infer or guess values not explicitly stated. Extract exactly what is written.

Be thorough — chief complaint and symptoms are always required. Patient name is required.
All other fields are optional but should be captured if present.""",
    ),
    (
        "human",
        """Extract all structured fields from this clinical intake form:

{intake_text}""",
    ),
])


class ExtractedFields(BaseModel):
    patient_name: str = Field(description="Full name of the patient")
    age: Optional[int] = Field(description="Patient age in years", default=None)
    date_of_birth: Optional[str] = Field(description="Date of birth (as written)", default=None)
    chief_complaint: str = Field(description="Primary reason for the visit or chief complaint")
    symptoms: List[str] = Field(
        description="List of current symptoms reported by patient or staff",
        default_factory=list,
    )
    medical_history: List[str] = Field(
        description="Relevant past medical history items",
        default_factory=list,
    )
    current_medications: List[str] = Field(
        description="Medications the patient is currently taking",
        default_factory=list,
    )
    allergies: List[str] = Field(
        description="Known allergies (medications, food, environmental)",
        default_factory=list,
    )
    insurance: Optional[str] = Field(
        description="Insurance provider or coverage type", default=None
    )
    referral_source: Optional[str] = Field(
        description="Who or where referred the patient", default=None
    )
    additional_notes: Optional[str] = Field(
        description="Any other clinically relevant information not captured above",
        default=None,
    )


def _get_handler(state: dict) -> CallbackHandler:
    """Return the pipeline-scoped handler if present, else a standalone one."""
    return state.get("langfuse_handler") or CallbackHandler()


def run(state: dict) -> dict:
    """Extract structured fields from intake form text. Writes to state['extraction_output']."""
    intake_text = state["input"].get("text", "")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ExtractedFields)
    chain = EXTRACTION_PROMPT | structured_llm

    handler = _get_handler(state)

    try:
        result: ExtractedFields = chain.invoke(
            {"intake_text": intake_text},
            config={"callbacks": [handler], "run_name": "extraction_agent"},
        )
        state["extraction_output"] = result.model_dump()
    except Exception as e:
        state["errors"].append(f"extraction_agent error: {str(e)}")
        state["extraction_output"] = None

    state["pipeline_step"] += 1
    return state


if __name__ == "__main__":
    test_state = {
        "input": {
            "text": (
                "Patient: Maria Gonzalez, DOB 09/14/1972 (52 y/o)\n"
                "Chief Complaint: Severe chest pain radiating to left arm, onset ~2 hours ago.\n"
                "Symptoms: chest tightness, shortness of breath, mild nausea, diaphoresis\n"
                "PMH: hypertension, hyperlipidemia, type 2 diabetes\n"
                "Medications: metformin 500mg BID, lisinopril 10mg daily, atorvastatin 40mg\n"
                "Allergies: penicillin (rash)\n"
                "Insurance: Blue Cross Blue Shield PPO\n"
                "Referred by: self-referral via ER walk-in"
            )
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }
    result = run(test_state)
    print("pipeline_step:", result["pipeline_step"])
    print("extraction_output present:", "extraction_output" in result)
    print("errors:", result["errors"])
    import json
    print(json.dumps(result.get("extraction_output"), indent=2, default=str))
