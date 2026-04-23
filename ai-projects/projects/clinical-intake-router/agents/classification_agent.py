"""
Classification Agent — Clinical Intake Router
Classifies urgency level and determines the appropriate department
based on extracted intake fields.
"""

import os
from enum import Enum
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical triage specialist. Based on extracted patient intake fields,
you will classify the urgency level and determine the most appropriate department.

Urgency levels:
- Emergent: Life-threatening symptoms, acute chest pain, stroke symptoms, severe breathing
  difficulty, active hemorrhage, altered mental status, or any condition requiring
  immediate intervention within minutes.
- Urgent: Significant symptoms requiring evaluation within hours. Worsening chronic
  conditions, high fever, moderate pain, concerning but non-life-threatening presentations.
- Routine: Non-acute, stable conditions, follow-up visits, preventive care, mild symptoms
  that can wait days to weeks for evaluation.

Department assignment should be based on clinical context. Examples include but are
not limited to: Emergency, Primary Care, Cardiology, Oncology, Neurology, Orthopedics,
Mental Health, Radiology, Gastroenterology, Pulmonology, Endocrinology, Nephrology.
Choose the most specific and appropriate department. Do not use a hardcoded list — reason
from the clinical presentation.

Your classification_reasoning must be 2-4 sentences explaining the urgency and department choice.""",
    ),
    (
        "human",
        """Classify this patient's intake:

Patient name: {patient_name}
Age: {age}
Chief complaint: {chief_complaint}
Symptoms: {symptoms}
Medical history: {medical_history}
Current medications: {current_medications}
Allergies: {allergies}
Additional notes: {additional_notes}""",
    ),
])


class UrgencyLevel(str, Enum):
    ROUTINE = "Routine"
    URGENT = "Urgent"
    EMERGENT = "Emergent"


class ClassificationResult(BaseModel):
    urgency_level: UrgencyLevel = Field(description="Urgency classification: Routine, Urgent, or Emergent")
    department: str = Field(description="The most appropriate department for this patient")
    classification_reasoning: str = Field(
        description="2-4 sentence explanation of the urgency level and department assignment"
    )
    confidence: float = Field(
        description="Confidence in this classification, between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    red_flags: List[str] = Field(
        description="Clinical red flags or warning signs identified in this presentation",
        default_factory=list,
    )


def _get_handler(state: dict) -> CallbackHandler:
    """Return the pipeline-scoped handler if present, else a standalone one."""
    return state.get("langfuse_handler") or CallbackHandler()


# Built once at module load — reused on every request.
_chain = CLASSIFICATION_PROMPT | ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0,
).with_structured_output(ClassificationResult)


async def run(state: dict) -> dict:
    """Classify urgency and department from extracted fields. Writes to state['classification_output']."""
    extraction = state.get("extraction_output")
    if not extraction:
        state["errors"].append("classification_agent: no extraction_output to classify")
        state["classification_output"] = None
        state["pipeline_step"] += 1
        return state

    handler = _get_handler(state)

    try:
        result: ClassificationResult = await _chain.ainvoke(
            {
                "patient_name": extraction.get("patient_name", "Unknown"),
                "age": extraction.get("age") or extraction.get("date_of_birth") or "Not provided",
                "chief_complaint": extraction.get("chief_complaint", "Not provided"),
                "symptoms": ", ".join(extraction.get("symptoms", [])) or "None listed",
                "medical_history": ", ".join(extraction.get("medical_history", [])) or "None listed",
                "current_medications": ", ".join(extraction.get("current_medications", [])) or "None listed",
                "allergies": ", ".join(extraction.get("allergies", [])) or "None listed",
                "additional_notes": extraction.get("additional_notes") or "None",
            },
            config={"callbacks": [handler], "run_name": "classification_agent"},
        )
        state["classification_output"] = result.model_dump()
    except Exception as e:
        state["errors"].append(f"classification_agent error: {str(e)}")
        state["classification_output"] = None

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
            "date_of_birth": "09/14/1972",
            "chief_complaint": "Severe chest pain radiating to left arm, onset ~2 hours ago",
            "symptoms": ["chest tightness", "shortness of breath", "mild nausea", "diaphoresis"],
            "medical_history": ["hypertension", "hyperlipidemia", "type 2 diabetes"],
            "current_medications": ["metformin 500mg BID", "lisinopril 10mg daily", "atorvastatin 40mg"],
            "allergies": ["penicillin (rash)"],
            "insurance": "Blue Cross Blue Shield PPO",
            "referral_source": "self-referral via ER walk-in",
            "additional_notes": None,
        },
    }
    result = asyncio.run(run(test_state))
    print("pipeline_step:", result["pipeline_step"])
    print("classification_output present:", "classification_output" in result)
    print("errors:", result["errors"])
    print(json.dumps(result.get("classification_output"), indent=2, default=str))
