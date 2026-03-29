"""
criteria_agent.py — Extracts and structures individual inclusion/exclusion criteria from raw clinical trial text
"""
from __future__ import annotations
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv(find_dotenv(), override=True)

PROJECT_NAME = "clinical-trial-eligibility-screener"
AGENT_NAME = "criteria_agent"


class CriterionItem(BaseModel):
    type: str = Field(description="inclusion or exclusion")
    criterion: str = Field(description="The specific criterion text")
    category: str = Field(description="Category like age, diagnosis, medication, lab_values, etc.")
    measurable: bool = Field(description="Whether this criterion can be objectively measured")


class CriteriaExtractionResult(BaseModel):
    inclusion_criteria: List[CriterionItem] = Field(description="List of inclusion criteria")
    exclusion_criteria: List[CriterionItem] = Field(description="List of exclusion criteria")
    total_criteria_count: int = Field(description="Total number of criteria extracted")
    extraction_confidence: float = Field(description="Confidence score 0.0 to 1.0")


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0,
    )


def _get_handler() -> CallbackHandler:
    # Langfuse v4: credentials read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars
    return CallbackHandler()


def run(state: dict) -> dict:
    """
    Extracts and structures individual inclusion/exclusion criteria from raw clinical trial text.
    Reads trial criteria text from state["input"]["trial_criteria"] and outputs structured criteria.
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        trial_criteria = state["input"].get("trial_criteria", "")
        if not trial_criteria:
            state["errors"].append(f"{AGENT_NAME}: no trial_criteria in input")
            state["criteria_agent_output"] = None
            return state

        llm = _get_llm()
        handler = _get_handler()
        structured_llm = llm.with_structured_output(CriteriaExtractionResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical research coordinator expert at parsing clinical trial eligibility criteria.

Your task is to extract and structure individual inclusion and exclusion criteria from raw clinical trial text.

For each criterion:
1. Identify if it's inclusion or exclusion
2. Extract the specific criterion text clearly
3. Categorize it (age, diagnosis, medication, lab_values, medical_history, pregnancy, etc.)
4. Determine if it's objectively measurable (lab values, age = measurable; "good general health" = not measurable)

Be thorough - extract ALL criteria mentioned. Each bullet point or numbered item should typically be a separate criterion.

Return structured output with high confidence scores for clear extractions."""),
            ("human", "Extract and structure all eligibility criteria from this clinical trial text:\n\n{trial_criteria}"),
        ])

        chain = prompt | structured_llm
        result = chain.invoke(
            {"trial_criteria": trial_criteria},
            config={"callbacks": [handler]}
        )

        state["criteria_agent_output"] = result.model_dump()

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")
        state["criteria_agent_output"] = None

    return state

if __name__ == "__main__":
    import json
    test_state = {
        "input": {
            "trial_criteria": (
                "Inclusion: Age 18-65. Diagnosed with Type 2 diabetes. HbA1c > 7%.\n"
                "Exclusion: Pregnant or nursing. History of diabetic ketoacidosis. Current insulin therapy."
            ),
            "patient_summary": "45-year-old male with Type 2 diabetes on metformin. HbA1c 8.2%. No DKA history.",
        },
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }
    result = run(test_state)
    print("pipeline_step:", result["pipeline_step"])
    print("criteria_agent_output present:", "criteria_agent_output" in result)
    print("errors:", result["errors"])
    print(json.dumps(result.get("criteria_agent_output"), indent=2, default=str))
