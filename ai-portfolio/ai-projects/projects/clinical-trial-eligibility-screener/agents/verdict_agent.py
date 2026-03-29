"""
verdict_agent.py — Synthesizes individual evaluations into final eligibility verdict with plain-English explanation
"""
from __future__ import annotations
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

PROJECT_NAME = "clinical-trial-eligibility-screener"
AGENT_NAME = "verdict_agent"


class VerdictResult(BaseModel):
    eligibility_status: str = Field(description="ELIGIBLE | INELIGIBLE | NEEDS_REVIEW")
    confidence_score: float = Field(description="0.0 to 1.0 confidence in the verdict")
    summary: str = Field(description="Plain-English explanation of the verdict for coordinators")
    key_factors: list[str] = Field(description="List of most important factors that influenced the decision")
    next_steps: str = Field(description="Recommended actions for the coordinator")


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
    )


def _get_handler() -> CallbackHandler:
    # Langfuse v4: credentials read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars
    return CallbackHandler()


def run(state: dict) -> dict:
    """
    Synthesizes individual criterion evaluations into final eligibility verdict.
    Reads criteria_agent_output and evaluation_agent_output to generate comprehensive verdict.
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        llm = _get_llm()
        structured_llm = llm.with_structured_output(VerdictResult)
        handler = _get_handler()

        # Get outputs from previous agents
        criteria_output = state.get("criteria_agent_output")
        evaluation_output = state.get("evaluation_agent_output")
        patient_summary = state["input"].get("patient_summary", "")

        if not criteria_output or not evaluation_output:
            state["errors"].append(f"{AGENT_NAME}: missing required outputs from previous agents")
            state["verdict_agent_output"] = None
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical trial coordinator assistant. Your job is to synthesize individual criterion evaluations into a final eligibility verdict.

VERDICT CATEGORIES:
- ELIGIBLE: Patient clearly meets all inclusion criteria and has no disqualifying exclusions
- INELIGIBLE: Patient has clear disqualifying factors that prevent participation
- NEEDS_REVIEW: Borderline case requiring human coordinator review

CONFIDENCE SCORING:
- 0.9-1.0: Very clear case with strong evidence
- 0.7-0.8: Good evidence but some minor uncertainties
- 0.5-0.6: Moderate uncertainty, human review recommended
- Below 0.5: High uncertainty, definitely needs coordinator review

Provide clear, actionable guidance that helps coordinators make informed decisions efficiently."""),
            ("human", """TRIAL CRITERIA:
{criteria}

INDIVIDUAL EVALUATIONS:
{evaluations}

PATIENT SUMMARY:
{patient_summary}

Synthesize these evaluations into a final verdict. Focus on the most critical factors and provide clear next steps for the coordinator."""),
        ])

        chain = prompt | structured_llm
        result = chain.invoke(
            {
                "criteria": str(criteria_output),
                "evaluations": str(evaluation_output),
                "patient_summary": patient_summary,
            },
            config={"callbacks": [handler]},
        )

        state["verdict_agent_output"] = result.model_dump()

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")
        state["verdict_agent_output"] = None

    return state

if __name__ == "__main__":
    import json
    test_state = {
        "input": {
            "trial_criteria": "Inclusion: Age 18-65. Type 2 diabetes.\nExclusion: Pregnant.",
            "patient_summary": "45-year-old male with Type 2 diabetes. Not pregnant.",
        },
        "pipeline_step": 2,
        "max_pipeline_steps": 10,
        "errors": [],
        "criteria_agent_output": {
            "inclusion_criteria": [{"criterion": "Age 18-65"}, {"criterion": "Type 2 diabetes"}],
            "exclusion_criteria": [{"criterion": "Pregnant or nursing"}],
        },
        "evaluation_agent_output": {
            "evaluations": [
                {"criterion_text": "Age 18-65", "meets_criterion": True, "confidence": 0.95, "reasoning": "Patient is 45.", "relevant_patient_info": "45-year-old male"},
                {"criterion_text": "Type 2 diabetes", "meets_criterion": True, "confidence": 0.98, "reasoning": "Explicitly stated.", "relevant_patient_info": "Type 2 diabetes"},
                {"criterion_text": "Pregnant or nursing", "meets_criterion": False, "confidence": 0.90, "reasoning": "Patient is male.", "relevant_patient_info": "male"},
            ],
            "overall_notes": "Patient appears to be a strong candidate.",
        },
    }
    result = run(test_state)
    print("pipeline_step:", result["pipeline_step"])
    print("verdict_agent_output present:", "verdict_agent_output" in result)
    print("errors:", result["errors"])
    print(json.dumps(result.get("verdict_agent_output"), indent=2, default=str))
