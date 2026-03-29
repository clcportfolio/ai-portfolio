"""
evaluation_agent.py — Evaluates patient summary against each criterion individually with detailed reasoning
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
AGENT_NAME = "evaluation_agent"


class CriterionEvaluation(BaseModel):
    criterion_id: str = Field(description="Unique identifier for the criterion")
    criterion_text: str = Field(description="The original criterion text")
    meets_criterion: bool = Field(description="True if patient meets this criterion, False otherwise")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed explanation of why the patient does or does not meet this criterion")
    relevant_patient_info: str = Field(description="Specific patient information that supports this evaluation")


class EvaluationResult(BaseModel):
    evaluations: List[CriterionEvaluation] = Field(description="Individual evaluation for each criterion")
    overall_notes: str = Field(description="General observations about the patient's eligibility profile")


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.3,
    )


def _get_handler() -> CallbackHandler:
    # Langfuse v4: credentials read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars
    return CallbackHandler()


def run(state: dict) -> dict:
    """
    Evaluates patient summary against each criterion individually with detailed reasoning.
    Reads criteria_agent_output and patient summary, writes to evaluation_agent_output.
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        # Get structured criteria from previous agent
        criteria_output = state.get("criteria_agent_output")
        if not criteria_output:
            state["errors"].append(f"{AGENT_NAME}: no criteria_agent_output found")
            state["evaluation_agent_output"] = None
            return state

        patient_summary = state["input"].get("patient_summary", "")
        if not patient_summary:
            state["errors"].append(f"{AGENT_NAME}: no patient_summary in input")
            state["evaluation_agent_output"] = None
            return state

        llm = _get_llm()
        handler = _get_handler()
        structured_llm = llm.with_structured_output(EvaluationResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical trial eligibility evaluator. Your task is to evaluate a patient summary against specific trial criteria.

For each criterion, you must:
1. Determine if the patient meets the criterion (true/false)
2. Provide a confidence score (0.0 to 1.0)
3. Give detailed reasoning explaining your decision
4. Identify the specific patient information that supports your evaluation

Be thorough and precise. Consider edge cases and ambiguous situations. If information is missing or unclear, note this in your reasoning and adjust confidence accordingly.

Return structured output with individual evaluations for each criterion."""),
            ("human", """Patient Summary:
{patient_summary}

Trial Criteria to Evaluate:
{criteria}

Evaluate the patient against each criterion individually. Provide detailed reasoning for each decision."""),
        ])

        # Format criteria for evaluation — criteria_agent returns inclusion_criteria / exclusion_criteria
        inclusion = criteria_output.get("inclusion_criteria", [])
        exclusion = criteria_output.get("exclusion_criteria", [])
        criteria_text = "INCLUSION CRITERIA:\n"
        for i, c in enumerate(inclusion, 1):
            criteria_text += f"  {i}. {c.get('criterion', '')} (category: {c.get('category', '')})\n"
        criteria_text += "\nEXCLUSION CRITERIA:\n"
        for i, c in enumerate(exclusion, 1):
            criteria_text += f"  {i}. {c.get('criterion', '')} (category: {c.get('category', '')})\n"

        chain = prompt | structured_llm
        result = chain.invoke(
            {
                "patient_summary": patient_summary,
                "criteria": criteria_text,
            },
            config={"callbacks": [handler]},
        )

        state["evaluation_agent_output"] = result.model_dump()

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")
        state["evaluation_agent_output"] = None

    return state

if __name__ == "__main__":
    import json
    test_state = {
        "input": {
            "trial_criteria": "Inclusion: Age 18-65. Type 2 diabetes.\nExclusion: Pregnant.",
            "patient_summary": "45-year-old male with Type 2 diabetes. Not pregnant.",
        },
        "pipeline_step": 1,
        "max_pipeline_steps": 10,
        "errors": [],
        "criteria_agent_output": {
            "inclusion_criteria": [
                {"type": "inclusion", "criterion": "Age 18-65", "category": "age", "measurable": True},
                {"type": "inclusion", "criterion": "Type 2 diabetes diagnosis", "category": "diagnosis", "measurable": True},
            ],
            "exclusion_criteria": [
                {"type": "exclusion", "criterion": "Pregnant or nursing", "category": "pregnancy", "measurable": True},
            ],
            "total_criteria_count": 3,
            "extraction_confidence": 0.95,
        },
    }
    result = run(test_state)
    print("pipeline_step:", result["pipeline_step"])
    print("evaluation_agent_output present:", "evaluation_agent_output" in result)
    print("errors:", result["errors"])
    print(json.dumps(result.get("evaluation_agent_output"), indent=2, default=str))
