"""
evaluation_agent.py — Evaluates a patient summary against all criteria in a single LLM call.

One call per patient instead of one call per criterion. Faster, fewer tokens, no semaphore
needed. Sonnet handles evaluating a full list of criteria in one pass without issue.
"""
from __future__ import annotations

import asyncio
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

PROJECT_NAME = "clinical-trial-eligibility-screener"
AGENT_NAME = "evaluation_agent"


# ── Output models ─────────────────────────────────────────────────────────────

class CriterionEvaluation(BaseModel):
    criterion_id: str
    criterion_text: str
    meets_criterion: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    relevant_patient_info: str


class EvaluationResult(BaseModel):
    evaluations: List[CriterionEvaluation]
    overall_assessment: str


class _EvalList(BaseModel):
    """Minimal model used for the LLM call — no overall_assessment, terse field descriptions."""
    evaluations: List[CriterionEvaluation] = Field(
        description="One entry per criterion. Use exact criterion_id values from the prompt."
    )


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        temperature=0,
    )


def _get_handler() -> CallbackHandler:
    return CallbackHandler()


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical trial eligibility evaluator. Evaluate every criterion in one pass and return a concise list.

meets_criterion semantics:
  INCLUSION: true = patient satisfies it | false = patient fails it
  EXCLUSION: true = patient HAS the disqualifying condition | false = patient does NOT have it

confidence: 0.85+ if explicitly stated | 0.5-0.84 if implied | <0.5 if absent/ambiguous

Keep reasoning to one short sentence. Keep relevant_patient_info to a brief phrase.
Use the exact criterion_id values provided."""),
    ("human", """CRITERIA:
{criteria_block}

PATIENT SUMMARY:
{patient_summary}

Return one evaluation per criterion."""),
])


def _build_criteria_block(tasks: list[tuple[str, str, str]]) -> str:
    """Format (id, text, type) tuples into a numbered block for the prompt."""
    lines = []
    for cid, ctext, ctype in tasks:
        lines.append(f"{cid} [{ctype.upper()}]: {ctext}")
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────

async def run(state: dict) -> dict:
    """
    Async. Evaluates all criteria in a single LLM call.
    Reads state["criteria_agent_output"] and state["input"]["patient_summary"].
    Writes state["evaluation_agent_output"].
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
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

        # Build flat list of (id, text, type)
        tasks: list[tuple[str, str, str]] = []
        for i, c in enumerate(criteria_output.get("inclusion_criteria", []), 1):
            tasks.append((f"INC_{i:03d}", c.get("criterion", ""), "inclusion"))
        for i, c in enumerate(criteria_output.get("exclusion_criteria", []), 1):
            tasks.append((f"EXC_{i:03d}", c.get("criterion", ""), "exclusion"))

        if not tasks:
            state["errors"].append(f"{AGENT_NAME}: no criteria found in criteria_agent_output")
            state["evaluation_agent_output"] = None
            return state

        handler = state.get("langfuse_handler") or _get_handler()
        llm = _get_llm()
        structured_llm = llm.with_structured_output(_EvalList)
        chain = _PROMPT | structured_llm

        result: _EvalList = await chain.ainvoke(
            {
                "criteria_block": _build_criteria_block(tasks),
                "patient_summary": patient_summary,
            },
            config={"callbacks": [handler], "run_name": AGENT_NAME},
        )

        # Compute overall assessment in Python — no extra LLM tokens needed
        inc_evals = [e for e in result.evaluations if e.criterion_id.startswith("INC")]
        exc_evals = [e for e in result.evaluations if e.criterion_id.startswith("EXC")]
        inc_satisfied = sum(1 for e in inc_evals if e.meets_criterion)
        exc_triggered = sum(1 for e in exc_evals if e.meets_criterion)

        state["evaluation_agent_output"] = EvaluationResult(
            evaluations=result.evaluations,
            overall_assessment=(
                f"{inc_satisfied}/{len(inc_evals)} inclusion criteria satisfied; "
                f"{exc_triggered}/{len(exc_evals)} exclusion criteria triggered."
            ),
        ).model_dump()

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")
        state["evaluation_agent_output"] = None

    return state


if __name__ == "__main__":
    import json

    test_state = {
        "input": {
            "trial_criteria": "Inclusion: Age 18-65. Type 2 diabetes.\nExclusion: Pregnant. Currently on insulin.",
            "patient_summary": "45-year-old male with Type 2 diabetes managed on metformin. Not on insulin. No DKA history.",
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
                {"type": "exclusion", "criterion": "Currently on insulin therapy", "category": "medication", "measurable": True},
            ],
            "total_criteria_count": 4,
            "extraction_confidence": 0.98,
        },
    }
    result = asyncio.run(run(test_state))
    print("pipeline_step:", result["pipeline_step"])
    print("errors:", result["errors"])
    print(json.dumps(result.get("evaluation_agent_output"), indent=2, default=str))
