"""
verdict_agent.py — Synthesizes per-criterion evaluations into a final eligibility verdict.

The verdict (ELIGIBLE / INELIGIBLE / NEEDS_REVIEW) and confidence score are computed
deterministically in Python from the boolean evaluation results. The LLM is only called
to generate the plain-English summary, key factors, and next steps — it cannot hedge or
override the verdict.

Deterministic rules:
  NEEDS_REVIEW  — any criterion has confidence < CONFIDENCE_THRESHOLD (0.5)
  INELIGIBLE    — any INC criterion has meets_criterion=False, OR
                  any EXC criterion has meets_criterion=True
  ELIGIBLE      — all INC criteria have meets_criterion=True AND
                  all EXC criteria have meets_criterion=False
"""
from __future__ import annotations

import asyncio

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

PROJECT_NAME = "clinical-trial-eligibility-screener"
AGENT_NAME = "verdict_agent"

# Any criterion evaluation below this confidence triggers NEEDS_REVIEW.
# 0.35 means "genuinely no usable information" — explicitly stated facts score 0.85+
CONFIDENCE_THRESHOLD = 0.35


class VerdictResult(BaseModel):
    eligibility_status: str = Field(description="ELIGIBLE | INELIGIBLE | NEEDS_REVIEW")
    confidence_score: float = Field(description="0.0 to 1.0 confidence in the verdict")
    summary: str = Field(description="Plain-English explanation of the verdict for coordinators")
    key_factors: list[str] = Field(description="Most important factors that influenced the decision")
    next_steps: str = Field(description="Recommended actions for the coordinator")


class _NarrativeOutput(BaseModel):
    """LLM only produces the narrative — verdict and confidence are set by Python."""
    summary: str
    key_factors: list[str]
    next_steps: str


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
    )


def _get_handler() -> CallbackHandler:
    return CallbackHandler()


def _compute_verdict(evaluations: list[dict]) -> tuple[str, float]:
    """
    Deterministically compute eligibility_status and confidence_score.

    Returns (status, confidence) where status is one of:
      NEEDS_REVIEW, INELIGIBLE, ELIGIBLE
    """
    if not evaluations:
        return "NEEDS_REVIEW", 0.0

    # --- Check for low-confidence evaluations first ---
    low_conf = [e for e in evaluations if e.get("confidence", 1.0) < CONFIDENCE_THRESHOLD]
    if low_conf:
        avg = sum(e.get("confidence", 0.0) for e in low_conf) / len(low_conf)
        return "NEEDS_REVIEW", round(avg, 3)

    inc_evals = [e for e in evaluations if e.get("criterion_id", "").startswith("INC")]
    exc_evals = [e for e in evaluations if e.get("criterion_id", "").startswith("EXC")]

    # --- Disqualifying checks ---
    failed_inc = [e for e in inc_evals if not e.get("meets_criterion", True)]
    triggered_exc = [e for e in exc_evals if e.get("meets_criterion", False)]

    disqualifying = failed_inc + triggered_exc
    if disqualifying:
        conf = sum(e.get("confidence", 0.0) for e in disqualifying) / len(disqualifying)
        return "INELIGIBLE", round(conf, 3)

    # --- All criteria clear → ELIGIBLE ---
    # Confidence = weakest link across all criteria
    min_conf = min(e.get("confidence", 0.0) for e in evaluations)
    return "ELIGIBLE", round(min_conf, 3)


def _format_evaluations(evaluations: list[dict]) -> str:
    """Format per-criterion evaluations for the narrative prompt."""
    lines = []
    for ev in evaluations:
        cid = ev.get("criterion_id", "?")
        conf = ev.get("confidence", 0.0)
        meets = ev.get("meets_criterion", False)

        if cid.startswith("INC"):
            status = "SATISFIED" if meets else "NOT SATISFIED (disqualifying)"
        else:
            status = "TRIGGERED — patient has this condition" if meets else "NOT TRIGGERED — patient clear"

        lines.append(
            f"[{cid}] {ev.get('criterion_text', '')} → {status} (confidence: {conf:.0%})\n"
            f"  Reasoning: {ev.get('reasoning', '')}"
        )
    return "\n".join(lines) if lines else "No evaluations available."


_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical trial coordinator assistant. The eligibility verdict has
already been determined. Your job is to write a clear, plain-English explanation for the
coordinator — do not second-guess or restate the verdict, just explain it.

Be concise and specific. Reference the actual criteria and patient findings."""),
    ("human", """VERDICT: {verdict}
CONFIDENCE: {confidence:.0%}

PER-CRITERION EVALUATIONS:
{evaluations}

PATIENT SUMMARY:
{patient_summary}

Write:
1. summary — 2-3 sentences explaining why this patient is {verdict}
2. key_factors — 3-5 bullet points listing the most important findings
3. next_steps — one sentence of recommended coordinator action"""),
])


async def run(state: dict) -> dict:
    """
    Async. Computes verdict deterministically, then calls LLM for narrative only.
    Reads state["evaluation_agent_output"]; writes state["verdict_agent_output"].
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        evaluation_output = state.get("evaluation_agent_output")
        patient_summary = state["input"].get("patient_summary", "")

        if not evaluation_output:
            state["errors"].append(f"{AGENT_NAME}: missing evaluation_agent_output")
            state["verdict_agent_output"] = None
            return state

        evaluations = evaluation_output.get("evaluations", [])

        # ── Step 1: deterministic verdict ──────────────────────────────────────
        status, confidence = _compute_verdict(evaluations)
        formatted = _format_evaluations(evaluations)

        # ── Step 2: LLM writes narrative only ──────────────────────────────────
        llm = _get_llm()
        structured_llm = llm.with_structured_output(_NarrativeOutput)
        chain = _NARRATIVE_PROMPT | structured_llm
        handler = state.get("langfuse_handler") or _get_handler()

        narrative: _NarrativeOutput = await chain.ainvoke(
            {
                "verdict": status,
                "confidence": confidence,
                "evaluations": formatted,
                "patient_summary": patient_summary,
            },
            config={"callbacks": [handler], "run_name": AGENT_NAME},
        )

        state["verdict_agent_output"] = VerdictResult(
            eligibility_status=status,
            confidence_score=confidence,
            summary=narrative.summary,
            key_factors=narrative.key_factors,
            next_steps=narrative.next_steps,
        ).model_dump()

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")
        state["verdict_agent_output"] = None

    return state


if __name__ == "__main__":
    import json

    # Test 1: clearly eligible patient — must return ELIGIBLE
    eligible_state = {
        "input": {
            "trial_criteria": "Inclusion: Age 18-65. Type 2 diabetes.\nExclusion: Pregnant.",
            "patient_summary": "45-year-old male with Type 2 diabetes. Not pregnant.",
        },
        "pipeline_step": 2,
        "max_pipeline_steps": 10,
        "errors": [],
        "evaluation_agent_output": {
            "evaluations": [
                {
                    "criterion_id": "INC_001",
                    "criterion_text": "Age 18-65",
                    "meets_criterion": True,
                    "confidence": 0.95,
                    "reasoning": "Patient is 45.",
                    "relevant_patient_info": "45-year-old male",
                },
                {
                    "criterion_id": "INC_002",
                    "criterion_text": "Type 2 diabetes",
                    "meets_criterion": True,
                    "confidence": 0.98,
                    "reasoning": "Explicitly stated.",
                    "relevant_patient_info": "Type 2 diabetes",
                },
                {
                    "criterion_id": "EXC_001",
                    "criterion_text": "Pregnant or nursing",
                    "meets_criterion": False,
                    "confidence": 0.90,
                    "reasoning": "Patient is male.",
                    "relevant_patient_info": "male",
                },
            ],
            "overall_assessment": "2 inclusion criteria satisfied, 1 exclusion criterion not triggered.",
        },
    }

    # Test 2: triggered exclusion — must return INELIGIBLE
    ineligible_state = {
        "input": {
            "trial_criteria": "Inclusion: Age 18-65. Type 2 diabetes.\nExclusion: Currently on insulin.",
            "patient_summary": "52-year-old female with Type 2 diabetes on insulin therapy.",
        },
        "pipeline_step": 2,
        "max_pipeline_steps": 10,
        "errors": [],
        "evaluation_agent_output": {
            "evaluations": [
                {
                    "criterion_id": "INC_001",
                    "criterion_text": "Age 18-65",
                    "meets_criterion": True,
                    "confidence": 0.97,
                    "reasoning": "Patient is 52.",
                    "relevant_patient_info": "52-year-old",
                },
                {
                    "criterion_id": "INC_002",
                    "criterion_text": "Type 2 diabetes",
                    "meets_criterion": True,
                    "confidence": 0.97,
                    "reasoning": "Explicitly stated.",
                    "relevant_patient_info": "Type 2 diabetes",
                },
                {
                    "criterion_id": "EXC_001",
                    "criterion_text": "Currently on insulin",
                    "meets_criterion": True,   # patient HAS this → disqualifying
                    "confidence": 0.95,
                    "reasoning": "On insulin therapy explicitly stated.",
                    "relevant_patient_info": "on insulin therapy",
                },
            ],
            "overall_assessment": "2 inclusion criteria satisfied, 1 exclusion criterion triggered.",
        },
    }

    for label, state in [("ELIGIBLE test", eligible_state), ("INELIGIBLE test", ineligible_state)]:
        print(f"\n{'='*50}\n{label}")
        result = asyncio.run(run(state))
        out = result.get("verdict_agent_output", {})
        print(f"status   : {out.get('eligibility_status')}")
        print(f"confidence: {out.get('confidence_score')}")
        print(f"errors   : {result['errors']}")
        print(json.dumps(out, indent=2, default=str))
