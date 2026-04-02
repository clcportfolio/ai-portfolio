"""
synthetic_generator.py — Generate synthetic patient summaries for a given trial.

Eligible patients are generated using a template-based prompt that enumerates every
inclusion criterion and every exclusion — the LLM must write a value that satisfies
each one explicitly. This eliminates the ambiguity that causes the evaluator to hedge.

Ineligible and borderline patients use a free-form prompt.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv(find_dotenv(), override=True)


class _SyntheticPatientList(BaseModel):
    patients: List[str] = Field(
        description="List of fictional patient summary strings, one per patient."
    )


def _get_llm() -> ChatAnthropic:
    # Sonnet is required here — Haiku doesn't follow the per-criterion eligible
    # template reliably enough, producing summaries that the evaluator hedges on.
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.9,
    )


# ── Eligible patient prompt ────────────────────────────────────────────────────
# Each eligible summary must address every criterion by name with a concrete value.

_ELIGIBLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are writing fictional patient summaries that must be CLEARLY ELIGIBLE
for a clinical trial. Each summary must leave zero ambiguity.

Rules:
1. Address every inclusion criterion with a specific confirming value.
2. Explicitly negate every exclusion criterion ("no history of X", "not pregnant", etc.).
3. Keep each summary to 3-5 sentences. Vary age (stay within any stated age range),
   sex, and secondary details across patients. Do not reuse the same combination."""),
    ("human", """INCLUSION CRITERIA — patient must satisfy ALL of these (state each explicitly):
{inclusion_list}

EXCLUSION CRITERIA — patient must satisfy NONE of these (negate each explicitly):
{exclusion_list}

Write {count} clearly eligible fictional patient summaries."""),
])

# ── Ineligible + borderline prompt ────────────────────────────────────────────

_INELIGIBLE_BORDERLINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are writing fictional patient summaries for clinical trial eligibility testing.

Write two types:
- INELIGIBLE: fail exactly one or two criteria clearly. State the disqualifying finding
  explicitly (e.g. "currently on insulin", "eGFR 22", "pregnant").
- BORDERLINE: meet most criteria but have exactly one ambiguous or marginal finding
  (e.g. a lab value just outside the stated range, a recent procedure, or missing
  information about one criterion).

Each summary should be 2-4 sentences. Vary age, sex, comorbidities, and lab values."""),
    ("human", """Trial criteria:
{criteria_text}

Write {ineligible_count} ineligible and {borderline_count} borderline patient summaries.
Label each with [INELIGIBLE] or [BORDERLINE] at the start so they can be distinguished,
then provide the summary text."""),
])


class _LabelledList(BaseModel):
    patients: List[str] = Field(
        description="List of patient summaries, each starting with [INELIGIBLE] or [BORDERLINE]."
    )


def _format_criteria_lists(structured_criteria: dict) -> tuple[str, str]:
    inc = structured_criteria.get("inclusion_criteria") or []
    exc = structured_criteria.get("exclusion_criteria") or []
    inc_lines = "\n".join(
        f"  {i}. {c.get('criterion', c.get('criterion_text', ''))}"
        for i, c in enumerate(inc, 1)
    )
    exc_lines = "\n".join(
        f"  {i}. {c.get('criterion', c.get('criterion_text', ''))}"
        for i, c in enumerate(exc, 1)
    )
    return inc_lines or "  (none)", exc_lines or "  (none)"


async def _generate_eligible(
    llm: ChatAnthropic,
    inc_list: str,
    exc_list: str,
    count: int,
) -> list[str]:
    """Generate exactly `count` eligible summaries, retrying once if short."""
    chain = _ELIGIBLE_PROMPT | llm.with_structured_output(_SyntheticPatientList)
    result = await chain.ainvoke(
        {"inclusion_list": inc_list, "exclusion_list": exc_list, "count": count},
        config={"run_name": "synthetic_generator_eligible"},
    )
    summaries = [p.strip() for p in result.patients if p.strip()]
    # Retry once if short
    if len(summaries) < count:
        shortfall = count - len(summaries)
        retry = await chain.ainvoke(
            {"inclusion_list": inc_list, "exclusion_list": exc_list, "count": shortfall},
            config={"run_name": "synthetic_generator_eligible_retry"},
        )
        summaries.extend(p.strip() for p in retry.patients if p.strip())
    return summaries[:count]


async def _generate_ineligible_borderline(
    llm: ChatAnthropic,
    criteria_text: str,
    ineligible_count: int,
    borderline_count: int,
) -> list[str]:
    """Generate ineligible + borderline summaries, retrying once if short."""
    chain = _INELIGIBLE_BORDERLINE_PROMPT | llm.with_structured_output(_LabelledList)
    total = ineligible_count + borderline_count

    async def _invoke(inel: int, bord: int) -> list[str]:
        result = await chain.ainvoke(
            {"criteria_text": criteria_text, "ineligible_count": inel, "borderline_count": bord},
            config={"run_name": "synthetic_generator_ineligible"},
        )
        return [
            p.replace("[INELIGIBLE]", "").replace("[BORDERLINE]", "").strip()
            for p in result.patients if p.strip()
        ]

    summaries = await _invoke(ineligible_count, borderline_count)
    if len(summaries) < total:
        shortfall = total - len(summaries)
        summaries.extend(await _invoke(shortfall, 0))
    return summaries[:total]


async def generate_patient_summaries(
    criteria_text: str,
    count: int,
    structured_criteria: Optional[dict] = None,
) -> list[str]:
    """
    Generate `count` synthetic patient summaries for a trial.

    Target distribution: 40% eligible, 50% ineligible, 10% borderline.

    Uses Sonnet for reliable template following — Haiku does not follow the
    per-criterion eligible template closely enough and produces ambiguous summaries
    that the evaluator hedges on. Each batch retries once if the LLM returns short.
    """
    llm = _get_llm()

    eligible_count = max(1, round(count * 0.40))
    ineligible_count = max(1, round(count * 0.50))
    borderline_count = max(0, count - eligible_count - ineligible_count)

    # ── Build criteria lists ───────────────────────────────────────────────────
    if structured_criteria:
        inc_list, exc_list = _format_criteria_lists(structured_criteria)
    else:
        lines = criteria_text.splitlines()
        inc_lines, exc_lines = [], []
        mode = None
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if "inclusion" in l.lower():
                mode = "inc"
            elif "exclusion" in l.lower():
                mode = "exc"
            elif mode == "inc":
                inc_lines.append(f"  {l.lstrip('-•').strip()}")
            elif mode == "exc":
                exc_lines.append(f"  {l.lstrip('-•').strip()}")
        inc_list = "\n".join(inc_lines) or criteria_text[:500]
        exc_list = "\n".join(exc_lines) or "  (see criteria above)"

    # ── Generate both batches in parallel ─────────────────────────────────────
    eligible_summaries, inelig_summaries = await asyncio.gather(
        _generate_eligible(llm, inc_list, exc_list, eligible_count),
        _generate_ineligible_borderline(llm, criteria_text, ineligible_count, borderline_count),
    )

    all_summaries = eligible_summaries + inelig_summaries
    return all_summaries[:count]


if __name__ == "__main__":
    sample_criteria_text = """Inclusion Criteria:
- Age 30-70
- Type 2 Diabetes, HbA1c 7-10%
- On metformin monotherapy >= 500mg/day for >= 3 months

Exclusion Criteria:
- Pregnant or breastfeeding
- Current insulin therapy
- eGFR < 45
- History of DKA"""

    sample_structured = {
        "inclusion_criteria": [
            {"criterion": "Age 30-70"},
            {"criterion": "Type 2 Diabetes with HbA1c between 7% and 10%"},
            {"criterion": "On metformin monotherapy >= 500mg/day for >= 3 months"},
        ],
        "exclusion_criteria": [
            {"criterion": "Pregnant or breastfeeding"},
            {"criterion": "Current insulin therapy"},
            {"criterion": "eGFR < 45"},
            {"criterion": "History of DKA"},
        ],
    }

    print("Generating 6 summaries (structured criteria)...\n")
    summaries = asyncio.run(
        generate_patient_summaries(sample_criteria_text, count=6, structured_criteria=sample_structured)
    )
    for i, s in enumerate(summaries, 1):
        print(f"[{i}] {s}\n")
