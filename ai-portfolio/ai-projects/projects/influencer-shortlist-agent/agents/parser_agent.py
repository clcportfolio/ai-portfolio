"""
parser_agent.py — Influencer Shortlist Agent

Turns the natural-language brief into a structured ParsedBrief Pydantic object.

Why structured output, not prompt-based JSON
--------------------------------------------
The downstream pipeline reads typed fields off the parsed object — hard_filter
needs `tiers: list[str]`, the cache needs `output_spec.count: int`. Asking the
LLM to "respond in JSON" leaves us validating strings; with_structured_output()
binds the schema so the model literally cannot return malformed shapes. Any
ValidationError is the LLM's fault, not the contract's, and we surface it as
an `ambiguity` for the user to clarify.

Brief-dependent scoring weights
-------------------------------
The parser also emits scoring_weights — how much each scoring dimension
(topic_fit, voice_fit, audience_fit, risk_penalty) should matter for THIS
brief. A skincare brief stressing authenticity weights voice_fit higher; a
demographic-precise brief weights audience_fit higher. The parser is the only
agent allowed to make this call, because only the brief text contains the
emphasis signal. Raw LLM weights are clamped to [0.1, 0.5] and normalized to
sum to 1.0 in Python — never trust the LLM to honour the constraint exactly.

Ambiguities
-----------
If the brief is missing required information, contradictory, or genuinely
ambiguous (e.g. "young creators" — age-tier or follower-tier?), the parser
records the issue in `ambiguities`. A non-empty list signals the pipeline to
stop and ask the user to clarify, rather than guessing and producing a
shortlist the user didn't mean.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field, field_validator

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

AGENT_NAME = "parser_agent"


# ── Schema ────────────────────────────────────────────────────────────────────

VALID_TIERS = {"nano", "micro", "mid", "macro", "mega"}
VALID_PLATFORMS = {"IG", "TikTok", "YouTube"}

DEFAULT_REQUIRED_FIELDS = ["name", "tier", "country", "rationale", "risk", "cited_post_ids"]


class HardConstraints(BaseModel):
    """Deterministic filters applied in SQL before the LLM ever sees a candidate."""
    countries: Optional[list[str]] = Field(
        default=None,
        description="ISO-2 country codes (US, CA, UK, AU, DE). Null/empty = any country.",
    )
    tiers: Optional[list[str]] = Field(
        default=None,
        description="Subset of {nano, micro, mid, macro, mega}. Null = any tier.",
    )
    platforms: Optional[list[str]] = Field(
        default=None,
        description="Subset of {IG, TikTok, YouTube}. Null = any platform.",
    )
    follower_min: Optional[int] = Field(default=None, description="Inclusive lower bound, or null.")
    follower_max: Optional[int] = Field(default=None, description="Inclusive upper bound, or null.")
    exclude_creator_ids: Optional[list[int]] = Field(
        default=None,
        description="Creator ids to drop. Almost always null unless the brief names them explicitly.",
    )
    exclude_brand_collabs: Optional[list[str]] = Field(
        default=None,
        description="Brand names — creators who recently collaborated with these brands are dropped.",
    )
    exclude_collab_window_days: Optional[int] = Field(
        default=None,
        description=(
            "Window for the brand exclusion. e.g. 180 means 'no creators who collabed with "
            "the listed brands in the last 6 months'. Defaults to 180 if exclude_brand_collabs "
            "is set but a window isn't specified."
        ),
    )


class SoftConstraint(BaseModel):
    """A semantic preference used as a query against post embeddings."""
    description: str = Field(description="A short phrase like 'ingredient-conscious voice'.")
    weight: float = Field(
        default=1.0,
        description="Relative weight 0-1. Defaults to 1.0; lower means 'nice to have'.",
    )


class ScoringWeights(BaseModel):
    """Per-dimension weights for the final ranking. Clamped + normalised in Python after parse."""
    topic_fit: float = Field(default=0.30, description="How well posts match the brief's topical scope.")
    voice_fit: float = Field(default=0.30, description="How closely the creator's voice matches.")
    audience_fit: float = Field(default=0.30, description="Demographics + geography alignment.")
    risk_penalty: float = Field(default=0.10, description="Brand-safety / controversial-content penalty.")


class OutputSpec(BaseModel):
    count: int = Field(default=20, description="Number of creators in the final shortlist.")
    required_fields: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REQUIRED_FIELDS),
        description="Fields the user wants in the output.",
    )

    @field_validator("count")
    @classmethod
    def _bound_count(cls, v: int) -> int:
        # The scorer is fixed at 50 candidates; we can never slice beyond that.
        if v < 1:
            return 1
        if v > 50:
            return 50
        return v


class ParsedBrief(BaseModel):
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_constraints: list[SoftConstraint] = Field(default_factory=list)
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    output_spec: OutputSpec = Field(default_factory=OutputSpec)
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Each entry is a question the parser would ask the user to clarify.",
    )


# ── Weight normalisation (post-LLM, deterministic) ───────────────────────────

WEIGHT_MIN = 0.10
WEIGHT_MAX = 0.50
WEIGHT_KEYS = ("topic_fit", "voice_fit", "audience_fit", "risk_penalty")


def normalise_weights(raw: ScoringWeights) -> ScoringWeights:
    """
    Clamp each weight to [0.1, 0.5], then renormalise so they sum to 1.0.
    Prevents single-dimension collapse (e.g. weights={1.0, 0, 0, 0}) without
    relying on the LLM to honour bounds it might ignore.

    Two-pass approach:
      1. Clamp every value to [WEIGHT_MIN, WEIGHT_MAX].
      2. Scale so the total is 1.0. If the clamped sum is itself outside
         [4*MIN, 4*MAX], renormalisation can push some values back outside
         the bounds — clamp once more and accept the small drift. (Sum of
         four 0.1-0.5 values lies in [0.4, 2.0], which always renormalises
         cleanly to a value within [0.1, 0.5] for any reasonable input.)
    """
    raw_dict = {k: max(WEIGHT_MIN, min(WEIGHT_MAX, getattr(raw, k))) for k in WEIGHT_KEYS}
    total = sum(raw_dict.values()) or 1.0
    scaled = {k: v / total for k, v in raw_dict.items()}
    # Final clamp after rescale
    scaled = {k: max(WEIGHT_MIN, min(WEIGHT_MAX, v)) for k, v in scaled.items()}
    # Re-rescale once more so they actually sum to 1.0
    total2 = sum(scaled.values()) or 1.0
    final = {k: v / total2 for k, v in scaled.items()}
    return ScoringWeights(**final)


# ── Prompt ───────────────────────────────────────────────────────────────────

PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You parse brand-marketing campaign briefs into a structured ParsedBrief object
for an influencer shortlist system.

Hard constraints are deterministic filters: countries, creator tiers, follower ranges,
platforms, and exclusion lists. Extract ONLY what the brief states explicitly. Never
infer a country if the brief doesn't mention one — leave it null and let the system
return creators from any country.

Tiers are EXACTLY one of: nano (1k-10k), micro (10k-100k), mid (100k-500k),
macro (500k-2M), mega (2M+). If the brief says "mid and macro", set tiers=["mid","macro"].
If the brief gives an explicit follower range (e.g. "between 50k and 200k followers"),
set follower_min/max instead of tiers — they're more precise.

Platforms are EXACTLY one of: IG, TikTok, YouTube.

Exclusions: only populate exclude_brand_collabs if the brief names competitor brands.
If a window like "in the last 6 months" is mentioned, convert to days
(week=7, month=30, quarter=90, half=180, year=365). If brands are excluded but
no window is given, default exclude_collab_window_days to 180.

Soft constraints are semantic preferences: voice, audience interests, content themes.
Each becomes a separate description string. Keep each phrase short (2-8 words).
Examples:
  - "ingredient-conscious voice"
  - "warm, parent-friendly tone"
  - "audience cares about sustainability"

Scoring weights are how much each ranking dimension matters for THIS brief.
The four dimensions are:
  topic_fit     — does the creator's content cover the campaign's topical area?
  voice_fit     — does their voice/tone match what the brand wants?
  audience_fit  — do their followers match the target demographic?
  risk_penalty  — should brand-safety risks be heavily weighted?

Read the brief and shift weights based on emphasis:
  - "authentic, real voice, must sound like one of us"      → voice_fit higher
  - "Gen-Z women, NYC, college-educated"                       → audience_fit higher
  - "must align with our clean-ingredient positioning"         → topic_fit higher
  - "no creators with controversy, family-friendly only"       → risk_penalty higher

Each weight should land between 0.1 and 0.5. They will be auto-normalised to sum to 1.0
after you respond, so don't worry about exact totals — emit honest relative weights.

output_spec.count: extract from phrases like "20 creators", "top 10", "shortlist of 15".
Default 20 if unspecified. Cap at 50 (the system's scoring pool size).

ambiguities: if the brief is missing required info, contradictory, or genuinely
ambiguous, add a clear, concise question to ambiguities. Examples:
  - "Brief says 'young creators' — does that mean nano/micro tier, or under-25 audience?"
  - "Brief lists competitor brands but no exclusion window — defaulting to 180 days; confirm?"
  - "No country specified — returning global; is that intended?"

Empty ambiguities means you're confident the brief was clear."""),
    ("human",
     "Parse this campaign brief into a ParsedBrief:\n\n{brief_text}"),
])


# ── Chain (built once, reused) ───────────────────────────────────────────────

_PARSER_LLM = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0,
)
_chain = PARSER_PROMPT | _PARSER_LLM.with_structured_output(ParsedBrief)


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_and_clean(parsed: ParsedBrief) -> ParsedBrief:
    """
    Drop or null-out invalid enum values, clamp weights, default the brand
    exclusion window. All deterministic — no LLM.
    """
    hc = parsed.hard_constraints

    if hc.tiers:
        cleaned = [t.lower() for t in hc.tiers if t.lower() in VALID_TIERS]
        hc.tiers = cleaned or None
    if hc.platforms:
        # case-normalise: map common variants
        norm = {"instagram": "IG", "ig": "IG", "tiktok": "TikTok", "tt": "TikTok",
                "youtube": "YouTube", "yt": "YouTube"}
        cleaned = []
        for p in hc.platforms:
            key = p.lower().strip()
            if key in norm:
                cleaned.append(norm[key])
            elif p in VALID_PLATFORMS:
                cleaned.append(p)
        hc.platforms = cleaned or None
    if hc.countries:
        # Just uppercase ISO-2; we don't whitelist (DE, FR etc. all valid)
        hc.countries = [c.upper().strip() for c in hc.countries if len(c.strip()) == 2] or None

    if hc.exclude_brand_collabs and hc.exclude_collab_window_days is None:
        hc.exclude_collab_window_days = 180

    parsed.scoring_weights = normalise_weights(parsed.scoring_weights)
    return parsed


# ── Run ──────────────────────────────────────────────────────────────────────

def _get_handler(state: dict) -> CallbackHandler:
    return state.get("langfuse_handler") or CallbackHandler()


async def run(state: dict) -> dict:
    """
    Parse state["input"]["brief_text"] into a ParsedBrief.
    Writes state["parser_output"] = parsed.model_dump() (or None on failure).
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    brief = (state.get("input") or {}).get("brief_text", "")
    if not brief.strip():
        state["errors"].append(f"{AGENT_NAME}: empty brief_text")
        state["parser_output"] = None
        return state

    handler = _get_handler(state)

    try:
        parsed: ParsedBrief = await _chain.ainvoke(
            {"brief_text": brief},
            config={"callbacks": [handler], "run_name": AGENT_NAME},
        )
        parsed = _validate_and_clean(parsed)
        state["parser_output"] = parsed.model_dump()
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: {e}")
        state["parser_output"] = None

    return state


if __name__ == "__main__":
    import asyncio
    import json

    sample = (
        "We're launching a clean-ingredient skincare line for women 25-40 in the US and "
        "Canada. Budget: $150K. Voice should be ingredient-conscious and authentic — no "
        "performative marketing. Mix of mid and macro creators. Skip anyone who's "
        "collaborated with PureSkin Co or Verde Beauty in the last 6 months. Want 20 "
        "creators total."
    )

    test_state = {
        "input": {"brief_text": sample},
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

    print("=== Parsing brief ===\n")
    print(sample)
    print("\n=== Result ===\n")
    result = asyncio.run(run(test_state))
    print(json.dumps(result.get("parser_output"), indent=2, default=str))
    print("\nerrors:", result["errors"])
