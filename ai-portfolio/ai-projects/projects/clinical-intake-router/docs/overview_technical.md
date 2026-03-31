# Clinical Intake Router — Technical Overview

## Architecture

Three-agent sequential pipeline with guardrails middleware:

```
User input (Streamlit UI)
    │
    ▼
guardrails.validate_input()          ← length, type, prompt injection scan
    │
    ▼
pipeline.run()
    ├─► extraction_agent.run(state)   ← ChatAnthropic + structured output (ExtractedFields)
    ├─► classification_agent.run(state) ← ChatAnthropic + structured output (ClassificationResult)
    └─► routing_agent.run(state)      ← ChatAnthropic + structured output (RoutingResult)
    │
    ▼
guardrails.sanitize_output()         ← HTML strip, PHI redaction stub
    │
    ▼
Streamlit UI → routing card + agent expanders
```

## Agent Roles

### extraction_agent
- **Model:** `claude-sonnet-4-20250514`, temperature 0, max_tokens 2048
- **Pattern:** `chain = EXTRACTION_PROMPT | llm.with_structured_output(ExtractedFields)`
- **Output schema (Pydantic):** `patient_name`, `age`, `date_of_birth`, `chief_complaint`,
  `symptoms`, `medical_history`, `current_medications`, `allergies`, `insurance`,
  `referral_source`, `additional_notes`
- **Design choice:** Temperature 0 for deterministic extraction. All optional fields use
  `Optional[...]` with `default=None` so missing data doesn't block the pipeline.

### classification_agent
- **Model:** `claude-sonnet-4-20250514`, temperature 0, max_tokens 1024
- **Pattern:** `chain = CLASSIFICATION_PROMPT | llm.with_structured_output(ClassificationResult)`
- **Output schema:** `urgency_level` (Enum: Routine/Urgent/Emergent), `department` (free text —
  LLM determines from context), `classification_reasoning` (2-4 sentence justification),
  `confidence` (0.0–1.0 float), `red_flags` (list of clinical warning signs)
- **Design choice:** Department is a free-text string, not an enum. The model reasons from
  clinical context to the most appropriate specialty — a hardcoded list would miss edge cases
  (e.g., routing a complex case to Hepatology vs. Gastroenterology).
- **Why "Emergent" not "Emergency":** The routing card header renders as `{urgency} — {department}`.
  Using "Emergency" for both the top urgency level and the Emergency department produces
  "Emergency — Emergency" or the ambiguous "Urgent — Emergency". "Emergent" is the correct
  clinical triage term and eliminates the namespace collision entirely.

### routing_agent
- **Model:** `claude-sonnet-4-20250514`, temperature 0.3, max_tokens 1024
- **Pattern:** `chain = ROUTING_PROMPT | llm.with_structured_output(RoutingResult)`
- **Output schema:** `routing_summary` (2-3 plain-English sentences), `department`,
  `urgency_level`, `recommended_next_steps` (ordered list), `follow_up_actions`,
  `estimated_response_time`
- **Design choice:** Temperature raised to 0.3 for generative output. The routing summary
  must be readable and natural — zero temperature produces overly mechanical prose.

## Shared State Dict

All agents communicate through a single mutable state dict, following the CLAUDE.md pattern:

```python
state = {
    "input": {"text": "..."},     # original input — never mutated
    "pipeline_step": 0,           # incremented after each agent
    "max_pipeline_steps": 10,     # safety ceiling
    "errors": [],                 # non-fatal errors accumulate here
    "extraction_output": {...},   # ExtractedFields.model_dump()
    "classification_output": {...}, # ClassificationResult.model_dump()
    "routing_output": {...},      # RoutingResult.model_dump()
    "output": {...},              # final output — same as routing_output
}
```

Each agent reads only what it needs and writes only to its own key.

## Guardrails Design

**validate_input:**
- Rejects empty text, non-string types, text over 8,000 characters (intake forms can be long)
- Scans for 9 prompt injection patterns using compiled regex (case-insensitive)

**sanitize_output:**
- Strips HTML/script tags from all string fields in routing_output (XSS prevention)
- PHI redaction stub: regex scan for SSN, DOB, long ID patterns — logs WARNING if found
- Comment in code: "Replace with AWS Comprehend Medical in production"

**rate_limit_check:**
- Stub — always returns True
- Comment: "Replace with Redis-backed counter in production"

## Observability (Langfuse v4)

One trace is created per pipeline run. All three agent LLM calls attach to it as child spans.

**Langfuse v4 API used (langfuse>=4.0.0):**
- `observe` and `get_client` are imported directly from `langfuse` — `langfuse.decorators` does not exist in v4
- `CallbackHandler` accepts `trace_context=TraceContext(trace_id, parent_span_id)` — not a bare `trace_id=` string
- `langfuse_context` (v2/v3 pattern) does not exist — `get_client()` is called inside an `@observe` context instead

```python
# pipeline.py — @observe creates the root span; TRACE_NAME pins the list-view name
@observe(name="intake_route")   # span name — visible in the trace hierarchy
def run(input_data, user_id="anonymous"):
    # Pin trace display name BEFORE any CallbackHandler runs.
    # Without this, the last agent's run_name overwrites the trace name in Langfuse.
    otel_trace.get_current_span().set_attribute(
        LangfuseOtelSpanAttributes.TRACE_NAME, "clinical-intake-router"
    )
    lf = get_client()
    state["langfuse_handler"] = CallbackHandler(
        trace_context=TraceContext(
            trace_id=lf.get_current_trace_id(),
            parent_span_id=lf.get_current_observation_id(),
        )
    )

# each agent — reads handler from state
handler = state.get("langfuse_handler") or CallbackHandler()
chain.invoke(..., config={"callbacks": [handler], "run_name": "extraction_agent"})
```

**Resulting Langfuse structure per form submission:**
```
clinical-intake-router   ← trace list display name (TRACE_NAME attribute)
  └── intake_route       ← @observe span (visible in trace detail hierarchy)
        └── extraction_agent    → ChatAnthropic
        └── classification_agent → ChatAnthropic
        └── routing_agent       → ChatAnthropic
```

The fallback `CallbackHandler()` is only reached when an agent is run standalone via
its `__main__` block — it creates its own trace named by the `run_name` on
`chain.invoke` (e.g. `extraction_agent`), so standalone runs are also identifiable.

Langfuse v4 note: Constructor takes no arguments — all credentials (`LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) are read from environment variables.

## PDF Support

The Streamlit UI uses `pypdf` to extract text from uploaded PDF intake forms:
```python
from pypdf import PdfReader
reader = PdfReader(io.BytesIO(uploaded_file.read()))
text = "\n".join(page.extract_text() or "" for page in reader.pages)
```

Text is extracted at the UI layer before entering the pipeline — the pipeline itself
only receives a `dict` with a `"text"` key.

## LangChain LCEL Pattern

All agents use the pipe operator pattern:
```python
chain = prompt | llm.with_structured_output(PydanticModel)
result = chain.invoke({...}, config={"callbacks": [handler]})
```

`with_structured_output()` uses Claude's tool use capability under the hood to enforce
Pydantic schema compliance. No manual parsing or error-prone regex extraction.

## Deployment Path

- Local: `streamlit run app.py` with `.env` file
- Production: Streamlit Community Cloud (free tier, public URL)
  - Environment variables set in Streamlit Cloud dashboard
  - `requirements.txt` is the source of truth for dependencies
  - `.venv` not committed, not used by Cloud

## Tradeoffs

| Decision | Alternative | Why this choice |
|---|---|---|
| Free-text department field | Enum of departments | Clinical routing is open-domain; enum creates false precision |
| Sequential 3-agent pipeline | Single LLM call | Separation of concerns: extraction errors don't corrupt classification logic |
| `pypdf` for PDF | LangChain document loader | Simpler, no loader class needed for a single file input |
| Temperature 0 for extraction/classification | Higher temperature | Determinism matters for clinical decisions |
| Temperature 0.3 for routing | Temperature 0 | Routing summaries are staff-facing prose — mechanical output is a UX problem |

## Stretch Goal (Not Built)

EHR lookup: validate patient records against existing entries before routing.
Notes in `build_walkthrough.md` describe the integration point.
