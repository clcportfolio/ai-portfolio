# Clinical Intake Router — Build Walkthrough

This document explains what was built, why each file exists, what every function does,
and the design decisions made at each step. Read this before a technical interview.

---

## Why This Project Exists

The Clinical Intake Router is the first portfolio project because it directly mirrors
what M3 USA's healthcare businesses do: read unstructured documents, extract structured
data, make routing or matching decisions, and communicate them to non-technical users.

Wake Research coordinates clinical trial intake. PracticeMatch routes physician candidates.
The Medicus Firm places healthcare professionals. All of them deal with high-volume document
intake and decision-making workflows. This project is a proof of concept for that class
of automation.

---

## Build Order

Files were built in dependency order:

1. `requirements.txt` and `.env.example` — dependency surface before any code
2. `agents/extraction_agent.py` — first, because classification depends on it
3. `agents/classification_agent.py` — second, because routing depends on it
4. `agents/routing_agent.py` — third, terminal step in the pipeline
5. `guardrails.py` — middleware layer, independent of agent logic
6. `pipeline.py` — wires everything together; imports all of the above
7. `app.py` — depends on pipeline.py; built last so all outputs are known
8. `README.md`, `docs/` — documentation after all code is final

---

## File-by-File Explanation

### agents/extraction_agent.py

**What it does:** Takes the raw intake form text and extracts every structured field —
patient name, age, DOB, chief complaint, symptoms, medical history, medications,
allergies, insurance, referral source, and additional notes.

**Why structured output:** The downstream classification agent needs structured Python
objects, not free text. Using `llm.with_structured_output(ExtractedFields)` forces Claude
to return a JSON object that maps exactly to our Pydantic model — no parsing, no guessing.

**Why temperature 0:** Extraction is deterministic. For a given intake form, there's one
correct set of fields. Temperature 0 minimizes hallucination of fields that aren't present.

**Why all optional fields are `Optional[...]`:** Intake forms vary. Some have insurance,
some don't. Making non-critical fields optional means a missing field becomes `None` rather
than a validation error that stops the pipeline.

**The `ExtractedFields` Pydantic model:**
- `patient_name: str` — required; if missing, the pipeline surfaces an error gracefully
- `chief_complaint: str` — required; the core clinical question
- `symptoms: List[str]` — list so classification can reason over individual items
- `medical_history: List[str]` — list for same reason
- Everything else `Optional[...]` — allows partial forms to proceed

**`run(state: dict) -> dict`:** Reads `state["input"]["text"]`, invokes the chain,
writes to `state["extraction_output"]`, increments `state["pipeline_step"]`. Catches
exceptions into `state["errors"]` without crashing the pipeline.

---

### agents/classification_agent.py

**What it does:** Takes the extracted fields and produces an urgency level (Routine,
Urgent, Emergency), a department assignment, a reasoning explanation, a confidence score,
and a list of clinical red flags.

**Why an Enum for urgency:** The three urgency levels are fixed — there are exactly three
of them and they map to UI colors in the Streamlit app. An Enum prevents the LLM from
inventing a fourth level like "Semi-urgent."

**Why "Emergent" not "Emergency":** The routing card header renders `{urgency_level} — {department}`.
If the top urgency level were named "Emergency", a patient routed to the Emergency department
at Urgent priority would display "Urgent — Emergency" (ambiguous), and an Emergency-level
case going to Emergency would display "Emergency — Emergency" (redundant). "Emergent" is
the correct clinical triage term — nurses and coordinators already use it — and it
cleanly separates the urgency namespace from the department namespace.

**Why department is free text (not an Enum):** This was a deliberate design decision.
Clinical routing is open-domain. A hardcoded department list would route everything into
Primary Care or "Other" when the right answer might be Hepatology, Geriatrics, or Infectious
Disease. Letting the LLM reason from clinical context produces more clinically appropriate
department assignments.

**The system prompt design:** The prompt defines the three urgency levels explicitly with
clear criteria (life-threatening → Emergency, significant symptoms → Urgent, stable →
Routine). Without this, the model tends to classify borderline cases inconsistently.
The prompt also specifies that `classification_reasoning` must be 2-4 sentences —
this enforces a reasoning trace that appears in the UI expander and makes the pipeline
auditable.

**`red_flags`:** This field was added specifically for the UI. When a case is Emergency
or Urgent, staff need to know *why* — not just that the urgency is high. Red flags like
"Chest pain radiating to left arm" and "Diaphoresis" give clinical credibility to the
routing decision.

**Confidence score:** A float 0.0–1.0. This is displayed in the UI classification expander.
It also gives downstream tooling (a future escalation agent, for instance) a signal to
flag low-confidence routing decisions for human review.

---

### agents/routing_agent.py

**What it does:** Takes the classification result and generates plain-English routing
instructions for non-clinical staff: a routing summary, recommended next steps, follow-up
actions, and an estimated response time.

**Why a separate agent instead of folding into classification:** Separation of concerns.
The classification agent makes a clinical judgment. The routing agent translates that
judgment into action. If the routing output format ever changes (e.g., to support a
ticketing system), only the routing agent needs to change — classification stays stable.

**Why temperature 0.3:** The routing summary is staff-facing prose. At temperature 0,
Claude tends to produce mechanical, repetitive text ("Route patient to department. Urgency
is Emergency. Contact department."). A small amount of temperature produces more natural,
readable output while keeping the factual content consistent.

**`recommended_next_steps` as ordered list:** Steps are numbered in the UI (1, 2, 3...).
The ordering matters — "Notify on-call cardiologist immediately" must come before
"Prepare intake paperwork." The prompt requests 3-5 ordered actions.

**Stretch goal note:** The stretch goal is an EHR lookup that would validate the patient
against existing records before finalizing the routing. The integration point would be
between the extraction and routing agents: after extraction, a new `ehr_lookup_agent`
would query the EHR by patient name + DOB, append a `patient_record_match` field to the
state, and the routing agent would incorporate that information. This requires HIPAA-
compliant EHR API access (e.g., FHIR-compliant endpoints), which is out of scope for a
portfolio demo but straightforward to add in a production system.

---

### guardrails.py

**What it does:** Validates input before it reaches any LLM, and sanitizes output before
it reaches the UI.

**`validate_input`:** Three checks:
1. Text must be a non-empty string
2. Max 8,000 characters (longer than the 4,000-char default — intake forms can be verbose)
3. Regex scan for 9 prompt injection patterns (case-insensitive compiled regex for speed)

Why 8,000 chars and not 4,000? Intake forms include medical history, medication lists,
insurance details, and clinical notes. A comprehensive intake can easily exceed 4,000
characters, and truncating it would degrade extraction quality.

**`sanitize_output`:** Two passes:
1. Strip HTML/script tags from all string fields in `routing_output` — prevents XSS if
   the output is ever rendered with `unsafe_allow_html=True` in Streamlit
2. PHI redaction stub — regex scan for SSN, DOB, and long ID number patterns. Logs a
   WARNING if triggered. The comment in the code reads: "Replace with AWS Comprehend
   Medical in production." This stub is present even though the pipeline doesn't process
   real PHI — it signals production and compliance instincts.

**`rate_limit_check`:** Stub that always returns True. Comment: "Replace with Redis-backed
counter in production." A Redis implementation would use `INCR` + `EXPIRE` on a key like
`f"rate:{user_id}"` and compare against a configured max requests per minute.

**Why PHI stub even in a demo?** CLAUDE.md mandates it for every project. Interviewers
from healthcare companies notice when PHI handling is absent. Having the stub present —
with the production note — demonstrates that the developer has thought about compliance
even at the prototype stage.

---

### pipeline.py

**What it does:** Wires the three agents together, calls guardrails as middleware, and
exposes a single `run(input_data, user_id)` function.

**`build_initial_state`:** A helper that constructs the state dict with all required keys
at their initial values. This ensures every key that an agent might read exists in the
state before any agent runs — no `KeyError` surprises.

**`max_pipeline_steps: 10`:** The safety ceiling. With 3 agents, the pipeline uses steps
0-2. The ceiling of 10 means there's room to add 7 more agents without touching the
wiring. The pipeline checks `pipeline_step >= max_pipeline_steps` after each agent and
returns early if it's reached — no infinite loops, no missing output.

**`--dry-run` flag:** Per the Terminal Runability Standard in CLAUDE.md. Dry run bypasses
all LLM calls and only exercises guardrails wiring. `python pipeline.py --dry-run` must
pass with no API keys. This is how you validate state dict wiring and guardrails
integration without burning API tokens.

---

### app.py

**What it does:** Streamlit UI that accepts intake form input (text or PDF), runs the
pipeline, and displays the routing card plus agent step expanders.

**Two-column layout:** Left column = input, right column = output. This is a deliberate
UX choice — the user can see the form they submitted alongside the routing result without
scrolling.

**PDF extraction:** `pypdf.PdfReader` reads uploaded PDFs in-memory (`io.BytesIO`). Text
is extracted at the UI layer, before the pipeline sees it. The pipeline only receives a
`dict` with a `"text"` key — it has no concept of file type.

**Routing card with `unsafe_allow_html=True`:** The routing card uses inline CSS for
color-coding (red/yellow/green border and header). Streamlit's native components don't
support per-element custom colors, so raw HTML is used. The output is sanitized in
`guardrails.sanitize_output()` before reaching this point, so this is safe.

**Urgency color mapping:**
```python
urgency_colors = {
    "Emergency": "#FF4B4B",  # red
    "Urgent": "#FFA500",     # orange/amber
    "Routine": "#21C354",    # green
}
```
These match intuitive traffic-light semantics. The urgency emoji (🔴/🟡/🟢) reinforces
the color for accessibility.

**Agent expanders:** Both expanders are `expanded=False` by default — the routing card
is the primary output and should be visible immediately. Interviewers can expand the
agent steps to inspect the full pipeline trace without the card being obscured.

**Error display:** Non-fatal errors are shown in a "Pipeline warnings" expander, not
as `st.error()` — they don't block the routing result from appearing. Only validation
failures and fatal pipeline exceptions use `st.error()` + `st.stop()`.

---

## Interview Talking Points

**"Why three agents instead of one LLM call?"**
Separation of concerns. Extraction errors don't corrupt classification logic. If the
extraction schema changes, only that agent changes. Each step is independently testable
(see the `__main__` blocks in each agent file).

**"Why use `with_structured_output` instead of parsing LLM text?"**
`with_structured_output` uses Claude's native tool use capability to guarantee schema
compliance. Manual parsing is brittle — the LLM might output "URGENT" or "Urgent" or
"urgent" and a regex might miss one. Pydantic validation catches it at the model level.

**"Why is the PHI stub in a demo that doesn't use real patient data?"**
Because production instincts should be visible in portfolio code. Healthcare companies
evaluating a hire want to see that the developer has thought about HIPAA compliance at
every layer, not just when it's mandatory. The stub signals that intention.

**"How would you add EHR integration?"**
Insert a `ehr_lookup_agent` between extraction and routing. After extraction, the agent
queries an EHR FHIR API with patient name + DOB, returns a `patient_record_match` field
(verified/unverified/new patient), and the routing agent incorporates it into the
routing summary. The state dict pattern makes this insertion straightforward — no
existing agents need to change.
