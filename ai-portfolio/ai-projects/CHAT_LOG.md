# CHAT_LOG.md — Session Log
## AI Portfolio Builder | Cody Culver

Tracks every ask/task performed in this Claude Code session.
Updated automatically after each task.

---

## Ask 1 — Create SKILLS.md
**Outcome:** Created `SKILLS.md` at the project root with 17 sections covering every
reusable LangChain pattern needed across all projects: LLM setup, LCEL chains,
structured output, tool use, vision, RAG, embeddings, memory, Langfuse observability,
shared state dict, agent module template, pipeline template, guardrails template,
Streamlit app template, web search, document loaders, and error handling.
Two quick-reference tables at the bottom: model/temperature picker and pattern picker.

---

## Ask 2 — Build foundational directory structure and all ai-agents
**Outcome:** Created the full project scaffold:
- `ai-agents/` with all 6 build-pipeline agents (flat `.py` files at this stage)
- `projects/` and `apps/` directories (with `.gitkeep`)
- `.env.example` with all required keys and comments
- `requirements.txt` with pinned base dependencies

Agents built:
| Agent | Role |
|---|---|
| `orchestrator.py` | Build loop controller; generates build plan, runs SE→eval loop (max 3 iter), calls guardrail/security/tech_writer, writes `build_log.md` |
| `software_engineer.py` | Generates `pipeline.py`, `agents/`, `app.py`, `requirements.txt` from build plan |
| `evaluator.py` | Structured pass/revise verdict (score 0-10, feedback list, strengths list) |
| `guardrail_engineer.py` | Writes `guardrails.py` with PHI stub; fallback template if LLM fails |
| `security_specialist.py` | Reviews code for secrets, injection vectors, HIPAA risks; writes `docs/security_report.md` |
| `tech_writer.py` | Writes `overview_nontechnical.md`, `overview_technical.md`, `build_walkthrough.md`, `README.md` |

---

## Ask 3 — Expand software_engineer, evaluator, tech_writer into subdirectories
**Outcome:** Refactored the three most complex agents from flat `.py` files into
subdirectory packages. Each now has:

```
ai-agents/
├── software_engineer/
│   ├── __init__.py     ← re-exports run()
│   ├── agent.py        ← orchestration logic
│   ├── prompts.py      ← AGENT_FILE_PROMPT, PIPELINE_PROMPT, APP_PROMPT, REQUIREMENTS_PROMPT
│   └── SKILLS.md       ← code generation patterns, model selection table, non-negotiables checklist
├── evaluator/
│   ├── __init__.py
│   ├── agent.py
│   ├── prompts.py      ← EVAL_PROMPT
│   └── SKILLS.md       ← scoring rubric with weights, state dict checklist, Langfuse checklist, feedback-writing standards
└── tech_writer/
    ├── __init__.py
    ├── agent.py
    ├── prompts.py      ← NONTECHNICAL_PROMPT, TECHNICAL_PROMPT, WALKTHROUGH_PROMPT, README_PROMPT
    └── SKILLS.md       ← audience profiles, doc templates, ASCII diagram syntax, Cody-specific interview analogies
```

Remaining agents (`orchestrator.py`, `guardrail_engineer.py`, `security_specialist.py`) stay flat.

Also fixed `orchestrator.py` import strategy: adds `ai-agents/` to `sys.path` and imports
agents by bare name (e.g. `importlib.import_module("software_engineer")`), which works
uniformly for both flat modules and subpackages via `__init__.py` re-exports.

Old flat files (`software_engineer.py`, `evaluator.py`, `tech_writer.py`) removed.

---

## Ask 4 — Build apps/stereogram-renderer/ and plan project naming
**Outcome:** Built `apps/stereogram-renderer/` — a pure algorithm module (no LLM)
that converts depth map images into stereograms using the SIRTS algorithm.

Files created:
- `apps/stereogram-renderer/main.py` — core `render()` function + `run(input: dict) -> dict` API + CLI entry point
- `apps/stereogram-renderer/requirements.txt` — numpy, Pillow ≥ 10
- `apps/stereogram-renderer/README.md` — module docs, format table, parameter guide, consumer list

Algorithm summary:
- For each row, each pixel: `shift = round((depth/255) × depth_factor × eye_separation)`
- `src = x - eye_separation + shift`; if `src ≥ 0` copy from result, else seed from texture tile
- Dependency is strictly backward → single left-to-right sweep per row is correct
- Texture tiled via numpy to canvas size for O(1) lookup

Input supports `.png`, `.jpg`, `.bmp`, `.webp`, `.tiff`, `.avif` (AVIF needs libavif + Pillow ≥ 10)
Resolution is maintained exactly. Random noise used if no texture provided.

Architecture decision documented: the three-folder rule has a gap for "Streamlit UI with
no LLM." Proposed naming convention for stereogram family:
- `apps/stereogram-renderer/` — algorithm only (built now)
- `projects/stereogram-converter/` — thin Streamlit wrapper, no AI
- `projects/stereogram-pipeline/` — full AI version (text → depth map → renderer)

---

## Ask 5 — Add Terminal Runability Standard to CLAUDE.md
**Outcome:** Updated `CLAUDE.md` with three changes:

1. **Directory structure corrected** — old flat `software_engineer.py / evaluator.py /
   tech_writer.py` entries replaced with the actual subpackage structure built in Ask 3.

2. **Three-folder rule updated** — added fourth row for "Streamlit UI wrapping an apps/
   module, no LLM" → `projects/`. Added prose clarifying `projects/stereogram-converter/`
   as the canonical example.

3. **Terminal Runability Standard added** (new section before Hard Rules):

   | File type | `__main__` depth | API keys needed? |
   |---|---|---|
   | `apps/*/main.py` | Full — real inputs, real output file | No |
   | `projects/*/agents/*.py` | Smoke test — hardcoded minimal input | Yes |
   | `projects/*/pipeline.py` | `--dry-run` flag skips LLM; full run available | `--dry-run`: No |
   | `ai-agents/` files | `--dry-run` validates shape; full run available | `--dry-run`: No |

   Rule added to Hard Rules: "Every `.py` file must have an `if __name__ == "__main__"` block."

---

## Ask 6 — Retrofit __main__ blocks to all existing agent files
**Outcome:** Added `if __name__ == "__main__"` blocks to all five agent files that were
missing them. Every block follows the Terminal Runability Standard from CLAUDE.md.

| File | `--dry-run` | Full run behaviour |
|---|---|---|
| `guardrail_engineer.py` | Prints context shape | Generates guardrails.py to `/tmp/{project}`, previews first 20 lines |
| `security_specialist.py` | Prints files to review | Generates security_report.md to `/tmp/{project}/docs/`, previews first 30 lines |
| `software_engineer/agent.py` | Prints context shape | Generates all project files to a temp dir, reports file sizes |
| `evaluator/agent.py` | Prints context shape | Runs real evaluation against sample pipeline.py + agent file, prints verdict |
| `tech_writer/agent.py` | Prints context shape | Generates all 4 docs to temp dir; `--doc nontechnical\|technical\|walkthrough\|readme` generates a single doc and previews it |

`tech_writer/agent.py` gets an extra `--doc` flag to generate and eyeball a single
document quickly without waiting for all four LLM calls.

---

## Ask 7 — Build projects/stereogram-converter/ (Streamlit wrapper)
**Outcome:** Built the full `projects/stereogram-converter/` project — a thin Streamlit
wrapper around `apps/stereogram-renderer/` with no LLM.

Files created:
```
projects/stereogram-converter/
├── guardrails.py          ← image size/format validation, PHI stub, rate-limit stub
│                             __main__ runs 6 inline test cases with ✓/✗ output
├── pipeline.py            ← bytes in/bytes out; importlib loads renderer by file path
│                             __main__: --dry-run checks imports, full run accepts --depth-map
├── app.py                 ← two-column Streamlit UI; upload, sliders, generate, download
├── requirements.txt       ← numpy, Pillow≥10, streamlit (no API keys)
├── .env.example           ← placeholder, documents future key locations
├── README.md
└── docs/
    ├── overview_nontechnical.md
    ├── overview_technical.md
    └── build_walkthrough.md
```

Key decisions:
- `importlib.util.spec_from_file_location` to import renderer (hyphen in dir name)
- Pipeline takes raw `bytes` — decoupled from Streamlit's UploadedFile type
- `tempfile.TemporaryDirectory()` via `with` block — guaranteed cleanup on exception
- Auto eye-separation checkbox defaults on (image_width // 8) for better UX
- PHI stub present even though pipeline produces no text output — habit over necessity

---

## Ask 8 — Fix CHAT_LOG.md ordering bug
**Outcome:** Rewrote the affected section of CHAT_LOG.md to correct an ordering bug
where Ask 7 appeared before Ask 6 had been written.

Root cause: when inserting Ask 7, the edit anchor matched a line inside Ask 5's content
block rather than the end of the file, causing Ask 7 to be spliced in mid-Ask-5 before
Ask 6 existed. Fixed by rewriting the entire affected section in chronological order.

---

## Ask 9 — Refactor guardrail_engineer and security_specialist into subpackages
**Outcome:** Moved `guardrail_engineer.py` and `security_specialist.py` from flat files
into subdirectory packages, matching the structure of software_engineer, evaluator, and
tech_writer. Each now has:

```
ai-agents/
├── guardrail_engineer/
│   ├── __init__.py     ← re-exports run()
│   ├── agent.py        ← run(), _fallback_guardrails(), _strip_code_fences()
│   ├── prompts.py      ← GUARDRAILS_PROMPT
│   └── SKILLS.md       ← required function signatures, validate_input/sanitize_output
│                           checklists, PHI stub rationale, common mistakes table
└── security_specialist/
    ├── __init__.py
    ├── agent.py        ← run(), _build_files_block(), _load_skills()
    ├── prompts.py      ← SECURITY_REVIEW_PROMPT (6-section report structure)
    └── SKILLS.md       ← review checklist, file priority order, specificity standards,
                            what NOT to flag
```

Old flat files (`guardrail_engineer.py`, `security_specialist.py`) removed.
`orchestrator.py` stays flat — documented in CLAUDE.md as the controller-vs-producer rule.

Also updated `CLAUDE.md`:
1. Directory listing updated to show both agents as subpackages
2. Controller-vs-producer rule added: content-producing agents are subpackages;
   orchestrator.py is flat because it is the build loop controller, not a content producer

---

## Ask 10 — Fix stereogram depth shelving via bilinear interpolation
**Outcome:** Fixed a fundamental quality issue in `apps/stereogram-renderer/main.py` where
smooth depth gradients produced visible stepped shelves in the stereogram output.

Root cause: the original algorithm rounded fractional pixel shifts to integers before
copying, collapsing 256 depth levels into only `max_shift` discrete steps (e.g. 42 steps
for a 1024px image at default settings). An artist using the same depth map on a
professional stereogram service would see smooth output; this renderer produced shelves.

Fix: replaced integer rounding with **bilinear interpolation**. The fractional source
position is now preserved and blended between its two neighbouring already-computed pixels:

```python
# Before — integer rounding, 42 discrete steps
shifts = np.round(depth_map[y] / 255.0 * max_shift).astype(np.int32)
row[x] = row[src]

# After — fractional shift, 256 smooth levels
frac_src = x - eye_separation + float_shifts[x]
src_lo = int(frac_src)
alpha = frac_src - src_lo
row[x] = (row[src_lo] * (1 - alpha) + row[src_hi] * alpha).astype(np.uint8)
```

`max_shift` is now kept as a float throughout — no rounding until the final uint8 cast.
Module docstring updated to describe the interpolated algorithm.
Result: smooth curved surfaces (horse cheek, forehead) with no visible depth banding.

---

## Ask 11 — Add Clinical Trial Eligibility Screener to CLAUDE.md + build project

**Outcome:** Added project spec to CLAUDE.md Suggested Projects section as #3, then
ran the full build loop and debugged the output to a fully working, end-to-end pipeline.

### CLAUDE.md update
Added `clinical-trial-eligibility-screener` to Suggested Projects with full spec:
pipeline agents, Streamlit UI requirements, stretch goal (ClinicalTrials.gov API dropdown),
and M3 alignment rationale.

### Infrastructure fixes (before build)
Two Langfuse v4 breaking changes required fixes across all 6 build agents:
1. `from langfuse.callback import CallbackHandler` → `from langfuse.langchain import CallbackHandler`
2. `CallbackHandler(public_key=..., secret_key=..., host=..., trace_name=...)` → `CallbackHandler()`
   (v4 reads all credentials from env vars; constructor no longer accepts secret_key/host/trace_name)

Also fixed `load_dotenv()` → `load_dotenv(override=True)` in `orchestrator.py` to prevent
empty shell env vars from blocking `.env` values from loading.

### Build loop result
Orchestrator ran 3 iterations. Evaluator scored 6/10 each time (status: `incomplete`).
Build proceeded — guardrail_engineer, security_specialist, and tech_writer all completed.

Consistent evaluator feedback across iterations:
- Missing `chain = prompt | structured_llm` (agents called `structured_llm.invoke(dict)` directly — invalid)
- Wrong field access in `evaluation_agent.py`: `.get("criteria", [])` vs actual `inclusion_criteria`/`exclusion_criteria`
- Missing `state["output"]` assignment before `sanitize_output()`
- `_get_handler()` in project agents still used Langfuse v3 constructor (software_engineer copied old template)
- App verdict card used wrong field names from `VerdictResult`

### Post-build manual fixes
All issues from the build log corrected manually:

| File | Fix |
|---|---|
| All 3 agents | `_get_handler()` → `CallbackHandler()` (Langfuse v4) |
| All 3 agents + pipeline + guardrails | `load_dotenv(find_dotenv(), override=True)` to search upward for `.env` |
| `criteria_agent.py` | `chain = prompt \| structured_llm` before `.invoke()` |
| `evaluation_agent.py` | Same chain fix + correct field access (`inclusion_criteria`/`exclusion_criteria`) |
| `verdict_agent.py` | Same chain fix |
| `pipeline.py` | Added `state["output"] = state.get("verdict_agent_output")` before `sanitize_output()` |
| `pipeline.py` | Added `--dry-run` flag (Terminal Runability Standard) |
| `app.py` | Fixed verdict card to use `eligibility_status`, `confidence_score`, `summary`, `key_factors`, `next_steps` |
| `app.py` | Fixed per-criterion display to use `criterion_text`, `meets_criterion`, `reasoning`, `relevant_patient_info` |
| All 3 agents + guardrails | Added `if __name__ == "__main__"` blocks (Terminal Runability Standard) |

### Verified working
- `python pipeline.py --dry-run` → passes with no API calls
- `python guardrails.py` → all 4 test cases pass; PHI stub fires correctly
- `python pipeline.py` → full run returns **ELIGIBLE, 95% confidence**, zero errors, all 3 agents fire

### Project structure
```
projects/clinical-trial-eligibility-screener/
├── pipeline.py              ← 3-agent orchestration + --dry-run flag
├── guardrails.py            ← validate_input, sanitize_output, rate_limit_check + PHI stub
├── app.py                   ← Streamlit UI: two inputs, verdict card, per-criterion expander
├── agents/
│   ├── __init__.py
│   ├── criteria_agent.py    ← extracts inclusion/exclusion criteria (CriteriaExtractionResult)
│   ├── evaluation_agent.py  ← evaluates patient per criterion (EvaluationResult)
│   └── verdict_agent.py     ← synthesizes final verdict (VerdictResult: ELIGIBLE/INELIGIBLE/NEEDS_REVIEW)
├── requirements.txt
├── .env.example             ← notes to create own Langfuse project for this screener
├── build_log.md
├── README.md
└── docs/
    ├── overview_nontechnical.md
    ├── overview_technical.md
    └── build_walkthrough.md
```

### Root cause note (for future build agent improvement)
The software_engineer agent consistently generated `structured_llm.invoke(dict)` instead of
`chain = prompt | structured_llm; chain.invoke(dict)`. This is a SKILLS.md gap — the
structured output snippet shows the chain pattern but the agent's prompt generation doesn't
enforce it. Consider adding an explicit checklist item to software_engineer's SKILLS.md.

---

## Ask 12 — Build projects/clinical-intake-router/

**Outcome:** Built the full `projects/clinical-intake-router/` project — a 3-agent
sequential pipeline that reads clinical intake forms, extracts structured fields,
classifies urgency and department, and generates plain-English routing instructions
for healthcare staff.

### Confirmation
`clinical-intake-router` spec was already present in `CLAUDE.md` as project #1 (lines
392–428) from a prior session. No update required.

### Build method
Direct construction by Claude Code (no orchestrator LLM retry loop). All conventions
from `CLAUDE.md` followed: LCEL chain pattern, Langfuse v4, Pydantic structured output,
shared state dict, guardrails middleware, `__main__` blocks on every `.py` file.

### Project structure
```
projects/clinical-intake-router/
├── pipeline.py              ← 3-agent orchestration + --dry-run flag
├── guardrails.py            ← validate_input, sanitize_output (PHI stub), rate_limit_check
├── app.py                   ← Streamlit UI: text/PDF input, routing card, 2 agent expanders
├── agents/
│   ├── __init__.py
│   ├── extraction_agent.py    ← ExtractedFields Pydantic model, temp=0
│   ├── classification_agent.py ← ClassificationResult + UrgencyLevel Enum, temp=0
│   └── routing_agent.py       ← RoutingResult, temp=0.3 (prose output)
├── requirements.txt
├── .env.example
├── build_log.md
├── README.md
└── docs/
    ├── overview_nontechnical.md
    ├── overview_technical.md
    └── build_walkthrough.md
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Free-text `department` field (not Enum) | Clinical routing is open-domain; Enum creates false precision |
| Temperature 0 for extraction + classification | Determinism matters for clinical decisions |
| Temperature 0.3 for routing | Routing summary is staff-facing prose — mechanical output is a UX problem |
| 8,000 char input limit | Intake forms with full PMH + meds exceed 4,000 chars |
| PHI stub in every project | Signals production/compliance instincts to healthcare interviewers |
| Lazy EHR stretch goal | Documented in build_walkthrough.md; integration point identified but not built |

### Verified working
- `python pipeline.py --dry-run` → passes, no API keys needed
- `python guardrails.py` → 5/5 test cases pass; PHI stub fires correctly; HTML strip confirmed

---

## Ask 13 — Rename urgency level Emergency → Emergent

**Problem:** The routing card header renders `{urgency_level} — {department}`. Using "Emergency"
for both the top urgency level and the Emergency department produced ambiguous output:
- "Urgent — Emergency" (patient going to Emergency dept at Urgent priority — is that one concept or two?)
- "Emergency — Emergency" (redundant and confusing)

**Fix:** Renamed `UrgencyLevel.EMERGENCY = "Emergency"` → `UrgencyLevel.EMERGENT = "Emergent"`.
"Emergent" is the correct clinical triage term — already used by nurses and coordinators —
and cleanly separates the urgency namespace from the department namespace.

**Files changed:**
| File | Change |
|---|---|
| `agents/classification_agent.py` | Enum value + system prompt: `Emergency` → `Emergent` |
| `agents/routing_agent.py` | Field description + `__main__` test hardcoded value |
| `app.py` | `urgency_colors` and `urgency_emoji` dict keys |
| `docs/overview_technical.md` | Enum schema docs + new "Why Emergent" design note |
| `docs/build_walkthrough.md` | New paragraph explaining the naming decision |
| `docs/overview_nontechnical.md` | Added plain-English note for non-technical readers |
| `CLAUDE.md` | Suggested Projects urgency level list updated |

---

## Ask 14 — Add explicit labels to routing card in app.py

**Problem:** Routing card header showed `🔴 Emergent — Emergency` with no labels. A reader
unfamiliar with the format had to infer which token was urgency and which was department.

**Fix:** Replaced the single `<h3>` header with a labeled flex row. Each field now has a
small-caps uppercase label ("URGENCY", "DEPARTMENT", "EXPECTED RESPONSE") above its value.
The summary text is separated by a faint divider line below the fields.

**File changed:** `app.py` — routing card HTML block only.

---

## Ask 15 — Simplify Langfuse run_name to agent name only

**Problem:** Traces were named `"clinical-intake-router/extraction_agent"` etc. — redundant
since the Langfuse project is already called `clinical-intake-router`.

**Fix:** Stripped the project prefix from `run_name` in all three agents.
Also updated `CLAUDE.md` observability convention so future projects don't repeat the pattern,
and updated `docs/overview_technical.md` to reflect the corrected trace names.

**Files changed:** `agents/extraction_agent.py`, `agents/classification_agent.py`,
`agents/routing_agent.py`, `CLAUDE.md`, `docs/overview_technical.md`.

---

## Ask 16 — Group all 3 agent LLM calls under one Langfuse trace per pipeline run

**Problem:** Each agent created its own independent Langfuse trace, so one form submission
produced 3 disconnected traces. No way to see a full pipeline run in one place.

**Fix:** Pipeline creates a root trace at the start of `run()` and stores `trace.id` in
`state["langfuse_trace_id"]`. Each agent's `_get_handler()` now accepts an optional
`trace_id` and passes it to `CallbackHandler(trace_id=...)`, making its LLM call a child
span of the root trace. Pipeline updates the trace at the end with urgency level, department,
and error metadata.

Fallback: when `trace_id` is None (agent run standalone via `__main__`), `_get_handler()`
falls back to `CallbackHandler()` and creates its own trace — solo agent runs remain observable.

**Files changed:** `pipeline.py`, all 3 agent files, `CLAUDE.md` observability convention,
`docs/overview_technical.md`.

**Verified:** `python pipeline.py --dry-run` — `langfuse_trace_id` key present in state.

---

## Ask 17 — Fix Langfuse still showing 3 traces; remove silent try/except from trace creation

**Problem (reported):** Still seeing 3 separate Langfuse traces per pipeline run instead
of 1. User also noted that CLAUDE.md did not document the try/except pattern that was
added around trace creation.

**Root cause:** The broad `try/except Exception` around `Langfuse().trace()` in
`pipeline.py` was silently swallowing any initialization error, leaving
`state["langfuse_trace_id"]` as `None`. Each agent then hit the fallback
(`CallbackHandler()`) and created its own independent trace.

**Fix:**
- Removed try/except from trace creation — if Langfuse can't create the root trace,
  the error now surfaces rather than being hidden
- `trace.update()` at pipeline end remains wrapped (it is metadata-only and must not
  break the return value)
- Updated `CLAUDE.md` observability convention to explicitly document: trace creation
  must NOT be wrapped in try/except; only `trace.update()` may be
- Updated `docs/overview_technical.md` to clarify the only intended fallback path

**Files changed:** `pipeline.py`, `CLAUDE.md`, `docs/overview_technical.md`.

---

## Ask 18 — Fix single-trace grouping using correct Langfuse v4 API

**Problem:** User confirmed 3 traces still appearing. `Langfuse().trace()` (raw SDK client)
and `CallbackHandler()` (LangChain integration) use different initialization paths in v4.
The raw client was failing silently (removed try/except in Ask 17 revealed the error).
User also noted this was working in their OpenAI projects — those used the `@observe`
decorator, not `Langfuse().trace()` directly.

**Root cause confirmed:** Langfuse v4 removed `langfuse.decorators` module, renamed APIs,
and `CallbackHandler` no longer accepts `trace_id=` as a string — it accepts
`trace_context=TraceContext(trace_id, parent_span_id)`. The `langfuse_context` object
from v2/v3 also does not exist in v4.

**Correct v4 pattern (discovered by inspecting installed package):**
```python
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext  # TypedDict: {trace_id, parent_span_id}

@observe(name="intake_route")
def run(...):
    lf = get_client()
    handler = CallbackHandler(trace_context=TraceContext(
        trace_id=lf.get_current_trace_id(),
        parent_span_id=lf.get_current_observation_id(),
    ))
    state["langfuse_handler"] = handler
    # agents read: state.get("langfuse_handler") or CallbackHandler()
```

**Files changed:** `pipeline.py` (full rewrite), `requirements.txt` (langfuse>=4.0.0),
all 3 agent files (unchanged logic, already correct from Ask 16), `CLAUDE.md`, `docs/overview_technical.md`.

**Verified:** `python pipeline.py --dry-run` passes with correct state keys.

---

## Ask 19 — Fix trace list-view name showing last agent's run_name

**Problem:** Trace list in Langfuse showed "routing_agent" or "classification_agent" as
the trace name (whichever agent ran last), not a meaningful pipeline name. The hierarchy
inside the trace was correct — `intake_route` → agent spans — but the top-level Name
column in the list was wrong.

**Root cause:** Langfuse v4 reads `LangfuseOtelSpanAttributes.TRACE_NAME`
(`langfuse.trace.name`) from OTel span attributes to set the trace display name. The
`CallbackHandler` was writing this attribute with each agent's `run_name`, and whichever
agent ran last won. `@observe(name="intake_route")` sets the SPAN name, not this attribute.

**Fix:** Set `langfuse.trace.name = "clinical-intake-router"` on the OTel span explicitly
at the top of `run()`, before any `CallbackHandler` executes. This pins the trace name
regardless of agent order or failures.

```python
otel_trace.get_current_span().set_attribute(
    LangfuseOtelSpanAttributes.TRACE_NAME, "clinical-intake-router"
)
```

`@observe(name="intake_route")` is kept — it remains the span name visible in the
trace hierarchy (the step below the trace root), which the user confirmed they liked.

**Resulting structure:**
```
clinical-intake-router   ← trace list name (TRACE_NAME attribute)
  └── intake_route       ← @observe span (trace detail hierarchy)
        └── extraction_agent
        └── classification_agent
        └── routing_agent
```

**Files changed:** `pipeline.py`, `CLAUDE.md`, `docs/overview_technical.md`.
**Verified:** `python pipeline.py --dry-run` passes.

---

## Ask 20 — Fix trace name: set_attribute insufficient; switch to propagate_attributes

**Problem:** `set_attribute(LangfuseOtelSpanAttributes.TRACE_NAME, ...)` on a single span
did not hold — the trace list name was still showing an agent name (not always the last
one, making the "last agent wins" theory wrong). The CallbackHandler writes
`langfuse.trace.name` to its own child spans; whichever child span Langfuse processes
last for naming purposes wins.

**Root cause clarified via `propagate_attributes` docstring:**
`propagate_attributes(trace_name=...)` sets the attribute on the current span AND
propagates it to ALL new child spans created within the context. `set_attribute` only
sets it on one span; subsequent CallbackHandler child spans are unaffected and overwrite.

**Fix:** Wrap the entire pipeline body (from handler creation through routing) in
`with propagate_attributes(trace_name="clinical-intake-router", user_id=user_id)`.
Every span the CallbackHandler creates inside that context inherits `trace_name`,
so no agent can overwrite the display name.

Also removed the `otel_trace` / `LangfuseOtelSpanAttributes` imports that were added
for the previous (failed) approach.

**Files changed:** `pipeline.py`, `CLAUDE.md`.
**Verified:** `python pipeline.py --dry-run` passes.

---

## Ask 19 — Rename Langfuse trace from "intake_route" to "clinical-intake-router"

**Problem:** Trace name appeared random/non-descriptive. Investigated `@observe` source
in v4 — it sets the root OTel span name, which Langfuse uses as the trace name. The
previous name `"intake_route"` was unclear; if it wasn't being picked up, traces fell
back to a generated identifier.

**Fix:** `@observe(name="intake_route")` → `@observe(name="clinical-intake-router")`.
Naming convention going forward:
- Full pipeline trace: `"clinical-intake-router"` (matches project name, unambiguous)
- Standalone agent runs: trace named by `run_name` on `chain.invoke` (e.g. `"extraction_agent"`)

**Files changed:** `pipeline.py`, `docs/overview_technical.md`.

---

## Ask 21 — Add Medical Triage Classifier to CLAUDE.md Suggested Projects

**Outcome:** Added `### 4. Medical Triage Classifier` to CLAUDE.md's Suggested Projects
section, positioned after project #3 (Clinical Trial Eligibility Screener) and before the
Stepwise build pattern section. No other entries modified.

### What was added
Full project spec for `projects/medical-triage-classifier/` — a PEFT/LoRA fine-tuning
project that trains Bio_ClinicalBERT or distilbert on MTSamples + Claude-generated
synthetic data to classify clinical text into Routine / Urgent / Emergency urgency levels.

### Key architectural decisions documented in the spec

| Decision | Rationale |
|---|---|
| Does NOT use pipeline.py/agents/ or build loop | This is a model training project, not a multi-agent pipeline |
| MLflow Tracking Server on EC2 (primary, not stretch) | Production ML requires centralized experiment tracking; localhost MLflow is not shareable |
| MLflow Model Registry as authoritative model store | `classifier.py` loads from registry at inference time — never local paths |
| S3 for dataset + checkpoints + artifacts | Raw CSV not committed; checkpoints enable resume; artifacts enable registry |
| Colab/SageMaker for training (GPU required) | LoRA fine-tuning on BERT needs GPU; local CPU training would take hours |
| Three-way Streamlit comparison (baseline vs fine-tuned vs Claude) | Shows cost/latency/accuracy tradeoffs — strongest interview artifact |
| `classifier.py` interface matches intake router's classification_agent | Same input (text) and output shape (urgency + confidence) for future drop-in replacement |
| Future Integration Notes in overview_technical.md | Documents how this connects to clinical-intake-router without modifying any intake router files |
| RLHF stretch goal with KLA JD alignment note | Clinician feedback loop for reward model training — noted but not built |

### Build order (9 steps)
1. `provision_infra.py` — EC2 MLflow + S3 bucket setup docs
2. `data_prep.py` — MTSamples load + Claude synthetic augmentation + MLflow logging
3. `trainer.py` — LoRA fine-tuning, EC2 MLflow tracking, S3 checkpoints, model registration
4. `evaluator.py` — baseline vs fine-tuned vs Claude comparison, all metrics to MLflow
5. `classifier.py` — loads from MLflow Model Registry, exposes `run()` interface
6. `guardrails.py` — standard input/output/PHI stub
7. `app.py` — Streamlit three-way comparison dashboard
8. tech_writer docs
9. README with architecture diagram

### Clinical intake router review
Read the full `projects/clinical-intake-router/` codebase to understand integration
points. Key findings:
- `classification_agent.py` uses Claude Sonnet at temperature 0 with `ClassificationResult`
  Pydantic model (urgency_level, department, classification_reasoning, confidence, red_flags)
- S3 client in `storage/s3_client.py` — shared bucket pattern viable
- Supabase for structured results — could store classification comparisons
- Langfuse v4 tracing already in place — fine-tuned model traces would appear alongside
- MLflow is not used anywhere in the intake router
- No intake router files were modified — integration suggestions documented in spec only

**Files changed:** `CLAUDE.md` (new project #4 entry), `CHAT_LOG.md` (this entry).
