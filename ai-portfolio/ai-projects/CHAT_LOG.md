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
