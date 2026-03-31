# CLAUDE.md — Project Constitution
## AI Portfolio Builder | Cody Culver

This is the single authoritative guide for Claude Code in this project.
Read it fully before taking any action. Do not deviate from these conventions.

---

## Who This Is For

**Developer:** Cody Culver — Data Scientist and AI Engineer. Strong in Python, LangChain,
RAG, multi-agent systems, Langfuse, Hugging Face, AWS, and Pydantic. Comfortable with
SQL, Git, Docker, REST APIs, and Kubernetes basics. Physics background — appreciates
clear mental models. Game dev hobbyist (Unity/C#) — state machine and game loop analogies
are useful. Less fluent in TypeScript, advanced DevOps, and raw cloud infra: keep patterns
simple and explainable.

**Target Role:** AI Engineer at M3 USA — healthcare/life sciences, agentic workflow
automation, LLM-based document processing, stakeholder communication, production systems.

**Goal:** Build a portfolio of agentic AI projects using Claude Code as the primary
development tool. Every project must be runnable, demoable via Streamlit, linkable on
GitHub, and fully explainable in a technical interview.

---

## What the JD Is Looking For (and How This Project Answers It)

| JD Requirement | How We Demonstrate It |
|---|---|
| Agentic coding with Claude Code | This entire repo is built with Claude Code |
| Multi-agent systems with tool use | Every project in `projects/` is a multi-agent pipeline |
| LLM APIs (Claude, OpenAI) | LangChain + ChatAnthropic throughout |
| Production deployment | Streamlit Community Cloud, shareable URL per project |
| Guardrails and security | Every project has `guardrails.py` as middleware |
| Observability | Langfuse instrumentation on every LLM call |
| Stakeholder communication | Non-technical doc per project, written for a clinic manager |
| Healthcare awareness | PHI redaction stubs and HIPAA flags even in non-clinical demos |
| Workflow automation | Projects automate real, recognizable business tasks |
| Portfolio / GitHub repos | Clean README, runnable in 3 commands, Streamlit demo link |

---

## Directory Structure

```
project-root/
│
├── CLAUDE.md                        ← You are here. Read before all else.
├── SKILLS.md                        ← Reusable LangChain patterns and snippets
│
├── ai-agents/                       ← Reusable build-pipeline agents (not user-facing)
│   ├── orchestrator.py              ← flat — build loop controller (see rule below)
│   ├── software_engineer/           ← subpackage
│   │   ├── __init__.py              ← re-exports run()
│   │   ├── agent.py                 ← orchestration logic
│   │   ├── prompts.py               ← ChatPromptTemplate definitions
│   │   └── SKILLS.md                ← code generation patterns
│   ├── evaluator/                   ← subpackage
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── SKILLS.md                ← review rubric and feedback standards
│   ├── guardrail_engineer/          ← subpackage
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompts.py               ← GUARDRAILS_PROMPT
│   │   └── SKILLS.md                ← required function signatures, PHI stub rationale
│   ├── security_specialist/         ← subpackage
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompts.py               ← SECURITY_REVIEW_PROMPT
│   │   └── SKILLS.md                ← 6-section report structure, review checklist
│   └── tech_writer/                 ← subpackage
│       ├── __init__.py
│       ├── agent.py
│       ├── prompts.py
│       └── SKILLS.md                ← doc templates and audience guides
│
├── projects/                        ← Everything demoable and linkable
│   └── [project-name]/
│       ├── pipeline.py              ← Multi-agent orchestration logic
│       ├── agents/                  ← Agent modules for this project
│       │   └── [agent_name].py
│       ├── guardrails.py            ← Input/output safety middleware
│       ├── app.py                   ← Streamlit demo UI (always present)
│       ├── requirements.txt
│       ├── .env.example
│       ├── docs/
│       │   ├── overview_nontechnical.md
│       │   ├── overview_technical.md
│       │   └── build_walkthrough.md
│       └── README.md
│
└── apps/                            ← Pure algorithms, zero AI (building blocks)
    └── [module-name]/
        ├── main.py                  ← Exposes def run(input: dict) -> dict
        ├── requirements.txt
        └── README.md
```

### The three-folder rule

| What you're building | Where it goes |
|---|---|
| A reusable build agent (software_engineer, tech_writer, etc.) | `ai-agents/` |
| A pure algorithm — no LLM, deterministic output | `apps/` |
| A Streamlit UI wrapping an `apps/` module — no LLM, but demoable | `projects/` |
| Anything with an LLM call — simple or complex | `projects/` |

If it has an LLM call, it belongs in `projects/`. A single-step classifier and a
five-agent pipeline both live here — they just have more or fewer agents. If you'd
show it to an interviewer, it's a project.

A `projects/` entry that wraps an `apps/` module with a Streamlit UI but no LLM is
valid — it still gets `pipeline.py`, `guardrails.py`, `app.py`, and docs. It just has
no agent files. See `projects/stereogram-converter/` as the canonical example.

### Controller-vs-producer rule (ai-agents/)

**Content-producing agents are subpackages.** Any agent that writes a substantial
prompt template belongs in its own subdirectory with `agent.py`, `prompts.py`, and
`SKILLS.md`. This makes prompts reviewable and editable without touching agent logic.

**`orchestrator.py` stays flat — intentionally.** The orchestrator does not produce
content. It runs the build loop: calls sub-agents in sequence, manages iteration count,
and writes `build_log.md`. It has no prompt templates worth extracting. Putting it in a
subpackage would add structure with no benefit. If an interview question comes up: the
orchestrator is the *controller*; the other five agents are *producers*.

---

## Agent Roles (ai-agents/)

These are the **build pipeline** agents — they construct and document projects.
They are not user-facing. Each exposes `run(context: dict) -> dict`.

### orchestrator.py
- Accepts a plain-language project description and target folder name
- Produces a `build_plan` dict: goal, tech stack, agent steps, `max_build_iterations` (default 3)
- `max_build_iterations` controls how many times the software_engineer → evaluator retry
  cycle runs before giving up. It does NOT control how many agents run inside a project.
- Runs the build loop: software_engineer → evaluator → (guardrail_engineer + security_specialist) → tech_writer
- Exits when evaluator returns `"pass"` OR after `max_build_iterations` — never loops indefinitely
- Writes `build_log.md` with each iteration result and final status

### software_engineer.py
- Receives `build_plan` plus any `evaluator_feedback` from prior iterations
- Writes `pipeline.py`, `agents/`, `app.py`, and `requirements.txt`
- Uses LangChain patterns from `SKILLS.md`
- Does NOT write guardrails or security logic — those are separate agents

### evaluator.py
- Reviews generated code for: correctness, LangChain best practices, appropriate LLM use,
  error handling, and Streamlit UI completeness
- Returns: `{ "status": "pass" | "revise", "score": 0-10, "feedback": [...] }`
- Feedback passed to software_engineer for the next iteration

### guardrail_engineer.py
- Writes `guardrails.py` for the project
- Must include: input validation, output sanitization, prompt injection detection,
  PHI/PII redaction stubs, content safety check, rate limiting hooks
- Guardrails are called as pre/post middleware in `pipeline.py`

### security_specialist.py
- Reviews `pipeline.py` and `guardrails.py` for: exposed secrets, prompt injection vectors,
  insecure data handling, missing auth, hardcoded credentials
- Writes `docs/security_report.md`
- Always flags HIPAA-adjacent risks even if the project doesn't use real health data

### tech_writer.py
- Writes three documents in `docs/`:
  1. `overview_nontechnical.md` — plain English for a non-technical business stakeholder.
     Imagine a clinic operations manager or a parent using the toy safety checker.
     No jargon. What it does, why it matters, what you see when you run it.
  2. `overview_technical.md` — for a technical interviewer. Pipeline architecture, agent
     roles, LLM choices and rationale, data flow, guardrails design, Langfuse setup,
     deployment path, tradeoffs made.
  3. `build_walkthrough.md` — for Cody. Step-by-step construction narrative: what was
     built first, why each file exists, what each function does, what design decisions
     were made and why. Cody should be able to read this and explain everything in an
     interview without looking at the code.

---

## Project Architecture Pattern

Every project follows this shape regardless of complexity:

```
User input (Streamlit UI)
    │
    ▼
guardrails.py → validate_input()
    │
    ▼
pipeline.py → run(input: dict) -> dict
    ├─► agent_1.run(state)     # e.g. vision, classification, extraction
    ├─► agent_2.run(state)     # e.g. lookup, search, retrieval
    └─► agent_3.run(state)     # e.g. report generation, summarization
    │
    ▼
guardrails.py → sanitize_output()
    │
    ▼
Streamlit UI → display result
```

### Shared state dict

Agents communicate through a shared `state` dict. Each agent reads what it needs
and writes only to its own key. Never delete or overwrite another agent's key.

```python
state = {
    "input": ...,           # Original user input — never mutated
    "pipeline_step": 0,        # which agent step is currently executing
    "max_pipeline_steps": 10,  # safety ceiling — not a target; most pipelines use 3-5 steps
    "errors": [],           # Non-fatal errors accumulated across steps
    # Each agent appends its own output key:
    "agent_1_output": ...,
    "agent_2_output": ...,
    "output": ...,          # Final result — written by last agent
}
```

---

## Technology Conventions

### LLMs
- Primary: `langchain_anthropic.ChatAnthropic`
- Model: `claude-sonnet-4-20250514` for reasoning, vision, and complex generation
- Model: `claude-haiku-4-5-20251001` for cheap, fast, high-frequency steps
- Always pass `max_tokens` explicitly
- Temperature: 0 for classification/structured output, 0.3–0.7 for generative tasks

### LangChain
- Use LCEL pipe syntax: `chain = prompt | llm | parser`
- Use `llm.with_structured_output(MyPydanticModel)` for structured responses
- Use `@tool` decorator + `llm.bind_tools([...])` for tool use
- Use `ChatPromptTemplate.from_messages(...)` — not raw string prompts
- For RAG: `Chroma` locally, `Qdrant` for production
- For document loaders: `langchain_community.document_loaders`
- For web search in agents: `langchain_community.tools.DuckDuckGoSearchRun` (free)
- Memory in chat apps: `ConversationBufferWindowMemory(k=10)`

### Vision (image inputs)
- Accept as file path or base64; convert to base64 at pipeline entry point
- Pass base64 through state dict — don't re-read files mid-pipeline
- Use `HumanMessage` with a content list containing an image block and a text block
- For algorithmic image processing (resize, depth maps, etc.): `Pillow` + `numpy`

### Embeddings
- `langchain_huggingface.HuggingFaceEmbeddings`
- Default model: `sentence-transformers/all-MiniLM-L6-v2` (free, no API key needed locally)

### Observability (Langfuse)
- Every LLM call gets a `CallbackHandler` from `langfuse`
- Pass handler to every `.invoke()` call
- Langfuse v4 API (langfuse>=4.0.0):
  - `observe`, `get_client` import from `langfuse` directly — `langfuse.decorators` does not exist
  - `CallbackHandler` accepts `trace_context=TraceContext(trace_id, parent_span_id)` — not `trace_id=` directly
  - `langfuse_context` does not exist — use `get_client()` inside an `@observe` context instead
- Decorate `pipeline.run()` with `@observe(name="intake_route")` — this creates the root span visible in the trace hierarchy
- Wrap the entire pipeline body in `propagate_attributes(trace_name="[project-name]", user_id=user_id)` — this pins the trace display name on the current span AND all child spans (including CallbackHandler spans), so no agent's `run_name` can overwrite it:
  ```python
  with propagate_attributes(trace_name="clinical-intake-router", user_id=user_id):
      # ... all agent calls go here ...
  ```
  `set_attribute(TRACE_NAME, ...)` on a single span is NOT sufficient — CallbackHandler writes to its own child spans and the last one written wins.
- Then get a scoped handler once and store it in state:
  ```python
  lf = get_client()
  state["langfuse_handler"] = CallbackHandler(
      trace_context=TraceContext(
          trace_id=lf.get_current_trace_id(),
          parent_span_id=lf.get_current_observation_id(),
      )
  )
  ```
- Each agent reads `state.get("langfuse_handler") or CallbackHandler()` — fall back to standalone only when running outside the pipeline (e.g. `__main__`)
- Tag each span: `run_name="[agent-name]"` in `chain.invoke(config=...)`
- Update trace output at pipeline end with `lf.set_current_trace_io(output={...})` — MAY be wrapped in try/except (metadata only, must never break the return)
- Log: prompt, response, latency, token count, eval scores where available
- Non-negotiable — observable systems are a direct signal to M3

### Streamlit (app.py)
- Every project has `app.py` — this is the demo and interview artifact
- Keep UI simple: input widget → run button → output display
- Show a spinner during pipeline execution: `st.spinner("Running pipeline...")`
- Display intermediate agent outputs in `st.expander` so the pipeline is visible,
  not a black box — interviewers want to see agent steps
- Sidebar must include: project description, tech stack, link to GitHub repo
- Must run with: `streamlit run app.py`
- Deploy to Streamlit Community Cloud for a shareable public URL

### AWS (free tier only)
- S3 for file storage, Lambda for lightweight endpoints
- Cost alarm set at $30 — never exceed free tier without explicit confirmation
- Use `boto3` for integrations

### Structured Storage (Supabase PostgreSQL)
- Use Supabase as the standard for structured output data (pipeline results, metadata)
- S3 holds raw files; Supabase holds structured results — keep them separate
- Connection: always use the **Session pooler** URI from the Supabase Connect modal
  (not the Direct or Transaction pooler). Session pooler works on IPv4 networks, which
  covers both local dev and Streamlit Community Cloud. Direct connection requires IPv6.
- Use `psycopg2-binary` for the DB driver
- SHA-256 hash file bytes at upload time; store in DB as the natural dedup key
- Standard table for intake-style projects:
  ```
  file_hash, s3_key, s3_bucket, original_filename, file_size_bytes,
  [agent output columns as JSONB], submitted_at
  ```
- Add `SUPABASE_DB_URI=` to `.env.example` for every project that uses it

### Virtual Environments
- Every project in `projects/` gets its own `.venv` — never shared across projects
- Create with: `python -m venv .venv && source .venv/bin/activate`
- Install with: `pip install -r requirements.txt`
- `.venv/` is in `.gitignore` — never committed
- Streamlit Community Cloud ignores `.venv` entirely and installs from `requirements.txt` directly
- `requirements.txt` is always the source of truth for dependencies

### Secrets and Environment
- All keys in `.env` — never hardcoded, never committed
- Always provide `.env.example` with comments explaining each key
- Required for every project:
  ```
  ANTHROPIC_API_KEY=
  LANGFUSE_PUBLIC_KEY=
  LANGFUSE_SECRET_KEY=
  LANGFUSE_HOST=https://cloud.langfuse.com
  HUGGINGFACE_API_KEY=        # only if using HF inference API
  ```

### Dependencies
- Always pin versions in `requirements.txt`
- Minimum for any project:
  ```
  langchain>=0.3.0
  langchain-anthropic>=0.3.0
  langchain-community>=0.3.0
  langchain-huggingface>=0.1.0
  langfuse>=2.0.0
  streamlit>=1.35.0
  python-dotenv>=1.0.0
  pydantic>=2.0.0
  ```

---

### RBAC (Role-Based Access Control)
Healthcare and multi-user projects use a `RoleConfig` Pydantic model as the single
source of truth for all role-specific settings. This keeps UI rendering, DB query
restrictions, and guardrail enforcement consistent without duplicating logic.

```python
class RoleConfig(BaseModel):
    role: str
    display_name: str
    allowed_columns: Optional[list[str]]  # None = unrestricted
    can_see_classification: bool
    can_see_full_extraction: bool
    can_delete_documents: bool            # admin only
    nl2sql_schema: str                    # role-scoped schema shown to NL2SQL agent
    badge_color: str
```

Standard demo roles for healthcare projects:
- `demo-admin` — full access + document deletion (purple badge)
- `demo-doctor` — full clinical access, no delete (green badge)
- `demo-reception` — routing/scheduling only, no clinical fields (orange badge)

`authenticate(username, password) -> Optional[RoleConfig]` is the single auth entry
point. All downstream role decisions read from the `RoleConfig` object — never from
raw role strings scattered through the code.

### NL2SQL Guardrail Layers
When a project exposes natural-language database queries, use 4 independent layers.
The key principle: deterministic checks first, LLM-dependent checks last.

- **L1 — Schema restriction**: show the LLM only the columns its role is allowed to see
- **L2 — AST validation**: parse LLM-generated SQL with `sqlglot` and check every
  `exp.Column` reference against the allowlist; block `SELECT *` for restricted roles.
  This is deterministic — it doesn't rely on the LLM complying with instructions.
- **L3 — Result column strip**: after query execution, drop any restricted columns
  that slipped through (defence against LLM non-compliance)
- **L4 — Output keyword scan**: scan the synthesised plain-English answer for clinical
  terms/patterns that shouldn't appear in restricted role output

Use `sqlglot>=23.0.0` for AST parsing. Never rely solely on prompt instructions to
enforce column restrictions.

### Multi-Store Delete Ordering
When deleting from both a database and a file store (S3), always delete the DB record
first. This is the safe failure ordering:

- DB fails → both intact. User can retry.
- DB succeeds, S3 fails → orphaned S3 file (invisible to users, not a broken reference).

The reverse (S3 first) leaves a DB row pointing to a deleted file — a broken reference
that causes errors every time a user selects it and is hard to recover from.

Queue any S3 failures for retry rather than silently swallowing them.

---

## Guardrails Standard (guardrails.py)

Every project gets these three functions, always:

```python
def validate_input(data: dict) -> dict:
    """Type checks, size limits, prompt injection scan. Raises ValueError on failure."""
    # Max image size: 10MB
    # Max text length: 4000 chars
    # Scan for prompt injection patterns

def sanitize_output(data: dict) -> dict:
    """Strip code injection from LLM output. PHI/PII redaction stub. Content safety flag."""
    # PHI stub: log warning if triggered — present in EVERY project, even non-clinical
    # Flag: violence, illegal activity, adult content
    # Note: "replace with production-grade scanner (e.g. AWS Comprehend Medical) in prod"

def rate_limit_check(user_id: str) -> bool:
    """Stub — returns True. Comment: 'replace with Redis-backed counter in production'."""
```

The PHI redaction stub is present in **every** project even when the app has nothing to
do with healthcare. This is deliberate — it signals production and compliance instincts.

---

## README Standard

Every project README must be runnable in 3 commands:

```markdown
# [Project Name]

[One sentence: what it does. One sentence: why it matters.]

## Run it
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py

## What you'll see
[2-3 sentences describing the UI and expected output]

## How it works
[ASCII pipeline diagram or brief agent step list]

## Tech stack
- LangChain + Claude (Anthropic)
- Langfuse observability
- Streamlit demo
- [project-specific tools]
```

---

## Build Loop Protocol

```
orchestrator.run(description, project_name)
  │
  ├─► software_engineer.run(build_plan)
  ├─► evaluator.run(code)             → "pass" or "revise" + feedback
  │      └─ if "revise" and iter < 3  → software_engineer.run(plan + feedback)  [loop]
  │
  ├─► guardrail_engineer.run(pipeline.py)  → writes guardrails.py
  ├─► security_specialist.run(...)    → writes security_report.md
  └─► tech_writer.run(all outputs)        → writes 3 docs
```

Max build iterations: 3. On reaching max, save last feedback to `build_log.md` as `"incomplete"`.
This only governs the build retry loop — it has no effect on product pipeline agent steps.

---

## Suggested Projects

### 1. Clinical Intake Router (M3-aligned — **BUILT**)
Location: `projects/clinical-intake-router/`

**Status: complete.** Do not rebuild. Read the existing code and docs before touching it.

A staff member at a healthcare organization pastes or uploads a clinical intake form.
The pipeline extracts key fields, classifies the urgency level, and routes the case to
the appropriate department — returning a plain-English routing summary that a
non-technical user can act on immediately.

Pipeline agents: `extraction_agent` → `classification_agent` → `routing_agent`

**What was built beyond the original spec:**
- AWS S3 for raw file storage; Supabase PostgreSQL for structured results
- SHA-256 dedup — files already in the database skip the pipeline
- Two-tab Streamlit UI: Tab 1 = intake routing, Tab 2 = NL2SQL chatbot
- RBAC with 3 roles (demo-admin, demo-doctor, demo-reception) via `RoleConfig`
- NL2SQL agent with 4-layer guardrail (schema → AST → column strip → keyword scan)
- Admin-only document deletion with DB-first ordering and orphaned S3 key retry queue
- `normalize_names.py` — one-time migration script for name format normalization

Stretch goal (do not build now, note in `docs/`): integrate with a real EHR system lookup
to validate patient records against existing entries before routing.

- Demonstrates: document processing, structured output, workflow automation, healthcare context,
  RBAC, NL2SQL with guardrails, S3 + Supabase storage, admin tooling
- Direct analog to M3's businesses: Wake Research, PracticeMatch, The Medicus Firm

### 2. Toy Safety Checker (general, visual, fun — build second)
User uploads a photo of a toy. Pipeline identifies it, checks CPSC recall database (free
public API), and generates a plain-English safety report for a parent.
- `vision_agent` → `lookup_agent` → `report_agent`
- Stepwise: build `apps/image-preprocessor/` first (pure algorithm), then the full project
- Demonstrates: vision, tool use, external API integration, consumer-facing output

### 3. Clinical Trial Eligibility Screener (healthcare/agentic — build third)
Location: `projects/clinical-trial-eligibility-screener/`

A coordinator pastes a trial's inclusion/exclusion criteria into one text box and a
patient summary into another. The pipeline evaluates the patient against every criterion
and returns a plain-English eligibility verdict with explicit reasoning per criterion.

Pipeline agents:
- `criteria_agent` — extracts and structures individual criteria from the raw trial text
- `evaluation_agent` — evaluates the patient summary against each criterion one at a time
- `verdict_agent` — synthesizes evaluations into a final verdict with plain-English reasoning

Streamlit UI:
- Two text area inputs: "Trial Criteria" and "Patient Summary"
- A "Run Eligibility Check" button
- A verdict card showing Eligible / Likely Ineligible / Needs Review with a color indicator
- An expander showing the per-criterion evaluation breakdown
- A sidebar with project description, tech stack, and GitHub link

Stretch goal (do not build now, note in `docs/`): Option C — user selects from a dropdown
of preloaded ClinicalTrials.gov trials via their free public API instead of pasting
criteria manually.

- Demonstrates: multi-step reasoning, structured output, healthcare context, agentic decomposition
- Direct analog to M3's clinical research business (Wake Research, trial coordination workflows)

### Stepwise build pattern (for complex projects)
When a project has a non-AI algorithmic component, build it in `apps/` first and
validate it standalone before wrapping it in the full project pipeline.

Example for stereogram generator:
1. `apps/stereogram-renderer/` — depth map → stereogram (pure `Pillow`/`numpy`, no AI)
2. `projects/stereogram-pipeline/` — text → image gen → depth → calls apps/ renderer

This makes debugging easier and each stage is independently demonstrable.

---

## Terminal Runability Standard

Every `.py` file in this repo must be independently runnable from the terminal via
`if __name__ == "__main__"`. The depth of the test is proportional to what the file
does — not everything needs a full end-to-end run, but everything must be validatable
without launching the Streamlit UI or triggering the full build pipeline.

### apps/ modules — full functionality

Pure algorithms have no LLM calls, no API cost, and no side effects beyond writing
a file. Run the complete logic from `__main__`.

```python
# apps/stereogram-renderer/main.py
if __name__ == "__main__":
    # Accept real file paths via sys.argv
    # Run the full conversion
    # Write the output file
    # Print the result dict
```

`python main.py depth.png out.png texture.avif` must produce a real output file.
This is how you validate a building block works before agents ever call it.

### projects/agents/ — lightweight smoke test

A full agent run costs API tokens. The `__main__` block uses a small hardcoded input
to confirm the chain initialises, the Langfuse handler is wired, the function returns
the expected state dict shape, and nothing crashes.

```python
# projects/[name]/agents/extraction_agent.py
if __name__ == "__main__":
    test_state = {
        "input": {"text": "Patient: John Doe, DOB 1985-03-12. Chief complaint: chest pain."},
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }
    result = run(test_state)
    print("pipeline_step:", result["pipeline_step"])
    print("output key present:", "extraction_output" in result)
    print("errors:", result["errors"])
    # Print the actual output so you can eyeball it
    import json
    print(json.dumps(result.get("extraction_output"), indent=2, default=str))
```

### projects/pipeline.py — --dry-run flag

The pipeline wires agents together and calls guardrails. A `--dry-run` flag bypasses
the LLM agents and just confirms state dict wiring, guardrails calls, and the
`pipeline_step` counter are all correct.

```python
# projects/[name]/pipeline.py
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; validate wiring only")
    args = parser.parse_args()

    test_input = {"text": "Sample input for local pipeline test."}

    if args.dry_run:
        # Run guardrails only — confirm validate_input and sanitize_output don't crash
        from guardrails import validate_input, sanitize_output
        validated = validate_input(test_input)
        state = build_initial_state(validated)
        state["output"] = "dry-run placeholder"
        state = sanitize_output(state)
        print("Dry run passed. State keys:", list(state.keys()))
    else:
        result = run(test_input)
        print(json.dumps(result, indent=2, default=str))
```

`python pipeline.py --dry-run` must pass with no LLM calls and no API keys required.

### ai-agents/ build agents — basic self-test

Build agents use LLMs, so `__main__` runs with a tiny hardcoded context dict and
confirms: the chain initialises, the Langfuse handler is constructed (not invoked),
and the function returns a dict with the expected keys. Use `--dry-run` to skip the
actual LLM call.

```python
# ai-agents/evaluator/agent.py
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM call; validate context shape only")
    args = parser.parse_args()

    test_context = {
        "project_name": "smoke-test",
        "goal": "Test that the evaluator initialises correctly.",
        "agents": ["agent_one"],
        "generated_files": {} if args.dry_run else {"pipeline.py": "# placeholder"},
    }

    if args.dry_run:
        print("Dry run: context shape valid.")
        print(json.dumps(test_context, indent=2))
    else:
        result = run(test_context)
        print("status:", result["status"])
        print("score:", result["score"])
        print("feedback:", result["feedback"])
```

`python agent.py --dry-run` must pass with no API keys. `python agent.py` makes a
real LLM call and should return a sensible verdict.

### Summary table

| File type | `__main__` depth | API keys needed? |
|---|---|---|
| `apps/*/main.py` | Full — real inputs, real output file | No |
| `projects/*/agents/*.py` | Smoke test — hardcoded minimal input | Yes (real LLM call) |
| `projects/*/pipeline.py` | `--dry-run` flag skips LLM; full run available | `--dry-run`: No |
| `ai-agents/*.py` / `ai-agents/*/agent.py` | `--dry-run` validates shape; full run available | `--dry-run`: No |

---

## Hard Rules (Never Violate)

- Never hardcode API keys or credentials anywhere
- Never skip `guardrails.py` — every project gets one, always with PHI stub
- Never use raw `openai` or `anthropic` SDK — use LangChain wrappers
- Never leave a pipeline without a `max_pipeline_steps` exit condition in the state dict
- Never skip `app.py` — every project is demoable via Streamlit
- Never write docs that Cody cannot read and explain in an interview
- Never use paid external APIs without confirming with Cody first
- Never deploy outside AWS free tier without explicit confirmation
- Always include `.env.example` — never commit `.env`
- Every `.py` file must have an `if __name__ == "__main__"` block — see Terminal Runability Standard
