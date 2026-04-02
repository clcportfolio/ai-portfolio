# Clinical Intake Router

A multi-agent LLM pipeline that reads a clinical intake form and routes the patient to the right department at the right urgency level — instantly. Built to demonstrate agentic workflow automation, structured LLM output, RBAC, NL2SQL with guardrails, and production observability in a healthcare context.
#### Demo built in ~8 work hours

> **Live demo:** [Streamlit app](https://clcportfolio-clinical-intake-router.streamlit.app/)
> **Live demo logins:** Contact Cody at: [clculver5@gmail.com](clculver5@gmail.com)
> **Portfolio:** [github.com/clcportfolio/ai-portfolio](https://github.com/clcportfolio/ai-portfolio)

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys (see Environment Variables below)
streamlit run app.py
```

**Demo credentials** — contact [Cody](mailto:cody@example.com) for access.

---

## What you'll see

**Tab 1 — Intake Router:** Paste or upload a clinical intake form (`.txt` or `.pdf`). Click **Route This Intake** to receive a color-coded routing card showing urgency level (Routine / Urgent / Emergent), assigned department, recommended next steps, and estimated response time. Expand the agent panels to inspect extracted fields and classification reasoning — each step is visible, not a black box.

**Tab 2 — Query Database:** Ask plain-English questions about the intake submissions database. The system generates and executes SQL, then returns a plain-English answer. Access is role-scoped — reception staff see a restricted schema and the response is scanned for clinical content they shouldn't receive.

---

## How it works

```
User input (Streamlit UI)
    │
    ▼
guardrails.py → validate_input()          # size limits, injection scan
    │
    ▼
pipeline.py → run(input)
    ├─► extraction_agent                   # ~10 structured fields from free text (Pydantic)
    ├─► classification_agent               # urgency level + department
    └─► routing_agent                      # plain-English routing card + next steps
    │
    ├─► storage: S3 (raw file) + Supabase (structured results)
    │
    ▼
guardrails.py → sanitize_output()         # HTML strip, PHI stub, content safety
    │
    ▼
Streamlit UI → routing card + agent expanders
```

**NL2SQL pipeline (Tab 2):**
```
Question
    │
    ▼
L1 — Role-scoped schema (LLM sees only allowed columns)
    │
    ▼
LLM → SQL generation (Claude Sonnet + structured output)
    │
    ▼
L2 — AST validation (sqlglot — deterministic, not LLM-dependent)
    │
    ▼
SQL execution (Supabase PostgreSQL)
    │
    ▼
L3 — Result column strip (restricted keys removed from rows)
    │
    ▼
LLM → answer synthesis (Claude Haiku)
    │
    ▼
L4 — Output keyword scan (clinical term detection)
    │
    ▼
Plain-English answer
```

---

## Features

### Three-agent intake pipeline
`extraction_agent` → `classification_agent` → `routing_agent`

Each agent has one job, one output key in the shared state dict, and is independently runnable from the terminal (`python agents/extraction_agent.py`). Agents communicate through a shared `state` dict — no agent mutates another's output.

### File storage — S3 + Supabase
Raw files are stored in AWS S3. Structured pipeline results (extracted fields, classification, routing) are written to Supabase PostgreSQL. Files are SHA-256 hashed at upload — duplicate files skip the pipeline and return the cached result.

### Role-based access control (RBAC)
`RoleConfig` is a Pydantic model that acts as the single source of truth for every role-dependent decision: which DB columns are visible, which UI panels render, what schema the NL2SQL agent sees, and whether the user can delete documents. No role strings scattered through the code.

### NL2SQL chatbot with 4-layer guardrail
Natural-language questions are converted to SQL and executed against the submissions database. Four independent enforcement layers ensure restricted roles cannot access clinical data even if the LLM is non-compliant:

| Layer | Mechanism | LLM-dependent? |
|---|---|---|
| L1 — Schema restriction | LLM prompt | Yes |
| L2 — AST validation | `sqlglot` column check | **No — deterministic** |
| L3 — Column strip | Post-query key removal | No |
| L4 — Keyword scan | Regex on answer text | No |

### Admin document deletion
`demo-admin` can delete documents from both S3 and the database via the directory UI. DB row is deleted first — if S3 deletion subsequently fails, the broken reference never surfaces to users (an orphaned S3 object is invisible; a DB row pointing to a deleted file is not). Failed S3 deletes are queued for retry in the session.

### Async pipeline
All LLM calls use `await chain.ainvoke()` throughout. S3 uploads and DB writes run via `asyncio.to_thread()` so synchronous I/O (boto3, psycopg2) never blocks the event loop. When multiple users submit intakes concurrently, each request yields the event loop during network I/O — other requests make progress instead of queuing behind a blocking call.

Each agent's LangChain chain (`prompt | llm | parser`) is constructed once at module load as a module-level constant and reused on every request — nothing on the hot path except the `ainvoke()` call itself.

### Observability
Every LLM call is traced in Langfuse with full span hierarchy. `pipeline.run()` is decorated with `@observe` and wrapped in `propagate_attributes` to pin the trace name across all child spans.

### Rate limiting
Redis-backed fixed-window rate limiter (10 requests / 60 seconds, global). A traffic indicator below the NL2SQL prompt shows current load (Low / Medium / High) with a colour-matched glow. Degrades gracefully to stub behaviour if `REDIS_URL` is not set.

---

## Production upgrade notes

These are the gaps between this demo and a production deployment. They are intentional — the architecture is designed so each can be swapped in without touching other components.

**PHI detection (`guardrails.py → sanitize_output`)**
The current `_phi_redaction_stub()` uses basic regex (SSN, DOB, long ID patterns). In production, replace with **AWS Comprehend Medical** or **Azure Text Analytics for Health** — both return typed PHI entities (name, DOB, MRN, address) and are HIPAA-eligible services.

**Clinical keyword scan (`guardrails.py → check_nl2sql_output`, L4)**
The current L4 guardrail uses a regex list of ~15 medications and diagnoses. In production, layer in a **biomedical NER model** (e.g. `scispaCy` with `en_ner_bc5cdr_md`, or AWS Comprehend Medical) to catch brand names, abbreviations (`HTN`, `DM2`, `MI`), and drugs not in the regex list. Regex and NER are complementary — regex handles structural patterns and dosages; NER handles named entities.

**Authentication (`auth.py`)**
Credentials are hardcoded for demo purposes. In production, replace `USERS` with a real auth provider — **Supabase Auth**, **Auth0**, or **Okta**. `RoleConfig` objects would be stored in a DB-backed permissions table and loaded on login.

**Rate limiting (`guardrails.py → rate_limit_check`)**
The current implementation uses a global Redis counter. In production, rate limit per authenticated user, add burst allowances, and back the counter with a managed Redis service (**Upstash**, **Redis Cloud**, or **AWS ElastiCache**).

**EHR integration (stretch goal)**
Before routing, validate the patient record against an existing EHR system (e.g. **Epic FHIR API**) to confirm the patient is registered and flag mismatches. Noted in `docs/overview_technical.md`.

---

## Environment variables

```bash
# Required
ANTHROPIC_API_KEY=          # Anthropic — LLM calls
LANGFUSE_PUBLIC_KEY=        # Langfuse — observability
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
AWS_ACCESS_KEY_ID=          # AWS — S3 file storage
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET_NAME=
SUPABASE_DB_URI=            # Supabase — PostgreSQL (use Session pooler URI)

# Optional
REDIS_URL=                  # Redis — rate limiting + traffic indicator
                            # Without this, rate limiting is disabled and the
                            # traffic light is hidden. Use Upstash for a free
                            # managed instance that works on Streamlit Cloud.
```

---

## Project structure

```
clinical-intake-router/
├── app.py                  # Streamlit UI (two tabs: intake router + NL2SQL)
├── pipeline.py             # Agent orchestration + S3/DB storage
├── guardrails.py           # Input validation, output sanitization, rate limiting
├── auth.py                 # RoleConfig + RBAC definitions
├── normalize_names.py      # One-time DB migration: normalize patient name format
├── agents/
│   ├── extraction_agent.py
│   ├── classification_agent.py
│   ├── routing_agent.py
│   └── nl2sql_agent.py
├── storage/
│   ├── s3_client.py        # Upload, download, delete, list, presigned URLs
│   └── db_client.py        # Supabase PostgreSQL: insert, query, delete, dedup
└── docs/
    ├── overview_nontechnical.md
    ├── overview_technical.md
    ├── build_walkthrough.md
    └── interview_feature_guide.docx
```

---

## Tech stack

| Category | Technology |
|---|---|
| LLM | Claude Sonnet 4 · Claude Haiku 4 (Anthropic) |
| Framework | LangChain — LCEL, structured output, callbacks |
| Observability | Langfuse v4 — full trace hierarchy |
| File storage | AWS S3 (boto3) |
| Database | Supabase PostgreSQL (psycopg2, Session pooler) |
| SQL validation | sqlglot — AST-level column enforcement |
| Auth / RBAC | Custom `RoleConfig` Pydantic model |
| Rate limiting | Redis fixed-window counter |
| Concurrency | `asyncio` — `ainvoke()` on all LLM calls, `to_thread()` for S3/DB I/O |
| UI | Streamlit |
| Structured output | Pydantic v2 |
| Deduplication | SHA-256 file hashing |

---

*Built with [Claude Code](https://claude.ai/claude-code)*
