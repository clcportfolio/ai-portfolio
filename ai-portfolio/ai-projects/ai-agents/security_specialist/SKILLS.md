# SKILLS.md — security_specialist agent
## Domain: Security Review for LangChain AI Projects

Reference for producing thorough, consistent `security_report.md` files.
Every report must cover all six sections below, even for simple projects.

---

## 1. Required Report Structure

```markdown
## 1. Secrets & Credentials
## 2. Prompt Injection
## 3. Data Handling
## 4. Authentication & Rate Limiting
## 5. HIPAA-Adjacent Risks
## 6. Summary
```

Always start directly with `## 1.` — no preamble, no "Here is the report."

---

## 2. Section-by-Section Review Guide

### Section 1 — Secrets & Credentials
Look for:
- Hardcoded strings matching: `sk-`, `key=`, `token=`, `password=`, `secret=`, `api_key=`
- API keys passed as function arguments (not via `os.getenv`)
- `.env` files committed (check for their absence in `.gitignore` / `.env.example`)

Green flags:
- All secrets via `os.getenv("KEY_NAME")`
- `.env.example` present with placeholder values
- `load_dotenv()` called at module level

### Section 2 — Prompt Injection
Look for:
- f-strings that embed user input directly into a prompt: `f"Answer this: {user_text}"`
- `PromptTemplate` with `{user_input}` passed as a template variable without guardrails
- `validate_input()` not called before the pipeline touches user text
- Missing injection pattern scan in `guardrails.py`

Green flags:
- `validate_input()` called at pipeline entry before any LLM interaction
- Injection regex patterns present in `guardrails.py`
- User input passed as a variable into `ChatPromptTemplate` (not interpolated into system prompt)

### Section 3 — Data Handling
Look for:
- Sensitive data written to disk outside temp dirs
- `print()` or `logging.info()` calls that could log full prompt text (which may contain PII)
- File paths constructed from user input without sanitization (path traversal risk)
- PHI redaction stub absent from `sanitize_output`

Green flags:
- Temp files used for intermediate storage and cleaned up
- PHI stub present in `sanitize_output` with production note
- No logging of full prompt or response content at INFO level

### Section 4 — Authentication & Rate Limiting
Look for:
- No auth on the Streamlit UI (note as a production gap — expected at portfolio stage)
- `rate_limit_check()` absent from `guardrails.py`
- `rate_limit_check()` never called from `pipeline.py` or `app.py`

Standard language for portfolio projects:
> "No authentication is implemented. This is acceptable for a local/demo deployment.
> Production hardening requires: session-based auth, API key validation, or OAuth.
> Rate limiting stub is present in guardrails.py — replace with Redis-backed counter."

### Section 5 — HIPAA-Adjacent Risks
This section is **always present**, even for non-clinical projects.

Template framing for non-clinical projects:
> "This project does not process health data. However, the following gaps would need
> to be addressed before it could safely handle PHI:
> 1. [gap 1 — e.g. no encryption at rest for temp files]
> 2. [gap 2 — e.g. no audit log for data access]
> 3. [gap 3 — e.g. PHI redaction is a stub, not a production scanner]"

For clinical projects, be specific about each gap relative to HIPAA safeguards:
- Administrative: access controls, workforce training
- Physical: workstation security, device controls
- Technical: encryption, audit controls, automatic logoff

### Section 6 — Summary
Always include:
- Overall risk level: **Low** / **Medium** / **High**
- Exactly 3 action items for production hardening (numbered list)

Risk level guidance:
- **Low**: secrets via env, guardrails present, PHI stub present, no hardcoded values
- **Medium**: any missing guardrail, or injection scan absent, or auth completely missing
- **High**: hardcoded secrets, raw user input in prompts, no sanitize_output

---

## 3. Files to Review (Priority Order)

1. `guardrails.py` — most important; covers input validation, PHI stub, rate limiting
2. `pipeline.py` — checks data flow, guardrails wiring, secret loading
3. `agents/*.py` — prompt construction, LLM call patterns, error handling
4. `app.py` — user input handling, error surface to UI

Trim each file to 2000 chars when sending to LLM to avoid token overflow.

---

## 4. Tone and Specificity

**Good finding:**
> `pipeline.py`: `build_initial_state()` does not call `rate_limit_check()`.
> The function exists in guardrails.py but is never invoked. Add:
> `if not rate_limit_check(user_id): raise ValueError("Rate limit exceeded")`

**Bad finding (too vague):**
> "Rate limiting is not implemented."

Always name the file, the function or line pattern, the risk, and the fix.

---

## 5. What NOT to Flag

- The PHI stub being a stub (it's intentional — note it as "stub, not production scanner")
- `rate_limit_check` returning `True` (it's an intentional stub — note it as such)
- Missing tests (out of scope for this report)
- Code style issues (not a security concern)
