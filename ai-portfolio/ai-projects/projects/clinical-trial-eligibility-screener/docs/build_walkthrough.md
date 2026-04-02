# Clinical Trial Eligibility Screener Build Walkthrough

## Why This Project Exists

Clinical trial coordinators spend hours manually reviewing patient records against complex eligibility criteria — a process that's both time-consuming and error-prone. This system automates that screening by breaking down trial criteria into structured components, evaluating each criterion against patient data, and synthesizing a clear verdict with reasoning. It's like having a specialized AI assistant that can read medical text and make systematic eligibility determinations, freeing coordinators to focus on patient care rather than paperwork.

## Build Order

**1. Core Pipeline Architecture** — Started with `pipeline.py` as the orchestrator, establishing the three-agent flow and shared state pattern. This is the "game loop" that coordinates everything.

**2. Pydantic Models** — Defined structured outputs for each agent to ensure type safety and clear data contracts between pipeline stages.

**3. Agent Implementation** — Built agents in dependency order: `criteria_agent` first (extracts structure), then `evaluation_agent` (uses that structure), finally `verdict_agent` (synthesizes evaluations).

**4. Streamlit UI** — Created `app.py` to provide a clean interface for coordinators to input trial criteria and patient summaries.

**5. Observability Layer** — Integrated Langfuse for LLM call tracking and performance monitoring.

**6. Security Hardening** — Added `guardrails.py` for input validation and PHI detection after evaluator feedback.

## File-by-File Breakdown

### `pipeline.py`
The main orchestrator that manages the three-agent workflow. Key functions:
- `build_initial_state()` — Initializes the shared state dict with input data and tracking variables
- `run()` — Executes the sequential agent pipeline with validation bookends

**Design Decision:** Used a shared state dict pattern instead of passing return values between agents. This allows each agent to read previous outputs and maintain pipeline metadata (step counts, errors) in one place.

### `guardrails.py`
Security and validation layer that sanitizes inputs and outputs. Key functions:
- `validate_input()` — Checks for prompt injection patterns, validates field types, enforces length limits
- `sanitize_output()` — Removes potential PHI patterns and ensures safe output format

**Design Decision:** Implemented as a separate module rather than inline validation to create a clear security boundary and make it easy to audit/update security rules.

### `agents/criteria_agent.py`
Extracts and structures individual inclusion/exclusion criteria from raw trial text. Uses LangChain's `with_structured_output()` to enforce Pydantic schema compliance. The agent parses free-form medical text into a standardized format that downstream agents can process systematically.

**Key Function:** `run(state)` — Takes raw trial criteria text and returns structured `CriteriaOutput` with separate inclusion/exclusion lists.

### `agents/evaluation_agent.py`
Evaluates patient summary against each structured criterion individually. For each criterion, it provides a verdict (ELIGIBLE/INELIGIBLE/INSUFFICIENT_INFO) with detailed medical reasoning.

**Key Function:** `run(state)` — Iterates through criteria from `criteria_agent`, evaluates each against patient data, returns `EvaluationOutput` with per-criterion assessments.

### `agents/verdict_agent.py`
Synthesizes individual criterion evaluations into a final eligibility determination. Provides plain-English explanation suitable for clinical coordinators who need to understand the reasoning.

**Key Function:** `run(state)` — Takes all individual evaluations and produces final `VerdictOutput` with overall eligibility status and coordinator-friendly explanation.

### `app.py`
Streamlit interface that provides clean UI for trial coordinators. Features expandable sections for each agent's output, allowing users to drill down into the reasoning at each pipeline stage.

**Key Components:** Text areas for input, expandable sections for agent outputs, error handling with user-friendly messages.

## Key Design Decisions

**Decision:** Three-agent sequential pipeline instead of single monolithic LLM call  
**Why:** Breaking the task into specialized agents improves reliability and debuggability. Each agent has a focused responsibility, making it easier to identify and fix issues. Also allows for different prompting strategies optimized for each subtask.

**Decision:** Shared state dict pattern for inter-agent communication  
**Why:** Similar to Unity's game state object — all agents can read/write to shared context while maintaining their own outputs. This enables pipeline metadata tracking (step counts, errors) and allows agents to reference previous outputs without complex parameter passing.

**Decision:** Pydantic models for all agent outputs with structured LLM calls  
**Why:** Enforces type safety and ensures consistent data contracts between pipeline stages. Using LangChain's `with_structured_output()` prevents hallucinated fields and makes the system more reliable for clinical use cases.

**Decision:** Langfuse integration for observability  
**Why:** Clinical applications need audit trails and performance monitoring. Langfuse provides LLM call tracking, token usage metrics, and latency monitoring — essential for production medical software.

**Decision:** Separate guardrails module for security validation  
**Why:** Medical data requires strict security controls. Isolating validation logic in a dedicated module makes it easier to audit, update security rules, and ensure consistent application across all inputs/outputs.

## What the Evaluator Flagged

The evaluator caught several integration issues:
- Missing `guardrails.py` import in pipeline — needed to create the security validation module
- Missing Langfuse callback handlers in all three agents — LLM calls weren't being tracked
- Missing expandable UI sections in Streamlit app — outputs were displayed directly without organization
- Missing `state['output']` assignment in pipeline — final output wasn't being set for downstream consumption

All issues were addressed by adding the missing imports, callback configurations, UI improvements, and state assignments.

## How to Explain This in an Interview

**Q: Why use three separate agents instead of one large prompt?**  
A: It's like having specialized systems in a game engine — each agent has a focused responsibility and can be optimized independently. The criteria agent is like a parser that structures raw text, evaluation agent is like a rules engine that applies logic, and verdict agent is like a UI system that presents results clearly. This modularity makes debugging easier and improves reliability.

**Q: How does the shared state pattern work?**  
A: Think of it like Unity's game state object — there's one central data structure that all systems can read from and write to. Each agent increments the pipeline step, checks against max steps (like a frame cap), and adds its output to the shared state. This allows agents to reference previous outputs while maintaining pipeline metadata in one place.

**Q: What makes this suitable for clinical use?**  
A: Three key factors: structured outputs using Pydantic ensure data consistency, Langfuse provides audit trails for compliance, and the guardrails module validates inputs and sanitizes outputs to protect PHI. It's like having collision detection and logging systems — essential safety layers for medical applications.