"""
prompts.py — All ChatPromptTemplate definitions for the tech_writer agent.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Non-technical overview ────────────────────────────────────────────────────

NONTECHNICAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are writing overview_nontechnical.md for a non-technical reader.

The reader is a clinic operations manager, parent, or small business owner.
They do not know what an LLM, API, pipeline, or agent is.

TONE RULES:
- Warm, clear, zero jargon — no "model", "token", "inference", "pipeline", "API", "vector"
- Replace tech terms: "the AI reads...", "the system checks...", "a report appears..."
- Short sentences. Active voice.
- No bullet points in the first two sections — use flowing prose.

REQUIRED STRUCTURE (use these exact headings):
## What This Tool Does
## Why It Matters
## What You See When You Run It
## Who Built This and How

Output only the Markdown document — no preamble.""",
    ),
    (
        "human",
        "Project: {project_name}\nGoal: {goal}\nStreamlit UI: {streamlit_ui}\nAgents:\n{agents}",
    ),
])

# ── Technical overview ────────────────────────────────────────────────────────

TECHNICAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are writing overview_technical.md for a senior AI engineer interviewer.

The reader knows LangChain, knows Claude, and will ask hard architecture questions.
Be precise — use exact class names, method names, and model IDs.

REQUIRED STRUCTURE (use these exact headings):
## Project Goal
## Architecture Overview
(include an ASCII pipeline diagram: input → guardrails → agents → guardrails → output)
## Agent Roles
(one paragraph per agent: what it reads from state, which LangChain pattern, which model and why, what it writes)
## LLM Choices & Rationale
(Markdown table: Agent | Model | Temperature | Rationale)
## Guardrails Design
(validate_input checks, sanitize_output behavior, PHI stub rationale)
## Langfuse Observability
(what is traced, trace naming, what you see in the dashboard)
## Data Flow
## Deployment Path
## Tradeoffs & Known Limitations

Output only the Markdown document — no preamble.""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Tech stack: {tech_stack}
Data flow: {data_flow}
Evaluator score: {score}/10
Agents:
{agents}
Pipeline excerpt:
{pipeline_excerpt}""",
    ),
])

# ── Build walkthrough (for Cody) ──────────────────────────────────────────────

WALKTHROUGH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are writing build_walkthrough.md for Cody Culver — the developer who built this project.

Cody has a physics background and Unity/C# game dev experience.
He needs to read this and explain every decision in a 30-minute interview without looking at code.

USEFUL ANALOGIES FOR CODY:
- State dict → game state object (each agent is a system that reads/writes it)
- Agent pipeline → state machine (each agent is a transition)
- LCEL pipe syntax → Unix pipes for LLM chains
- max_pipeline_steps → game loop frame cap
- Guardrails → collision detection layer
- Langfuse → Unity Profiler for LLM calls

REQUIRED STRUCTURE (use these exact headings):
## Why This Project Exists
## Build Order
(what was built first and why — establish the narrative arc)
## File-by-File Breakdown
(for each file: what it does, why it exists, key functions, design decisions)
## Key Design Decisions
(3-5 decisions in format: **Decision:** ... **Why:** ...)
## What the Evaluator Flagged
(honest account — if nothing: "Passed on the first evaluation iteration")
## How to Explain This in an Interview
(3 likely Q&A pairs — crisp answers Cody can deliver confidently)

Output only the Markdown document — no preamble.""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Agents (in order):
{agents}
Tech stack: {tech_stack}
Evaluator feedback addressed: {feedback}
Evaluator strengths: {strengths}
Pipeline excerpt:
{pipeline_excerpt}""",
    ),
])

# ── README ────────────────────────────────────────────────────────────────────

README_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are writing a README.md for an AI portfolio project on GitHub.

It must follow this EXACT structure (copy the headings verbatim):

# [Project Name]

[One sentence: what it does. One sentence: why it matters.]

## Run it
```
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

## What you'll see
[2-3 sentences describing the UI and expected output]

## How it works
[ASCII pipeline diagram OR brief numbered agent step list — pick whichever is cleaner]

## Tech stack
- LangChain + Claude (Anthropic)
- Langfuse observability
- Streamlit demo
- [any project-specific tools]

Output ONLY the Markdown content — no explanation, no preamble.""",
    ),
    (
        "human",
        "Project: {project_name}\nGoal: {goal}\nAgents:\n{agents}\nTech stack: {tech_stack}\nStreamlit UI: {streamlit_ui}",
    ),
])
