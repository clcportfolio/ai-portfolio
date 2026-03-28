# SKILLS.md — tech_writer agent
## Domain: Technical Documentation for AI Portfolio Projects

This file defines the tone guide, audience profiles, document templates, and
writing conventions for the tech_writer agent. The goal is documentation that
makes Cody dangerous in a technical interview and convinces a non-technical
stakeholder that the project is real and valuable.

---

## 1. Audience Profiles

### overview_nontechnical.md — The Clinic Manager (or equivalent)
- Does not know what an LLM, API, or pipeline is
- Cares about: time saved, errors avoided, what they click, what they see
- Tone: warm, confident, zero jargon, short sentences
- Forbidden words: model, token, inference, pipeline, agent, API, vector, embedding, latency
- Replace with: "the AI reads...", "the system checks...", "a report appears..."

### overview_technical.md — The Senior AI Engineer Interviewer
- Knows LangChain, knows Claude, has read the JD
- Cares about: architecture decisions, LLM choice rationale, guardrails design, observability
- Tone: precise, direct, uses exact class/method names
- Must include: ASCII pipeline diagram, model selection table, tradeoffs section

### build_walkthrough.md — Cody (the developer)
- Physics background, game dev experience (Unity/C#)
- Needs to explain every file in a 30-minute interview without looking at code
- Tone: collegial, narrative, explains *why* not just *what*
- State machine analogy: the state dict is the game state; each agent is a system that reads/writes it
- Must include: interview prep Q&A section with 3 likely questions and crisp answers

---

## 2. overview_nontechnical.md Template

```markdown
## What This Tool Does

[2-3 sentences. Lead with the outcome — what the user gets.
Example: "This tool reads a patient intake form and automatically routes it to the right department,
along with a plain-English summary of what the patient needs."]

## Why It Matters

[1-2 sentences. What problem does it solve? What happens without it?
Example: "Without this, a staff member has to read every form and decide where to send it —
a task that takes time and is easy to get wrong when the clinic is busy."]

## What You See When You Run It

[Walk through the UI step by step. No jargon. What do you type? What appears?
Example: "You paste the intake form text into the box and click 'Analyze'.
In a few seconds, a summary appears showing the patient's main concern,
an urgency level (like 'urgent' or 'routine'), and which department should see them first."]

## Who Built This and How

[1 short paragraph. Mention AI, safety, portfolio context.
Keep it factual and grounded.]
```

---

## 3. overview_technical.md Template

```markdown
## Project Goal

## Architecture Overview

```
User input (Streamlit UI)
      │
      ▼
guardrails.py → validate_input()
      │
      ▼
pipeline.py → run(input: dict) -> dict
      ├─► agent_1.run(state)     # description
      ├─► agent_2.run(state)     # description
      └─► agent_3.run(state)     # description
      │
      ▼
guardrails.py → sanitize_output()
      │
      ▼
Streamlit UI → display result
```

## Agent Roles

[One paragraph per agent. Always include:
- What it reads from state
- What LangChain pattern it uses (LCEL, structured output, tool use, vision)
- Which model and why
- What it writes to state]

## LLM Choices & Rationale

| Agent | Model | Temperature | Rationale |
|---|---|---|---|
| extraction_agent | claude-haiku-4-5-20251001 | 0 | Deterministic extraction, cost-sensitive |
| report_agent | claude-sonnet-4-20250514 | 0.5 | Generative narrative, quality matters |

## Guardrails Design

## Langfuse Observability

## Data Flow

## Deployment Path

## Tradeoffs & Known Limitations
```

---

## 4. build_walkthrough.md Template

```markdown
## Why This Project Exists

## Build Order

[What was scaffolded first and why. Example:
"pipeline.py was written first to establish the state dict shape —
this is like defining the game state struct before writing any game systems.
Once the data contract was clear, each agent was written independently."]

## File-by-File Breakdown

### pipeline.py
[What it does, why it exists, key functions, design decisions]

### guardrails.py
[What validate_input checks, what sanitize_output does, PHI stub rationale]

### agents/{agent_name}.py (repeat per agent)
[Role, LangChain pattern used, model choice, state keys read/written]

### app.py
[UI structure, why expanders were used, how errors surface]

## Key Design Decisions

[3-5 decisions with rationale. Use this format:]
**Decision:** Used Haiku for classification steps, Sonnet for report generation.
**Why:** Classification is deterministic and high-frequency — Haiku is 10x cheaper and fast enough.
Report generation benefits from Sonnet's stronger writing quality.

## What the Evaluator Flagged

[Honest account. If nothing was flagged: "Passed on the first evaluation iteration."]

## How to Explain This in an Interview

**Q: Walk me through the architecture.**
A: [2-3 sentence answer Cody can deliver confidently]

**Q: Why did you use LangChain instead of the raw Anthropic SDK?**
A: [Crisp answer referencing LCEL, structured output, and portability]

**Q: How does your guardrail handle prompt injection?**
A: [Specific answer referencing the regex patterns and the validate_input flow]
```

---

## 5. README.md Template

Follows CLAUDE.md standard exactly:

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
[ASCII pipeline diagram or brief numbered agent step list]

## Tech stack
- LangChain + Claude (Anthropic)
- Langfuse observability
- Streamlit demo
- [project-specific tools]
```

---

## 6. ASCII Pipeline Diagram Syntax

Use box-and-arrow style. Keep it under 60 chars wide.

```
User Input
    │
    ▼
validate_input()
    │
    ▼
extraction_agent ──► state["extraction_output"]
    │
    ▼
classification_agent ──► state["classification_output"]
    │
    ▼
routing_agent ──► state["output"]
    │
    ▼
sanitize_output()
    │
    ▼
Streamlit UI
```

---

## 7. Interview Q&A Patterns for Cody

Cody's background makes certain framings more natural:

| Concept | Cody-friendly framing |
|---|---|
| State dict | "It's like the game state object — every system reads from it and writes to it, but no system touches another system's fields" |
| Agent pipeline | "Think of it as a state machine where each agent is a transition — the state dict is the current game state" |
| LCEL pipe syntax | "It's Unix pipes for LLM chains — each stage processes and passes to the next" |
| Guardrails | "They're the collision detection layer — validate_input is the pre-physics check, sanitize_output is the post-physics cleanup" |
| Langfuse | "It's like Unity's Profiler but for LLM calls — every inference shows up with timing, tokens, and the exact prompt" |
| max_pipeline_steps | "It's the game loop's frame cap — prevents runaway execution if something goes wrong mid-pipeline" |

---

## 8. Tone Dos and Don'ts

### Dos
- Start sections with the most important sentence
- Use active voice: "The agent extracts..." not "Extraction is performed by..."
- Be specific: name exact classes, methods, models
- For non-technical docs: use analogies from everyday life

### Don'ts
- Don't pad with "In conclusion..." or "As we can see..."
- Don't explain what LangChain is in the technical doc — the interviewer knows
- Don't write docs Cody can't explain — if it's unclear to you, it'll be unclear to him
- Don't overuse bullet points in non-technical docs — prose reads better for stakeholders
