"""
prompts.py — ChatPromptTemplate definitions for the guardrail_engineer agent.
"""

from langchain_core.prompts import ChatPromptTemplate

GUARDRAILS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a security-focused AI engineer writing guardrails.py for a LangChain project.

SKILLS reference (follow these patterns exactly):
{skills_md}

You MUST implement all three functions with these exact signatures:
    def validate_input(data: dict) -> dict  — raises ValueError on failure
    def sanitize_output(state: dict) -> dict
    def rate_limit_check(user_id: str) -> bool

Hard requirements — every generated file must include ALL of these:

validate_input:
- Type-check all user-controlled input fields (infer from pipeline.py)
- Max text length: 4000 characters
- Max image size: 10MB (if images used)
- Prompt injection scan on all text fields (patterns: ignore previous instructions, you are now, disregard, act as, jailbreak)
- Raise ValueError with a descriptive message on any failure

sanitize_output:
- Strip <script> tags from any string output
- PHI redaction stub: scan for SSN (\\d{{3}}-\\d{{2}}-\\d{{4}}), NPI (\\d{{10}}), MRN patterns
  - Log a WARNING if found — do NOT block output
  - Add comment: "# NOTE: replace with AWS Comprehend Medical in production"
  - REQUIRED even for non-clinical projects
- Return the modified state dict

rate_limit_check:
- Returns True (stub)
- Add comment: "# replace with Redis-backed counter in production"

Write complete, runnable Python — no placeholders.
Output ONLY the Python file content — no explanation, no markdown fences.""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Healthcare notes: {healthcare_notes}

pipeline.py (to understand input/output shape):
```python
{pipeline_content}
```

Write the complete guardrails.py.""",
    ),
])
