"""
agent.py — guardrail_engineer agent.

Writes guardrails.py for a project. Output always includes:
  - validate_input(): type checks, size limits, prompt injection scan
  - sanitize_output(): script tag removal, PHI/PII redaction stub
  - rate_limit_check(): stub returning True

The PHI redaction stub is present in EVERY project even when the app has
nothing to do with healthcare — it signals production and compliance instincts.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler

from .prompts import GUARDRAILS_PROMPT

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0,
    )


def _get_handler(project_name: str) -> CallbackHandler:
    # Langfuse v4: credentials read from env vars
    return CallbackHandler()


def _load_skills() -> str:
    skills_path = Path(__file__).parent / "SKILLS.md"
    return skills_path.read_text() if skills_path.exists() else ""


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _fallback_guardrails(project_name: str) -> str:
    """Minimal complete guardrails.py returned if the LLM call fails."""
    return f'''"""
guardrails.py — Input/output safety middleware for {project_name}.
Auto-generated fallback (LLM call failed during build).
"""

import re
import logging

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4000
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"disregard (your |all )?instructions",
    r"act as (a |an )?",
    r"jailbreak",
]

_PHI_PATTERNS = [
    r"\\b\\d{{3}}-\\d{{2}}-\\d{{4}}\\b",   # SSN
    r"\\b\\d{{10}}\\b",                      # NPI
    r"\\bMRN[:\\s]?\\d+\\b",                # Medical record number
]


def validate_input(data: dict) -> dict:
    text = data.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Input 'text' must be a string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Input exceeds {{MAX_TEXT_LENGTH}} character limit.")
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Potential prompt injection detected.")
    return data


def sanitize_output(state: dict) -> dict:
    output = state.get("output", "")
    if isinstance(output, str):
        output = re.sub(r"<script.*?>.*?</script>", "[REMOVED]", output, flags=re.DOTALL | re.IGNORECASE)
        for pattern in _PHI_PATTERNS:
            if re.search(pattern, output):
                logger.warning(
                    "PHI pattern detected in output. "
                    "Replace with AWS Comprehend Medical in production."  # NOTE: stub
                )
                break
        state["output"] = output
    return state


def rate_limit_check(user_id: str) -> bool:
    # replace with Redis-backed counter in production
    return True
'''


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Generate and write guardrails.py for the project.

    Input keys:
        project_name (str), project_path (str), goal (str),
        agents (list[str]), generated_files (dict),
        healthcare_notes (str)

    Returns dict with:
        generated_files (dict: adds "guardrails.py"),
        project_path (str),
        errors (list[str])
    """
    errors: list[str] = []
    project_name = context.get("project_name", "unknown")
    project_path = Path(context.get("project_path", "."))
    existing_files = context.get("generated_files", {})
    pipeline_content = existing_files.get("pipeline.py", "(pipeline.py not available)")

    llm = _get_llm()
    handler = _get_handler(project_name)
    chain = GUARDRAILS_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke(
            {
                "skills_md": _load_skills(),
                "project_name": project_name,
                "goal": context.get("goal", ""),
                "healthcare_notes": context.get(
                    "healthcare_notes",
                    "Not a clinical project — include PHI stub as standard practice.",
                ),
                "pipeline_content": pipeline_content[:3000],
            },
            config={"callbacks": [handler]},
        )
        content = _strip_code_fences(raw)
    except Exception as e:
        errors.append(f"guardrail_engineer LLM call failed: {e}")
        content = _fallback_guardrails(project_name)

    out_path = project_path / "guardrails.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)

    return {
        "generated_files": {**existing_files, "guardrails.py": content},
        "project_path": str(project_path),
        "errors": errors,
    }


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="guardrail_engineer — writes guardrails.py for a project")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate context shape only; no LLM call, no files written")
    parser.add_argument("--project-name", default="smoke-test",
                        help="Project name to use in the test (default: smoke-test)")
    args = parser.parse_args()

    test_context = {
        "project_name": args.project_name,
        "project_path": f"/tmp/{args.project_name}",
        "goal": "Extract fields from a clinical intake form and route to the correct department.",
        "agents": ["extraction_agent", "classification_agent", "routing_agent"],
        "healthcare_notes": "Handles patient intake data — PHI stub required.",
        "generated_files": {
            "pipeline.py": (
                "from guardrails import validate_input, sanitize_output\n"
                "def run(user_input):\n"
                "    validated = validate_input(user_input)\n"
                "    state = {'input': validated, 'pipeline_step': 0, 'max_pipeline_steps': 10, 'errors': []}\n"
                "    state = sanitize_output(state)\n"
                "    return state\n"
            )
        },
    }

    if args.dry_run:
        print("[guardrail_engineer] Dry run — context shape valid.")
        print(json.dumps({k: v for k, v in test_context.items() if k != "generated_files"}, indent=2))
        print("generated_files keys:", list(test_context["generated_files"].keys()))
    else:
        print(f"[guardrail_engineer] Running against project: {args.project_name}")
        print(f"  Output will be written to: {test_context['project_path']}/guardrails.py")
        result = run(test_context)
        print("\nResult:")
        print("  generated_files keys:", list(result.get("generated_files", {}).keys()))
        print("  errors:", result.get("errors", []))
        if "guardrails.py" in result.get("generated_files", {}):
            print("\n--- guardrails.py preview (first 20 lines) ---")
            lines = result["generated_files"]["guardrails.py"].splitlines()[:20]
            print("\n".join(lines))
