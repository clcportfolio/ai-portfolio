"""
agent.py — security_specialist agent.

Reviews pipeline.py, guardrails.py, and agents/ for:
  - Exposed secrets or hardcoded credentials
  - Prompt injection vectors
  - Insecure data handling
  - Missing authentication
  - HIPAA-adjacent risks (always flagged, even for non-clinical projects)

Writes docs/security_report.md into the project directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler

from .prompts import SECURITY_REVIEW_PROMPT

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0,
    )


def _get_handler(project_name: str) -> CallbackHandler:
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_name=f"{project_name}/security_specialist",
    )


def _load_skills() -> str:
    skills_path = Path(__file__).parent / "SKILLS.md"
    return skills_path.read_text() if skills_path.exists() else ""


def _build_files_block(generated_files: dict) -> str:
    """Format relevant files for the LLM review. Trims each to 2000 chars."""
    files_to_review = {
        k: v for k, v in generated_files.items()
        if k in ("pipeline.py", "guardrails.py") or k.startswith("agents/")
    }
    return "\n\n".join(
        f"### {path}\n```python\n{content[:2000]}\n```"
        for path, content in files_to_review.items()
    )


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Review project code for security issues and write docs/security_report.md.

    Input keys:
        project_name (str), project_path (str), goal (str),
        healthcare_notes (str), generated_files (dict)

    Returns dict with:
        security_report (str), errors (list[str])
    """
    errors: list[str] = []
    project_name = context.get("project_name", "unknown")
    project_path = Path(context.get("project_path", "."))
    generated_files = context.get("generated_files", {})

    files_block = _build_files_block(generated_files)
    llm = _get_llm()
    handler = _get_handler(project_name)
    chain = SECURITY_REVIEW_PROMPT | llm | StrOutputParser()

    try:
        report = chain.invoke(
            {
                "skills_md": _load_skills(),
                "project_name": project_name,
                "goal": context.get("goal", ""),
                "healthcare_notes": context.get("healthcare_notes", "Not a clinical project."),
                "files_block": files_block or "(no files provided)",
            },
            config={"callbacks": [handler]},
        )
    except Exception as e:
        errors.append(f"security_specialist LLM call failed: {e}")
        report = f"# Security Report — {project_name}\n\nReview failed: {e}\n"

    docs_path = project_path / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)
    (docs_path / "security_report.md").write_text(report)

    return {
        "security_report": report,
        "errors": errors,
    }


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="security_specialist — reviews code and writes security_report.md")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate context shape only; no LLM call, no files written")
    parser.add_argument("--project-name", default="smoke-test",
                        help="Project name to use in the test (default: smoke-test)")
    args = parser.parse_args()

    _sample_pipeline = """\
from guardrails import validate_input, sanitize_output
import os

def run(user_input: dict) -> dict:
    validated = validate_input(user_input)
    state = {"input": validated, "pipeline_step": 0, "max_pipeline_steps": 10, "errors": []}
    state = sanitize_output(state)
    return state
"""
    _sample_guardrails = """\
import re, os, logging
logger = logging.getLogger(__name__)

def validate_input(data):
    text = data.get("text", "")
    if len(text) > 4000:
        raise ValueError("Input too long")
    return data

def sanitize_output(state):
    return state

def rate_limit_check(user_id):
    return True  # replace with Redis-backed counter in production
"""

    test_context = {
        "project_name": args.project_name,
        "project_path": f"/tmp/{args.project_name}",
        "goal": "Extract and classify clinical intake forms.",
        "healthcare_notes": "Processes patient intake data — PHI risk present.",
        "generated_files": {
            "pipeline.py": _sample_pipeline,
            "guardrails.py": _sample_guardrails,
        },
    }

    if args.dry_run:
        print("[security_specialist] Dry run — context shape valid.")
        print(f"  project_name : {test_context['project_name']}")
        print(f"  project_path : {test_context['project_path']}")
        print(f"  files to review: {list(test_context['generated_files'].keys())}")
    else:
        print(f"[security_specialist] Running against project: {args.project_name}")
        print(f"  Report will be written to: {test_context['project_path']}/docs/security_report.md")
        result = run(test_context)
        print("\nErrors:", result.get("errors", []))
        print("\n--- security_report.md preview (first 30 lines) ---")
        lines = result.get("security_report", "").splitlines()[:30]
        print("\n".join(lines))
