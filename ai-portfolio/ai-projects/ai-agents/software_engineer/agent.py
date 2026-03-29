"""
agent.py — software_engineer agent.

Receives a build_plan (from orchestrator) plus any evaluator_feedback from
prior iterations. Writes pipeline.py, agents/, app.py, and requirements.txt
into the target project directory.

Does NOT write guardrails.py or security logic — those are separate agents.

Usage:
    from software_engineer import run   # via __init__.py
    result = run({
        "project_name": "clinical-intake-router",
        "project_path": "/path/to/projects/clinical-intake-router",
        "goal": "...",
        "agents": ["extraction_agent", ...],
        "agent_descriptions": {"extraction_agent": "Extracts fields from intake form"},
        "tech_stack": [...],
        "data_flow": "...",
        "streamlit_ui": "...",
        "skills_md": "...",
        "evaluator_feedback": [],
        "generated_files": {},
    })
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler

from .prompts import (
    AGENT_FILE_PROMPT,
    APP_PROMPT,
    PIPELINE_PROMPT,
    REQUIREMENTS_PROMPT,
)

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        temperature=0.2,
    )


def _get_handler(project_name: str, step: str) -> CallbackHandler:
    # Langfuse v4: credentials read from env vars
    return CallbackHandler()


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _strip_code_fences(text: str) -> str:
    """Remove markdown ```python ... ``` fences if the LLM wrapped its output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _feedback_block(feedback: list[str]) -> str:
    if not feedback:
        return "No prior feedback — this is the first iteration."
    return "Evaluator feedback to address:\n" + "\n".join(f"- {f}" for f in feedback)


def _agents_list_str(context: dict) -> str:
    return "\n".join(
        f"- {name}: {context['agent_descriptions'].get(name, '')}"
        for name in context["agents"]
    )


# ── File generators ────────────────────────────────────────────────────────────

def _generate_agent_file(
    agent_name: str,
    description: str,
    context: dict,
    skills_with_feedback: str,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> str:
    chain = AGENT_FILE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "skills_md": skills_with_feedback,
            "project_name": context["project_name"],
            "goal": context["goal"],
            "data_flow": context["data_flow"],
            "agent_name": agent_name,
            "agent_description": description,
            "all_agents": ", ".join(context["agents"]),
            "feedback_block": _feedback_block(context.get("evaluator_feedback", [])),
        },
        config={"callbacks": [handler]},
    )
    return _strip_code_fences(raw)


def _generate_pipeline(
    context: dict,
    skills_with_feedback: str,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> str:
    chain = PIPELINE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "skills_md": skills_with_feedback,
            "project_name": context["project_name"],
            "goal": context["goal"],
            "data_flow": context["data_flow"],
            "agents_list": _agents_list_str(context),
            "feedback_block": _feedback_block(context.get("evaluator_feedback", [])),
        },
        config={"callbacks": [handler]},
    )
    return _strip_code_fences(raw)


def _generate_app(
    context: dict,
    skills_with_feedback: str,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> str:
    chain = APP_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "skills_md": skills_with_feedback,
            "project_name": context["project_name"],
            "goal": context["goal"],
            "streamlit_ui": context["streamlit_ui"],
            "agents": ", ".join(context["agents"]),
            "feedback_block": _feedback_block(context.get("evaluator_feedback", [])),
        },
        config={"callbacks": [handler]},
    )
    return _strip_code_fences(raw)


def _generate_requirements(
    context: dict,
    llm: ChatAnthropic,
    handler: CallbackHandler,
) -> str:
    chain = REQUIREMENTS_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "tech_stack": ", ".join(context.get("tech_stack", [])),
            "goal": context["goal"],
        },
        config={"callbacks": [handler]},
    ).strip()


def _env_example() -> str:
    return textwrap.dedent("""\
        # ── Anthropic ─────────────────────────────────────────────────────────
        ANTHROPIC_API_KEY=

        # ── Langfuse ──────────────────────────────────────────────────────────
        LANGFUSE_PUBLIC_KEY=
        LANGFUSE_SECRET_KEY=
        LANGFUSE_HOST=https://cloud.langfuse.com

        # ── HuggingFace (only if using HF Inference API) ──────────────────────
        HUGGINGFACE_API_KEY=
    """)


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Generate all project code files from the build plan.

    Input keys (from orchestrator):
        project_name, project_path, goal, agents, agent_descriptions,
        tech_stack, data_flow, streamlit_ui, skills_md,
        evaluator_feedback (list[str]), generated_files (dict)

    Returns dict with:
        generated_files (dict: relative_path -> content),
        project_path (str),
        errors (list[str])
    """
    errors: list[str] = []
    project_name = context["project_name"]
    project_path = Path(context["project_path"])

    # Load this agent's own SKILLS.md and merge with top-level SKILLS.md
    _own_skills = Path(__file__).parent / "SKILLS.md"
    own_skills = _own_skills.read_text() if _own_skills.exists() else ""
    top_skills = context.get("skills_md", "")
    skills_with_feedback = own_skills + "\n\n---\n\n" + top_skills

    llm = _get_llm()
    generated: dict[str, str] = {}

    # ── agents/__init__.py ─────────────────────────────────────────────────────
    agents_init = "# Auto-generated agents package\n" + "\n".join(
        f"from . import {name}" for name in context["agents"]
    ) + "\n"
    _write_file(project_path / "agents" / "__init__.py", agents_init)
    generated["agents/__init__.py"] = agents_init

    # ── Individual agent files ─────────────────────────────────────────────────
    for agent_name in context["agents"]:
        description = context["agent_descriptions"].get(agent_name, f"Handles {agent_name} step.")
        print(f"  [software_engineer] Generating agents/{agent_name}.py...")
        try:
            handler = _get_handler(project_name, agent_name)
            content = _generate_agent_file(agent_name, description, context, skills_with_feedback, llm, handler)
            _write_file(project_path / "agents" / f"{agent_name}.py", content)
            generated[f"agents/{agent_name}.py"] = content
        except Exception as e:
            errors.append(f"Failed to generate agents/{agent_name}.py: {e}")

    # ── pipeline.py ────────────────────────────────────────────────────────────
    print(f"  [software_engineer] Generating pipeline.py...")
    try:
        content = _generate_pipeline(context, skills_with_feedback, llm, _get_handler(project_name, "pipeline"))
        _write_file(project_path / "pipeline.py", content)
        generated["pipeline.py"] = content
    except Exception as e:
        errors.append(f"Failed to generate pipeline.py: {e}")

    # ── app.py ─────────────────────────────────────────────────────────────────
    print(f"  [software_engineer] Generating app.py...")
    try:
        content = _generate_app(context, skills_with_feedback, llm, _get_handler(project_name, "app"))
        _write_file(project_path / "app.py", content)
        generated["app.py"] = content
    except Exception as e:
        errors.append(f"Failed to generate app.py: {e}")

    # ── requirements.txt ───────────────────────────────────────────────────────
    print(f"  [software_engineer] Generating requirements.txt...")
    try:
        content = _generate_requirements(context, llm, _get_handler(project_name, "requirements"))
        _write_file(project_path / "requirements.txt", content)
        generated["requirements.txt"] = content
    except Exception as e:
        errors.append(f"Failed to generate requirements.txt: {e}")

    # ── .env.example ───────────────────────────────────────────────────────────
    env_content = _env_example()
    _write_file(project_path / ".env.example", env_content)
    generated[".env.example"] = env_content

    return {
        "generated_files": generated,
        "project_path": str(project_path),
        "errors": errors,
    }


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import tempfile

    parser = argparse.ArgumentParser(description="software_engineer — generates project code from a build plan")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate context shape only; no LLM call, no files written")
    parser.add_argument("--project-name", default="smoke-test",
                        help="Project name to use in the test (default: smoke-test)")
    args = parser.parse_args()

    # Use a temp dir so the dry run never pollutes projects/
    tmp_dir = tempfile.mkdtemp(prefix=f"{args.project_name}-")

    test_context = {
        "project_name": args.project_name,
        "project_path": tmp_dir,
        "goal": "Extract fields from a clinical intake form and route to the correct department.",
        "agents": ["extraction_agent", "classification_agent"],
        "agent_descriptions": {
            "extraction_agent": "Extracts patient name, DOB, chief complaint, and urgency from raw intake text.",
            "classification_agent": "Classifies urgency level and selects the routing department.",
        },
        "tech_stack": ["LangChain", "ChatAnthropic", "Pydantic", "Langfuse", "Streamlit"],
        "data_flow": "Raw intake text → extraction → classification → routing decision + plain-English summary",
        "streamlit_ui": "Text area for intake form paste, run button, output card with urgency badge and department, expanders for each agent step.",
        "healthcare_notes": "Handles patient intake data — PHI stub required in guardrails.",
        "skills_md": "(skills_md omitted in smoke test)",
        "evaluator_feedback": [],
        "generated_files": {},
    }

    if args.dry_run:
        print("[software_engineer] Dry run — context shape valid.")
        print(f"  project_name : {test_context['project_name']}")
        print(f"  project_path : {tmp_dir}")
        print(f"  agents       : {test_context['agents']}")
        print(f"  tech_stack   : {test_context['tech_stack']}")
        print("No LLM calls made. Pass without --dry-run to generate real files.")
    else:
        print(f"[software_engineer] Generating project files for: {args.project_name}")
        print(f"  Output directory: {tmp_dir}")
        result = run(test_context)
        print(f"\nGenerated files ({len(result['generated_files'])}):")
        for path, content in result["generated_files"].items():
            print(f"  {path:40s}  ({len(content)} chars)")
        print("\nErrors:", result["errors"] or "none")
        print(f"\nFiles written to: {tmp_dir}")
