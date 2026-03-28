"""
orchestrator.py — Build pipeline controller.

Accepts a plain-language project description and target folder name.
Produces a build_plan, runs the build loop, and writes build_log.md.

Build loop:
    software_engineer → evaluator → [loop if revise, max 3 iterations]
    → guardrail_engineer + security_specialist → tech_writer

Usage:
    from ai_agents import orchestrator
    result = orchestrator.run({
        "description": "Build a clinical intake router that ...",
        "project_name": "clinical-intake-router",
    })
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MAX_BUILD_ITERATIONS = 3


# ── Pydantic models ────────────────────────────────────────────────────────────

class BuildPlan(BaseModel):
    goal: str = Field(description="One sentence: what this project does and why it matters.")
    tech_stack: list[str] = Field(description="List of key libraries and services used.")
    agents: list[str] = Field(description="Ordered list of agent names in the pipeline (e.g. extraction_agent, classification_agent).")
    agent_descriptions: dict[str, str] = Field(description="Map of agent_name -> one-line description of its role.")
    data_flow: str = Field(description="Brief description of how data flows through the pipeline.")
    streamlit_ui: str = Field(description="Description of the Streamlit UI: inputs, outputs, and any expanders.")
    max_build_iterations: int = Field(default=3, description="Max software_engineer→evaluator retry cycles.")
    healthcare_notes: str = Field(description="Any HIPAA/PHI considerations, even if not a clinical project.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.3) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=temperature,
    )


def _get_handler(step: str) -> CallbackHandler:
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_name=f"build-pipeline/orchestrator/{step}",
    )


def _skills_md() -> str:
    """Load SKILLS.md content to pass as context to build agents."""
    skills_path = PROJECT_ROOT / "SKILLS.md"
    if skills_path.exists():
        return skills_path.read_text()
    return "(SKILLS.md not found)"


def _write_build_log(project_path: Path, log_entries: list[dict], final_status: str) -> None:
    lines = [
        f"# Build Log — {project_path.name}",
        f"Generated: {datetime.now().isoformat()}",
        f"Final status: **{final_status}**",
        "",
    ]
    for i, entry in enumerate(log_entries, 1):
        lines += [
            f"## Iteration {i}",
            f"- Evaluator status: `{entry.get('status', 'unknown')}`",
            f"- Evaluator score: {entry.get('score', 'N/A')}/10",
            "- Feedback:",
        ]
        for fb in entry.get("feedback", []):
            lines.append(f"  - {fb}")
        lines.append("")
    (project_path / "build_log.md").write_text("\n".join(lines))


# ── Core: plan generation ──────────────────────────────────────────────────────

def _generate_build_plan(description: str, project_name: str) -> BuildPlan:
    """Use LLM to turn a plain-language description into a structured BuildPlan."""
    llm = _get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(BuildPlan)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a senior AI engineer designing a LangChain multi-agent pipeline.
Given a project description, produce a structured build plan.

Rules:
- Use LangChain + ChatAnthropic (claude-sonnet-4-20250514 for complex tasks, claude-haiku-4-5-20251001 for simple/fast tasks)
- Every project has guardrails.py with PHI stub, even non-clinical projects
- Every project has a Streamlit app.py
- Every project uses Langfuse for observability
- Agents communicate via a shared state dict
- Keep agents focused: each does one job
- 3-5 agents is typical; never plan more than 7
""",
        ),
        (
            "human",
            "Project name: {project_name}\n\nDescription:\n{description}",
        ),
    ])

    chain = prompt | structured_llm
    return chain.invoke(
        {"project_name": project_name, "description": description},
        config={"callbacks": [_get_handler("plan_generation")]},
    )


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Orchestrate the full build pipeline for a new project.

    Input keys:
        description (str):    Plain-language description of the project.
        project_name (str):   Snake-case or kebab-case folder name under projects/.
        dry_run (bool):       If True, generate build_plan only; don't invoke sub-agents.

    Returns dict with:
        project_name, project_path, build_plan, final_status, build_log_path, errors
    """
    import importlib
    import sys

    errors: list[str] = []

    description = context.get("description", "").strip()
    project_name = context.get("project_name", "").strip()
    dry_run = context.get("dry_run", False)

    if not description:
        return {"errors": ["'description' is required."], "final_status": "failed"}
    if not project_name:
        return {"errors": ["'project_name' is required."], "final_status": "failed"}

    project_path = PROJECT_ROOT / "projects" / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate build plan ────────────────────────────────────────────
    print(f"[orchestrator] Generating build plan for '{project_name}'...")
    try:
        build_plan = _generate_build_plan(description, project_name)
    except Exception as e:
        return {"errors": [f"Build plan generation failed: {e}"], "final_status": "failed"}

    build_plan_dict = build_plan.model_dump()
    build_plan_dict["project_name"] = project_name
    build_plan_dict["project_path"] = str(project_path)
    build_plan_dict["skills_md"] = _skills_md()

    if dry_run:
        print("[orchestrator] Dry run — skipping code generation.")
        return {
            "project_name": project_name,
            "project_path": str(project_path),
            "build_plan": build_plan_dict,
            "final_status": "dry_run",
            "errors": errors,
        }

    # Lazy-import sub-agents to avoid circular imports at module load time.
    # Add the ai-agents/ directory itself to sys.path so each agent is importable
    # by its bare name (e.g. "software_engineer"). Subpackage agents rely on their
    # own __init__.py re-exporting run(), so se.run() works uniformly for both flat
    # and subpackage agents.
    agents_dir = Path(__file__).parent  # .../ai-agents/
    if str(agents_dir) not in sys.path:
        sys.path.insert(0, str(agents_dir))

    try:
        se = importlib.import_module("software_engineer")   # subpackage → __init__.py → agent.py
        ev = importlib.import_module("evaluator")           # subpackage → __init__.py → agent.py
        ge = importlib.import_module("guardrail_engineer")  # flat module
        ss = importlib.import_module("security_specialist") # flat module
        tw = importlib.import_module("tech_writer")         # subpackage → __init__.py → agent.py
    except ModuleNotFoundError as e:
        return {"errors": [f"Sub-agent import failed: {e}"], "final_status": "failed"}

    # ── Step 2: software_engineer → evaluator loop ─────────────────────────────
    log_entries: list[dict] = []
    se_context = {**build_plan_dict, "evaluator_feedback": []}
    eval_result: dict = {}

    for iteration in range(1, MAX_BUILD_ITERATIONS + 1):
        print(f"[orchestrator] Build iteration {iteration}/{MAX_BUILD_ITERATIONS}...")

        se_result = se.run(se_context)
        if se_result.get("errors"):
            errors.extend(se_result["errors"])

        eval_result = ev.run({**build_plan_dict, "generated_files": se_result.get("generated_files", {})})
        log_entries.append(eval_result)

        print(f"[orchestrator] Evaluator: {eval_result.get('status')} (score {eval_result.get('score')}/10)")

        if eval_result.get("status") == "pass":
            break

        # Feed feedback back to software_engineer for next iteration
        se_context["evaluator_feedback"] = eval_result.get("feedback", [])
        se_context["generated_files"] = se_result.get("generated_files", {})

    final_status = "complete" if eval_result.get("status") == "pass" else "incomplete"

    # ── Step 3: guardrail_engineer + security_specialist (parallel-ish) ────────
    print("[orchestrator] Running guardrail_engineer...")
    ge_result = ge.run({**build_plan_dict, "generated_files": se_context.get("generated_files", {})})
    if ge_result.get("errors"):
        errors.extend(ge_result["errors"])

    print("[orchestrator] Running security_specialist...")
    ss_result = ss.run({
        **build_plan_dict,
        "generated_files": {
            **se_context.get("generated_files", {}),
            **ge_result.get("generated_files", {}),
        },
    })
    if ss_result.get("errors"):
        errors.extend(ss_result["errors"])

    # ── Step 4: tech_writer ────────────────────────────────────────────────────
    print("[orchestrator] Running tech_writer...")
    tw_result = tw.run({
        **build_plan_dict,
        "eval_result": eval_result,
        "generated_files": {
            **se_context.get("generated_files", {}),
            **ge_result.get("generated_files", {}),
        },
        "security_report": ss_result.get("security_report", ""),
    })
    if tw_result.get("errors"):
        errors.extend(tw_result["errors"])

    # ── Step 5: write build log ────────────────────────────────────────────────
    _write_build_log(project_path, log_entries, final_status)
    print(f"[orchestrator] Done. Status: {final_status}. Project at: {project_path}")

    return {
        "project_name": project_name,
        "project_path": str(project_path),
        "build_plan": build_plan_dict,
        "final_status": final_status,
        "build_log_path": str(project_path / "build_log.md"),
        "errors": errors,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI project build pipeline.")
    parser.add_argument("project_name", help="Folder name for the new project (kebab-case)")
    parser.add_argument("description", help="Plain-language description of the project")
    parser.add_argument("--dry-run", action="store_true", help="Generate build plan only, skip code generation")
    args = parser.parse_args()

    result = run({
        "description": args.description,
        "project_name": args.project_name,
        "dry_run": args.dry_run,
    })
    print(json.dumps({k: v for k, v in result.items() if k != "build_plan"}, indent=2))
    if result.get("build_plan"):
        print("\nBuild plan:")
        print(json.dumps(result["build_plan"], indent=2, default=str))
