"""
agent.py — tech_writer agent.

Writes four documents for a project:
  docs/overview_nontechnical.md  — plain English for a non-technical stakeholder
  docs/overview_technical.md     — architecture deep-dive for a technical interviewer
  docs/build_walkthrough.md      — construction narrative for Cody to use in interviews
  README.md                      — GitHub-ready, runnable in 3 commands
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler

from .prompts import (
    NONTECHNICAL_PROMPT,
    README_PROMPT,
    TECHNICAL_PROMPT,
    WALKTHROUGH_PROMPT,
)

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.4,
    )


def _get_handler(project_name: str, doc: str) -> CallbackHandler:
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_name=f"{project_name}/tech_writer/{doc}",
    )


def _agent_summary(context: dict) -> str:
    return "\n".join(
        f"- **{name}**: {context.get('agent_descriptions', {}).get(name, '')}"
        for name in context.get("agents", [])
    )


def _pipeline_excerpt(generated_files: dict) -> str:
    excerpt = ""
    for key in ("pipeline.py", "guardrails.py"):
        if key in generated_files:
            excerpt += f"\n\n### {key}\n```python\n{generated_files[key][:1500]}\n```"
    return excerpt.strip()


# ── Document generators ────────────────────────────────────────────────────────

def _write_nontechnical(context: dict, llm: ChatAnthropic) -> str:
    chain = NONTECHNICAL_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "project_name": context["project_name"],
            "goal": context.get("goal", ""),
            "streamlit_ui": context.get("streamlit_ui", ""),
            "agents": _agent_summary(context),
        },
        config={"callbacks": [_get_handler(context["project_name"], "nontechnical")]},
    )


def _write_technical(context: dict, llm: ChatAnthropic) -> str:
    chain = TECHNICAL_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "project_name": context["project_name"],
            "goal": context.get("goal", ""),
            "tech_stack": ", ".join(context.get("tech_stack", [])),
            "data_flow": context.get("data_flow", ""),
            "score": context.get("eval_result", {}).get("score", "N/A"),
            "agents": _agent_summary(context),
            "pipeline_excerpt": _pipeline_excerpt(context.get("generated_files", {})),
        },
        config={"callbacks": [_get_handler(context["project_name"], "technical")]},
    )


def _write_walkthrough(context: dict, llm: ChatAnthropic) -> str:
    eval_result = context.get("eval_result", {})
    chain = WALKTHROUGH_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "project_name": context["project_name"],
            "goal": context.get("goal", ""),
            "agents": _agent_summary(context),
            "tech_stack": ", ".join(context.get("tech_stack", [])),
            "feedback": "; ".join(eval_result.get("feedback", [])) or "None — passed on first iteration.",
            "strengths": "; ".join(eval_result.get("strengths", [])) or "Not recorded.",
            "pipeline_excerpt": _pipeline_excerpt(context.get("generated_files", {})),
        },
        config={"callbacks": [_get_handler(context["project_name"], "walkthrough")]},
    )


def _write_readme(context: dict, llm: ChatAnthropic) -> str:
    chain = README_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "project_name": context["project_name"],
            "goal": context.get("goal", ""),
            "agents": _agent_summary(context),
            "tech_stack": ", ".join(context.get("tech_stack", [])),
            "streamlit_ui": context.get("streamlit_ui", ""),
        },
        config={"callbacks": [_get_handler(context["project_name"], "readme")]},
    )


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Write all documentation files for the project.

    Input keys:
        project_name, project_path, goal, agents, agent_descriptions,
        tech_stack, data_flow, streamlit_ui, eval_result,
        security_report, generated_files

    Returns dict with:
        docs_written (list[str]), errors (list[str])
    """
    errors: list[str] = []
    project_path = Path(context.get("project_path", "."))
    docs_path = project_path / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    llm = _get_llm()
    docs_written: list[str] = []

    doc_jobs = [
        ("overview_nontechnical.md", _write_nontechnical),
        ("overview_technical.md", _write_technical),
        ("build_walkthrough.md", _write_walkthrough),
    ]

    for filename, generator in doc_jobs:
        print(f"  [tech_writer] Writing docs/{filename}...")
        try:
            content = generator(context, llm)
            (docs_path / filename).write_text(content)
            docs_written.append(f"docs/{filename}")
        except Exception as e:
            errors.append(f"Failed to write docs/{filename}: {e}")

    print(f"  [tech_writer] Writing README.md...")
    try:
        readme = _write_readme(context, llm)
        (project_path / "README.md").write_text(readme)
        docs_written.append("README.md")
    except Exception as e:
        errors.append(f"Failed to write README.md: {e}")

    return {
        "docs_written": docs_written,
        "errors": errors,
    }


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="tech_writer — writes project documentation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate context shape only; no LLM call, no files written")
    parser.add_argument("--project-name", default="smoke-test",
                        help="Project name to use in the test (default: smoke-test)")
    parser.add_argument("--doc", choices=["nontechnical", "technical", "walkthrough", "readme", "all"],
                        default="all", help="Which document(s) to generate (default: all)")
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix=f"{args.project_name}-docs-")

    test_context = {
        "project_name": args.project_name,
        "project_path": tmp_dir,
        "goal": "Extract fields from a clinical intake form and route to the correct department.",
        "agents": ["extraction_agent", "classification_agent", "routing_agent"],
        "agent_descriptions": {
            "extraction_agent": "Extracts patient name, DOB, chief complaint from raw intake text.",
            "classification_agent": "Classifies urgency: low / medium / high / critical.",
            "routing_agent": "Selects department and generates a plain-English routing summary.",
        },
        "tech_stack": ["LangChain", "ChatAnthropic", "Pydantic", "Langfuse", "Streamlit"],
        "data_flow": "Raw intake text → extraction → urgency classification → department routing + summary",
        "streamlit_ui": "Text area for intake paste, run button, urgency badge, department label, agent step expanders.",
        "eval_result": {
            "status": "pass",
            "score": 9,
            "feedback": [],
            "strengths": [
                "Used Haiku for classification (good cost optimisation)",
                "All agents check max_pipeline_steps",
                "Langfuse handler wired on every .invoke() call",
            ],
        },
        "security_report": "## Security Report\nOverall risk: Low.\n(smoke test placeholder)",
        "generated_files": {
            "pipeline.py": "# pipeline stub for smoke test",
            "guardrails.py": "# guardrails stub for smoke test",
        },
    }

    if args.dry_run:
        print("[tech_writer] Dry run — context shape valid.")
        print(f"  project_name : {test_context['project_name']}")
        print(f"  project_path : {tmp_dir}")
        print(f"  agents       : {test_context['agents']}")
        print(f"  eval score   : {test_context['eval_result']['score']}/10")
        print("No LLM calls made. Pass without --dry-run to generate real documents.")
    else:
        # Optionally narrow to a single document for faster iteration
        if args.doc != "all":
            doc_map = {
                "nontechnical": ("overview_nontechnical.md", _write_nontechnical),
                "technical":    ("overview_technical.md",    _write_technical),
                "walkthrough":  ("build_walkthrough.md",     _write_walkthrough),
                "readme":       ("README.md",                None),
            }
            filename, generator = doc_map[args.doc]
            llm = _get_llm()
            docs_path = Path(tmp_dir) / "docs"
            docs_path.mkdir(parents=True, exist_ok=True)
            if args.doc == "readme":
                content = _write_readme(test_context, llm)
                (Path(tmp_dir) / "README.md").write_text(content)
            else:
                content = generator(test_context, llm)
                (docs_path / filename).write_text(content)
            print(f"[tech_writer] Written: {filename}")
            print(f"  Path: {tmp_dir}")
            print(f"\n--- {filename} preview (first 25 lines) ---")
            print("\n".join(content.splitlines()[:25]))
        else:
            print(f"[tech_writer] Generating all docs for: {args.project_name}")
            print(f"  Output directory: {tmp_dir}")
            result = run(test_context)
            print(f"\nDocs written ({len(result['docs_written'])}):")
            for doc in result["docs_written"]:
                print(f"  {doc}")
            print("\nErrors:", result["errors"] or "none")
            print(f"\nFiles written to: {tmp_dir}")
