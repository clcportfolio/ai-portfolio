"""
agent.py — evaluator agent.

Reviews generated project code for correctness, LangChain best practices,
appropriate LLM use, error handling, and Streamlit UI completeness.

Returns a structured verdict: pass or revise, with a score and feedback list.
Feedback is passed back to software_engineer for the next build iteration.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field

from .prompts import EVAL_PROMPT

load_dotenv()


# ── Pydantic model ─────────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    status: str = Field(
        description="'pass' if the code is production-ready (score >= 8), 'revise' if changes are needed."
    )
    score: int = Field(description="Quality score 0-10 based on the rubric.")
    feedback: list[str] = Field(
        description=(
            "Specific, actionable feedback items. Each must name the file, the "
            "function or line pattern, the exact problem, and the exact fix. "
            "Empty list if status is 'pass'."
        )
    )
    strengths: list[str] = Field(description="1–3 specific things done well.")


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
        trace_name=f"{project_name}/evaluator",
    )


def _format_files(generated_files: dict[str, str]) -> str:
    sections = []
    for path, content in generated_files.items():
        # Trim very long files to avoid token overflow — show first 3000 chars
        preview = content[:3000] + ("\n... [truncated]" if len(content) > 3000 else "")
        sections.append(f"### {path}\n```python\n{preview}\n```")
    return "\n\n".join(sections)


def _load_rubric() -> str:
    """Load this agent's own SKILLS.md as additional context for the LLM."""
    skills_path = Path(__file__).parent / "SKILLS.md"
    return skills_path.read_text() if skills_path.exists() else ""


# ── Run ────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Review generated code and return a structured evaluation.

    Input keys:
        project_name (str), goal (str), agents (list[str]),
        generated_files (dict: relative_path -> file_content)

    Returns dict with:
        status ("pass" | "revise"), score (int), feedback (list[str]),
        strengths (list[str]), errors (list[str])
    """
    errors: list[str] = []
    project_name = context.get("project_name", "unknown")
    generated_files = context.get("generated_files", {})

    if not generated_files:
        return {
            "status": "revise",
            "score": 0,
            "feedback": ["No files were generated. software_engineer produced no output."],
            "strengths": [],
            "errors": ["No generated_files provided to evaluator."],
        }

    llm = _get_llm().with_structured_output(EvalResult)
    handler = _get_handler(project_name)

    chain = EVAL_PROMPT | llm

    try:
        result: EvalResult = chain.invoke(
            {
                "project_name": project_name,
                "goal": context.get("goal", ""),
                "agents": ", ".join(context.get("agents", [])),
                "files": _format_files(generated_files),
            },
            config={"callbacks": [handler]},
        )
    except Exception as e:
        errors.append(f"Evaluator LLM call failed: {e}")
        return {
            "status": "revise",
            "score": 0,
            "feedback": [f"Evaluation failed: {e}"],
            "strengths": [],
            "errors": errors,
        }

    return {
        "status": result.status,
        "score": result.score,
        "feedback": result.feedback,
        "strengths": result.strengths,
        "errors": errors,
    }


# ── CLI / self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="evaluator — reviews generated project code and returns pass/revise verdict")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate context shape only; no LLM call")
    parser.add_argument("--project-name", default="smoke-test",
                        help="Project name to use in the test (default: smoke-test)")
    args = parser.parse_args()

    # Minimal but realistic generated files — enough to exercise the rubric
    _sample_pipeline = """\
from guardrails import validate_input, sanitize_output
from agents import extraction_agent, classification_agent

def build_initial_state(user_input):
    return {"input": user_input, "pipeline_step": 0, "max_pipeline_steps": 10, "errors": []}

def run(user_input: dict) -> dict:
    validated = validate_input(user_input)
    state = build_initial_state(validated)
    state = extraction_agent.run(state)
    state = classification_agent.run(state)
    state = sanitize_output(state)
    return state
"""

    _sample_agent = """\
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
load_dotenv()

PROJECT_NAME = "smoke-test"
AGENT_NAME   = "extraction_agent"

def _get_llm():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512, temperature=0)

def _get_handler():
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_name=f"{PROJECT_NAME}/{AGENT_NAME}",
    )

def run(state: dict) -> dict:
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state
    try:
        llm = _get_llm()
        handler = _get_handler()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract patient name and chief complaint."),
            ("human", "{text}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": state["input"]["text"]}, config={"callbacks": [handler]})
        state["extraction_output"] = result
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: {e}")
        state["extraction_output"] = None
    return state
"""

    test_context = {
        "project_name": args.project_name,
        "goal": "Extract and classify clinical intake forms.",
        "agents": ["extraction_agent", "classification_agent"],
        "generated_files": {} if args.dry_run else {
            "pipeline.py": _sample_pipeline,
            "agents/extraction_agent.py": _sample_agent,
        },
    }

    if args.dry_run:
        print("[evaluator] Dry run — context shape valid.")
        print(f"  project_name : {test_context['project_name']}")
        print(f"  agents       : {test_context['agents']}")
        print("No LLM calls made. Pass without --dry-run to run a real evaluation.")
    else:
        print(f"[evaluator] Evaluating generated code for: {args.project_name}")
        result = run(test_context)
        print(f"\n  status   : {result['status']}")
        print(f"  score    : {result['score']}/10")
        print(f"  strengths: {result['strengths']}")
        print(f"  feedback : {result['feedback']}")
        print(f"  errors   : {result['errors']}")
