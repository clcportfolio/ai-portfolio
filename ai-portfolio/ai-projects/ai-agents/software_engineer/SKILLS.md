# SKILLS.md — software_engineer agent
## Domain: LangChain Code Generation

This file is the generation reference for the software_engineer agent.
Every pattern here is what the agent must produce in output files.

---

## 1. The Non-Negotiables (check every generated file)

| Rule | Where it applies |
|---|---|
| Every `.invoke()` gets `config={"callbacks": [handler]}` | All agent files |
| Every agent checks `pipeline_step > max_pipeline_steps` first | All agent `run()` functions |
| Errors are appended to `state["errors"]`, never raised | All agent `run()` functions |
| `validate_input` called before state is built | `pipeline.py` |
| `sanitize_output` called before returning | `pipeline.py` |
| `state["output"]` assigned to final agent's output key before `sanitize_output` | `pipeline.py` |
| `max_tokens` always explicit on every LLM instantiation | All agent files |
| No raw `anthropic` or `openai` SDK — always LangChain wrappers | All files |
| `structured_llm` MUST be composed into a chain — never call `.invoke()` directly on it | All agents using structured output |
| `CallbackHandler()` takes NO constructor args — Langfuse v4 reads keys from env vars | All `_get_handler()` functions |
| `load_dotenv(find_dotenv(), override=True)` — searches parent dirs, overrides stale env | All agent and pipeline files |

---

## 2. Agent File Template

Every `agents/{agent_name}.py` must follow this exact structure:

```python
"""
{agent_name}.py — {one_line_description}
"""
from __future__ import annotations
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)  # searches parent dirs; override=True forces .env values

PROJECT_NAME = "{project_name}"
AGENT_NAME   = "{agent_name}"


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",  # or haiku for cheap/fast steps
        max_tokens=1024,
        temperature=0,  # 0 for classification, 0.3-0.7 for generation
    )


def _get_handler() -> CallbackHandler:
    # Langfuse v4: reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env vars
    # Do NOT pass public_key, secret_key, host, or trace_name as constructor args — they are ignored or error
    return CallbackHandler()


def run(state: dict) -> dict:
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        llm     = _get_llm()
        handler = _get_handler()
        prompt  = ChatPromptTemplate.from_messages([...])
        chain   = prompt | llm | StrOutputParser()
        result  = chain.invoke({...}, config={"callbacks": [handler]})
        state[f"{AGENT_NAME}_output"] = result
    except Exception as e:
        state["errors"].append(f"{AGENT_NAME}: {e}")
        state[f"{AGENT_NAME}_output"] = None

    return state
```

---

## 3. Model Selection Quick Reference

| Task type | Model | Temperature |
|---|---|---|
| Classification, routing, entity extraction | `claude-haiku-4-5-20251001` | 0 |
| Vision, complex reasoning, narrative generation | `claude-sonnet-4-20250514` | 0.3–0.7 |
| Structured output (Pydantic) | Either — match complexity | 0 |
| Summarization | `claude-haiku-4-5-20251001` | 0.3 |

---

## 4. Structured Output Pattern

Use when the agent must return typed fields (classification, extraction, routing).

```python
from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    patient_name: str = Field(description="Full name of the patient")
    urgency: str      = Field(description="low | medium | high | critical")
    summary: str      = Field(description="Plain-English summary for staff")

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512, temperature=0)
structured_llm = llm.with_structured_output(ExtractionResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract fields from the intake form."),
    ("human", "{text}"),
])
chain = prompt | structured_llm
result: ExtractionResult = chain.invoke({"text": ...}, config={"callbacks": [handler]})

state["extraction_output"] = result.model_dump()
```

---

## 5. Tool Use Pattern

Use when the agent must call an external API or lookup.

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def lookup_department_hours(department: str) -> str:
    """Return operating hours for a hospital department."""
    hours = {"cardiology": "Mon-Fri 8am-6pm"}
    return hours.get(department.lower(), "Contact admissions for hours.")

llm_with_tools = llm.bind_tools([lookup_department_hours])
messages = [HumanMessage(content=user_input)]
response = llm_with_tools.invoke(messages, config={"callbacks": [handler]})

if response.tool_calls:
    for tc in response.tool_calls:
        tool_result = lookup_department_hours.invoke(tc["args"])
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
    final = llm_with_tools.invoke(messages, config={"callbacks": [handler]})
    state["lookup_output"] = final.content
```

---

## 6. Vision Pattern

Use when the agent processes an image. Image must already be base64 in state.

```python
from langchain_core.messages import HumanMessage

def run(state: dict) -> dict:
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append("vision_agent: max steps exceeded")
        return state

    image_b64 = state["input"].get("image_b64")
    if not image_b64:
        state["errors"].append("vision_agent: no image_b64 in input")
        return state

    llm     = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0)
    handler = _get_handler()

    message = HumanMessage(content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
        {"type": "text",  "text": "Identify this object and describe it in detail."},
    ])
    response = llm.invoke([message], config={"callbacks": [handler]})
    state["vision_output"] = response.content
    return state
```

---

## 7. Pipeline Template

```python
# pipeline.py
from guardrails import validate_input, sanitize_output
from agents import agent_one, agent_two, agent_three

def build_initial_state(user_input: dict) -> dict:
    return {
        "input": user_input,
        "pipeline_step": 0,
        "max_pipeline_steps": 10,
        "errors": [],
    }

def run(user_input: dict) -> dict:
    validated = validate_input(user_input)
    state     = build_initial_state(validated)
    state     = agent_one.run(state)
    state     = agent_two.run(state)
    state     = agent_three.run(state)
    state["output"] = state.get("agent_three_output")  # surface final output for sanitize_output + app.py
    state     = sanitize_output(state)
    return state

if __name__ == "__main__":
    result = run({"text": "Test input."})
    print(result.get("output"))
    print("Errors:", result["errors"])
```

---

## 8. Addressing Evaluator Feedback

When `evaluator_feedback` is non-empty, the software_engineer MUST:

1. **Read each feedback item carefully** — they name a specific file and issue.
2. **Fix in priority order:**
   - Missing Langfuse handler on `.invoke()` → add `config={"callbacks": [handler]}`
   - Missing `max_pipeline_steps` check → add at top of `run()`
   - Guardrails not wired in `pipeline.py` → add `validate_input` / `sanitize_output` calls
   - Incorrect LangChain API usage → switch to LCEL pipe syntax
   - Streamlit missing expanders → add one `st.expander` per agent
3. **Do not change what was already correct** — only address the flagged items.

---

## 9. agents/__init__.py Template

```python
# Auto-generated by software_engineer
from . import agent_one
from . import agent_two
from . import agent_three
```

---

## 10. Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| `structured_llm.invoke({"key": val})` | WRONG — `structured_llm` is not a runnable alone. Always: `chain = prompt \| structured_llm` then `chain.invoke({"key": val})` |
| `llm.invoke(prompt_str)` | Use `chain = prompt \| llm \| parser; chain.invoke({...})` |
| `CallbackHandler(public_key=..., secret_key=..., host=..., trace_name=...)` | Langfuse v4: use `CallbackHandler()` only — keys come from env vars |
| `load_dotenv()` | Use `load_dotenv(find_dotenv(), override=True)` — finds .env in parent dirs, overrides stale shell env |
| `state["errors"] = [str(e)]` | `state["errors"].append(str(e))` — never overwrite |
| `state["output"]` never set | Must assign `state["output"] = state.get("final_agent_output")` before `sanitize_output()` |
| `temperature` omitted | Always pass `temperature=` explicitly |
| `max_tokens` omitted | Always pass `max_tokens=` explicitly |
| Importing `anthropic` directly | Import from `langchain_anthropic` |
| Reading state["input"] mid-pipeline | Read once at pipeline entry; pass results via state |
