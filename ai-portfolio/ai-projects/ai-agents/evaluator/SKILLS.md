# SKILLS.md — evaluator agent
## Domain: LangChain Code Review & Quality Scoring

This file defines the review rubric, scoring logic, and feedback-writing standards
for the evaluator agent. The goal is consistent, actionable verdicts that help
software_engineer converge to passing code within 3 iterations.

---

## 1. Scoring Rubric (weighted)

| Criterion | Weight | What to look for |
|---|---|---|
| Correctness | 30% | Imports valid, LangChain APIs used correctly, code would run without errors |
| LangChain best practices | 20% | LCEL pipe syntax, `with_structured_output`, `bind_tools`, `ChatPromptTemplate.from_messages` |
| Shared state dict | 15% | `pipeline_step` incremented, `max_pipeline_steps` checked, `errors` appended not overwritten |
| Langfuse observability | 15% | `CallbackHandler` created and passed to **every** `.invoke()` call |
| Streamlit UI | 10% | Spinner present, one `st.expander` per agent, sidebar populated, `ValueError` caught |
| Guardrails wiring | 10% | `validate_input` called at top of `pipeline.run()`, `sanitize_output` called before return |

**Threshold:** score ≥ 8 → `"pass"`. Score < 8 → `"revise"`.

---

## 2. Correctness Checks (the 30%)

Run these mentally against each file:

### Imports
- [ ] `from langchain_anthropic import ChatAnthropic` (not `from anthropic import ...`)
- [ ] `from langchain_core.prompts import ChatPromptTemplate`
- [ ] `from langfuse.callback import CallbackHandler`
- [ ] `from dotenv import load_dotenv`
- [ ] `load_dotenv()` called at module level

### LangChain API usage
- [ ] Chain built with pipe syntax: `chain = prompt | llm | parser`
- [ ] `.invoke()` used (not `.run()`, `.predict()`, `.call()`)
- [ ] `ChatPromptTemplate.from_messages([...])` (not raw strings or `PromptTemplate`)
- [ ] `max_tokens` and `temperature` always passed to `ChatAnthropic()`

### Pydantic (if structured output used)
- [ ] Model inherits from `BaseModel`, fields use `Field(description=...)`
- [ ] `llm.with_structured_output(Model)` pattern (not manual JSON parsing)

---

## 3. State Dict Checks (the 15%)

In every `agents/{agent_name}.py`:
```python
def run(state: dict) -> dict:
    state["pipeline_step"] += 1                           # ← must be first
    if state["pipeline_step"] > state["max_pipeline_steps"]:  # ← must be second
        state["errors"].append(f"{AGENT_NAME}: max steps exceeded")
        return state
    # ... logic ...
    state["errors"].append(...)   # ← append, never state["errors"] = [...]
    return state
```
Flag if any of these are missing.

---

## 4. Langfuse Checks (the 15%)

Strict: every single `.invoke()` must have the callback.

```python
# CORRECT
result = chain.invoke({"input": text}, config={"callbacks": [handler]})

# WRONG — missing callbacks
result = chain.invoke({"input": text})

# WRONG — handler created but not passed
handler = _get_handler()
result = chain.invoke({"input": text})  # handler unused
```

Count the number of `.invoke()` calls in each file. Each one must have `config={"callbacks": [...]}`.

---

## 5. Streamlit UI Checks (the 10%)

In `app.py`:
```python
# Required elements
st.set_page_config(...)                         # top of file
with st.sidebar: ...                            # project info
with st.spinner("Running pipeline..."): ...     # wraps pipeline.run()
st.expander(f"Agent: {agent_name}")             # one per agent, shows intermediate output
try: result = pipeline.run(...)                 # wrapped
except ValueError as e: st.error(...); st.stop()  # error handling
if result["errors"]: st.warning(...)            # surface non-fatal errors
```

---

## 6. Writing Actionable Feedback

Each feedback item must:
1. Name the specific file
2. Name the specific function or line pattern
3. State exactly what is wrong
4. State exactly what the fix is

**Good feedback:**
- `"agents/extraction_agent.py: run() does not check max_pipeline_steps. Add: if state['pipeline_step'] > state['max_pipeline_steps']: state['errors'].append(...); return state"`
- `"pipeline.py: sanitize_output(state) is imported but never called. Call it before the return statement."`
- `"app.py: pipeline.run() is not wrapped in st.spinner(). Wrap with: with st.spinner('Running pipeline...'): result = pipeline.run(...)"`

**Bad feedback (too vague):**
- `"The agent doesn't follow best practices"`
- `"Langfuse is not set up correctly"`
- `"The UI needs work"`

---

## 7. Pass Conditions

Issue a `"pass"` when ALL of the following are true:
- Every agent has the `pipeline_step` check
- Every `.invoke()` has a Langfuse callback
- `pipeline.py` calls both `validate_input` and `sanitize_output`
- `app.py` has at least one `st.expander`, a spinner, and catches `ValueError`
- No bare `anthropic` SDK imports
- No hardcoded secrets

A score of 8 is the minimum for pass. 9–10 means no significant issues found.

---

## 8. Strengths to Note

Always identify 1–3 things done well. These go in `result["strengths"]` and
inform the build_walkthrough doc. Examples:
- Clean separation of agent responsibilities
- Appropriate model selection (Haiku for cheap steps, Sonnet for complex)
- Pydantic models used for all structured outputs
- Guardrails have comprehensive injection patterns
- Streamlit expanders make the pipeline transparent

---

## 9. Iteration Strategy

- **Iteration 1 failure (score 5-7):** Focus feedback on the highest-weight criteria first (correctness, state dict, Langfuse). Don't flood with 10 feedback items — pick the top 5.
- **Iteration 2 failure (score 6-7):** Prior feedback should be addressed. If it wasn't, repeat it verbatim with `"(REPEAT FROM PRIOR ITERATION)"` prefix.
- **Iteration 3:** Give your best verdict. If still failing, note `"incomplete"` in feedback — the orchestrator will log it and stop.
