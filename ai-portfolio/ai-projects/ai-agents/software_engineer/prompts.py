"""
prompts.py — All ChatPromptTemplate definitions for the software_engineer agent.

Imported by agent.py. Keeping prompts separate makes them easy to iterate on
without touching orchestration logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Agent file generation ──────────────────────────────────────────────────────

AGENT_FILE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior AI engineer writing a LangChain agent module in Python.

SKILLS reference (use these exact patterns):
{skills_md}

Hard rules — every generated file MUST follow these:
- Filename: projects/{project_name}/agents/{agent_name}.py
- Expose exactly one public function: run(state: dict) -> dict
- First two lines of run(): increment pipeline_step, check max_pipeline_steps
- Write only to state["{agent_name}_output"] — never touch other agents' keys
- Pass Langfuse CallbackHandler to every single .invoke() call
- Trace name: "{project_name}/{agent_name}"
- Use LCEL pipe syntax: chain = prompt | llm | parser
- Use with_structured_output(PydanticModel) if typed output is needed
- Handle ALL exceptions: append to state["errors"], never raise, set output key to None
- All imports at the top of the file
- Complete, runnable Python — zero placeholders, zero TODOs
""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Data flow: {data_flow}

Agent to write: {agent_name}
Role: {agent_description}

All agents in pipeline (for context on state keys): {all_agents}

Prior evaluator feedback to address (may be empty):
{feedback_block}

Write the complete agents/{agent_name}.py file. Output only Python code.""",
    ),
])

# ── pipeline.py generation ────────────────────────────────────────────────────

PIPELINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior AI engineer writing pipeline.py for a LangChain multi-agent project.

SKILLS reference:
{skills_md}

Hard rules:
- Import and call agents in order: state = agent_name.run(state)
- Call validate_input(user_input) from guardrails at the very start
- Call sanitize_output(state) from guardrails at the very end, before returning
- build_initial_state() sets: input, pipeline_step=0, max_pipeline_steps=10, errors=[]
- Expose: run(user_input: dict) -> dict
- Include if __name__ == "__main__": block for quick local test
- Complete, runnable Python — zero placeholders
""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Data flow: {data_flow}

Agents (in order):
{agents_list}

Prior evaluator feedback to address (may be empty):
{feedback_block}

Write the complete pipeline.py. Output only Python code.""",
    ),
])

# ── app.py generation ─────────────────────────────────────────────────────────

APP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior AI engineer writing app.py (Streamlit UI) for a LangChain project.

SKILLS reference:
{skills_md}

Hard rules:
- import pipeline; call result = pipeline.run(user_input_dict)
- st.set_page_config(page_title=..., layout="wide") at the top
- Sidebar: project description, tech stack bullet list, GitHub link placeholder
- Wrap pipeline.run() in: with st.spinner("Running pipeline..."): ...
- Show final result (state["output"]) prominently after the spinner
- Show each intermediate agent output in its own st.expander — interviewers need to see steps
- Catch ValueError from pipeline.run() → st.error(str(e)); st.stop()
- Show state["errors"] as st.warning items if the list is non-empty
- Must run with: streamlit run app.py
- Complete, runnable Python — zero placeholders
""",
    ),
    (
        "human",
        """Project: {project_name}
Goal: {goal}
Streamlit UI plan: {streamlit_ui}
Agents: {agents}

Prior evaluator feedback to address (may be empty):
{feedback_block}

Write the complete app.py. Output only Python code.""",
    ),
])

# ── requirements.txt generation ───────────────────────────────────────────────

REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are writing a requirements.txt for a Python LangChain project.

Always include at minimum (with these exact version pins):
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
langfuse>=2.0.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pydantic>=2.0.0

Add project-specific packages required by the tech stack.
Do NOT add packages that are not actually used.
Output ONLY the requirements.txt content — no explanation, no markdown fences.""",
    ),
    (
        "human",
        "Tech stack: {tech_stack}\nProject goal: {goal}",
    ),
])
