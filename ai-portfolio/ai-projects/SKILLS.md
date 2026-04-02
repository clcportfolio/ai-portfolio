# SKILLS.md — Reusable LangChain Patterns & Snippets
## AI Portfolio Builder | Cody Culver

This is the pattern library for all agents in this repo. Every snippet here
follows the conventions in `CLAUDE.md`. Copy and adapt — do not reinvent.

---

## Table of Contents

1. [LLM Setup](#1-llm-setup)
2. [LCEL Chains](#2-lcel-chains)
3. [Structured Output](#3-structured-output)
4. [Tool Use](#4-tool-use)
5. [Vision (Image Input)](#5-vision-image-input)
6. [RAG — Retrieval-Augmented Generation](#6-rag--retrieval-augmented-generation)
7. [Embeddings](#7-embeddings)
8. [Memory (Chat Apps)](#8-memory-chat-apps)
9. [Langfuse Observability](#9-langfuse-observability)
10. [Shared State Dict Pattern](#10-shared-state-dict-pattern)
11. [Agent Module Template](#11-agent-module-template)
12. [Pipeline Template](#12-pipeline-template)
13. [Guardrails Template](#13-guardrails-template)
14. [Streamlit App Template](#14-streamlit-app-template)
15. [Web Search Tool](#15-web-search-tool)
16. [Document Loaders](#16-document-loaders)
17. [Error Handling Pattern](#17-error-handling-pattern)

---

## 1. LLM Setup

```python
from langchain_anthropic import ChatAnthropic

# Reasoning, vision, complex generation
llm_sonnet = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0.3,  # 0 for classification, 0.3-0.7 for generative
)

# Cheap, fast, high-frequency steps
llm_haiku = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    temperature=0,
)
```

**Rule:** Always pass `max_tokens`. Never use raw `anthropic` SDK — always LangChain wrappers.

---

## 2. LCEL Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. {context}"),
    ("human", "{input}"),
])

chain = prompt | llm_sonnet | StrOutputParser()

result = chain.invoke({
    "context": "You specialize in clinical intake forms.",
    "input": "Summarize this intake form: ...",
})
```

**Rule:** Use pipe syntax `prompt | llm | parser`. Never call `.run()` or `.predict()`.

---

## 3. Structured Output

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

class IntakeClassification(BaseModel):
    urgency: str = Field(description="low | medium | high | critical")
    department: str = Field(description="department to route to")
    summary: str = Field(description="plain-English summary for clinic staff")
    confidence: float = Field(description="0.0 to 1.0 confidence score")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0)
structured_llm = llm.with_structured_output(IntakeClassification)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify this clinical intake form. Return structured output only."),
    ("human", "{text}"),
])

chain = prompt | structured_llm
result: IntakeClassification = chain.invoke({"text": "..."})
print(result.urgency, result.department)
```

**Rule:** Use `with_structured_output(PydanticModel)` for any step that needs typed output.
Temperature 0 for classification tasks.

---

## 4. Tool Use

```python
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

@tool
def lookup_cpsc_recalls(product_name: str) -> str:
    """Search CPSC recall database for a product by name. Returns recall info or 'No recalls found'."""
    # implementation here
    return f"No active recalls found for: {product_name}"

@tool
def get_department_contact(department: str) -> str:
    """Return contact information for a given hospital department."""
    contacts = {"cardiology": "ext. 4200", "oncology": "ext. 4300"}
    return contacts.get(department.lower(), "Contact info not available")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0)
llm_with_tools = llm.bind_tools([lookup_cpsc_recalls, get_department_contact])

# Simple tool-calling loop
messages = [HumanMessage(content="Check for recalls on the Fisher-Price Rock-n-Play sleeper.")]

response = llm_with_tools.invoke(messages)
messages.append(response)

# Process tool calls
if response.tool_calls:
    from langchain_core.messages import ToolMessage
    for tc in response.tool_calls:
        tool_fn = {"lookup_cpsc_recalls": lookup_cpsc_recalls,
                   "get_department_contact": get_department_contact}[tc["name"]]
        tool_result = tool_fn.invoke(tc["args"])
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

    final_response = llm_with_tools.invoke(messages)
```

---

## 5. Vision (Image Input)

```python
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

def load_image_base64(file_path: str) -> str:
    """Convert image file to base64 string. Call once at pipeline entry — pass b64 through state."""
    return base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")

def run_vision_chain(image_b64: str, prompt_text: str, llm: ChatAnthropic) -> str:
    """Send image + text prompt to Claude vision model."""
    message = HumanMessage(content=[
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",  # or image/png
                "data": image_b64,
            },
        },
        {"type": "text", "text": prompt_text},
    ])
    response = llm.invoke([message])
    return response.content
```

**Rules:**
- Convert to base64 once at pipeline entry; pass the string through the state dict.
- Use `claude-sonnet-4-20250514` for vision — Haiku supports vision but Sonnet is more accurate.
- Max image size: 10MB (enforced in `guardrails.py → validate_input`).

---

## 6. RAG — Retrieval-Augmented Generation

### Local (Chroma)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load and chunk documents
loader = PyPDFLoader("docs/clinical_guidelines.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm_sonnet, retriever=retriever)

answer = qa_chain.invoke({"query": "What is the protocol for high-urgency cardiology intake?"})
```

### Production (Qdrant)

```python
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
vectorstore = Qdrant(client=client, collection_name="intake_docs", embeddings=embeddings)
```

---

## 7. Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # Free, no API key needed locally. Fast and good enough for most portfolio use cases.
)

# Embed a single string
vector = embeddings.embed_query("patient reports chest pain")

# Embed a list of documents
vectors = embeddings.embed_documents(["doc1 text", "doc2 text"])
```

**Rule:** Default to `all-MiniLM-L6-v2` unless the use case demands higher quality.
No API key needed locally — important for demos and free-tier constraints.

---

## 8. Memory (Chat Apps)

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

memory = ConversationBufferWindowMemory(k=10)  # keep last 10 exchanges

conversation = ConversationChain(
    llm=llm_sonnet,
    memory=memory,
    verbose=False,
)

response = conversation.invoke({"input": "Tell me about the patient's history."})
```

**Rule:** Use `k=10` as the default window. For longer context needs, switch to
`ConversationSummaryBufferMemory` with a token limit.

---

## 9. Langfuse Observability

```python
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv, find_dotenv

# find_dotenv() searches parent directories — works from any project subdirectory
# override=True forces .env values even if the shell has stale empty vars
load_dotenv(find_dotenv(), override=True)

def _get_handler() -> CallbackHandler:
    """Return a Langfuse v4 callback handler.
    Langfuse v4 reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env vars.
    Do NOT pass these as constructor args — they will raise or be silently ignored.
    """
    return CallbackHandler()

# Usage — pass handler to every .invoke() call
handler = _get_handler()
result = chain.invoke({"input": text}, config={"callbacks": [handler]})
```

**Rule:** Every LLM call gets a handler. Trace name format: `"[project-name]/[agent-name]"`.
Non-negotiable — this is a direct interview signal for production instincts.

---

## 10. Shared State Dict Pattern

```python
# Initialize at pipeline entry in pipeline.py
def build_initial_state(user_input: dict) -> dict:
    return {
        "input": user_input,        # Original input — NEVER mutate this key
        "pipeline_step": 0,
        "max_pipeline_steps": 10,   # Safety ceiling. Most pipelines use 3-5 steps.
        "errors": [],               # Non-fatal errors — append, never overwrite
    }

# Each agent reads what it needs, writes only to its own key
def extraction_agent_run(state: dict) -> dict:
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append("Max pipeline steps exceeded — halting.")
        return state

    # ... do work ...
    state["extraction_output"] = {"fields": [...], "raw_text": "..."}
    return state

# Final agent writes to "output"
def report_agent_run(state: dict) -> dict:
    state["pipeline_step"] += 1
    state["output"] = {"report": "...", "status": "complete"}
    return state
```

**Rules:**
- Never delete or overwrite another agent's key.
- Always check `pipeline_step > max_pipeline_steps` at the start of each agent.
- `errors` list is append-only — non-fatal issues accumulate; fatal ones raise exceptions.

---

## 11. Agent Module Template

```python
# projects/[project-name]/agents/[agent_name].py

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "your-project-name"
AGENT_NAME = "your_agent_name"


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
    )


def _get_handler() -> CallbackHandler:
    # Langfuse v4 reads keys from env vars automatically — do NOT pass them as kwargs
    return CallbackHandler()


def run(state: dict) -> dict:
    """
    Agent entry point. Reads from state, writes to state["[agent_name]_output"].
    Never mutates state["input"]. Always increments pipeline_step first.
    """
    state["pipeline_step"] += 1
    if state["pipeline_step"] > state["max_pipeline_steps"]:
        state["errors"].append(f"{AGENT_NAME}: max pipeline steps exceeded")
        return state

    try:
        llm = _get_llm()
        handler = _get_handler()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Your system prompt here."),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(
            {"input": state["input"]},
            config={"callbacks": [handler]},
        )

        state[f"{AGENT_NAME}_output"] = result

    except Exception as e:
        state["errors"].append(f"{AGENT_NAME} error: {str(e)}")

    return state
```

---

## 12. Pipeline Template

```python
# projects/[project-name]/pipeline.py

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
    """
    Main pipeline entry point. Called by app.py.
    Returns the final state dict with state["output"] populated.
    """
    validated = validate_input(user_input)
    state = build_initial_state(validated)

    state = agent_one.run(state)
    state = agent_two.run(state)
    state = agent_three.run(state)

    state = sanitize_output(state)
    return state


if __name__ == "__main__":
    # Quick local test
    result = run({"text": "Sample input for local testing."})
    print(result.get("output"))
    if result["errors"]:
        print("Errors:", result["errors"])
```

---

## 13. Guardrails Template

```python
# projects/[project-name]/guardrails.py

import re
import logging

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4000
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now",
    r"disregard (your |all )?instructions",
    r"act as (a |an )?",
    r"jailbreak",
]

PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",      # SSN
    r"\b\d{10}\b",                  # NPI
    r"\bMRN[:\s]?\d+\b",           # Medical record number
]


def validate_input(data: dict) -> dict:
    """
    Type checks, size limits, prompt injection scan.
    Raises ValueError on failure — pipeline stops.
    """
    text = data.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Input 'text' must be a string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Input text exceeds {MAX_TEXT_LENGTH} character limit.")

    image_b64 = data.get("image_b64")
    if image_b64:
        import base64
        raw = base64.b64decode(image_b64)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds 10MB size limit.")

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError(f"Potential prompt injection detected: pattern '{pattern}'")

    return data


def sanitize_output(state: dict) -> dict:
    """
    Strip code injection from LLM output. PHI/PII redaction stub. Content safety flag.
    NOTE: Replace PHI detection with AWS Comprehend Medical in production.
    """
    output = state.get("output", "")
    if isinstance(output, str):
        # Strip script tags
        output = re.sub(r"<script.*?>.*?</script>", "[REMOVED]", output, flags=re.DOTALL | re.IGNORECASE)

        # PHI redaction stub — logs a warning; does NOT block output in this stub version
        for pattern in PHI_PATTERNS:
            if re.search(pattern, output):
                logger.warning(
                    "PHI pattern detected in output. "
                    "Replace with production-grade scanner (e.g. AWS Comprehend Medical) in prod."
                )
                break

        state["output"] = output

    return state


def rate_limit_check(user_id: str) -> bool:
    """
    Stub — always returns True (allow).
    Replace with Redis-backed counter in production.
    """
    return True
```

---

## 14. Streamlit App Template

```python
# projects/[project-name]/app.py

import streamlit as st
from dotenv import load_dotenv
import pipeline

load_dotenv()

st.set_page_config(page_title="Project Name", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Project Name")
    st.markdown("**What it does:** One sentence description.")
    st.markdown("**Tech stack:**")
    st.markdown("- LangChain + Claude (Anthropic)")
    st.markdown("- Langfuse observability")
    st.markdown("- Streamlit")
    st.markdown("[GitHub Repo](https://github.com/your-repo)")

st.title("Project Name")
st.caption("One sentence tagline.")

# Input
user_text = st.text_area("Paste input here:", height=200)

if st.button("Run Pipeline", type="primary"):
    if not user_text.strip():
        st.warning("Please provide some input.")
    else:
        with st.spinner("Running pipeline..."):
            try:
                result = pipeline.run({"text": user_text})
            except ValueError as e:
                st.error(f"Input validation failed: {e}")
                st.stop()

        # Show final output
        st.subheader("Result")
        st.write(result.get("output", "No output generated."))

        # Show intermediate agent outputs — makes pipeline visible to interviewers
        with st.expander("Agent 1: Extraction", expanded=False):
            st.json(result.get("agent_one_output", {}))

        with st.expander("Agent 2: Classification", expanded=False):
            st.json(result.get("agent_two_output", {}))

        with st.expander("Agent 3: Report", expanded=False):
            st.json(result.get("agent_three_output", {}))

        if result.get("errors"):
            with st.expander("Pipeline warnings"):
                for e in result["errors"]:
                    st.warning(e)
```

---

## 15. Web Search Tool

```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic. Returns a text summary."""
    return search.run(query)

# Bind to LLM
llm_with_search = llm_sonnet.bind_tools([web_search])
```

**Rule:** Use `DuckDuckGoSearchRun` — it's free and requires no API key.
Do not use SerpAPI, Tavily, or Brave unless Cody explicitly approves the cost.

---

## 16. Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,        # PDF files
    TextLoader,         # .txt files
    CSVLoader,          # CSV files
    UnstructuredWordDocumentLoader,  # .docx files
)

# PDF
loader = PyPDFLoader("intake_form.pdf")
docs = loader.load()  # returns List[Document]

# Text
loader = TextLoader("notes.txt", encoding="utf-8")
docs = loader.load()

# CSV
loader = CSVLoader("patient_data.csv", csv_args={"delimiter": ","})
docs = loader.load()

# Each Document has .page_content (str) and .metadata (dict)
for doc in docs:
    print(doc.page_content[:200])
    print(doc.metadata)  # e.g. {"source": "intake_form.pdf", "page": 0}
```

---

## 17. Error Handling Pattern

```python
# Non-fatal error: append to state["errors"], continue pipeline
def safe_agent_run(state: dict) -> dict:
    try:
        result = chain.invoke(...)
        state["my_agent_output"] = result
    except Exception as e:
        state["errors"].append(f"my_agent: {str(e)}")
        state["my_agent_output"] = None  # downstream agents must handle None
    return state

# Fatal error: raise — pipeline.py catches and surfaces to Streamlit
def strict_validate(value: str) -> str:
    if not value:
        raise ValueError("Input cannot be empty.")
    return value

# In app.py — catch ValueError from guardrails/pipeline
try:
    result = pipeline.run({"text": user_text})
except ValueError as e:
    st.error(f"Input validation failed: {e}")
    st.stop()
```

**Rule:** Guardrails raise `ValueError` for bad input (pipeline stops).
Agent failures are non-fatal — append to `errors`, set output key to `None`,
let the pipeline continue and surface warnings in the Streamlit UI.

---

## Quick Reference — Model Selection

| Task | Model | Temp |
|---|---|---|
| Classification, routing, structured output | `claude-haiku-4-5-20251001` | 0 |
| Vision, reasoning, complex generation | `claude-sonnet-4-20250514` | 0.3–0.7 |
| Document summarization | `claude-haiku-4-5-20251001` | 0.3 |
| Report / narrative generation | `claude-sonnet-4-20250514` | 0.5 |
| Prompt injection / safety check | `claude-haiku-4-5-20251001` | 0 |

**Haiku caveat — medical/boolean reasoning:** Haiku hedges aggressively on tasks that
require strict boolean semantics (e.g. "does this patient satisfy this criterion?").
In the clinical trial screener, Haiku consistently misclassified eligible patients as
ineligible by failing to correctly apply inclusion vs. exclusion logic even with explicit
prompt instructions. Use Sonnet for any evaluation task where correctness of
true/false judgments is critical, not just output quality.

## Quick Reference — Pattern Picker

| Need | Pattern |
|---|---|
| Typed output from LLM | `with_structured_output(PydanticModel)` |
| External API or tool call | `@tool` + `bind_tools` |
| Image analysis | `HumanMessage` with image content block |
| Search over documents | Chroma + `HuggingFaceEmbeddings` + `RetrievalQA` |
| Track every LLM call | `langfuse.CallbackHandler` → pass to `.invoke()` |
| Chat with history | `ConversationBufferWindowMemory(k=10)` |
| Free web search | `DuckDuckGoSearchRun` |
| PHI / PII detection | Stub in `guardrails.py` → AWS Comprehend Medical in prod |
