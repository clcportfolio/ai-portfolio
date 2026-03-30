# Clinical Intake Router

Routes clinical intake forms to the right department at the right urgency level in seconds. Demonstrates multi-agent LLM pipelines, structured output, and healthcare workflow automation.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

## What you'll see
Paste or upload a clinical intake form (text or PDF). Click **Route This Intake** to receive a color-coded routing card (green/yellow/red) showing the assigned department, urgency level, and recommended next steps. Expand the agent step panels to inspect the raw extracted fields and the classification reasoning.

## How it works
```
User input (Streamlit UI)
    │
    ▼
guardrails.py → validate_input()
    │
    ▼
pipeline.py → run(input)
    ├─► extraction_agent    → structured patient fields (Pydantic)
    ├─► classification_agent → urgency level + department (Routine/Urgent/Emergency)
    └─► routing_agent       → plain-English routing summary + next steps
    │
    ▼
guardrails.py → sanitize_output()
    │
    ▼
Streamlit UI → routing card + agent expanders
```

## Tech stack
- LangChain + Claude Sonnet (Anthropic)
- Langfuse observability
- Pydantic structured output
- Streamlit demo
- pypdf (PDF intake form support)
