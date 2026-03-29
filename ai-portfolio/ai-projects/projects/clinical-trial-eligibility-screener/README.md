# Clinical Trial Eligibility Screener

Automates clinical trial eligibility screening by evaluating patient summaries against trial criteria and providing clear verdicts with reasoning. Helps clinical coordinators make faster, more informed decisions about patient enrollment.

## Run it
```
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

## What you'll see
Two text areas for entering trial criteria and patient summary, with a "Run Eligibility Check" button. Results display as a color-coded verdict card (Eligible/Likely Ineligible/Needs Review) with an expandable section showing detailed per-criterion evaluation breakdown.

## How it works
```
Trial Criteria Text → criteria_agent → Structured Criteria
                                            ↓
Patient Summary → evaluation_agent → Individual Evaluations
                                            ↓
All Evaluations → verdict_agent → Final Verdict + Reasoning
```

## Tech stack
- LangChain + Claude (Anthropic)
- Langfuse observability
- Streamlit demo
- Pydantic data validation