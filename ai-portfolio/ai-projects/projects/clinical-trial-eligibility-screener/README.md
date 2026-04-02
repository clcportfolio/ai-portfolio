# Clinical Trial Eligibility Screener

Automates clinical trial eligibility screening — evaluates a patient summary against a trial's inclusion/exclusion criteria and returns a plain-English verdict with per-criterion reasoning. Helps clinical coordinators make faster, more consistent enrollment decisions.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY, LANGFUSE_*, SUPABASE_DB_URI
streamlit run app.py
```

To pre-populate the trial dropdown and analytics with sample data:
```bash
python scripts/seed_trials.py                              # insert 3 sample trials
python scripts/seed_synthetic_data.py --trial-id 1        # seed synthetic patients
```

## What you'll see

**Tab 1 — Eligibility Check:** Select a stored trial from the dropdown (or paste custom criteria), enter a patient summary, and click Run. A color-coded verdict card shows Eligible / Ineligible / Needs Review with a confidence score, summary, and expandable per-criterion breakdown.

**Tab 2 — Analytics:** Choose a trial and view 6 Plotly charts: eligibility status donut, four-category eligibility breakdown (High/Moderate Eligibility, Needs Review, Ineligible), confidence violin by status, average confidence bar, patient age violin, and sex breakdown by status. Use the "Generate Synthetic Patients" button to seed the dashboard with LLM-generated test data.

## How it works

```
Trial Criteria Text
       │
       ▼
 criteria_agent          ← extracts structured inclusion/exclusion criteria
       │                   (skipped if trial already has cached structured criteria)
       ▼
 evaluation_agent        ← evaluates patient against all criteria in a single LLM call
                           (one call per patient regardless of criterion count)
       │
       ▼
 verdict_agent           ← synthesizes per-criterion results into final verdict
       │
       ▼
 storage (Supabase)      ← stores result by trial_id + SHA-256(patient_summary)
```

## Verdict vs. Confidence — two separate signals

Every screening produces two independent outputs:

- **`eligibility_status`** — the model's verdict: `ELIGIBLE`, `INELIGIBLE`, or `NEEDS_REVIEW`. NEEDS_REVIEW means the model genuinely couldn't determine a clear answer (missing information, contradictory findings, borderline values) — it is not a low-confidence downgrade.

- **`confidence_score`** — how certain the model is about that verdict (0.0–1.0). A score of 0.55 on an ELIGIBLE verdict means the patient looks eligible but the model is uncertain. A score of 0.92 on an INELIGIBLE verdict means the model is highly sure the patient is disqualified.

The **confidence threshold slider** in the UI is a policy control: any verdict (including ELIGIBLE or INELIGIBLE) with a confidence score below your threshold is *displayed* as Needs Review. This does not change what is stored in the database — the raw `eligibility_status` and `confidence_score` are always preserved.

The analytics summary metrics (Eligible / Ineligible / Needs Review counts) reflect the raw stored `eligibility_status`, independent of the threshold slider.

## Tech stack
- LangChain + Claude (Anthropic) — Sonnet for all reasoning and evaluation
- Single-call evaluation — all criteria evaluated in one LLM call per patient
- Langfuse observability — single trace per pipeline run, all agents nested
- Supabase PostgreSQL — trial storage with cached structured criteria, screening history
- Plotly — analytics charts
- Streamlit — demo UI

## Key design decisions

**Criteria caching:** After the first screening of a stored trial, the structured criteria JSON is cached in the `trials` table. Every subsequent screening skips `criteria_agent` entirely — zero LLM cost for re-extraction.

**Deduplication:** Screenings are deduplicated by `(trial_id, SHA-256(patient_summary))`. Running the same patient against the same trial twice returns the cached result instantly.

**Synthetic data:** The "Generate Synthetic Patients" button uses the trial's own criteria (structured when cached, raw text as fallback) to generate diverse fictional profiles via Claude Sonnet (40% eligible, 50% ineligible, 10% borderline), then runs each through the full pipeline. Synthetic screenings are flagged `is_synthetic=True` and can be filtered in analytics.
