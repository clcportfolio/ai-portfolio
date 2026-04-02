# Clinical Trial Eligibility Screener

Automates clinical trial eligibility screening — evaluates a patient summary against a trial's inclusion/exclusion criteria and returns a plain-English verdict with per-criterion reasoning. Helps clinical coordinators make faster, more consistent enrollment decisions.
#### Demo built in ~8 work hours

> **Live demo:** [Streamlit app](https://clcportfolio-clinical-trial-eligibility-screener.streamlit.app/)
> **Live demo logins:** Contact Cody at: [clculver5@gmail.com](clculver5@gmail.com)
> **Portfolio:** [github.com/clcportfolio/ai-portfolio](https://github.com/clcportfolio/ai-portfolio)

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY, LANGFUSE_*, SUPABASE_DB_URI
streamlit run app.py
```

**Demo credentials** — contact [Cody](clculver5@gmail.com) for access.

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

## Production considerations

This is a portfolio demo. Here is what would change before putting it in front of real coordinators:

**Real patient data and PHI compliance:** The demo accepts free-text patient summaries with no de-identification step. In production, patient input would be de-identified before entering the pipeline (using AWS Comprehend Medical or an equivalent), all storage would be HIPAA-compliant, and audit logs would track every screening by user, timestamp, and trial. The SHA-256 dedup key would need to be replaced — hashing raw PHI is not sufficient for compliance.

**Structured patient records instead of free text:** Free-text summaries work for a demo but are unreliable in production — the evaluation quality depends entirely on how well the summary was written. A real deployment would pull structured fields (age, diagnoses, medications, lab values, comorbidities) from an EHR integration (HL7 FHIR, Epic, Cerner) and format them deterministically before passing to the evaluation agent. This removes ambiguity from the most common source of NEEDS_REVIEW verdicts: missing information.

**ClinicalTrials.gov integration:** The demo requires coordinators to paste criteria manually. Production would connect to the ClinicalTrials.gov API to fetch structured trial criteria by NCT ID, eliminating manual entry and the risk of copy-paste errors.

**Human-in-the-loop review queue:** NEEDS_REVIEW verdicts currently surface in the analytics dashboard but have no workflow attached. A production system would route these to a coordinator queue with a UI for confirming or overriding the verdict, capturing ground-truth labels that can be used to evaluate model performance over time.

**Model evaluation and drift detection:** There is currently no way to know if the model's verdicts are correct. In production, a sample of ELIGIBLE and INELIGIBLE verdicts would be reviewed by a clinical coordinator, with agreement rates tracked in Langfuse. If agreement drops below a threshold, the evaluation prompt or model would be flagged for review. The synthetic data pipeline would be repurposed to generate regression test cases for prompt changes.

**Authentication and role-based access:** The app is gated behind a login screen with two demo roles — coordinator (run screenings, view verdicts) and admin (full access including raw agent outputs and synthetic data seeding). `RoleConfig` is a Pydantic model that acts as the single source of truth for every role-dependent UI decision. Production would replace the hardcoded credential store with SSO, restrict trial visibility by sponsor/site, and log every screening action by user ID for audit purposes.

**Scalability:** The current pipeline runs synchronously in Streamlit's process, which blocks the UI during screening. A production deployment would decouple screening into a background task queue (Celery, AWS SQS) with a webhook or polling mechanism to return results, allowing the UI to remain responsive and supporting batch screening of hundreds of patients at once.

## Tech stack
- LangChain + Claude (Anthropic) — Sonnet for all reasoning and evaluation (Haiku was tested but hedged too aggressively on exclusion criteria, producing false INELIGIBLE verdicts; Sonnet follows the inclusion/exclusion boolean semantics reliably)
- Single-call evaluation — all criteria evaluated in one LLM call per patient
- Langfuse observability — single trace per pipeline run, all agents nested
- Supabase PostgreSQL — trial storage with cached structured criteria, screening history
- Plotly — analytics charts
- Streamlit — demo UI

## Key design decisions

**Deterministic verdict logic:** The eligibility verdict (`ELIGIBLE` / `INELIGIBLE` / `NEEDS_REVIEW`) is computed in Python from the boolean evaluation results — the LLM is never asked to decide the verdict. Early versions let the LLM synthesize the verdict directly, but it consistently hedged to `NEEDS_REVIEW` even when all criteria were clearly resolved. Moving the verdict to deterministic code eliminated the hedging entirely: if all inclusion criteria are satisfied and no exclusion criteria are triggered, the verdict is `ELIGIBLE`, period. The LLM's role in `verdict_agent` is now limited to writing the plain-English explanation.

**Exclusion criterion semantics:** The `meets_criterion` boolean has opposite meaning for inclusion vs. exclusion criteria. For inclusion: `true` means the patient satisfies the requirement. For exclusion: `true` means the patient *has* the disqualifying condition — i.e., they are excluded. This asymmetry is the most common source of bugs in this kind of pipeline. The evaluation prompt makes it explicit with concrete examples for both directions, and the verdict logic encodes it deterministically rather than relying on the LLM to reason about it correctly every time.

**Single evaluation call per patient:** The original design fired one LLM call per criterion in parallel via `asyncio.gather`. With 7 criteria and 5 patients running concurrently, that's 35 simultaneous Sonnet calls — consistently hitting API rate limits, causing silent failures that returned `confidence=0.0` and triggered `NEEDS_REVIEW` for every patient. Collapsing all criteria into one call per patient reduced concurrent API calls from 35 to 5, eliminated the rate limit problem entirely, and cut total token overhead (one system prompt instead of N).

**NEEDS_REVIEW is confidence-gated, not LLM-decided:** `NEEDS_REVIEW` is only assigned when any individual criterion evaluation returns `confidence < 0.35` — meaning the patient summary genuinely lacked the information needed to assess that criterion. A threshold of 0.35 rather than 0.5 ensures that only truly absent information triggers review, not mild uncertainty on clearly-stated facts (which score 0.85+).

**Criteria caching:** After the first screening of a stored trial, the structured criteria JSON is cached in the `trials` table. Every subsequent screening skips `criteria_agent` entirely — zero LLM cost for re-extraction. This also improves synthetic data quality: the generator receives the already-structured criteria list rather than raw text, so it can address each criterion precisely.

**Deduplication:** Screenings are deduplicated by `(trial_id, SHA-256(patient_summary))`. Running the same patient against the same trial twice returns the cached result instantly without an API call.

**Synthetic data distribution:** The generator targets 40% eligible, 50% ineligible, and 10% borderline per batch. Eligible patients are generated with a separate prompt that explicitly confirms every inclusion criterion and negates every exclusion criterion — this was necessary because a combined prompt produced summaries that were too ambiguous to score as clearly eligible. The two batches (eligible and ineligible/borderline) run in parallel and are shuffled before being passed to the pipeline.
