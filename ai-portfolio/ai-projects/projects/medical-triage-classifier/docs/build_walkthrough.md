# Medical Triage Classifier — Build Walkthrough

Step-by-step construction narrative. Read this to explain every design decision
in an interview without looking at the code.

---

## Step 1: Project Scaffold

**Files:** `.env.example`, `requirements.txt`, `.gitignore`

Started with the standard CLAUDE.md scaffold. This project is different from
other portfolio projects because it does NOT use the `pipeline.py` / `agents/`
pattern — it's a model training and evaluation project. Still follows all other
conventions: guardrails.py, app.py, .env.example, `__main__` blocks, docs.

Key dependency additions beyond the base:
- `transformers`, `peft`, `datasets` — HuggingFace model training
- `mlflow` — experiment tracking and model registry
- `torch`, `accelerate` — PyTorch training runtime
- `scikit-learn` — evaluation metrics
- `plotly` — visualization in Streamlit

---

## Step 2: Infrastructure (provision_infra.py)

**What it does:** Creates S3 bucket with prefix structure for datasets, checkpoints,
and model artifacts. Documents EC2 MLflow server setup.

**Design decision — local MLflow first:**
`MLFLOW_TRACKING_URI` defaults to `http://localhost:5000`. Run `mlflow ui` locally
during development. When EC2 is set up, change one env var. No code changes needed.

**S3 prefix structure:**
```
medical-triage-classifier/
  datasets/    ← train.csv, val.csv, test.csv
  checkpoints/ ← per-epoch model saves
  models/      ← final registered model artifacts
```

---

## Step 3: Data Preparation (data_prep.py)

**What it does:** Loads MTSamples CSV, maps specialties to urgency labels, generates
synthetic examples to balance classes, splits data, uploads to S3, logs stats to MLflow.

**Labeling strategy:**
MTSamples has a `medical_specialty` column. Rather than manually annotating thousands
of notes, we map specialties to urgency levels based on clinical heuristic:
- Emergency Room Reports → Emergency (obvious)
- Cardiovascular/Pulmonary → Emergency (acute presentations common)
- General Medicine → Routine (scheduled visits)
- Orthopedic → Urgent (injuries needing timely care)

This is a pragmatic choice for a portfolio project. Production would use expert annotation.

**Synthetic augmentation:**
After labeling, some classes are underrepresented. Claude Haiku generates synthetic
clinical notes to reach ~300 per class. Separate prompts per urgency level — combining
them in one prompt produced ambiguous examples that were hard to classify.

The synthetic ratio is tracked in MLflow so we can measure its impact on model quality.

**Split:** 70/15/15 stratified. Stratification ensures each split has the same class
proportions, which is critical for reliable evaluation on imbalanced datasets.

---

## Step 4: Training (trainer.py)

**What it does:** Fine-tunes DistilBERT with PEFT/LoRA, logs to MLflow, saves
checkpoints to S3, registers final model.

**Why DistilBERT?**
- 67M parameters (vs BERT's 110M) — 40% smaller
- Trains in minutes on free Colab T4 GPU
- Sufficient for 3-class classification
- Bio_ClinicalBERT can be tested later as an MLflow experiment

**Why LoRA?**
- Only trains ~300K parameters (0.5% of the model)
- Avoids catastrophic forgetting of pre-trained knowledge
- Adapter weights are tiny (~1MB vs 250MB for full model)
- Standard PEFT approach for efficient fine-tuning

**LoRA config explained:**
- `r=16` — rank of the low-rank matrices (higher = more capacity, more params)
- `alpha=32` — scaling factor (alpha/r = 2 is standard)
- Target modules: `q_lin`, `v_lin` — DistilBERT's attention query and value projections.
  These are the most impactful layers for classification tasks.

**Training loop:**
HuggingFace `Trainer` handles the training loop, gradient accumulation, mixed precision
(fp16 on GPU), and evaluation. We evaluate after every epoch and keep the best model
by F1 score.

**MLflow integration:**
- Hyperparams logged at start
- Metrics logged per epoch (automatic via Trainer callbacks)
- Final model registered to MLflow Model Registry

---

## Step 5: Evaluation (evaluator.py)

**What it does:** Three-way comparison on held-out test set.

**Three classifiers:**
1. **Baseline** — vanilla DistilBERT with a random classification head. This shows
   what you get without any fine-tuning. Expected: near-random performance (33% accuracy
   on 3 classes).
2. **Fine-tuned** — our LoRA model. Expected: 70-90% accuracy depending on data quality.
3. **Claude Haiku** — LLM-based classification via structured output. Expected: 60-80%
   accuracy. Good but expensive and slow.

**Metrics:**
- Accuracy and macro F1 (accounts for class imbalance)
- Per-class precision and recall (which urgency levels does each model get wrong?)
- Average latency in milliseconds
- Cost per 1000 classifications (fine-tuned = $0, Claude = ~$0.18)

**Why this comparison matters for interviews:**
It demonstrates that fine-tuning a small model can match or beat an expensive LLM API
for specific, narrow classification tasks. This is exactly the kind of cost/performance
tradeoff that production ML teams care about.

---

## Step 6: Inference (classifier.py)

**What it does:** Loads the fine-tuned model and exposes `run()` for classification.

**Model loading priority:**
1. Explicit local path (for development)
2. MLflow Model Registry (for production)
3. Error (no silent fallback to untrained model)

**Claude fallback:**
If the fine-tuned model isn't available (e.g., first deployment before training),
`run()` falls back to Claude Haiku classification. This means the Streamlit app
always works even without a trained model — it just uses the more expensive path.

**Langfuse tracing:**
`run_observed()` wraps `run()` with `@observe(name="triage_classify")`. This traces
every inference call in Langfuse, including when fallback to Claude happens. Training
metrics stay in MLflow — different tools for different concerns.

---

## Step 7: Guardrails (guardrails.py)

Standard CLAUDE.md guardrails. The PHI redaction stub is present even though this
project doesn't use real patient data — it signals production and compliance instincts
to interviewers.

Input validation catches prompt injection attempts (important because Claude fallback
accepts raw text). Output sanitization strips any HTML that might have leaked through.

---

## Step 8: Streamlit App (app.py)

**Layout:**
- Text input with sample clinical notes (Emergency, Urgent, Routine) for easy demo
- Three-column results showing all three classifiers side by side
- Color-coded urgency badges (red/orange/green)
- Confidence scores and latency for each model

**Expanders (for interviewers):**
- Model details & hyperparameters table
- Dataset composition with actual stats (reads from `data/full.csv` if available)
- Cost comparison table showing the value proposition
- Integration notes explaining how this replaces the intake router's classification agent

**Sidebar:** Tech stack, MLflow link, GitHub link, model directory config.

---

## Key Interview Talking Points

1. **"Why not just use Claude?"** — Fine-tuned model is ~100x cheaper and ~30x faster
   for this specific task. Claude is better for general-purpose reasoning, but narrow
   classification doesn't need it.

2. **"How does LoRA work?"** — Instead of updating all 67M parameters, LoRA adds small
   low-rank matrices (~300K params) to the attention layers. These capture task-specific
   patterns without forgetting pre-trained knowledge. Like adding a thin adapter layer
   on top of a frozen foundation.

3. **"How would you deploy this?"** — Model artifact in S3, loaded by a FastAPI endpoint
   behind an ALB. MLflow Registry tracks which version is production. Langfuse monitors
   inference quality. Canary deployment: route 5% of traffic to the model, compare
   against Claude on the remaining 95%, promote when F1 matches.

4. **"What would you change for production?"** — Expert-annotated labels instead of
   specialty mapping. Calibrated confidence scores. A/B testing framework. HIPAA-compliant
   infrastructure. Drift detection on input distribution.
