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

**Design decision — local MLflow first, then EC2:**
Started with `mlflow ui` locally during development to avoid blocking on infra.
`MLFLOW_TRACKING_URI` is an env var — switching from `http://127.0.0.1:5000` to
the EC2 public IP was a one-line `.env` change, no code changes needed. EC2 runs
on a free-tier t2.micro with SQLite backend and S3 artifact store.

**S3 prefix structure:**
```
medical-triage-classifier/
  datasets/    ← train.csv, val.csv, test.csv
  checkpoints/ ← per-epoch model saves
  models/      ← final registered model artifacts (timestamped + latest)
```

---

## Step 3: Data Preparation (data_prep.py)

**What it does:** Loads MTSamples CSV, labels urgency, splits data, uploads to S3,
logs stats to MLflow.

**Critical pivot — specialty mapping → Sonnet content labeling:**

The initial approach mapped `medical_specialty` to urgency via heuristic (e.g.,
"Orthopedic" → Urgent). This produced 58% accuracy. Investigation revealed the
problem: an ER note about a minor sprain was labeled "Emergency" because of the
department, not the content. The model learned department writing style, not
clinical urgency.

The fix: `load_and_label_with_llm()` sends each note to Claude Sonnet with explicit
urgency definitions. Sonnet reads the content — vital signs, symptoms, acuity — and
assigns the label. This single change improved accuracy from 58% to 74%. The model
architecture, hyperparameters, and training procedure were identical.

**Lesson:** Data quality > model tuning. Fixing the labels had 4x more impact than
any hyperparameter change could have.

**Class distribution after Sonnet labeling:**
- Routine: 59% — most clinical notes describe stable or scheduled care
- Urgent: 31% — timely intervention needed
- Emergency: 10% — acute, life-threatening

This distribution is realistic but imbalanced, which required class weights in
training (see Step 4).

**Split:** 70/15/15 stratified. Stratification ensures each split has the same class
proportions, which is critical for reliable evaluation on imbalanced datasets.

---

## Step 4: Training (trainer.py + notebooks/train_colab.ipynb)

**What it does:** Fine-tunes DistilBERT with PEFT/LoRA on a Colab T4 GPU, logs to
MLflow, saves model to S3.

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

**Class-weighted loss (WeightedTrainer):**
With only 10% Emergency examples, the model initially defaulted to predicting Routine
for everything (which gives ~59% accuracy for free). `WeightedTrainer` subclasses
HuggingFace's `Trainer` and applies inverse-frequency class weights to the cross-entropy
loss. This penalizes mistakes on Emergency 6x more than Routine, forcing the model to
actually learn the minority class patterns.

**Training loop:**
HuggingFace `Trainer` handles the training loop, gradient accumulation, mixed precision
(fp16 on GPU), and evaluation. We evaluate after every epoch and keep the best model
by F1 score.

**Two Colab notebooks:**
- `train_colab.ipynb` — manual CSV upload/download
- `train_colab_s3.ipynb` — pulls data from S3, uploads model to S3 (AWS credentials
  entered via `getpass`, not saved to notebook)

---

## Step 5: Evaluation (evaluator.py)

**What it does:** Three-way comparison on held-out test set.

**Three classifiers:**
1. **Baseline** — vanilla DistilBERT with a randomly initialized classification head.
   Scores ~31% accuracy — confirms it's guessing randomly across 3 classes.
2. **Fine-tuned** — our LoRA model. 74% accuracy, 0.68 F1.
3. **Claude Haiku** — LLM-based classification via structured output. 78% accuracy,
   0.72 F1. Better accuracy but ~19x slower and ~$0.25/1K calls.

**Actual results:**
```
Model             Accuracy   F1 (macro)   Latency (ms)    Cost/1K
-----------------------------------------------------------------
baseline            0.31       0.16           64ms        $0.00
finetuned           0.74       0.68           70ms        $0.00
claude (haiku)      0.78       0.72         1420ms        $0.25
```

**Why this comparison matters for interviews:**
The fine-tuned model is within 4 points of Claude Haiku at zero marginal cost and
19x faster latency. For a narrow, well-defined classification task, fine-tuning
a small model is the right production choice. The intake router currently uses
Sonnet (~$3/1K calls) for classification — this model could replace it as a
drop-in swap.

---

## Step 6: Inference (classifier.py)

**What it does:** Loads the fine-tuned model and exposes `run()` for classification.

**Model loading priority:**
1. Explicit local path (for development)
2. MLflow Model Registry (for production)
3. Claude fallback (if no trained model available)

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

1. **"Why not just use Claude?"** — Fine-tuned model is ~100x cheaper and ~19x faster
   for this specific task. Claude is better for general-purpose reasoning, but narrow
   classification doesn't need it. This is knowledge distillation — teach a small model
   to replicate the large model's decisions.

2. **"How does LoRA work?"** — Instead of updating all 67M parameters, LoRA adds small
   low-rank matrices (~300K params) to the attention layers. These capture task-specific
   patterns without forgetting pre-trained knowledge. Like adding a thin adapter layer
   on top of a frozen foundation.

3. **"What was the biggest improvement?"** — Fixing the labels. Switching from
   specialty-based mapping to Sonnet content-based labeling improved accuracy from
   58% to 74%. No model or hyperparameter changes. Data quality was the bottleneck.

4. **"How would you deploy this?"** — Model artifact in S3, loaded by a FastAPI endpoint
   behind an ALB. MLflow Registry tracks which version is production. Langfuse monitors
   inference quality. Canary deployment: route 5% of traffic to the model, compare
   against Claude on the remaining 95%, promote when F1 matches.

5. **"What would you change for production?"** — Clinician-annotated labels instead of
   LLM labeling. Hyperparameter search via Optuna. Bio_ClinicalBERT as an alternative
   base model. Calibrated confidence scores. A/B testing framework. HIPAA-compliant
   infrastructure. Drift detection on input distribution.
