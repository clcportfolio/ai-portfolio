# Medical Triage Classifier — Technical Overview

## Architecture

```
MTSamples CSV (HuggingFace Hub)
    │
    ▼
fetch_mtsamples.py ── download from HuggingFace ──► data/mtsamples.csv
    │
    ▼
data_prep.py ─────────────────────────────────────────────┐
    │ Claude Sonnet content-based labeling (not specialty) │
    │ Split: 70/15/15 stratified                           │
    ├─► data/train.csv, val.csv, test.csv (local)         │
    └─► S3: medical-triage-classifier/datasets/ ───────────┘
                                                           │
trainer.py (Google Colab T4 GPU) ◄─────────────────────────┘
    │ DistilBERT + PEFT/LoRA (r=16, alpha=32)
    │ Class-weighted loss (WeightedTrainer)
    │ HuggingFace Trainer, 5 epochs
    ├─► S3: model artifacts (timestamped + latest)
    └─► MLflow (EC2): metrics per epoch
                                                           │
evaluator.py ◄─────────────────────────────────────────────┘
    │ 3-way comparison on held-out test set
    │ Baseline (pre-trained) vs Fine-tuned vs Claude Haiku
    ├─► accuracy, F1 (macro), per-class precision/recall
    ├─► latency, cost estimates
    └─► MLflow (EC2): comparison run
                                                           │
classifier.py ◄────────────────────────────────────────────┘
    │ run({"text": str}) -> {"urgency", "confidence", "model"}
    │ Loads from MLflow Registry (or local fallback)
    │ Langfuse @observe for inference tracing
    └─► app.py (Streamlit dashboard)
```

## Model Choice: DistilBERT + LoRA

**Why DistilBERT over Bio_ClinicalBERT?**
- DistilBERT is 40% smaller, trains faster, and is sufficient for 3-class classification
- Bio_ClinicalBERT is tracked as a future MLflow experiment for comparison
- LoRA reduces trainable parameters to ~0.5% of the model (from 67M to ~300K)
- This means training completes in minutes on a free Colab GPU

**LoRA Configuration:**
- `r=16`, `alpha=32` (alpha/r ratio = 2, standard for classification)
- Target modules: `q_lin`, `v_lin` (DistilBERT's attention query and value projections)
- Dropout: 0.1
- No bias adaptation

## Data Pipeline

**Source:** MTSamples via HuggingFace Hub (`harishnair04/mtsamples`) — ~5000 anonymized
medical transcription samples across 40+ specialties.

**Labeling evolution (key design decision):**

The initial approach mapped `medical_specialty` to urgency via heuristic (e.g.,
"Orthopedic" = Urgent, "Emergency Room" = Emergency). This produced 58% accuracy
because the labels didn't reflect clinical content — an ER note about a sprained
ankle was labeled "Emergency" when it was actually routine.

The fix: Claude Sonnet reads each clinical note and assigns urgency based on the
actual content — vital signs, symptoms, acuity, and clinical presentation. This
improved accuracy from 58% to 74% with no other changes. The labeling heuristic
was the bottleneck, not the model architecture or hyperparameters.

Both labeling modes are available in `data_prep.py`:
- Default: `load_and_label_with_llm()` — Sonnet content-based (recommended)
- Flag: `--use-specialty-labels` — original specialty mapping (for comparison)

**Class distribution (Sonnet-labeled):**
- Routine: 59% — most clinical notes describe stable or scheduled care
- Urgent: 31% — timely intervention needed
- Emergency: 10% — acute, life-threatening

**Class-weighted training:**
The imbalanced distribution required class weights to prevent the model from
defaulting to "Routine" for everything. `WeightedTrainer` applies inverse-frequency
weights to the cross-entropy loss, penalizing mistakes on minority classes more
heavily.

**Split:** 70/15/15 stratified by urgency label.

## Evaluation Results

```
Model             Accuracy   F1 (macro)   Latency (ms)    Cost/1K
-----------------------------------------------------------------
baseline            0.31       0.16           64ms        $0.00
finetuned           0.74       0.68           70ms        $0.00
claude (haiku)      0.78       0.72         1420ms        $0.25
```

The fine-tuned model is within 4 points of Claude Haiku at zero cost and 19x faster.
Baseline (~31%) confirms random performance — the randomly initialized classification
head has never been trained, so it guesses.

## Experiment Tracking (MLflow)

- **Tracking server:** EC2 t2.micro (free tier) with SQLite backend
- **Artifact store:** S3 (`medical-triage-classifier/` prefix)
- **Tracking URI:** configurable via `MLFLOW_TRACKING_URI` env var
- **Experiment:** `medical-triage-classifier`
- **Runs:**
  - `data_prep` — dataset stats, class distribution, source breakdown
  - `lora_finetune` — hyperparams, per-epoch metrics, model registration
  - `model_comparison` — 3-way accuracy/F1/latency/cost comparison

## Guardrails

Standard CLAUDE.md guardrails:
- `validate_input`: text type check, 4000 char limit, prompt injection regex
- `sanitize_output`: HTML strip, PHI redaction stub (always present even though
  this project doesn't use real patient data — signals compliance awareness)
- `rate_limit_check`: Redis stub, falls back to allowing all requests

## Langfuse Observability

- `classifier.py:run_observed()` is decorated with `@observe(name="triage_classify")`
- Traces inference calls when the fine-tuned model falls back to Claude
- Training metrics go to MLflow (not Langfuse) — different concerns, different tools

## Integration with Clinical Intake Router

`classifier.py:run()` has the same interface as the intake router's
`classification_agent`:
- Input: `{"text": str}`
- Output: `{"urgency": str, "confidence": float}`

Drop-in replacement path:
1. Import `classifier.run` in the intake router's pipeline
2. Replace `classification_agent.run(state)` call
3. Map `urgency` to the intake router's `UrgencyLevel` enum

Benefits: ~100x cheaper (no API cost), ~19x faster (~70ms vs ~1400ms).
The intake router currently uses Sonnet (~$3/1K calls), making the savings even larger.

## Deployment

- **Demo:** Streamlit Community Cloud (shareable URL)
- **MLflow:** EC2 free tier (t2.micro, Amazon Linux 2023)
- **Training:** Google Colab (free T4 GPU)
- **Artifacts:** S3 (datasets, checkpoints, registered models)
- **Cost:** $0/mo within AWS free tier limits

## Tradeoffs

1. **Sonnet-labeled data vs. expert annotation:** Using an LLM to label training
   data is a form of knowledge distillation — we're teaching a small model to mimic
   a large one. This is effective but means the fine-tuned model inherits Sonnet's
   biases. Production would use clinician-annotated data as ground truth.

2. **LoRA vs. full fine-tuning:** LoRA trains ~0.5% of parameters. This is faster
   and cheaper but may underperform full fine-tuning on edge cases. MLflow makes
   it easy to compare both approaches.

3. **Single hyperparameter configuration:** The current model uses one set of
   hyperparameters (lr=3e-4, r=16, alpha=32, 5 epochs). Automated hyperparameter
   search via Optuna or Ray Tune would likely improve results by 3-5%, but the
   primary bottleneck remains data quality, not model tuning.

4. **Class imbalance:** Only 10% of Sonnet-labeled notes are Emergency. Class
   weights help but don't fully solve the problem — the model has fewer Emergency
   examples to learn from. More Emergency training data (real or synthetic) would
   improve recall on the most critical class.
