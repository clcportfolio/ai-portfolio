# Medical Triage Classifier — Technical Overview

## Architecture

```
MTSamples CSV (local)
    │
    ▼
data_prep.py ─────────────────────────────────────────────┐
    │ Load, label, synthetic augment (Claude Haiku)        │
    │ Split: 70/15/15 stratified                           │
    ├─► data/train.csv, val.csv, test.csv (local)         │
    └─► S3: medical-triage-classifier/datasets/ ───────────┘
                                                           │
trainer.py (Colab/GPU) ◄──────────────────────────────────┘
    │ DistilBERT + PEFT/LoRA (r=16, alpha=32)
    │ HuggingFace Trainer, 5 epochs
    ├─► MLflow: metrics per epoch
    ├─► S3: checkpoints per epoch
    └─► MLflow Model Registry: final model
                                                           │
evaluator.py ◄─────────────────────────────────────────────┘
    │ 3-way comparison on test set
    │ Baseline (pre-trained) vs Fine-tuned vs Claude
    ├─► accuracy, F1 (macro), per-class precision/recall
    ├─► latency, cost estimates
    └─► MLflow: comparison run
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

**Source:** MTSamples (mtsamples.com) — ~5000 anonymized medical transcription samples
across 40+ specialties.

**Labeling strategy:** Map `medical_specialty` to urgency via clinical heuristic:
- Emergency: ER reports, cardiovascular, neurosurgery, surgery, neurology
- Urgent: orthopedic, gastro, urology, hematology-oncology, nephrology, OB/GYN, etc.
- Routine: general medicine, ophthalmology, radiology, dermatology, dentistry, etc.

**Augmentation:** Claude Haiku generates synthetic clinical notes for underrepresented
classes. Target: ~300 per class. Separate prompts for each urgency level to avoid
ambiguous outputs.

**Split:** 70/15/15 stratified by urgency label.

## Experiment Tracking (MLflow)

- **Tracking URI:** configurable via `MLFLOW_TRACKING_URI` env var
- **Local development:** `mlflow ui` on localhost:5000
- **Production:** EC2 t2.micro with SQLite backend, S3 artifact store
- **Experiment:** `medical-triage-classifier`
- **Runs:**
  - `data_prep` — dataset stats, class distribution, synthetic ratio
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

Benefits: ~100x cheaper (no API cost), ~30x faster (~10ms vs ~400ms).

## Deployment

- **Demo:** Streamlit Community Cloud (shareable URL)
- **MLflow:** EC2 free tier (t2.micro, Amazon Linux 2023)
- **Artifacts:** S3 (datasets, checkpoints, registered models)
- **Cost:** $0/mo within AWS free tier limits

## Tradeoffs

1. **Specialty-based labeling vs. expert annotation:** Mapping specialties to urgency
   is a heuristic, not ground truth. A radiologist's note could describe an emergency
   finding. This is acceptable for a portfolio demo but would need expert review for
   production.

2. **LoRA vs. full fine-tuning:** LoRA trains ~0.5% of parameters. This is faster
   and cheaper but may underperform full fine-tuning on edge cases. MLflow makes
   it easy to compare both approaches.

3. **Claude Haiku for synthetic data:** Generated examples may have distributional
   drift from real clinical notes. The synthetic ratio is tracked in MLflow so its
   impact on model performance can be measured.
