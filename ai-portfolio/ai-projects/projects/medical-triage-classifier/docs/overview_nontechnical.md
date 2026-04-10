# Medical Triage Classifier — Overview for Stakeholders

## What It Does

This tool reads a short clinical note — the kind a nurse or intake coordinator writes
when a patient arrives — and instantly tells you how urgent the case is:
**Routine**, **Urgent**, or **Emergency**.

Think of it as a second pair of eyes that helps clinical staff prioritize patients
faster, especially during high-volume periods when triage decisions need to happen
quickly and consistently.

## Why It Matters

- **Speed:** The classifier returns a result in under 15 milliseconds — fast enough
  to run on every intake form without slowing anyone down.
- **Consistency:** Human triage is subject to fatigue, bias, and variation between
  shifts. This model applies the same criteria every time.
- **Cost:** Once trained, running the model costs nothing per classification.
  The alternative (sending every note to an AI service like Claude) costs money on
  every single request.

## What You See When You Run It

A simple web page where you paste a clinical note and click "Classify." You'll see
three results side by side:

1. **Fine-tuned model** — the one we trained specifically for this task
2. **Baseline model** — a general-purpose model with no triage training (for comparison)
3. **Claude AI** — a commercial AI service (for comparison)

This three-way comparison makes it easy to see that the fine-tuned model matches
or beats the expensive AI service while being faster and free to run.

## Who This Is For

- **Clinic operations managers** evaluating AI triage tools
- **Clinical coordinators** who want to understand what automated triage looks like
- **Technical interviewers** who want to see a real ML training and deployment project

## What It Does NOT Do

- This is a demonstration, not a production medical device
- It does not replace clinical judgment — it supports it
- It does not process real patient data (all examples are fictional or anonymized)
- It has no access to patient records or electronic health systems
