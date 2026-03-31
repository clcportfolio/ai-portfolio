# Clinical Intake Router — What It Does and Why It Matters

## The Problem It Solves

When a patient arrives at a healthcare facility or calls in to schedule care, a staff member
typically reads through the intake form by hand — the patient's name, their reason for visiting,
their medical history, their insurance — and then decides where to send them and how quickly.

For routine cases, this works fine. For urgent or emergency presentations, every minute counts.
And even for routine cases, inconsistent routing decisions can mean a patient ends up waiting
in the wrong department or gets triaged at the wrong urgency level.

The Clinical Intake Router automates the hardest part of that decision: reading the form,
understanding what it means clinically, and telling staff exactly where to send the patient
and what to do next.

## What You Do

1. **Paste or upload the intake form.** You can copy and paste text directly into the tool,
   or upload a `.txt` or `.pdf` file of the form. The tool accepts forms in any free-text
   format — it doesn't require a specific template.

2. **Click "Route This Intake."** The tool reads the form, extracts every relevant detail,
   evaluates the urgency, and determines the best department.

3. **Read the routing card.** Within seconds, you see a color-coded result:
   - **Red** = Emergency — act immediately
   - **Yellow** = Urgent — act within hours
   - **Green** = Routine — schedule normally

   The card also tells you exactly which department the patient should go to and gives you
   a step-by-step list of actions to take right now.

   *(Note: the top urgency level is labeled **Emergent** rather than Emergency to avoid
   confusion — a patient can be routed to the Emergency department at Urgent priority,
   or to Cardiology at Emergent priority. Keeping the two words distinct prevents
   ambiguity in the routing card.)*

## What the Tool Does Behind the Scenes

The tool uses three specialized AI steps, each focused on a specific task:

1. **Reading the form** — The tool picks out every key piece of information: patient name,
   age, chief complaint, symptoms, medical history, medications, allergies, insurance,
   and referral source. It reads free text the same way a trained coordinator would.

2. **Evaluating the situation** — The tool looks at the full clinical picture and assigns
   an urgency level. It doesn't use a simple keyword list. It reasons about the combination
   of symptoms, age, history, and risk factors to make a judgment about how quickly this
   patient needs to be seen and by whom.

3. **Writing the routing instructions** — The tool produces a plain-English summary
   written specifically for non-clinical staff. It tells you what to do, not just what
   the diagnosis might be.

## What This Is Not

This tool is a **routing assistant**, not a diagnostic tool. It does not diagnose patients,
prescribe treatment, or replace clinical judgment. It helps the right person see the patient
faster by getting them to the right place.

All routing decisions should be reviewed by a qualified clinician for emergency and urgent
cases before action is taken.

## Why It Matters for M3's Businesses

Organizations like Wake Research (clinical trial coordination), PracticeMatch (physician
placement), and The Medicus Firm (healthcare staffing) all deal with high-volume intake and
matching workflows. Automating the initial read and route of incoming documents — whether
patient intake, physician credentials, or trial eligibility — reduces time-to-decision and
reduces manual burden on coordinators.

This tool is a direct proof of concept for that class of automation.
