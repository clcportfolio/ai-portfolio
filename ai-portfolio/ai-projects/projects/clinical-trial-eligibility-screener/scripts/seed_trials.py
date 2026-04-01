"""
seed_trials.py — Seed three sample clinical trials into the database.

Run this once to populate the trial dropdown in the Streamlit UI before demoing
or before seeding synthetic patient data.

Usage:
    python scripts/seed_trials.py               # insert all 3 trials
    python scripts/seed_trials.py --dry-run     # print trials without inserting
    python scripts/seed_trials.py --list        # list trials currently in DB

After running this script, use seed_synthetic_data.py to populate screenings:
    python scripts/seed_synthetic_data.py --trial-id 1
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# ── Sample trials ─────────────────────────────────────────────────────────────

SAMPLE_TRIALS = [
    {
        "name": "DIAB-METFORMIN-PILOT-2025",
        "criteria_text": """Inclusion Criteria:
- Age 30–70 years at time of enrollment
- Diagnosed with Type 2 Diabetes Mellitus for at least 6 months
- HbA1c between 7.0% and 10.0% (most recent value within 3 months)
- Currently on metformin monotherapy (≥ 500 mg/day) for at least 3 months
- BMI between 25 and 40 kg/m²
- eGFR ≥ 45 mL/min/1.73m² (adequate renal function for metformin)
- Willing and able to provide written informed consent

Exclusion Criteria:
- Type 1 Diabetes Mellitus or latent autoimmune diabetes in adults (LADA)
- Currently pregnant, breastfeeding, or planning pregnancy during trial period
- History of diabetic ketoacidosis (DKA) within the past 12 months
- Currently on insulin therapy of any kind
- Currently on any GLP-1 receptor agonist or SGLT-2 inhibitor
- Severe hepatic impairment (ALT or AST > 3× upper limit of normal)
- Active malignancy or receiving chemotherapy within the past 2 years
- History of acute or chronic pancreatitis
- eGFR < 45 mL/min/1.73m² (contraindication to metformin)
- Uncontrolled thyroid disease (TSH outside 0.5–4.5 mIU/L)
- Current participation in another interventional clinical trial""",
    },
    {
        "name": "HYPER-ACE-COMBO-2024",
        "criteria_text": """Inclusion Criteria:
- Age 40–80 years
- Diagnosed with essential (primary) hypertension for at least 12 months
- Systolic blood pressure (SBP) 140–179 mmHg OR diastolic blood pressure (DBP) 90–109 mmHg on ≥ 2 separate measurements
- Currently on ACE inhibitor monotherapy at stable dose for ≥ 8 weeks
- Willing to attend all scheduled visits and complete study procedures

Exclusion Criteria:
- Secondary hypertension (renovascular, primary aldosteronism, pheochromocytoma, etc.)
- Severe hypertension: SBP ≥ 180 mmHg or DBP ≥ 110 mmHg
- History of myocardial infarction, stroke, or TIA within the past 6 months
- Heart failure with reduced ejection fraction (EF < 40%)
- Significant renal impairment: eGFR < 30 mL/min/1.73m² or serum creatinine > 2.0 mg/dL
- Serum potassium > 5.5 mEq/L at screening
- Known hypersensitivity or angioedema related to ACE inhibitors or ARBs
- Current use of potassium-sparing diuretics, potassium supplements, or NSAIDs
- Pregnant, breastfeeding, or women of childbearing potential not using effective contraception
- Uncontrolled diabetes (HbA1c > 10%) at screening
- Active liver disease or ALT/AST > 2× upper limit of normal
- Current or recent (< 30 days) participation in another clinical trial""",
    },
    {
        "name": "CARDIO-STATIN-RRX-2025",
        "criteria_text": """Inclusion Criteria:
- Age 45–75 years
- Established cardiovascular disease, defined as ≥ 1 of: prior MI, prior ischemic stroke, symptomatic peripheral arterial disease, or prior coronary revascularization
- LDL-C ≥ 70 mg/dL (1.8 mmol/L) despite being on maximally tolerated statin therapy for ≥ 3 months
- Currently on a high-intensity statin (atorvastatin ≥ 40 mg/day or rosuvastatin ≥ 20 mg/day) OR documented statin intolerance to ≥ 2 different statins
- Willing and able to self-administer subcutaneous injections

Exclusion Criteria:
- Active liver disease or unexplained persistent elevations in hepatic transaminases (ALT or AST > 3× ULN)
- Creatine kinase (CK) > 3× ULN without a clear non-drug cause
- Severe renal impairment: eGFR < 30 mL/min/1.73m²
- Uncontrolled hypothyroidism (TSH > 10 mIU/L)
- Active or history of rhabdomyolysis on statin therapy
- Currently taking fibrates (other than fenofibrate), niacin > 1 g/day, or cyclosporine
- Known hypersensitivity to PCSK9 inhibitors or any component of the study drug
- Acute coronary syndrome, stroke, TIA, or revascularization within the past 4 weeks
- Planned cardiac surgery or revascularization within 3 months of screening
- Life expectancy < 12 months due to non-cardiovascular disease
- Pregnancy, breastfeeding, or inadequate contraception in women of childbearing potential
- Current enrollment in another investigational drug study""",
    },
]


def seed_trials(dry_run: bool = False) -> None:
    from storage.db_client import init_db, insert_trial, trial_name_exists

    if not dry_run:
        init_db()

    print(f"{'[DRY RUN] ' if dry_run else ''}Seeding {len(SAMPLE_TRIALS)} sample trial(s)...\n")

    for trial in SAMPLE_TRIALS:
        name = trial["name"]
        criteria = trial["criteria_text"]
        line_count = len([l for l in criteria.splitlines() if l.strip()])

        if dry_run:
            print(f"  Would insert: {name!r} ({line_count} criteria lines)")
            continue

        if trial_name_exists(name):
            print(f"  SKIP  {name!r} (already exists)")
            continue

        row = insert_trial(name=name, criteria_text=criteria)
        print(f"  OK    {name!r} → id={row['id']} ({line_count} criteria lines)")

    if not dry_run:
        print(
            "\nTrials seeded. You can now seed synthetic patients:\n"
            "  python scripts/seed_synthetic_data.py --trial-id 1\n"
            "  python scripts/seed_synthetic_data.py --trial-id 2\n"
            "  python scripts/seed_synthetic_data.py --trial-id 3"
        )


def list_trials() -> None:
    from storage.db_client import init_db, get_trials

    init_db()
    trials = get_trials()
    if not trials:
        print("No trials in database.")
        return
    print(f"{'ID':<5} {'Name':<40} {'Cached criteria'}")
    print("-" * 65)
    for t in trials:
        cached = "yes" if t.get("structured_criteria") else "no"
        print(f"{t['id']:<5} {t['name']:<40} {cached}")


def main():
    parser = argparse.ArgumentParser(description="Seed sample clinical trials into the database.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing to DB.")
    parser.add_argument("--list", action="store_true", help="List all trials currently in the database.")
    args = parser.parse_args()

    if args.list:
        list_trials()
    else:
        seed_trials(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
