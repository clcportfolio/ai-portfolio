# Clinical Trial Eligibility Screener

## What This Tool Does

This tool helps medical staff quickly determine if patients might qualify for clinical trials. Instead of manually reading through complex trial requirements and patient records, you simply paste the trial criteria and patient information into the system. The AI reads both documents, compares them carefully, and tells you whether the patient is eligible, likely not eligible, or needs a closer human review.

The system breaks down trial requirements into individual criteria, then checks each one against the patient's medical history. It explains its reasoning in plain English, so you can understand exactly why it reached its conclusion. This turns what used to be a time-consuming manual process into a quick, reliable screening step.

## Why It Matters

Clinical trial coordinators often spend hours reviewing patient files against complex eligibility criteria. This manual process is slow, prone to human error, and can delay getting patients into potentially life-saving treatments. Small clinics especially struggle because they lack dedicated research staff to handle this screening.

This tool speeds up the initial screening process dramatically. It catches obvious matches and mismatches right away, letting coordinators focus their time on borderline cases that truly need human judgment. Patients get faster answers about trial opportunities, and clinics can participate in more research studies without overwhelming their staff.

## What You See When You Run It

The screen shows two large text boxes. You paste the clinical trial eligibility requirements into the first box and the patient's medical summary into the second. When you click "Run Eligibility Check," the system processes both documents.

Within moments, a colored card appears showing the verdict. Green means "Eligible" - the patient meets the requirements. Red means "Likely Ineligible" - they don't qualify based on the information provided. Yellow means "Needs Review" - there isn't enough information to make a clear decision, or the case is complex enough to require human judgment.

Below the verdict, you can expand a detailed breakdown that shows how each individual criterion was evaluated. The system explains in simple terms why the patient does or doesn't meet each requirement, pointing to specific information from their medical summary. A sidebar provides background information about the project and links to additional resources.

## Who Built This and How

This tool was created using artificial intelligence technology that can read and understand medical text. The system uses three specialized components working together: one that identifies individual trial requirements, another that compares each requirement to the patient information, and a third that combines all the evaluations into a final recommendation.

The interface was built using Streamlit, a platform that makes it easy to create user-friendly web applications. All the code is available on GitHub for transparency and collaboration with other healthcare technology developers. The system is designed to assist medical professionals, not replace their expertise - the final enrollment decisions always remain with qualified clinical staff.