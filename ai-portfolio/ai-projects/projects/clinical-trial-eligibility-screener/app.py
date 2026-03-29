import streamlit as st
from dotenv import load_dotenv
import pipeline

load_dotenv()

st.set_page_config(page_title="Clinical Trial Eligibility Screener", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Clinical Trial Eligibility Screener")
    st.markdown("**What it does:** Automate clinical trial eligibility screening by evaluating patient summaries against trial criteria and providing clear verdicts with reasoning to help coordinators make informed decisions.")
    st.markdown("**Tech stack:**")
    st.markdown("- LangChain + Claude (Anthropic)")
    st.markdown("- Langfuse observability")
    st.markdown("- Streamlit")
    st.markdown("[GitHub Repo](https://github.com/your-repo)")

st.title("Clinical Trial Eligibility Screener")
st.caption("Automated screening to help coordinators make informed eligibility decisions.")

# Input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trial Criteria")
    trial_criteria = st.text_area(
        "Enter the clinical trial inclusion/exclusion criteria:",
        height=300,
        placeholder="Example:\nInclusion:\n- Age 18-65\n- Diagnosed with Type 2 diabetes\n- HbA1c > 7%\n\nExclusion:\n- Pregnant or nursing\n- History of diabetic ketoacidosis\n- Current insulin therapy"
    )

with col2:
    st.subheader("Patient Summary")
    patient_summary = st.text_area(
        "Enter the patient summary for evaluation:",
        height=300,
        placeholder="Example:\n45-year-old female with Type 2 diabetes diagnosed 3 years ago. Current HbA1c: 8.2%. Takes metformin 1000mg BID. No history of DKA. Not pregnant. BMI: 32. Recent labs show normal kidney function."
    )

if st.button("Run Eligibility Check", type="primary"):
    if not trial_criteria.strip() or not patient_summary.strip():
        st.warning("Please provide both trial criteria and patient summary.")
    else:
        with st.spinner("Running eligibility screening..."):
            try:
                result = pipeline.run({
                    "trial_criteria": trial_criteria,
                    "patient_summary": patient_summary
                })
            except ValueError as e:
                st.error(f"Input validation failed: {e}")
                st.stop()

        # Show verdict with color-coded indicator
        # VerdictResult fields: eligibility_status, confidence_score, summary, key_factors, next_steps
        verdict_output = result.get("verdict_agent_output") or {}
        if verdict_output:
            status = verdict_output.get("eligibility_status", "UNKNOWN")
            confidence = verdict_output.get("confidence_score", 0.0)
            summary = verdict_output.get("summary", "No summary provided.")
            key_factors = verdict_output.get("key_factors", [])
            next_steps = verdict_output.get("next_steps", "")

            st.subheader("Eligibility Verdict")

            if status == "ELIGIBLE":
                st.success(f"✅ **Eligible** (Confidence: {confidence:.1%})")
            elif status == "INELIGIBLE":
                st.error(f"❌ **Likely Ineligible** (Confidence: {confidence:.1%})")
            else:
                st.warning(f"⚠️ **Needs Review** (Confidence: {confidence:.1%})")

            st.write("**Summary:**", summary)
            if key_factors:
                st.write("**Key factors:**")
                for f in key_factors:
                    st.write(f"- {f}")
            if next_steps:
                st.write("**Recommended next steps:**", next_steps)
        else:
            st.error("No verdict generated.")

        # Show detailed per-criterion evaluation breakdown
        # CriterionEvaluation fields: criterion_id, criterion_text, meets_criterion, confidence, reasoning, relevant_patient_info
        evaluation_output = result.get("evaluation_agent_output") or {}
        if evaluation_output and evaluation_output.get("evaluations"):
            with st.expander("Per-Criterion Evaluation Breakdown", expanded=True):
                evaluations = evaluation_output.get("evaluations", [])
                for eval_item in evaluations:
                    criterion_text = eval_item.get("criterion_text", "—")
                    meets = eval_item.get("meets_criterion", None)
                    conf = eval_item.get("confidence", 0.0)
                    reasoning = eval_item.get("reasoning", "")
                    patient_info = eval_item.get("relevant_patient_info", "")

                    if meets is True:
                        st.success(f"✅ **{criterion_text}** (confidence: {conf:.0%})")
                    elif meets is False:
                        st.error(f"❌ **{criterion_text}** (confidence: {conf:.0%})")
                    else:
                        st.warning(f"⚠️ **{criterion_text}** (confidence: {conf:.0%})")

                    if reasoning:
                        st.write(f"*Reasoning:* {reasoning}")
                    if patient_info:
                        st.write(f"*Relevant info:* {patient_info}")
                    st.divider()

        # Show intermediate agent outputs
        with st.expander("Criteria Agent Output", expanded=False):
            st.json(result.get("criteria_agent_output", {}))

        with st.expander("Evaluation Agent Output", expanded=False):
            st.json(result.get("evaluation_agent_output", {}))

        with st.expander("Verdict Agent Output", expanded=False):
            st.json(result.get("verdict_agent_output", {}))

        if result.get("errors"):
            with st.expander("Pipeline Warnings"):
                for e in result["errors"]:
                    st.warning(e)