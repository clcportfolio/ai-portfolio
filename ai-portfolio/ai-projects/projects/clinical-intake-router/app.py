"""
Streamlit UI — Clinical Intake Router
Demo interface: paste or upload an intake form → routing card + agent step expanders.
"""

import sys
import os
import io

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Intake Router",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Clinical Intake Router")
    st.markdown(
        """
A healthcare staff tool that reads a clinical intake form and routes the
patient to the right department at the right urgency level — instantly.

**Pipeline**
1. **Extraction** — pulls structured fields from free-text
2. **Classification** — assigns urgency + department
3. **Routing** — generates plain-English instructions

**Tech Stack**
- LangChain + Claude Sonnet (Anthropic)
- Langfuse observability
- Streamlit demo
- Pydantic structured output
        """
    )
    st.divider()
    st.markdown(
        "[GitHub](https://github.com/codyculver/ai-portfolio)"
        " · Built with Claude Code"
    )

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Clinical Intake Router")
st.caption("Paste or upload an intake form. Get an immediate routing decision.")

col_input, col_output = st.columns([1, 1], gap="large")

# ── Left column: Input ────────────────────────────────────────────────────────
with col_input:
    st.subheader("Intake Form")

    uploaded_file = st.file_uploader(
        "Upload intake form (.txt or .pdf)",
        type=["txt", "pdf"],
        help="Accepted formats: plain text (.txt) or PDF (.pdf)",
    )

    intake_text = st.text_area(
        "— or paste intake form text here —",
        height=300,
        placeholder=(
            "Patient: Jane Smith, 45 y/o\n"
            "Chief Complaint: Sudden onset chest pain radiating to jaw...\n"
            "PMH: Hypertension, type 2 diabetes\n"
            "Medications: metformin, lisinopril\n"
            "Allergies: aspirin\n"
            "Insurance: Aetna PPO\n"
            "Referred by: ER walk-in"
        ),
    )

    run_button = st.button("Route This Intake", type="primary", use_container_width=True)

# ── Right column: Output ──────────────────────────────────────────────────────
with col_output:
    st.subheader("Routing Decision")

    if run_button:
        # Resolve input text from upload or text area
        final_text = ""

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(uploaded_file.read()))
                    final_text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    ).strip()
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
                    st.stop()
            else:
                final_text = uploaded_file.read().decode("utf-8", errors="replace").strip()
        elif intake_text.strip():
            final_text = intake_text.strip()

        if not final_text:
            st.warning("Please paste intake text or upload a file before routing.")
            st.stop()

        # Run the pipeline
        with st.spinner("Running pipeline — extracting, classifying, routing..."):
            try:
                from pipeline import run as pipeline_run
                result = pipeline_run({"text": final_text})
            except ValueError as e:
                st.error(f"Input validation failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        # ── Routing card ──────────────────────────────────────────────────────
        routing = result.get("routing_output") or {}
        urgency = routing.get("urgency_level", "Unknown")
        department = routing.get("department", "Unknown")
        summary = routing.get("routing_summary", "")
        next_steps = routing.get("recommended_next_steps", [])
        follow_ups = routing.get("follow_up_actions", [])
        response_time = routing.get("estimated_response_time")

        # Urgency color mapping
        # Note: top level is "Emergent" (triage term) to avoid collision with "Emergency" department
        urgency_colors = {
            "Emergent": "#FF4B4B",
            "Urgent": "#FFA500",
            "Routine": "#21C354",
        }
        urgency_color = urgency_colors.get(urgency, "#888888")
        urgency_emoji = {"Emergent": "🔴", "Urgent": "🟡", "Routine": "🟢"}.get(urgency, "⚪")

        # Render routing card
        st.markdown(
            f"""
<div style="border:2px solid {urgency_color}; border-radius:8px; padding:16px; margin-bottom:12px;">
  <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;">
    <div>
      <div style="font-size:0.72em; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:2px;">Urgency</div>
      <div style="font-size:1.25em; font-weight:700; color:{urgency_color};">{urgency_emoji} {urgency}</div>
    </div>
    <div>
      <div style="font-size:0.72em; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:2px;">Department</div>
      <div style="font-size:1.25em; font-weight:700;">{department}</div>
    </div>
    {"<div><div style='font-size:0.72em; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:2px;'>Expected Response</div><div style='font-size:1.0em;'>" + response_time + "</div></div>" if response_time else ""}
  </div>
  <p style="margin:12px 0 0 0; border-top:1px solid #e0e0e022; padding-top:10px;">{summary}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Next steps
        if next_steps:
            st.markdown("**Recommended Next Steps**")
            for i, step in enumerate(next_steps, 1):
                st.markdown(f"{i}. {step}")

        # Follow-up actions
        if follow_ups:
            st.markdown("**Follow-up Actions**")
            for action in follow_ups:
                st.markdown(f"- {action}")

        # Non-fatal errors
        if result.get("errors"):
            with st.expander("Pipeline warnings", expanded=False):
                for err in result["errors"]:
                    st.warning(err)

        # ── Agent step expanders ──────────────────────────────────────────────
        st.divider()

        # Extraction output
        extraction = result.get("extraction_output") or {}
        with st.expander("Extracted Fields (extraction_agent)", expanded=False):
            if extraction:
                field_map = {
                    "Patient Name": extraction.get("patient_name"),
                    "Age": extraction.get("age"),
                    "Date of Birth": extraction.get("date_of_birth"),
                    "Chief Complaint": extraction.get("chief_complaint"),
                    "Symptoms": ", ".join(extraction.get("symptoms", [])) or None,
                    "Medical History": ", ".join(extraction.get("medical_history", [])) or None,
                    "Current Medications": ", ".join(extraction.get("current_medications", [])) or None,
                    "Allergies": ", ".join(extraction.get("allergies", [])) or None,
                    "Insurance": extraction.get("insurance"),
                    "Referral Source": extraction.get("referral_source"),
                    "Additional Notes": extraction.get("additional_notes"),
                }
                for label, value in field_map.items():
                    if value:
                        st.markdown(f"**{label}:** {value}")
            else:
                st.info("No extraction output available.")

        # Classification output
        classification = result.get("classification_output") or {}
        with st.expander("Classification Reasoning (classification_agent)", expanded=False):
            if classification:
                conf = classification.get("confidence", 0)
                st.markdown(f"**Urgency:** {classification.get('urgency_level', 'N/A')}")
                st.markdown(f"**Department:** {classification.get('department', 'N/A')}")
                st.markdown(f"**Confidence:** {conf:.0%}")
                st.markdown(f"**Reasoning:** {classification.get('classification_reasoning', '')}")
                red_flags = classification.get("red_flags", [])
                if red_flags:
                    st.markdown("**Red Flags:**")
                    for flag in red_flags:
                        st.markdown(f"- {flag}")
            else:
                st.info("No classification output available.")

    else:
        st.info("Enter an intake form on the left and click **Route This Intake** to begin.")
