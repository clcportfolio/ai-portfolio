"""
Streamlit UI — Clinical Trial Eligibility Screener

Tab 1 — Eligibility Check:
  - Select a stored trial from a dropdown, or paste custom criteria
  - Optional "Save as new trial" so custom criteria become reusable
  - Enter patient summary → Run → verdict card + per-criterion breakdown
  - Confidence threshold slider: verdicts below the threshold are shown as NEEDS_REVIEW

Tab 2 — Analytics:
  - Select a stored trial
  - 6 Plotly charts: status distribution, confidence histogram, volume over time,
    criteria pass rate, synthetic vs real mix, avg confidence by status
  - Confidence threshold slider filters which screenings count as reliable
  - "Generate synthetic patients" button seeds the analytics with pre-built profiles
"""

import asyncio
import logging

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trial Eligibility Screener",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("last_result", None),
    ("db_available", None),    # None = not checked yet
    ("trials_cache", None),
    ("seed_just_completed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_available() -> bool:
    """Check whether Supabase is configured and reachable. Only caches True — failures
    are retried on every render so a paused/recovering project reconnects automatically."""
    if st.session_state["db_available"] is True:
        return True
    try:
        import os
        if not os.getenv("SUPABASE_DB_URI"):
            return False
        from storage.db_client import init_db
        init_db()
        st.session_state["db_available"] = True
        return True
    except Exception:
        return False


def _load_trials() -> list[dict]:
    """Load all stored trials from DB, with simple caching in session_state."""
    if not _db_available():
        return []
    try:
        from storage.db_client import get_trials
        trials = get_trials()
        st.session_state["trials_cache"] = trials
        return trials
    except Exception as e:
        st.warning(f"Could not load trials from database: {e}")
        return []


def _load_screenings(trial_id: int) -> list[dict]:
    if not _db_available():
        return []
    try:
        from storage.db_client import get_screenings_for_trial
        return get_screenings_for_trial(trial_id)
    except Exception as e:
        st.warning(f"Could not load screenings: {e}")
        return []


def _get_stats(trial_id: int) -> dict:
    if not _db_available():
        return {}
    try:
        from storage.db_client import get_screening_stats
        return get_screening_stats(trial_id)
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Clinical Trial Eligibility Screener")
    st.markdown(
        "**What it does:** Automates clinical trial eligibility screening — "
        "evaluates a patient summary against trial criteria and returns a "
        "plain-English verdict (Eligible / Ineligible / Needs Review) with "
        "per-criterion reasoning."
    )
    st.markdown("**Tech stack:**")
    st.markdown("- LangChain + Claude (Anthropic)")
    st.markdown("- Parallel async evaluation (`asyncio.gather`)")
    st.markdown("- Langfuse observability")
    st.markdown("- Supabase PostgreSQL")
    st.markdown("- Streamlit")
    st.markdown("[GitHub Repo](https://github.com/clcportfolio/ai-portfolio)")

    st.divider()
    db_ok = _db_available()
    if db_ok:
        st.success("Database connected")
    else:
        st.warning("No database (SUPABASE_DB_URI not set). Stored trials unavailable.")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Eligibility Check", "Analytics"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ELIGIBILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Eligibility Check")
    st.caption("Evaluate a patient against a trial's inclusion/exclusion criteria.")

    # ── Trial source ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Trial Criteria")

        trials = _load_trials()
        trial_options = {t["name"]: t for t in trials}
        CUSTOM_LABEL = "Custom (paste criteria below)"

        if trial_options:
            selected_label = st.selectbox(
                "Select a stored trial or enter custom criteria:",
                options=[CUSTOM_LABEL] + list(trial_options.keys()),
                index=0,
            )
        else:
            selected_label = CUSTOM_LABEL
            if db_ok:
                st.info("No trials saved yet. Paste criteria below to screen ad-hoc or save a new trial.")

        is_custom = selected_label == CUSTOM_LABEL
        selected_trial = trial_options.get(selected_label)

        if is_custom:
            trial_criteria_text = st.text_area(
                "Paste inclusion/exclusion criteria:",
                height=220,
                placeholder=(
                    "Inclusion:\n"
                    "- Age 18-65\n"
                    "- Diagnosed with Type 2 diabetes\n"
                    "- HbA1c > 7%\n\n"
                    "Exclusion:\n"
                    "- Pregnant or nursing\n"
                    "- History of diabetic ketoacidosis\n"
                    "- Current insulin therapy"
                ),
                key="criteria_text_input",
            )
            # Optional: save as a new trial
            if db_ok:
                save_trial = st.checkbox("Save as a new trial for future reuse")
                if save_trial:
                    trial_name_input = st.text_input(
                        "Trial name (must be unique):",
                        placeholder="e.g. DIABETES-PILOT-2025",
                        key="new_trial_name",
                    )
                else:
                    trial_name_input = ""
            else:
                save_trial = False
                trial_name_input = ""
        else:
            # Show stored criteria as read-only
            trial_criteria_text = selected_trial["criteria_text"]
            st.text_area(
                "Stored criteria (read-only):",
                value=trial_criteria_text,
                height=220,
                disabled=True,
            )
            cached = selected_trial.get("structured_criteria") is not None
            if cached:
                st.caption("Criteria already extracted — criteria agent will be skipped.")
            save_trial = False
            trial_name_input = ""

    with col_right:
        st.subheader("Patient Summary")
        patient_summary = st.text_area(
            "Enter the patient summary for evaluation:",
            height=220,
            placeholder=(
                "45-year-old female with Type 2 diabetes diagnosed 3 years ago. "
                "Current HbA1c: 8.2%. Takes metformin 1000mg BID. No history of DKA. "
                "Not pregnant. BMI: 32. Recent labs show normal kidney function."
            ),
            key="patient_summary_input",
        )

    # ── Confidence threshold ──────────────────────────────────────────────────
    confidence_threshold_pct = st.slider(
        "Confidence threshold — verdicts below this level are shown as Needs Review",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        format="%d%%",
        key="confidence_threshold_tab1",
    )
    confidence_threshold = confidence_threshold_pct / 100

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("Run Eligibility Check", type="primary"):
        if not trial_criteria_text or not trial_criteria_text.strip():
            st.warning("Please provide trial criteria (either select a stored trial or paste custom criteria).")
        elif not patient_summary or not patient_summary.strip():
            st.warning("Please enter a patient summary.")
        elif save_trial and not trial_name_input.strip():
            st.warning("Please enter a trial name, or uncheck 'Save as a new trial'.")
        else:
            with st.spinner("Running eligibility screening..."):
                try:
                    import pipeline

                    pipeline_input: dict = {"patient_summary": patient_summary}

                    if is_custom:
                        pipeline_input["trial_criteria"] = trial_criteria_text
                        if save_trial and trial_name_input.strip():
                            pipeline_input["trial_name"] = trial_name_input.strip()
                    else:
                        pipeline_input["trial_id"] = selected_trial["id"]

                    result = asyncio.run(pipeline.run(pipeline_input))
                    st.session_state["last_result"] = result

                    # Refresh trials cache in case a new trial was just saved
                    if save_trial:
                        st.session_state["trials_cache"] = None

                except ValueError as e:
                    st.error(f"Input validation failed: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    logger.exception("Pipeline error")
                    st.stop()

    # ── Results display ───────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        verdict_output = result.get("verdict_agent_output") or {}
        if verdict_output:
            raw_status = verdict_output.get("eligibility_status", "NEEDS_REVIEW")
            confidence = verdict_output.get("confidence_score", 0.0)
            summary = verdict_output.get("summary", "No summary provided.")
            key_factors = verdict_output.get("key_factors", [])
            next_steps = verdict_output.get("next_steps", "")

            # Apply confidence threshold
            display_status = raw_status
            threshold_applied = False
            if isinstance(confidence, (int, float)) and confidence < confidence_threshold:
                display_status = "NEEDS_REVIEW"
                threshold_applied = True

            st.subheader("Eligibility Verdict")

            if display_status == "ELIGIBLE":
                st.success(f"Eligible  (Confidence: {confidence:.0%})")
            elif display_status == "INELIGIBLE":
                st.error(f"Likely Ineligible  (Confidence: {confidence:.0%})")
            else:
                st.warning(f"Needs Review  (Confidence: {confidence:.0%})")

            if threshold_applied:
                st.caption(
                    f"Original verdict was **{raw_status}** but confidence {confidence:.0%} "
                    f"is below your threshold ({confidence_threshold:.0%}) — showing as Needs Review."
                )

            st.write("**Summary:**", summary)
            if key_factors:
                st.write("**Key factors:**")
                for f in key_factors:
                    st.write(f"- {f}")
            if next_steps:
                st.write("**Recommended next steps:**", next_steps)

            # Storage feedback
            storage = result.get("storage") or {}
            if storage.get("duplicate"):
                st.info("This patient / trial combination was already screened. Showing cached result.")
            elif storage.get("db"):
                st.caption(f"Screening saved to database (id={storage['db'].get('id')}).")

        else:
            st.error("No verdict generated. Check pipeline warnings below.")

        # Per-criterion breakdown
        evaluation_output = result.get("evaluation_agent_output") or {}
        if evaluation_output and evaluation_output.get("evaluations"):
            with st.expander("Per-Criterion Evaluation Breakdown", expanded=True):
                for ev in evaluation_output["evaluations"]:
                    criterion_text = ev.get("criterion_text", "—")
                    meets = ev.get("meets_criterion")
                    conf = ev.get("confidence", 0.0)
                    reasoning = ev.get("reasoning", "")
                    patient_info = ev.get("relevant_patient_info", "")

                    if meets is True:
                        st.success(f"{criterion_text}  (confidence: {conf:.0%})")
                    elif meets is False:
                        st.error(f"{criterion_text}  (confidence: {conf:.0%})")
                    else:
                        st.warning(f"{criterion_text}  (confidence: {conf:.0%})")

                    if reasoning:
                        st.write(f"*Reasoning:* {reasoning}")
                    if patient_info:
                        st.write(f"*Relevant info:* {patient_info}")
                    st.divider()

        # Agent output expanders
        with st.expander("Criteria Agent Output", expanded=False):
            st.json(result.get("criteria_agent_output") or {})
        with st.expander("Evaluation Agent Output", expanded=False):
            st.json(result.get("evaluation_agent_output") or {})
        with st.expander("Verdict Agent Output", expanded=False):
            st.json(result.get("verdict_agent_output") or {})

        if result.get("errors"):
            with st.expander("Pipeline Warnings"):
                for e in result["errors"]:
                    st.warning(e)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Screening Analytics")

    if not _db_available():
        st.info(
            "Analytics require a database connection. "
            "Add SUPABASE_DB_URI to your .env file and restart the app."
        )
        st.stop()

    trials = _load_trials()
    if not trials:
        st.info(
            "No trials in the database yet. "
            "To seed sample trials, run: `python scripts/seed_trials.py` from the project directory. "
            "Or go to Tab 1, paste criteria, check 'Save as a new trial', and run a screening."
        )
        st.stop()

    trial_names = {t["name"]: t for t in trials}
    selected_analytics_trial = st.selectbox(
        "Select a trial to analyse:",
        options=list(trial_names.keys()),
        key="analytics_trial_select",
    )
    trial_row = trial_names[selected_analytics_trial]
    trial_id = trial_row["id"]

    # Confidence threshold for analytics filtering
    analytics_threshold_pct = st.slider(
        "Confidence threshold — screenings below this level are excluded from charts",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%",
        key="confidence_threshold_tab2",
        help="Set to 0% to include all screenings regardless of confidence.",
    )
    analytics_threshold = analytics_threshold_pct / 100

    # Synthetic data seeding
    col_seed_count, col_seed_btn, _ = st.columns([1, 1, 2])
    with col_seed_count:
        seed_count = st.number_input(
            "Patients to generate",
            min_value=1,
            max_value=25,
            value=15,
            key="seed_count_input",
        )
    with col_seed_btn:
        st.write("")  # vertical alignment spacer
        st.write("")
        if st.button("Generate Synthetic Patients", key="seed_btn"):
            try:
                from scripts.seed_synthetic_data import seed_trial as _seed
                progress_bar = st.progress(0, text=f"Generating {seed_count} patient profile(s) with LLM...")
                status_text = st.empty()

                def _on_progress(completed, total):
                    pct = completed / total
                    progress_bar.progress(pct, text=f"Screening patients — {completed}/{total} complete ({pct:.0%})")

                summary = asyncio.run(_seed(trial_id=trial_id, count=seed_count, use_llm=True, on_progress=_on_progress))
                progress_bar.progress(1.0, text="Screening complete.")

                if summary and summary.get("ok", 0) > 0:
                    status_text.success(f"{summary['ok']} new screening(s) inserted. {summary.get('dups', 0)} duplicate(s) skipped.")
                    st.session_state["seed_just_completed"] = True
                elif summary and summary.get("dups", 0) > 0:
                    status_text.warning(f"All {summary['dups']} patient(s) were already screened for this trial (duplicates). Delete existing screenings in Supabase to re-seed.")
                elif summary and summary.get("errors", 0) > 0:
                    status_text.error(f"{summary['errors']} error(s) during screening. Check terminal output for details.")
                else:
                    status_text.warning("No patients were generated. Check that ANTHROPIC_API_KEY is set and the trial exists.")
            except Exception as e:
                st.error(f"Seeding failed: {e}")

    if st.session_state.pop("seed_just_completed", False):
        st.rerun()

    screenings = _load_screenings(trial_id)
    stats = _get_stats(trial_id)

    if not screenings:
        st.info(
            "No screenings for this trial yet. Run some eligibility checks on Tab 1, "
            "or click 'Generate Synthetic Patients' above."
        )
        st.stop()

    # Build DataFrame
    df = pd.DataFrame(screenings)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["screened_at"] = pd.to_datetime(df["screened_at"], errors="coerce")

    # Apply confidence threshold
    df_filtered = df[df["confidence_score"] >= analytics_threshold].copy() if analytics_threshold > 0 else df.copy()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Screenings", stats.get("total", len(df)))
    m2.metric("Eligible", stats.get("eligible", 0))
    m3.metric("Ineligible", stats.get("ineligible", 0))
    m4.metric("Needs Review", stats.get("needs_review", 0))
    avg_conf = stats.get("avg_confidence")
    m5.metric("Avg Confidence", f"{avg_conf:.0%}" if avg_conf else "—")

    if analytics_threshold > 0:
        st.caption(
            f"Charts below show {len(df_filtered)} of {len(df)} screenings "
            f"(confidence ≥ {analytics_threshold:.0%})."
        )

    if df_filtered.empty:
        st.warning("No screenings meet the confidence threshold. Lower the slider to see more data.")
        st.stop()

    import plotly.express as px

    color_map = {"ELIGIBLE": "#22c55e", "INELIGIBLE": "#ef4444", "NEEDS_REVIEW": "#f59e0b"}
    four_cat_colors = {
        "High Eligibility":      "#16a34a",
        "Moderate Eligibility":  "#86efac",
        "Needs Review":          "#f59e0b",
        "Ineligible":            "#ef4444",
    }

    # ── Derive four-category column ───────────────────────────────────────────
    def _four_category(row):
        s = row["eligibility_status"]
        c = row["confidence_score"] if pd.notna(row["confidence_score"]) else 0.0
        if s == "ELIGIBLE" and c >= 0.75:
            return "High Eligibility"
        elif s == "ELIGIBLE":
            return "Moderate Eligibility"
        elif s == "NEEDS_REVIEW":
            return "Needs Review"
        else:
            return "Ineligible"

    df_filtered = df_filtered.copy()
    df_filtered["category"] = df_filtered.apply(_four_category, axis=1)

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    col_e, col_f = st.columns(2)

    # Chart 1 — Raw status donut
    with col_a:
        st.subheader("Eligibility Status")
        status_counts = df_filtered["eligibility_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig1 = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.45,
            color="Status",
            color_discrete_map=color_map,
        )
        fig1.update_traces(textinfo="percent+label")
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — Four-category breakdown (status × confidence)
    with col_b:
        st.subheader("Eligibility Category")
        cat_order = ["High Eligibility", "Moderate Eligibility", "Needs Review", "Ineligible"]
        cat_counts = (
            df_filtered["category"]
            .value_counts()
            .reindex(cat_order, fill_value=0)
            .reset_index()
        )
        cat_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.45,
            color="Category",
            color_discrete_map=four_cat_colors,
            category_orders={"Category": cat_order},
        )
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 — Confidence violin by status
    with col_c:
        st.subheader("Confidence Distribution by Status")
        df_viol = df_filtered[df_filtered["confidence_score"].notna()].copy()
        df_viol["confidence_pct"] = df_viol["confidence_score"] * 100
        if not df_viol.empty:
            fig3 = px.violin(
                df_viol,
                x="eligibility_status",
                y="confidence_pct",
                color="eligibility_status",
                color_discrete_map=color_map,
                box=True,
                points="all",
                labels={"confidence_pct": "Confidence (%)", "eligibility_status": "Status"},
            )
            fig3.update_layout(
                yaxis=dict(ticksuffix="%", range=[0, 105]),
                showlegend=False,
                xaxis_title="",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No confidence data yet.")

    # Chart 4 — Average confidence by status (bar)
    with col_d:
        st.subheader("Average Confidence by Status")
        avg_by_status = (
            df_filtered.groupby("eligibility_status")["confidence_score"]
            .mean()
            .reset_index()
        )
        avg_by_status.columns = ["Status", "Avg Confidence"]
        avg_by_status["Avg Confidence %"] = (avg_by_status["Avg Confidence"] * 100).round(1)
        fig4 = px.bar(
            avg_by_status,
            x="Status",
            y="Avg Confidence %",
            color="Status",
            color_discrete_map=color_map,
            text="Avg Confidence %",
        )
        fig4.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig4.update_layout(
            yaxis=dict(ticksuffix="%", range=[0, 115]),
            showlegend=False,
            xaxis_title="",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Chart 5 — Age distribution (violin)
    with col_e:
        st.subheader("Patient Age Distribution")
        if "patient_age" in df_filtered.columns:
            age_df = df_filtered[df_filtered["patient_age"].notna()].copy()
            if len(age_df) >= 3:
                fig5 = px.violin(
                    age_df,
                    x="eligibility_status",
                    y="patient_age",
                    color="eligibility_status",
                    color_discrete_map=color_map,
                    box=True,
                    points="all",
                    labels={"patient_age": "Age (years)", "eligibility_status": "Status"},
                )
                fig5.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Not enough age data yet — re-seed patients to populate.")
        else:
            st.info("Age data not available — re-seed patients to populate.")

    # Chart 6 — Sex breakdown by eligibility status
    with col_f:
        st.subheader("Sex Breakdown by Status")
        if "patient_sex" in df_filtered.columns:
            sex_df = df_filtered[df_filtered["patient_sex"].notna()].copy()
            if not sex_df.empty:
                sex_counts = (
                    sex_df.groupby(["patient_sex", "eligibility_status"])
                    .size()
                    .reset_index(name="Count")
                )
                sex_counts["patient_sex"] = sex_counts["patient_sex"].str.capitalize()
                fig6 = px.bar(
                    sex_counts,
                    x="patient_sex",
                    y="Count",
                    color="eligibility_status",
                    color_discrete_map=color_map,
                    labels={"patient_sex": "Sex", "eligibility_status": "Status"},
                    barmode="group",
                )
                fig6.update_layout(legend_title="Status", xaxis_title="")
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("No sex data available yet.")
        else:
            st.info("Sex data not available — re-seed patients to populate.")
