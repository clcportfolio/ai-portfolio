"""
Streamlit UI — Clinical Intake Router
Two-tab interface with RBAC login.

Roles:
  demo-doctor    — full access: all extracted fields, classification reasoning,
                   unrestricted NL2SQL
  demo-reception — restricted access: routing card + basic fields only,
                   NL2SQL limited to non-clinical columns, guardrail on output

Tab 1 — Intake Router: upload/select/paste, run pipeline, view routing decision
Tab 2 — Query Database: NL2SQL chatbot with role-scoped schema + 4-layer guardrail
"""

import io
import logging
import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Intake Router",
    page_icon="🏥",
    layout="wide",
)


# ── Session state defaults ────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "role_config" not in st.session_state:
    st.session_state["role_config"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "pending_delete" not in st.session_state:
    st.session_state["pending_delete"] = None   # file_hash of record awaiting delete confirm
if "orphaned_s3_keys" not in st.session_state:
    st.session_state["orphaned_s3_keys"] = []   # s3_keys deleted from DB but not yet from S3


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            """
<div style="text-align:center; margin-bottom:24px;">
  <span style="font-size:2.5em;">🏥</span>
  <h2 style="margin:8px 0 4px 0;">Clinical Intake Router</h2>
  <p style="color:#888; margin:0;">Please sign in to continue</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                from auth import authenticate
                role_config = authenticate(username, password)
                if role_config:
                    st.session_state["authenticated"] = True
                    st.session_state["role_config"] = role_config
                    st.session_state["username"] = username.strip().lower()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    st.stop()


# ── Authenticated from here down ──────────────────────────────────────────────
role_config = st.session_state["role_config"]
username    = st.session_state["username"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Clinical Intake Router")

    # Role badge
    st.markdown(
        f"""
<div style="display:inline-block; padding:4px 12px; border-radius:12px;
     background:{role_config.badge_color}22; border:1px solid {role_config.badge_color};
     color:{role_config.badge_color}; font-size:0.85em; font-weight:600; margin-bottom:8px;">
  {role_config.display_name}
</div>
<div style="font-size:0.8em; color:#888; margin-bottom:12px;">Logged in as <strong>{username}</strong></div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Log Out", use_container_width=True):
        for key in ["authenticated", "role_config", "username", "prefilled_result",
                    "nl2sql_prefill", "nl2sql_question", "nl2sql_autorun",
                    "pending_delete", "orphaned_s3_keys"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    page = st.radio(
        "Navigation",
        ["🏥 Intake Router", "🔍 Query Database"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
A healthcare staff tool that reads a clinical intake form and routes the
patient to the right department at the right urgency level — instantly.

**Pipeline**
1. **Extraction** — pulls structured fields from free-text
2. **Classification** — assigns urgency + department
3. **Routing** — generates plain-English instructions

**Storage**
- Raw files → AWS S3
- Structured results → Supabase (PostgreSQL)
- Duplicate detection via SHA-256 hash

**Tech Stack**
- LangChain + Claude Sonnet / Haiku (Anthropic)
- Langfuse observability
- AWS S3 · Supabase PostgreSQL
- RBAC with 4-layer NL2SQL guardrail
- Streamlit demo · Pydantic structured output
        """
    )
    st.divider()
    st.markdown("[GitHub](https://github.com/codyculver/ai-portfolio) · Built with Claude Code")

# ── DB init on startup ────────────────────────────────────────────────────────
try:
    from storage.db_client import init_db
    init_db()
except Exception:
    pass

# ── Helpers ───────────────────────────────────────────────────────────────────
URGENCY_COLORS = {"Emergent": "#FF4B4B", "Urgent": "#FFA500", "Routine": "#21C354"}
URGENCY_EMOJI  = {"Emergent": "🔴", "Urgent": "🟡", "Routine": "🟢"}


def _traffic_light() -> None:
    """
    Render the agentic traffic indicator — a glowing dot + label showing
    current request volume against the Redis rate limit window.
    Hidden silently if Redis is not configured.
    """
    from guardrails import get_traffic_stats
    stats = get_traffic_stats()
    if not stats["available"]:
        return
    color = stats["color"]
    level = stats["level"]
    count = stats["count"]
    limit = stats["limit"]
    st.markdown(
        f"""
<div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
  <div style="width:10px; height:10px; border-radius:50%;
              background:{color}; box-shadow:0 0 7px {color};
              flex-shrink:0;"></div>
  <span style="font-size:0.8em; color:#888;">
    Agentic traffic:&nbsp;<strong style="color:{color};">{level}</strong>
    <span style="color:#555;">&nbsp;({count}&thinsp;/&thinsp;{limit} req/min)</span>
  </span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return file_bytes.decode("utf-8", errors="replace").strip()


def _routing_card(routing: dict) -> None:
    urgency       = routing.get("urgency_level", "Unknown")
    department    = routing.get("department", "Unknown")
    summary       = routing.get("routing_summary", "")
    next_steps    = routing.get("recommended_next_steps", [])
    follow_ups    = routing.get("follow_up_actions", [])
    response_time = routing.get("estimated_response_time")

    color = URGENCY_COLORS.get(urgency, "#888888")
    emoji = URGENCY_EMOJI.get(urgency, "⚪")

    st.markdown(
        f"""
<div style="border:2px solid {color}; border-radius:8px; padding:16px; margin-bottom:12px;">
  <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;">
    <div>
      <div style="font-size:0.72em; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:2px;">Urgency</div>
      <div style="font-size:1.25em; font-weight:700; color:{color};">{emoji} {urgency}</div>
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

    if next_steps:
        st.markdown("**Recommended Next Steps**")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")

    if follow_ups:
        st.markdown("**Follow-up Actions**")
        for action in follow_ups:
            st.markdown(f"- {action}")


def _agent_expanders(result: dict, rc) -> None:
    """Render agent detail expanders — content gated by role."""
    st.divider()

    extraction = result.get("extraction_output") or {}
    with st.expander("Extracted Fields (extraction_agent)", expanded=False):
        if extraction:
            if rc.can_see_full_extraction:
                field_map = {
                    "Patient Name":        extraction.get("patient_name"),
                    "Age":                 extraction.get("age"),
                    "Date of Birth":       extraction.get("date_of_birth"),
                    "Chief Complaint":     extraction.get("chief_complaint"),
                    "Symptoms":            ", ".join(extraction.get("symptoms", [])) or None,
                    "Medical History":     ", ".join(extraction.get("medical_history", [])) or None,
                    "Current Medications": ", ".join(extraction.get("current_medications", [])) or None,
                    "Allergies":           ", ".join(extraction.get("allergies", [])) or None,
                    "Insurance":           extraction.get("insurance"),
                    "Referral Source":     extraction.get("referral_source"),
                    "Additional Notes":    extraction.get("additional_notes"),
                }
            else:
                # Reception: name + chief complaint only
                field_map = {
                    "Patient Name":    extraction.get("patient_name"),
                    "Chief Complaint": extraction.get("chief_complaint"),
                }
            for label, value in field_map.items():
                if value:
                    st.markdown(f"**{label}:** {value}")
            if not rc.can_see_full_extraction:
                st.caption("🔒 Additional clinical fields restricted to physician access.")
        else:
            st.info("No extraction output available.")

    if rc.can_see_classification:
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


def _run_pipeline(text: str, file_bytes=None, filename=None, content_type="text/plain") -> dict:
    import asyncio
    from pipeline import run as pipeline_run
    return asyncio.run(pipeline_run(
        {"text": text},
        file_bytes=file_bytes,
        original_filename=filename,
        content_type=content_type,
    ))


# ── Module-level cached loaders ───────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_directory():
    try:
        from storage.db_client import get_recent_submissions
        return get_recent_submissions(limit=500), None
    except Exception as e:
        return [], str(e)


@st.cache_data(ttl=15)
def _load_recent_submissions():
    try:
        from storage.db_client import get_recent_submissions
        return get_recent_submissions(limit=20), None
    except Exception as e:
        return [], str(e)


# ── Tabs ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INTAKE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏥 Intake Router":

    st.title("Clinical Intake Router")
    st.caption("Select an existing file from the directory, upload a new one, or paste text.")

    input_source = st.radio(
        "Input source",
        ["Select from directory", "Upload & route", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        file_bytes   = None
        filename     = None
        content_type = "text/plain"
        final_text   = ""

        # ── Directory table ───────────────────────────────────────────────────
        if input_source == "Select from directory":
            dir_col, refresh_col = st.columns([4, 1])
            with refresh_col:
                if st.button("🔄 Refresh", key="dir_refresh", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()

            dir_rows, dir_error = _load_directory()

            if dir_error:
                st.warning(f"Could not load directory: {dir_error}")
            elif not dir_rows:
                st.info("No files in the directory yet. Upload one above to get started.")
            else:
                import pandas as pd

                search = st.text_input(
                    "Search",
                    placeholder="Patient name, department, urgency, filename...",
                )

                df = pd.DataFrame(dir_rows)
                display_cols = ["patient_name", "urgency_level", "department",
                                "original_filename", "submitted_at"]
                display_cols = [c for c in display_cols if c in df.columns]
                df_display = df[display_cols].copy()
                df_display["submitted_at"] = df_display["submitted_at"].astype(str).str[:16]
                df_display.columns = ["Patient", "Urgency", "Department", "Filename", "Submitted"]

                if search.strip():
                    q = search.strip()
                    mask = df_display.apply(
                        lambda col: col.astype(str).str.contains(q, case=False, na=False, regex=False)
                    ).any(axis=1)
                    # Also match "First Last" searches against "Last, First" stored names
                    # e.g. typing "Nathaniel Price" should find "Price, Nathaniel Owen"
                    if "Patient" in df_display.columns:
                        parts = q.split()
                        if len(parts) == 2:
                            flipped = f"{parts[1]}, {parts[0]}"
                            flipped_mask = df_display["Patient"].astype(str).str.contains(
                                flipped, case=False, na=False, regex=False
                            )
                            mask = mask | flipped_mask
                    df_display = df_display[mask]

                st.caption(f"{len(df_display)} record(s)")

                selection = st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                )

                selected_rows = selection.selection.rows if selection.selection.rows else []

                if selected_rows:
                    original_idx = df_display.index[selected_rows[0]]
                    selected_record = dir_rows[original_idx]
                    s3_key = selected_record.get("s3_key")
                    file_hash = selected_record.get("file_hash")
                    filename = selected_record.get("original_filename", "intake.txt")
                    has_routing = bool(selected_record.get("has_routing"))
                    patient_label = selected_record.get("patient_name") or filename

                    if has_routing:
                        st.success(f"Selected: **{patient_label}** — previously routed.")
                        try:
                            from storage.db_client import submission_exists
                            full_record = submission_exists(file_hash) or selected_record
                        except Exception:
                            full_record = selected_record
                        st.session_state["prefilled_result"] = {
                            "extraction_output": full_record.get("extraction_output"),
                            "classification_output": full_record.get("classification_output"),
                            "routing_output": full_record.get("routing_output"),
                            "errors": [],
                            "storage": {"duplicate": True, "s3": None, "db": full_record, "storage_errors": []},
                        }
                    else:
                        st.info(f"Selected: **{patient_label}** — not yet routed. Click **Route This Intake** to process.")

                    if s3_key:
                        try:
                            from storage.s3_client import download_file
                            file_bytes = download_file(s3_key)
                            final_text = _extract_text(file_bytes, filename)
                            if filename.lower().endswith(".pdf"):
                                content_type = "application/pdf"
                        except Exception as e:
                            st.error(f"Could not load file from S3: {e}")

                    # ── Admin-only delete ─────────────────────────────────────
                    if role_config.can_delete_documents and file_hash:
                        st.divider()
                        pending = st.session_state.get("pending_delete")

                        if pending == file_hash:
                            st.warning(
                                f"⚠️ Permanently delete **{patient_label}**?\n\n"
                                "This will remove the record from the database"
                                + (" and the file from S3." if s3_key else ".")
                            )
                            col_confirm, col_cancel, _ = st.columns([1, 1, 2])
                            with col_confirm:
                                if st.button("✅ Confirm Delete", type="primary",
                                             use_container_width=True, key="confirm_delete_btn"):
                                    delete_errors = []
                                    # Delete DB row first — if this fails, S3 is untouched
                                    # and the record remains fully intact (safe failure).
                                    # If DB succeeds but S3 fails, the worst case is an
                                    # orphaned S3 object (invisible to users, not a broken row).
                                    try:
                                        from storage.db_client import delete_submission
                                        delete_submission(file_hash)
                                    except Exception as e:
                                        st.error(f"Database deletion failed: {e}")
                                        st.session_state["pending_delete"] = None
                                        st.stop()
                                    # Delete from S3 after DB is clean (non-fatal if missing)
                                    if s3_key:
                                        try:
                                            from storage.s3_client import delete_file as s3_delete
                                            s3_delete(s3_key)
                                        except Exception as e:
                                            # DB row is already gone — queue the key for retry
                                            # rather than just warning. The Pending S3 Cleanup
                                            # panel will surface it with a direct retry button.
                                            orphans = st.session_state.get("orphaned_s3_keys", [])
                                            if s3_key not in orphans:
                                                orphans.append(s3_key)
                                            st.session_state["orphaned_s3_keys"] = orphans
                                            delete_errors.append(
                                                f"S3 file not deleted (queued for cleanup): {e}"
                                            )

                                    st.session_state["pending_delete"] = None
                                    st.cache_data.clear()
                                    if delete_errors:
                                        for err in delete_errors:
                                            st.warning(err)
                                    st.success(f"Deleted **{patient_label}** successfully.")
                                    st.rerun()
                            with col_cancel:
                                if st.button("❌ Cancel", use_container_width=True,
                                             key="cancel_delete_btn"):
                                    st.session_state["pending_delete"] = None
                                    st.rerun()
                        else:
                            if st.button("🗑️ Delete Document", use_container_width=False,
                                         key="delete_doc_btn"):
                                st.session_state["pending_delete"] = file_hash
                                st.rerun()

        # ── Upload & route ────────────────────────────────────────────────────
        elif input_source == "Upload & route":
            uploaded = st.file_uploader(
                "Upload intake form (.txt or .pdf)",
                type=["txt", "pdf"],
                key="route_uploader",
            )
            if uploaded is not None:
                file_bytes = uploaded.read()
                filename = uploaded.name
                content_type = "application/pdf" if uploaded.name.endswith(".pdf") else "text/plain"
                try:
                    final_text = _extract_text(file_bytes, filename)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    st.stop()

        # ── Paste text ────────────────────────────────────────────────────────
        else:
            final_text = st.text_area(
                "Paste intake form text",
                height=320,
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

    # ── Output column ─────────────────────────────────────────────────────────
    with col_output:
        st.subheader("Routing Decision")

        if "prefilled_result" in st.session_state and not run_button:
            result = st.session_state.pop("prefilled_result")
            st.info("Showing previously stored routing result.", icon="📋")
            _routing_card(result.get("routing_output") or {})
            _agent_expanders(result, role_config)

        elif run_button:
            if not final_text.strip():
                st.warning("Please provide intake form text before routing.")
                st.stop()

            from guardrails import rate_limit_check
            if not rate_limit_check(username):
                st.error("⚠️ Rate limit reached — too many requests this minute. Please wait and try again.")
                st.stop()

            with st.spinner("Running pipeline — extracting, classifying, routing..."):
                try:
                    result = _run_pipeline(final_text.strip(), file_bytes, filename, content_type)
                except ValueError as e:
                    st.error(f"Input validation failed: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            st.cache_data.clear()

            storage = result.get("storage") or {}
            if storage.get("duplicate"):
                st.info("This file was already in the database. Showing previously stored result.", icon="📋")

            s3_info = storage.get("s3")
            db_info = storage.get("db")
            if s3_info or db_info:
                with st.expander("Storage", expanded=False):
                    if s3_info:
                        st.markdown(f"**S3:** `{s3_info.get('s3_key')}`")
                    if db_info and not storage.get("duplicate"):
                        st.markdown(f"**Database row ID:** {db_info.get('id')}")
                    for err in storage.get("storage_errors", []):
                        st.warning(f"Storage warning: {err}")

            routing = result.get("routing_output") or {}
            if routing:
                _routing_card(routing)
            else:
                st.warning("Pipeline completed but no routing output was produced.")

            if result.get("errors"):
                with st.expander("Pipeline warnings", expanded=False):
                    for err in result["errors"]:
                        st.warning(err)

            _agent_expanders(result, role_config)

        else:
            st.info("Choose an input source on the left and click **Route This Intake** to begin.")

    # ── Upload to S3 only (utilities section) ────────────────────────────────
    st.divider()
    orphaned_keys = st.session_state.get("orphaned_s3_keys", [])
    expander_label = (
        f"📤 File Management — Upload to S3 / Sync  ⚠️ {len(orphaned_keys)} pending S3 cleanup"
        if orphaned_keys and role_config.can_delete_documents
        else "📤 File Management — Upload to S3 / Sync"
    )
    with st.expander(expander_label, expanded=bool(orphaned_keys and role_config.can_delete_documents)):
        st.caption("Save a file to the cloud directory without running the intake pipeline, or sync existing S3 files into the database.")

        # ── Pending S3 Cleanup (admin only) ──────────────────────────────────
        if role_config.can_delete_documents and orphaned_keys:
            st.warning(
                f"**{len(orphaned_keys)} S3 file(s) could not be deleted** when their database "
                "records were removed. The files are orphaned in S3 — click Retry to clean them up.",
                icon="⚠️",
            )
            still_orphaned = []
            for okey in list(orphaned_keys):
                short = okey.split("/")[-1]
                col_label, col_retry = st.columns([4, 1])
                with col_label:
                    st.code(okey, language=None)
                with col_retry:
                    if st.button("Retry", key=f"retry_orphan_{okey}", use_container_width=True):
                        try:
                            from storage.s3_client import delete_file as s3_delete
                            s3_delete(okey)
                            st.success(f"Deleted `{short}` from S3.")
                        except Exception as e:
                            st.error(f"Retry failed: {e}")
                            still_orphaned.append(okey)
            st.session_state["orphaned_s3_keys"] = still_orphaned
            if not still_orphaned and len(orphaned_keys) > len(still_orphaned):
                st.rerun()
            st.divider()
        s3_upload_file = st.file_uploader(
            "Choose file (.txt or .pdf)",
            type=["txt", "pdf"],
            key="s3_only_uploader",
        )
        col_save, col_sync = st.columns([1, 1])

        with col_sync:
            if st.button("🔄 Sync S3 → Database", key="s3_sync_button", use_container_width=True,
                         help="Backfill DB rows for files already in S3 that aren't in the directory yet."):
                with st.spinner("Scanning S3 and syncing missing files..."):
                    try:
                        from storage.s3_client import list_files, download_file, hash_file
                        from storage.db_client import get_existing_s3_keys, insert_file_only

                        all_s3_files = list_files()
                        existing_keys = get_existing_s3_keys()
                        missing = [f for f in all_s3_files if f["s3_key"] not in existing_keys]

                        st.write(f"**S3:** {len(all_s3_files)} file(s) found")
                        st.write(f"**DB:** {len(all_s3_files) - len(missing)} already indexed, {len(missing)} missing")

                        if not missing:
                            st.info("All S3 files are already in the database.")
                        else:
                            synced = 0
                            for f in missing:
                                try:
                                    fb = download_file(f["s3_key"])
                                    fh = hash_file(fb)
                                    raw_name = f["filename"]
                                    clean_name = raw_name[13:] if len(raw_name) > 13 else raw_name
                                    row = insert_file_only(fh, {
                                        "s3_key": f["s3_key"],
                                        "s3_bucket": os.getenv("S3_BUCKET_NAME", ""),
                                        "file_size_bytes": f["file_size_bytes"],
                                    }, clean_name)
                                    if row.get("_duplicate"):
                                        st.write(f"⏭ Already exists: `{clean_name}`")
                                    else:
                                        st.write(f"✅ Synced: `{clean_name}`")
                                        synced += 1
                                except Exception as e:
                                    st.write(f"❌ Failed: `{f['filename']}` — {e}")

                            st.cache_data.clear()
                            if synced:
                                st.success(f"Done — {synced} new file(s) added to the database.")
                    except Exception as e:
                        st.error(f"Sync failed: {e}")

        with col_save:
            save_clicked = st.button("Save to S3", key="s3_only_button", use_container_width=True)

        if save_clicked:
            if s3_upload_file is None:
                st.warning("Select a file first.")
            else:
                file_bytes = s3_upload_file.read()
                with st.spinner("Uploading..."):
                    try:
                        from storage.s3_client import upload_file, hash_file
                        from storage.db_client import submission_exists, insert_file_only

                        file_hash = hash_file(file_bytes)
                        existing = submission_exists(file_hash)
                        if existing:
                            st.info(f"Already in directory (uploaded {str(existing.get('submitted_at', ''))[:10]}).")
                        else:
                            ct = "application/pdf" if s3_upload_file.name.endswith(".pdf") else "text/plain"
                            s3_result = upload_file(file_bytes, s3_upload_file.name, ct)
                            insert_file_only(file_hash, s3_result, s3_upload_file.name)
                            st.cache_data.clear()
                            st.success(f"Uploaded: `{s3_upload_file.name}` — now visible in the directory.")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — QUERY DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Query Database":
    st.title("Query Intake Database")
    st.caption("Ask a question in plain English. The system generates and runs the SQL for you.")

    # Role access note
    if not role_config.can_see_full_extraction:
        st.info(
            "🔒 **Reception access:** queries are limited to routing information "
            "(patient name, urgency, department, dates). Clinical fields are restricted.",
            icon="ℹ️",
        )

    st.subheader("Ask a Question")

    # Role-appropriate example questions
    if role_config.can_see_full_extraction:
        example_questions = [
            "How many submissions are in the database?",
            "Show me all Emergent cases.",
            "Which department received the most referrals?",
            "List patients referred this week.",
            "How many cases per urgency level?",
        ]
    else:
        example_questions = [
            "How many patients are in the database?",
            "Show me all Emergent cases.",
            "Which department is busiest?",
            "How many patients came in today?",
            "Show me the routing summary for recent patients.",
        ]

    st.markdown("**Example questions:**")
    ex_cols = st.columns(len(example_questions))
    for i, (col, q) in enumerate(zip(ex_cols, example_questions)):
        with col:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                # Set the input value AND flag an immediate run — no extra click needed
                st.session_state["nl2sql_question"] = q
                st.session_state["nl2sql_autorun"] = True
                st.rerun()

    # Consume the autorun flag before rendering the form so it fires exactly once
    autorun = st.session_state.pop("nl2sql_autorun", False)

    # st.form submits on Enter key as well as button click
    with st.form("nl2sql_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            key="nl2sql_question",
            placeholder="e.g. How many emergent cases came in this month?",
        )
        ask_button = st.form_submit_button("Ask", type="primary")

    _traffic_light()

    # Run if the form was submitted (Enter or button) OR an example was just clicked
    run_question = question.strip() if (ask_button and question.strip()) else (
        st.session_state.get("nl2sql_question", "").strip() if autorun else None
    )

    if run_question:
        from guardrails import rate_limit_check
        if not rate_limit_check(username):
            st.error("⚠️ Rate limit reached — too many requests this minute. Please wait and try again.")
            st.stop()

        with st.spinner("Generating SQL and querying database..."):
            try:
                import asyncio
                from agents.nl2sql_agent import run as nl2sql_run
                nl_result = asyncio.run(nl2sql_run(
                    run_question,
                    role_config=role_config,
                ))
            except Exception as e:
                st.error(f"NL2SQL agent error: {e}")
                st.stop()

        # ── Guardrail banner ──────────────────────────────────────────────────
        if nl_result.get("guardrail_triggered"):
            st.error(
                "🛡️ **Access control guardrail triggered.** "
                "This query touched information outside your access level.",
                icon="🔒",
            )
            with st.expander("Guardrail details", expanded=True):
                for reason in nl_result.get("guardrail_reasons", []):
                    st.markdown(f"- {reason}")

        # ── Answer ────────────────────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(nl_result.get("answer", "No answer generated."))

        with st.expander("SQL & Raw Results", expanded=False):
            st.code(nl_result.get("sql", ""), language="sql")
            if nl_result.get("sql_reasoning"):
                st.caption(nl_result["sql_reasoning"])
            st.markdown(f"**Rows returned:** {nl_result.get('row_count', 0)}")
            rows = nl_result.get("rows", [])
            if rows:
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                except Exception:
                    st.json(rows)

        if nl_result.get("error") and not nl_result.get("guardrail_triggered"):
            st.warning(f"Note: {nl_result['error']}")

    elif ask_button:
        st.warning("Please enter a question.")
    else:
        st.info("Type a question above or click an example to query the intake submissions database.")

    st.divider()

    with st.expander("Recent Submissions", expanded=False):
        recent, recent_error = _load_recent_submissions()

        if recent_error:
            st.warning(f"Could not load submissions: {recent_error}")
        elif not recent:
            st.info("No submissions in the database yet. Route an intake form to get started.")
        else:
            import pandas as pd
            df = pd.DataFrame(recent)

            # Reception sees fewer columns
            if role_config.can_see_full_extraction:
                display_cols = ["id", "patient_name", "chief_complaint", "urgency_level",
                                "department", "original_filename", "submitted_at"]
            else:
                display_cols = ["id", "patient_name", "urgency_level",
                                "department", "original_filename", "submitted_at"]

            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
