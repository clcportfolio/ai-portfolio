"""
Auth — Clinical Intake Router
Role-based access control definitions.

RoleConfig is a Pydantic model that carries all role-specific settings —
allowed DB columns, UI visibility flags, and the schema string shown to the
NL2SQL agent. Everything role-dependent in the app reads from one of these
objects, so adding a new role is a single entry in ROLE_CONFIGS + USERS.

Production note: credentials are hardcoded here for demo purposes only.
In production, replace USERS with a real auth provider (Supabase Auth,
Auth0, Okta, etc.) and store RoleConfig in a DB-backed permission table.
"""

from typing import Optional
from pydantic import BaseModel


# ── Allowed columns per role ──────────────────────────────────────────────────

RECEPTION_ALLOWED_COLUMNS = [
    "id",
    "patient_name",
    "urgency_level",
    "department",
    "routing_summary",
    "original_filename",
    "file_size_bytes",
    "submitted_at",
]

# ── NL2SQL schema strings per role ────────────────────────────────────────────
# The NL2SQL agent only sees the schema for its role — it has no knowledge
# of columns outside its allowlist.

RECEPTION_NL2SQL_SCHEMA = """
Table: intake_submissions (reception access — clinical fields restricted)

Columns:
  id                    INTEGER       — auto-increment primary key
  patient_name          TEXT          — extracted patient name
  urgency_level         TEXT          — 'Routine', 'Urgent', or 'Emergent'
  department            TEXT          — routed department (e.g. 'Cardiology')
  routing_summary       TEXT          — plain-English routing summary
  original_filename     TEXT          — original upload filename
  file_size_bytes       INTEGER       — file size in bytes
  submitted_at          TIMESTAMPTZ   — timestamp of submission (UTC)

Example queries:
  SELECT patient_name, urgency_level, department FROM intake_submissions ORDER BY submitted_at DESC LIMIT 10;
  SELECT department, COUNT(*) FROM intake_submissions GROUP BY department ORDER BY COUNT(*) DESC;
  SELECT patient_name, urgency_level FROM intake_submissions WHERE urgency_level = 'Emergent';
""".strip()

DOCTOR_NL2SQL_SCHEMA = """
Table: intake_submissions

Columns:
  id                    INTEGER       — auto-increment primary key
  file_hash             TEXT          — SHA-256 of file content; unique per document
  s3_key                TEXT          — S3 object key (null for text-only submissions)
  s3_bucket             TEXT          — S3 bucket name
  original_filename     TEXT          — original upload filename
  file_size_bytes       INTEGER       — file size in bytes
  patient_name          TEXT          — extracted patient name
  chief_complaint       TEXT          — extracted chief complaint
  urgency_level         TEXT          — 'Routine', 'Urgent', or 'Emergent'
  department            TEXT          — routed department (free text, e.g. 'Cardiology')
  routing_summary       TEXT          — plain-English routing summary from routing agent
  extraction_output     JSONB         — full extraction agent output as JSON
  classification_output JSONB         — full classification agent output as JSON
  routing_output        JSONB         — full routing agent output as JSON
  submitted_at          TIMESTAMPTZ   — timestamp of submission (UTC)

Example queries:
  SELECT patient_name, urgency_level, department FROM intake_submissions ORDER BY submitted_at DESC LIMIT 10;
  SELECT department, COUNT(*) FROM intake_submissions GROUP BY department ORDER BY COUNT(*) DESC;
  SELECT * FROM intake_submissions WHERE urgency_level = 'Emergent';
  SELECT patient_name, extraction_output->>'current_medications' AS medications FROM intake_submissions;
""".strip()


# ── RoleConfig ────────────────────────────────────────────────────────────────

class RoleConfig(BaseModel):
    """
    All role-specific settings in one typed object.
    Passed through the app so every rendering and enforcement decision
    reads from a single source of truth.
    """
    role: str
    display_name: str
    allowed_columns: Optional[list[str]]  # None = all columns (doctor/admin)
    can_see_classification: bool          # classification reasoning expander
    can_see_full_extraction: bool         # full extracted fields vs. name+complaint only
    can_delete_documents: bool            # S3 + DB document deletion (admin only)
    nl2sql_schema: str                    # schema string shown to NL2SQL agent
    badge_color: str                      # sidebar role badge color


ROLE_CONFIGS: dict[str, RoleConfig] = {
    "demo-admin": RoleConfig(
        role="demo-admin",
        display_name="Administrator",
        allowed_columns=None,
        can_see_classification=True,
        can_see_full_extraction=True,
        can_delete_documents=True,
        nl2sql_schema=DOCTOR_NL2SQL_SCHEMA,
        badge_color="#9B59B6",
    ),
    "demo-doctor": RoleConfig(
        role="demo-doctor",
        display_name="Physician",
        allowed_columns=None,
        can_see_classification=True,
        can_see_full_extraction=True,
        can_delete_documents=False,
        nl2sql_schema=DOCTOR_NL2SQL_SCHEMA,
        badge_color="#21C354",
    ),
    "demo-reception": RoleConfig(
        role="demo-reception",
        display_name="Reception",
        allowed_columns=RECEPTION_ALLOWED_COLUMNS,
        can_see_classification=False,
        can_see_full_extraction=False,
        can_delete_documents=False,
        nl2sql_schema=RECEPTION_NL2SQL_SCHEMA,
        badge_color="#FFA500",
    ),
}

# ── Users ─────────────────────────────────────────────────────────────────────
# Hardcoded for demo. Replace with auth provider in production.

USERS: dict[str, dict] = {
    "demo-admin":     {"password": "admin-demo",     "role": "demo-admin"},
    "demo-doctor":    {"password": "doctor-demo",    "role": "demo-doctor"},
    "demo-reception": {"password": "reception-demo", "role": "demo-reception"},
}


# ── Auth function ─────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> Optional[RoleConfig]:
    """
    Validate credentials and return the RoleConfig for the user, or None.

    Args:
        username: Submitted username string.
        password: Submitted password string.

    Returns:
        RoleConfig if credentials are valid, None otherwise.
    """
    user = USERS.get(username.strip().lower())
    if user and user["password"] == password.strip():
        return ROLE_CONFIGS[user["role"]]
    return None


if __name__ == "__main__":
    # Smoke test
    rc_admin = authenticate("demo-admin", "admin-demo")
    assert rc_admin is not None
    assert rc_admin.role == "demo-admin"
    assert rc_admin.allowed_columns is None
    assert rc_admin.can_see_classification is True
    assert rc_admin.can_delete_documents is True

    rc = authenticate("demo-doctor", "doctor-demo")
    assert rc is not None
    assert rc.role == "demo-doctor"
    assert rc.allowed_columns is None
    assert rc.can_see_classification is True
    assert rc.can_delete_documents is False

    rc2 = authenticate("demo-reception", "reception-demo")
    assert rc2 is not None
    assert rc2.role == "demo-reception"
    assert "extraction_output" not in rc2.allowed_columns
    assert rc2.can_see_classification is False
    assert rc2.can_delete_documents is False

    bad = authenticate("demo-doctor", "wrongpassword")
    assert bad is None

    print("auth.py smoke test passed.")
    print(f"  demo-admin role: {rc_admin.display_name}, can_delete={rc_admin.can_delete_documents}")
    print(f"  demo-doctor role: {rc.display_name}, allowed_columns: all")
    print(f"  demo-reception role: {rc2.display_name}, allowed_columns: {rc2.allowed_columns}")
