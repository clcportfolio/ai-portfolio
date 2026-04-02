"""
auth.py — Clinical Trial Eligibility Screener
Role-Based Access Control (RBAC) for the Streamlit UI.

Two roles:
  demo-coordinator — standard user; can run screenings, view verdicts and per-criterion
                     breakdown; cannot view raw agent JSONB dumps or synthetic-only stats
  demo-admin       — full access; can view all agent outputs, manage trials, and see the
                     synthetic/real split in analytics

authenticate(username, password) -> Optional[RoleConfig]
  Single auth entry point. All downstream access decisions read from the returned
  RoleConfig — never from raw role strings scattered through the code.

In production, replace USERS with a real credential store and hash passwords properly
(e.g. bcrypt). The USERS dict here is demo-only.
"""

from typing import Optional
from pydantic import BaseModel


class RoleConfig(BaseModel):
    role: str
    display_name: str
    can_see_agent_outputs: bool       # raw JSON expanders for each agent
    can_manage_trials: bool           # save / delete trials
    can_see_synthetic_stats: bool     # synthetic vs real split in analytics
    can_seed_synthetic_data: bool     # "Generate Synthetic Patients" button
    badge_color: str                  # CSS color string for the login badge


ROLE_CONFIGS: dict[str, RoleConfig] = {
    "coordinator": RoleConfig(
        role="coordinator",
        display_name="Coordinator",
        can_see_agent_outputs=False,
        can_manage_trials=True,
        can_see_synthetic_stats=False,
        can_seed_synthetic_data=False,
        badge_color="#f97316",    # orange
    ),
    "admin": RoleConfig(
        role="admin",
        display_name="Administrator",
        can_see_agent_outputs=True,
        can_manage_trials=True,
        can_see_synthetic_stats=True,
        can_seed_synthetic_data=True,
        badge_color="#8b5cf6",    # purple
    ),
}

# Demo credential store — plaintext for demo purposes only.
# Production: replace with hashed credentials + secure storage.
USERS: dict[str, dict] = {
    "demo-coordinator": {"password": "coordinator-demo", "role": "coordinator"},
    "demo-admin": {"password": "admin-demo", "role": "admin"},
}


def authenticate(username: str, password: str) -> Optional[RoleConfig]:
    """
    Validate credentials and return the associated RoleConfig.

    Args:
        username: Username string.
        password: Plaintext password (demo only).

    Returns:
        RoleConfig if credentials are valid, None otherwise.
    """
    user = USERS.get(username.lower().strip())
    if user and user["password"] == password:
        return ROLE_CONFIGS.get(user["role"])
    return None


if __name__ == "__main__":
    print("=== Auth Smoke Test ===\n")

    # Valid logins
    for uname, pwd in [("demo-coordinator", "coordinator-demo"), ("demo-admin", "admin-demo")]:
        cfg = authenticate(uname, pwd)
        assert cfg is not None, f"Expected auth success for {uname}"
        print(f"  {uname!r} → role={cfg.role!r}, badge={cfg.badge_color}")

    # Invalid login
    cfg = authenticate("coordinator", "wrongpassword")
    assert cfg is None, "Expected auth failure for wrong password"
    print("  Invalid password → correctly rejected")

    cfg = authenticate("unknown", "password")
    assert cfg is None, "Expected auth failure for unknown user"
    print("  Unknown user → correctly rejected")

    print("\n=== All checks passed ===")
