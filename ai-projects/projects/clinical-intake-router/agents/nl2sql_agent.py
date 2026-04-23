"""
NL2SQL Agent — Clinical Intake Router
Converts a natural-language question into a SQL SELECT query, executes it
against the intake_submissions Postgres table, and returns a plain-English
answer synthesised from the results.

RBAC enforcement layers (in order):
  L1 — Schema restriction: agent only sees columns allowed for the role
  L2 — SQL AST validation (sqlglot): deterministically rejects any query that
       references columns outside the role's allowlist — not LLM-dependent
  L3 — Result column stripping: strips restricted keys from returned rows
  L4 — Output guardrail (guardrails.check_nl2sql_output): scans synthesised
       answer for clinical keywords and restricted field content

Two-step chain:
  1. sql_chain    — LLM generates SELECT query given role-scoped schema + question
  2. answer_chain — LLM synthesises plain-English answer from result rows

Safety:
  - Only SELECT queries are permitted (enforced in db_client.query_submissions)
  - SQL AST validation is deterministic — cannot be prompt-injected
  - Max 100 rows returned
"""

import logging
from typing import Optional

import sqlglot
import sqlglot.expressions as exp
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)


# ── Pydantic model for structured SQL output ──────────────────────────────────

class SQLQuery(BaseModel):
    sql: str = Field(description="A valid PostgreSQL SELECT query. No trailing semicolon.")
    reasoning: str = Field(description="One sentence explaining what this query does.")


# ── Prompts ───────────────────────────────────────────────────────────────────

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a PostgreSQL expert generating SELECT queries against a clinical intake submissions database.

You will be given:
1. The database schema (scoped to your access level)
2. A natural-language question from a healthcare administrator

Your job is to write a precise, correct PostgreSQL SELECT query that answers the question.

Rules:
- Only write SELECT queries. Never INSERT, UPDATE, DELETE, DROP, or any DDL.
- Use exact column names from the schema. Do not reference columns not listed in the schema.
- For urgency_level, valid values are exactly: 'Routine', 'Urgent', 'Emergent'
- For text searches, use ILIKE for case-insensitive matching.
- Always include LIMIT 100 unless the question asks for a count or aggregate.
- For date filtering, submitted_at is a TIMESTAMPTZ column.
- Do not use SELECT * — always name the specific columns from the schema.
- Return only the SQL — no markdown fences, no explanation in the sql field.

Schema:
{schema}
""",
    ),
    ("human", "Question: {question}"),
])

ANSWER_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant for healthcare administrators.
You have run a database query against clinical intake submissions and received the results.
Write a clear, concise plain-English answer to the original question based on the query results.

Guidelines:
- Be direct and factual — answer the question, don't pad.
- If the result is empty, say "No matching records found."
- If the result is a count or aggregate, state the number clearly.
- For lists of patients, format as a readable summary (not raw JSON).
- Never include SQL or technical jargon in your answer.
- Keep the answer under 200 words unless the result set requires more detail.
""",
    ),
    (
        "human",
        "Question: {question}\n\nSQL executed: {sql}\n\nQuery results ({row_count} rows):\n{results}",
    ),
])


# ── Chains — built once at module load, reused on every request ───────────────

_sql_chain = SQL_GENERATION_PROMPT | ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=512,
).with_structured_output(SQLQuery)

_answer_chain = ANSWER_SYNTHESIS_PROMPT | ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=1024,
) | StrOutputParser()


def _get_handler(langfuse_handler=None):
    if langfuse_handler:
        return langfuse_handler
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None


# ── SQL AST validation (Layer 2) ──────────────────────────────────────────────

def validate_sql_columns(sql: str, allowed_columns: Optional[list[str]]) -> tuple[bool, str]:
    """
    Parse the generated SQL with sqlglot and check every column reference
    against the role's allowlist. Deterministic — not LLM-dependent.

    Args:
        sql:             Generated SQL string.
        allowed_columns: List of allowed column names, or None (all allowed).

    Returns:
        (is_valid: bool, reason: str)
        reason is empty string on success, error message on failure.
    """
    if allowed_columns is None:
        return True, ""

    allowed = {c.lower() for c in allowed_columns}

    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as e:
        # Unparseable SQL — reject to be safe
        return False, f"Could not parse query: {e}"

    # Block SELECT * for restricted roles
    stars = list(ast.find_all(exp.Star))
    if stars:
        return False, (
            "SELECT * is not permitted at your access level. "
            "Please ask about specific fields available to you."
        )

    # Check all column references
    restricted_found = []
    for col in ast.find_all(exp.Column):
        col_name = col.name.lower()
        if col_name and col_name not in allowed:
            restricted_found.append(col_name)

    if restricted_found:
        unique = list(dict.fromkeys(restricted_found))
        return False, (
            f"Query references column(s) not available at your access level: "
            f"{', '.join(unique)}. Please ask about patient name, urgency, "
            f"department, or submission date."
        )

    return True, ""


# ── Result column stripping (Layer 3) ─────────────────────────────────────────

def strip_restricted_columns(rows: list[dict], allowed_columns: Optional[list[str]]) -> list[dict]:
    """Remove any restricted column keys from result rows before returning."""
    if allowed_columns is None:
        return rows
    allowed = set(allowed_columns)
    return [{k: v for k, v in row.items() if k in allowed} for row in rows]


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(question: str, role_config=None, langfuse_handler=None) -> dict:
    """
    Convert a natural-language question to SQL, execute it, and synthesise an answer.
    Applies all four RBAC enforcement layers.

    Args:
        question:        Plain-English question from the user.
        role_config:     RoleConfig from auth.py. None = full access (doctor).
        langfuse_handler: Optional Langfuse CallbackHandler.

    Returns:
        dict with keys:
          question          — original question
          sql               — generated SELECT query
          sql_reasoning     — one-sentence explanation
          rows              — result row dicts (stripped of restricted columns)
          row_count         — number of rows returned
          answer            — plain-English answer (may be redacted by guardrail)
          error             — error message if something failed, else None
          guardrail_triggered — True if L4 guardrail fired
          guardrail_reasons — list of reason strings from L4
    """
    result = {
        "question": question,
        "sql": None,
        "sql_reasoning": None,
        "rows": [],
        "row_count": 0,
        "answer": None,
        "error": None,
        "guardrail_triggered": False,
        "guardrail_reasons": [],
    }

    handler = _get_handler(langfuse_handler)
    invoke_config = {"callbacks": [handler], "run_name": "nl2sql_agent"} if handler else {}

    # ── L1: Role-scoped schema ────────────────────────────────────────────────
    try:
        if role_config is not None:
            schema = role_config.nl2sql_schema
        else:
            from storage.db_client import get_table_schema
            schema = get_table_schema()
    except Exception as e:
        result["error"] = f"Could not load schema: {e}"
        result["answer"] = "Database schema is unavailable. Check your SUPABASE_DB_URI configuration."
        return result

    # ── SQL generation ────────────────────────────────────────────────────────
    try:
        sql_output: SQLQuery = await _sql_chain.ainvoke(
            {"schema": schema, "question": question},
            config=invoke_config,
        )
        result["sql"] = sql_output.sql.strip()
        result["sql_reasoning"] = sql_output.reasoning
        logger.info("Generated SQL: %s", result["sql"])
    except Exception as e:
        logger.error("SQL generation failed: %s", e)
        result["error"] = f"SQL generation failed: {e}"
        result["answer"] = "Sorry, I could not generate a query for that question. Try rephrasing."
        return result

    # ── L2: SQL AST validation ────────────────────────────────────────────────
    allowed_columns = role_config.allowed_columns if role_config else None
    is_valid, validation_msg = validate_sql_columns(result["sql"], allowed_columns)
    if not is_valid:
        logger.warning(
            "[GUARDRAIL L2] SQL rejected for role=%s: %s",
            role_config.role if role_config else "none",
            validation_msg,
        )
        result["error"] = f"Query blocked by access control: {validation_msg}"
        result["answer"] = f"🔒 {validation_msg}"
        result["guardrail_triggered"] = True
        result["guardrail_reasons"] = [f"L2 SQL validation: {validation_msg}"]
        return result

    # ── SQL execution ─────────────────────────────────────────────────────────
    try:
        from storage.db_client import query_submissions
        rows = query_submissions(result["sql"])
    except ValueError as e:
        result["error"] = str(e)
        result["answer"] = "The generated query was not a SELECT statement. Please rephrase your question."
        return result
    except Exception as e:
        logger.error("SQL execution failed: %s", e)
        result["error"] = f"SQL execution failed: {e}"
        result["answer"] = "The query could not be executed. The database may be unavailable."
        return result

    # ── L3: Result column stripping ───────────────────────────────────────────
    rows = strip_restricted_columns(rows, allowed_columns)
    result["rows"] = rows
    result["row_count"] = len(rows)
    logger.info("Query returned %d rows (after column stripping).", result["row_count"])

    # ── Answer synthesis ──────────────────────────────────────────────────────
    try:
        import json
        rows_text = json.dumps(result["rows"][:100], indent=2, default=str)

        result["answer"] = await _answer_chain.ainvoke(
            {
                "question": question,
                "sql": result["sql"],
                "row_count": result["row_count"],
                "results": rows_text,
            },
            config={**(invoke_config), "run_name": "nl2sql_answer"},
        )
    except Exception as e:
        logger.error("Answer synthesis failed: %s", e)
        result["error"] = f"Answer synthesis failed: {e}"
        result["answer"] = f"Query returned {result['row_count']} row(s). Answer synthesis failed: {e}"
        return result

    # ── L4: Output guardrail ──────────────────────────────────────────────────
    from guardrails import check_nl2sql_output
    result = check_nl2sql_output(result, role_config)

    return result


if __name__ == "__main__":
    import asyncio
    import json

    print("=== NL2SQL Agent Smoke Test ===\n")

    # Test SQL AST validation directly (deterministic — no async needed)
    from auth import ROLE_CONFIGS
    reception = ROLE_CONFIGS["demo-reception"]
    doctor = ROLE_CONFIGS["demo-doctor"]

    print("--- L2 SQL validation tests ---")
    tests = [
        ("SELECT patient_name FROM intake_submissions", reception, True),
        ("SELECT extraction_output FROM intake_submissions", reception, False),
        ("SELECT * FROM intake_submissions", reception, False),
        ("SELECT * FROM intake_submissions", doctor, True),
        ("SELECT patient_name, urgency_level FROM intake_submissions WHERE urgency_level = 'Emergent'", reception, True),
    ]
    for sql, role, expected_valid in tests:
        valid, msg = validate_sql_columns(sql, role.allowed_columns)
        status = "✅" if valid == expected_valid else "❌"
        print(f"  {status} [{role.role}] {sql[:60]}... → valid={valid}" + (f" | {msg}" if msg else ""))

    print("\n--- Live query test (requires ANTHROPIC_API_KEY + SUPABASE_DB_URI) ---")
    result = asyncio.run(run("How many submissions are in the database?"))
    print(f"SQL: {result['sql']}")
    print(f"Rows: {result['row_count']}")
    print(f"Answer: {result['answer']}")
    print(f"Guardrail triggered: {result['guardrail_triggered']}")
