"""text2sql.py — Dual-model Text2SQL engine with parallel execution.

Model priority:
  1. PRIMARY:  debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned  (fast, 8s)
  2. FALLBACK: a-kore/Arctic-Text2SQL-R1-7B                      (accurate, 30s)

Both fire in parallel — first valid SQL wins.
If parallel fails → retry with fallback model alone (more thorough).

Key design:
  • CREATE TABLE schema format (what fine-tuned Text2SQL models expect)
  • Ollama /api/chat endpoint so each model uses its correct template
  • Accepts BOTH direct column access AND document->>'field' JSONB access
    (both work: flat columns AND document JSONB column are populated)
  • JOIN key: always use _id (the actual PK text column)

Public API:
    run(query, agent)          -> Optional[Dict]
    generate_sql(query, agent) -> Optional[str]
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from pipeline.schema import (
    build_text2sql_schema,
    get_table_names,
    run_sql,
)
from pipeline.utils import (
    Timer,
    coerce_number,
    fmt_number,
    format_rows_as_markdown_table,
)

LOGGER = logging.getLogger("sql_chatbot")

# ── Constants ──────────────────────────────────────────────────────────────────
_MAX_SQL_RETRIES    = 1     # retry once with fallback model on failure
_PARALLEL_TIMEOUT_S = 35    # max wait for the parallel pool
_MAX_RESULT_ROWS    = 500   # return all rows; user can add "top N" to limit
_SQL_EXEC_TIMEOUT_S = 30    # SQL execution timeout


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA — CREATE TABLE format (what fine-tuned Text2SQL models understand)
# ══════════════════════════════════════════════════════════════════════════════

# Static CREATE TABLE schema built from the actual DB columns.
# Includes only the fields most relevant to CRM queries.
# Both direct column access (deals.name) AND document->>'name' work — the DB
# has flat columns AND a populated JSONB document column.

_CREATE_TABLE_SCHEMA = """-- PostgreSQL CRM Database
-- PRIMARY KEY: _id TEXT on every table
-- JOINs: JOIN "users" u ON u._id = d.owner   (use _id flat column)
-- SOFT DELETE: WHERE NOT deleted   (boolean)  |  outreaches: WHERE NOT "isDeleted"
--
-- ⚠️ FIELD ACCESS RULE — VERY IMPORTANT:
--   Mixed-case fields (dealWonAt, companyName, firstName…) MUST use JSONB:
--     document->>'dealWonAt'      ← CORRECT (JSONB preserves case)
--     dealWonAt                   ← WRONG  (PostgreSQL lowercases → not found)
--   All-lowercase fields can use either: _id, deleted, stage, name, email, etc.
--   NUMERIC FIELDS: use direct column (already numeric, no cast needed):
--     grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal, month, year
--   DATE TEXT FIELDS: cast with NULLIF to avoid empty-string error:
--     NULLIF(document->>'invoice_date','')::timestamptz
--     NULLIF(invoice_date,'')::timestamptz     (both work the same)

CREATE TABLE "deals" (
    _id TEXT PRIMARY KEY,
    name TEXT,                  -- deal name (lowercase — use directly)
    stage TEXT,                 -- 'Analysis - To be Quoted','Quotation Sent','Negotiation',
                                -- 'Contract Under Review','On Hold','Closed Won','Closed Lost'
    owner TEXT,                 -- references users._id
    company TEXT,               -- references companies._id
    grand_total_in_usd NUMERIC, -- deal value in USD (use directly, already numeric)
    deleted BOOLEAN,            -- WHERE NOT deleted
    -- MIXED-CASE columns — MUST use document->>:
    -- document->>'closeDate'   TEXT date, cast: NULLIF(document->>'closeDate','')::timestamptz
    -- document->>'dealWonAt'   NULL=not won / NOT NULL=won
    -- document->>'dealLostAt'  NULL=not lost / NOT NULL=lost
    -- document->>'createdAt'   TEXT date
    document JSONB              -- always populated; use for mixed-case fields
);
-- ✓ Open deals:  WHERE document->>'dealWonAt' IS NULL AND document->>'dealLostAt' IS NULL AND NOT deleted
-- ✓ Won deals:   WHERE document->>'dealWonAt' IS NOT NULL AND NOT deleted
-- ✓ Lost deals:  WHERE document->>'dealLostAt' IS NOT NULL AND NOT deleted
-- ✓ Close date:  WHERE NULLIF(document->>'closeDate','')::timestamptz < NOW()

CREATE TABLE "invoices" (
    _id TEXT PRIMARY KEY,
    invoice_number TEXT,        -- e.g. 'ELSN/2025/101'
    payment_status TEXT,        -- 'paid','unpaid','cancelled','draft','partial_payment',
                                -- 'approved','rejected','submitted','confirmed'
    approval_status TEXT,       -- 'approved','rejected','pending','submitted'
    grand_total NUMERIC,        -- invoice total in base currency
    grandtotal_in_usd NUMERIC,  -- invoice total in USD (use this for revenue)
    currency TEXT,
    company TEXT,               -- references companies._id
    invoice_date TEXT,          -- invoice date (TEXT, cast ::timestamptz)
    due_date TEXT,              -- payment due date (TEXT, cast ::timestamptz)
    payment_date TEXT,
    deleted BOOLEAN,
    createdBy TEXT,             -- references users._id
    createdAt TEXT
);
-- Revenue = SUM(grandtotal_in_usd) WHERE payment_status='paid' AND NOT deleted
-- Overdue  = WHERE due_date::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')

CREATE TABLE "sales" (
    _id TEXT PRIMARY KEY,
    sales_number TEXT,          -- e.g. 'S000080'
    status TEXT,                -- 'Confirm','Draft','Cancel'
    salesOwner TEXT,            -- references users._id
    company TEXT,               -- references companies._id
    grand_total NUMERIC,
    grand_total_in_usd NUMERIC, -- use this for revenue calculations
    currency TEXT,
    sales_date TEXT,            -- TEXT, cast ::timestamptz for date filtering
    deleted BOOLEAN,
    createdAt TEXT
);
-- Confirmed sales revenue = SUM(grand_total_in_usd) WHERE status='Confirm' AND NOT deleted

CREATE TABLE "companies" (
    _id TEXT PRIMARY KEY,
    industry TEXT,      -- (lowercase — use directly)
    email TEXT,
    country TEXT,
    source TEXT,
    deleted BOOLEAN,    -- WHERE NOT deleted
    -- MIXED-CASE — use document->>'fieldName':
    -- document->>'companyName'    company display name
    -- document->>'lifecycleStage' 'Lead','Customer','Partner','Inactive Customer','Dead Customer'
    -- document->>'leadStatus'
    -- document->>'companyOwner'   references users._id
    -- document->>'createdAt'
    document JSONB
);
-- ✓ Company name:    document->>'companyName'
-- ✓ Active companies: WHERE document->>'lifecycleStage' NOT IN ('Inactive Customer','Dead Customer') AND NOT deleted

CREATE TABLE "contacts" (
    _id TEXT PRIMARY KEY,
    email TEXT,
    company TEXT,           -- references companies._id
    deleted BOOLEAN,
    -- MIXED-CASE — use document->>'fieldName':
    -- document->>'firstName', document->>'lastName'
    -- document->>'jobTitle', document->>'phoneNumber'
    -- document->>'lifecycleStage', document->>'leadStatus'
    -- document->>'contactOwner'   references users._id
    document JSONB
);
-- ✓ Full name: document->>'firstName' || ' ' || document->>'lastName'

CREATE TABLE "users" (
    _id TEXT PRIMARY KEY,
    name TEXT,                  -- full name
    email TEXT,
    department TEXT,            -- references departments._id
    "isActive" BOOLEAN,
    "isAdmin" BOOLEAN,
    "createdAt" TEXT
);

CREATE TABLE "createtasks" (    -- NOTE: table name is 'createtasks' NOT 'tasks'
    _id TEXT PRIMARY KEY,
    "Task" TEXT,                -- task title (capital T)
    status TEXT,                -- 'Pending','Completed','Open'
    priority TEXT,              -- 'Low','Medium','High'
    "createdBy" TEXT,           -- references users._id (task owner/assignee)
    due_date TEXT,              -- TEXT, cast ::timestamptz for date filtering
    company TEXT,               -- references companies._id
    deleted BOOLEAN,
    "createdAt" TEXT
);
-- Pending tasks: WHERE status='Pending' AND NOT deleted
-- Overdue tasks: WHERE due_date::timestamptz < NOW() AND status!='Completed' AND NOT deleted

CREATE TABLE "targets" (
    _id TEXT PRIMARY KEY,
    "userId" TEXT,              -- references users._id
    "targetInUSD" NUMERIC,      -- monthly target in USD
    month NUMERIC,              -- 1-12
    year NUMERIC,               -- e.g. 2025
    "teamName" TEXT,
    "createdAt" TEXT
);

CREATE TABLE "meetings" (
    _id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    start TEXT,                 -- meeting start datetime (TEXT, cast ::timestamptz)
    "end" TEXT,                 -- meeting end datetime
    location TEXT,
    "createdBy" TEXT,           -- references users._id
    "createdAt" TEXT
);
-- Meetings today: WHERE start::timestamptz::date = CURRENT_DATE
-- Meetings for date: WHERE start::timestamptz::date = '2026-05-14'::date

CREATE TABLE "outreaches" (
    _id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    status TEXT,                -- 'New','Contacted','Interested','Converted to Deal'
    "leadStatus" TEXT,
    campaign TEXT,              -- references campaigns._id
    "assignedTo" TEXT,          -- references users._id
    "isDeleted" BOOLEAN,        -- NOTE: isDeleted not deleted!
    "createdAt" TEXT
);
-- Filter: WHERE NOT "isDeleted"

CREATE TABLE "departments" (
    _id TEXT PRIMARY KEY,
    name TEXT
);

CREATE TABLE "regions" (
    _id TEXT PRIMARY KEY,
    "regionName" TEXT
);

CREATE TABLE "products" (
    _id TEXT PRIMARY KEY,
    name TEXT,
    "isActive" BOOLEAN
);

-- ── KEY JOIN PATTERNS ────────────────────────────────────────────────────────
-- Deals with owner name:
--   FROM deals d LEFT JOIN users u ON u._id = d.owner
-- Invoices with company name:
--   FROM invoices i LEFT JOIN companies c ON c._id = i.company
-- Tasks with user name:
--   FROM createtasks t LEFT JOIN users u ON u._id = t."createdBy"
-- Targets with user name:
--   FROM targets t LEFT JOIN users u ON u._id = t."userId"
-- Sales with user name:
--   FROM sales s LEFT JOIN users u ON u._id = s."salesOwner"
-- Companies with region:
--   FROM companies c LEFT JOIN regions r ON r._id = c.region
-- Users with department:
--   FROM users u LEFT JOIN departments d ON d._id = u.department
"""


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CALLER  (uses /api/chat so each model applies its own template)
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama_chat(
    model: str,
    system_msg: str,
    user_msg: str,
    timeout: int,
    max_tokens: int = 600,
) -> Optional[str]:
    """Call Ollama /api/chat endpoint with system + user messages.

    Using the chat endpoint ensures each model applies its correct prompt
    template (Qwen ChatML, Arctic instruct format, etc.) rather than raw text.
    """
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream":  False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "stop": ["Question:", "\n\n\n", "```\n\n"],
        },
    }
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data   = json.loads(resp.read())
            result = data.get("message", {}).get("content", "").strip()
            if result:
                LOGGER.debug("Ollama [%s] responded (%d chars)", model, len(result))
            return result or None
    except Exception as exc:
        LOGGER.debug("Ollama [%s] chat error: %s", model, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING  (model-specific)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_SQL_EXPERT = """You are an expert PostgreSQL SQL generator for a CRM database.
Given a schema and a question, generate ONE correct SELECT SQL query.

STRICT OUTPUT RULES:
1. Output ONLY the SQL — no explanation, no markdown, no text before or after.
2. Start directly with SELECT.
3. Table names in double-quotes: FROM "deals", FROM "createtasks"
4. JOIN key is always _id: LEFT JOIN "users" u ON u._id = d.owner

CRITICAL — MIXED-CASE COLUMNS:
  Many columns have mixed case (dealWonAt, companyName, firstName, etc.).
  ALWAYS use document->>'fieldName' (JSONB) for these — it is case-safe.
  NEVER use bare column names for mixed-case fields without double-quoting.
  Examples:
    ✓ document->>'dealWonAt'     (JSONB — always works)
    ✓ "dealWonAt"                (quoted flat — also works)
    ✗ dealWonAt                  (unquoted — PostgreSQL lowercases it, FAILS)

SOFT DELETE:
  Most tables: WHERE NOT deleted          (deleted is BOOLEAN)
  outreaches:  WHERE NOT "isDeleted"      (isDeleted is BOOLEAN)

DATE FILTERING:
  Date columns are TEXT — cast when comparing:
    WHERE due_date::timestamptz < NOW()
    WHERE start::timestamptz::date = CURRENT_DATE
    WHERE invoice_date::timestamptz >= date_trunc('month', NOW())

NUMERIC COLUMNS (no casting needed — already numeric):
  grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal, "targetInUSD"

OPEN/WON/LOST DEALS — use JSONB (avoids mixed-case quoting):
  Open:  document->>'dealWonAt' IS NULL AND document->>'dealLostAt' IS NULL
  Won:   document->>'dealWonAt' IS NOT NULL
  Lost:  document->>'dealLostAt' IS NOT NULL

TASKS TABLE:
  Table name:    "createtasks"   (NOT "tasks")
  Title column:  document->>'Task'   (capital T — use JSONB to avoid case issues)
  Assignee:      document->>'createdBy' = users._id

SELECT COLUMNS — always include a human-readable name first:
  deals:       d.name, d.stage, d.grand_total_in_usd
  invoices:    i.invoice_number, i.payment_status, i.grandtotal_in_usd
  companies:   document->>'companyName' (JSONB — mixed case)
  contacts:    document->>'firstName', document->>'lastName' (JSONB — mixed case)
  users:       u.name, u.email
  createtasks: document->>'Task', t.status, t.priority

ONLY generate SELECT. Never UPDATE, DELETE, INSERT, DROP, CREATE."""


def _build_primary_prompt(query: str) -> Tuple[str, str]:
    """Prompt for Qwen2.5-Coder 3B fine-tuned (fast, direct SQL generation)."""
    user_msg = (
        f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
        f"Question: {query}\n\n"
        "Write ONLY the SQL query:"
    )
    return _SYSTEM_SQL_EXPERT, user_msg


def _build_fallback_prompt(query: str, prev_sql: str = "", error: str = "") -> Tuple[str, str]:
    """Prompt for Arctic-Text2SQL-R1-7B (accurate, with reasoning context)."""
    system = (
        _SYSTEM_SQL_EXPERT
        + "\n\nThink carefully about the correct table names, JOIN keys, and filters. "
        "The primary key is always _id. Use it for all JOINs."
    )
    if prev_sql and error:
        user_msg = (
            f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
            f"Question: {query}\n\n"
            f"Previous SQL attempt failed:\n{prev_sql}\n"
            f"Error: {error}\n\n"
            "Fix the SQL and write ONLY the corrected query:"
        )
    else:
        user_msg = (
            f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
            f"Question: {query}\n\n"
            "Write ONLY the SQL query:"
        )
    return system, user_msg


# ══════════════════════════════════════════════════════════════════════════════
# SQL EXTRACTION AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sql(raw: str) -> Optional[str]:
    """Extract a clean SELECT statement from model output."""
    if not raw:
        return None

    cleaned = raw.strip()
    cleaned = re.sub(r"^(?:Here is|The SQL|SQL query|Answer|Result)\s*:?\s*", "", cleaned, flags=re.I)

    # ```sql ... ``` code block
    code_blocks = re.findall(r"```(?:sql|SQL|postgresql)?\s*(SELECT.+?)```", cleaned, re.DOTALL)
    if code_blocks:
        return _clean_sql(code_blocks[-1])

    # <execute>...</execute>
    exec_m = re.search(r"<execute>\s*(SELECT.+?)(?:</execute>|```|$)", cleaned, re.DOTALL | re.I)
    if exec_m:
        return _clean_sql(exec_m.group(1))

    # Bare SELECT statement
    m = re.search(
        r"(SELECT\b.+?)(?:;|\n\n\n|Explanation:|Note:|Question:|$)",
        cleaned, re.DOTALL | re.I,
    )
    if m:
        return _clean_sql(m.group(1))

    return None


def _clean_sql(sql: str) -> str:
    """Normalise extracted SQL: strip semicolons, trailing comments, whitespace."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s*--[^\n]*$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def _validate_sql(sql: str, table_names: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate SQL for safety and correctness.

    Accepts BOTH:
      • Direct column access:  SELECT name FROM "deals"
      • JSONB document access: SELECT document->>'name' FROM "deals"
    Both work because the DB has flat columns AND a populated document JSONB.
    """
    if not sql or len(sql) < 10:
        return False, "empty or too short SQL"

    if not sql.upper().strip().startswith("SELECT"):
        return False, "not a SELECT statement"

    # Block destructive operations
    if re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|GRANT|REVOKE|CREATE)\b", sql, re.I):
        return False, "destructive operation detected"

    # Wrong dialect functions
    if re.search(r"\bIFNULL\b|\bNVL\b|\bDATEDIFF\b|\bSTRFTIME\b", sql, re.I):
        return False, "wrong SQL dialect (MySQL/SQLite function)"

    # Unknown table reference check
    if table_names:
        used_tables = re.findall(r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', sql, re.I)
        unknown = [t for t in used_tables if t.lower() not in [x.lower() for x in table_names]]
        if unknown:
            return False, f"unknown tables: {unknown}"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL SQL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_model(
    model:       str,
    system_msg:  str,
    user_msg:    str,
    timeout:     int,
    table_names: List[str],
    label:       str,
) -> Optional[str]:
    """Generate + validate SQL with one model. Returns valid SQL or None."""
    raw = _call_ollama_chat(model, system_msg, user_msg, timeout)
    if not raw:
        LOGGER.debug("[%s] no response from Ollama", label)
        return None

    sql = _extract_sql(raw)
    if not sql:
        LOGGER.debug("[%s] could not extract SQL from: %.100s", label, raw)
        return None

    ok, err = _validate_sql(sql, table_names)
    if not ok:
        LOGGER.debug("[%s] SQL validation failed: %s | SQL: %.80s", label, err, sql)
        return None

    LOGGER.debug("[%s] valid SQL (%d chars): %.80s", label, len(sql), sql)
    return sql


def generate_sql(query: str, agent) -> Optional[str]:
    """Generate SQL for a query using parallel primary + fallback models.

    Strategy:
      1. Fire PRIMARY (Qwen 3B) + FALLBACK (Arctic 7B) in parallel
      2. Return the first valid SQL — prefer primary if both finish close together
      3. If both fail → retry with FALLBACK model alone (more thorough)

    Returns validated SQL string, or None if all attempts fail.
    """
    table_names = get_table_names(agent)

    primary_sys,  primary_user  = _build_primary_prompt(query)
    fallback_sys, fallback_user = _build_fallback_prompt(query)

    # ── Parallel execution ────────────────────────────────────────────────────
    with Timer("text2sql_parallel") as timer:
        primary_sql:  Optional[str] = None
        fallback_sql: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2s") as pool:
            f_primary = pool.submit(
                _run_model,
                settings.ollama_primary_model,
                primary_sys,
                primary_user,
                settings.ollama_primary_timeout,
                table_names,
                "primary(Qwen3B)",
            )
            f_fallback = pool.submit(
                _run_model,
                settings.ollama_fallback_model,
                fallback_sys,
                fallback_user,
                settings.ollama_fallback_timeout,
                table_names,
                "fallback(Arctic7B)",
            )

            futures = {f_primary: "primary", f_fallback: "fallback"}
            for future in as_completed(futures, timeout=_PARALLEL_TIMEOUT_S):
                label = futures[future]
                try:
                    sql = future.result(timeout=2)
                    if sql:
                        if label == "primary":
                            primary_sql = sql
                        else:
                            fallback_sql = sql
                        LOGGER.info(
                            "Text2SQL: %s model produced SQL (%.0fms): %.80s",
                            label, timer.elapsed_ms, sql,
                        )
                except Exception as exc:
                    LOGGER.debug("Text2SQL %s future error: %s", label, exc)

    # Prefer primary if both succeeded; otherwise use whichever worked
    best_sql = primary_sql or fallback_sql
    if best_sql:
        LOGGER.info("Text2SQL parallel: selected %s SQL", "primary" if primary_sql else "fallback")
        return best_sql

    # ── Retry with fallback model (more powerful, more time) ─────────────────
    LOGGER.info("Text2SQL parallel both failed → retry with Arctic-7B fallback")
    retry_sys, retry_user = _build_fallback_prompt(query)
    retry_raw = _call_ollama_chat(
        settings.ollama_fallback_model,
        retry_sys,
        retry_user,
        timeout=settings.ollama_fallback_timeout,
        max_tokens=800,
    )
    if retry_raw:
        retry_sql = _extract_sql(retry_raw)
        if retry_sql:
            ok, err = _validate_sql(retry_sql, table_names)
            if ok:
                LOGGER.info("Text2SQL retry success: %.80s", retry_sql)
                return retry_sql
            # One more attempt with error feedback
            LOGGER.debug("Text2SQL retry SQL invalid (%s) — trying with error hint", err)
            fix_sys, fix_user = _build_fallback_prompt(query, retry_sql, err)
            fix_raw = _call_ollama_chat(
                settings.ollama_fallback_model,
                fix_sys,
                fix_user,
                timeout=settings.ollama_fallback_timeout,
                max_tokens=800,
            )
            if fix_raw:
                fix_sql = _extract_sql(fix_raw)
                if fix_sql:
                    ok2, _ = _validate_sql(fix_sql, table_names)
                    if ok2:
                        LOGGER.info("Text2SQL fix-retry success: %.80s", fix_sql)
                        return fix_sql

    LOGGER.warning("Text2SQL: all attempts failed for query: %.60s", query)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names referenced in a SQL query."""
    return sorted({
        m for m in re.findall(
            r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            sql, re.IGNORECASE,
        ) if m
    })


def _extract_column_headers(sql: str) -> List[str]:
    """Extract column aliases or names from the SELECT clause."""
    select_m = re.search(r"SELECT\s+(.+?)\s+FROM\b", sql, re.DOTALL | re.I)
    if not select_m:
        return []

    select_clause = select_m.group(1)
    headers: List[str] = []

    # Split by top-level commas
    depth, current = 0, ""
    for ch in select_clause:
        if ch == "(":   depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            headers.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        headers.append(current.strip())

    result = []
    for expr in headers:
        as_m = re.search(r"\bAS\s+\"?(\w+)\"?\s*$", expr, re.I)
        if as_m:
            result.append(as_m.group(1))
            continue
        doc_m = re.search(r"document->>'(\w+)'\s*$", expr)
        if doc_m:
            result.append(doc_m.group(1))
            continue
        agg_m = re.search(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, re.I)
        if agg_m:
            result.append(agg_m.group(1).lower())
            continue
        # bare column or table.column
        bare = re.search(r'(?:\w+\.)?"?(\w+)"?\s*$', expr)
        if bare:
            result.append(bare.group(1))
            continue
        result.append(expr[:20].strip())

    return result


def _format_result(rows: List[Any], sql: str, query: str) -> str:
    """Format SQL result rows into user-friendly output."""
    if not rows:
        return "No data found for this query."

    first = rows[0]
    if isinstance(first, (list, tuple)) and len(rows) == 1 and all(v is None for v in first):
        return "No data found for this query."
    if first is None:
        return "No data found for this query."

    # Single scalar value
    if len(rows) == 1 and not isinstance(first, (list, tuple)):
        return f"**{fmt_number(first)}**"

    if len(rows) == 1 and isinstance(first, (list, tuple)) and len(first) == 1:
        val = first[0]
        n   = coerce_number(val)
        if isinstance(n, (int, float)):
            return f"**{fmt_number(n)}**"
        return f"**{val}**"

    headers      = _extract_column_headers(sql)
    display_rows = rows[:_MAX_RESULT_ROWS]

    table = format_rows_as_markdown_table(
        display_rows,
        headers=headers or None,
        max_rows=_MAX_RESULT_ROWS,
        show_overflow=len(rows) > _MAX_RESULT_ROWS,
    )

    count_str = (
        f"**{len(rows)}**" if len(rows) <= _MAX_RESULT_ROWS
        else f"**{_MAX_RESULT_ROWS}** of **{len(rows)}**"
    )
    return f"Found {count_str} result(s):\n\n{table}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """Generate SQL → execute → return formatted result dict.

    Returns result dict {answer, tables_used, confidence, sql_queries}
    or None if SQL generation failed entirely.
    """
    with Timer("text2sql_total") as timer:
        LOGGER.info("Text2SQL: processing: %.60s", query)

        sql = generate_sql(query, agent)
        if not sql:
            LOGGER.warning("Text2SQL: no valid SQL generated for: %.60s", query)
            return None

        _res = run_sql(agent, sql)
        if _res.error:
            LOGGER.warning("Text2SQL exec failed: %s | SQL: %.80s", _res.error, sql)
            return None

        rows        = _res.rows
        tables_used = _extract_tables_from_sql(sql)
        body        = _format_result(rows, sql, query)

        confidence = 0.90
        if not rows or "No data found" in body:
            confidence = 0.70
        elif len(rows) == 1:
            confidence = 0.92

    LOGGER.info(
        "Text2SQL: done %.0fms | rows=%d | tables=%s | sql=%.80s",
        timer.elapsed_ms, len(rows) if rows else 0, tables_used, sql,
    )

    return {
        "answer":      body,
        "tables_used": tables_used,
        "confidence":  confidence,
        "sql_queries": [sql],
    }
