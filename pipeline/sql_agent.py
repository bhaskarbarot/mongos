"""sql_agent.py — Dynamic LLM-driven SQL generation and execution engine.

Zero hardcoding. Zero regex for intent detection. 100% LLM-driven.

Flow:
    user_query
        ↓
    build_text2sql_schema()      ← full DB schema (cached 10 min)
        ↓
    generate_sql()               ← LLM: Groq 70b → Gemini → OR → Ollama 7b
        ↓
    validate_sql()               ← SELECT-only safety check
        ↓
    run_sql()                    ← execute against PostgreSQL
        ↓ (on error → retry with error feedback, max 2 retries)
    synthesize_response()        ← LLM: rich business-language answer
        ↓
    return result dict

Public API:
    run(query, agent, memory_context="") -> Optional[Dict]
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pipeline.llm import call as llm_call
from pipeline.schema import (
    build_db_metadata,
    build_text2sql_schema,
    discover_schema_links,
    get_table_names,
    run_sql,
)
from pipeline.utils import format_rows_as_markdown_table

LOGGER = logging.getLogger("sql_chatbot")

_MAX_ROWS_FOR_SYNTHESIS = 150   # send at most N rows to synthesis LLM
_MAX_RETRIES = 2                # retry SQL generation on error


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_SQL_SYSTEM = """\
You are a senior PostgreSQL expert for a CRM business database.
Your job: translate ANY natural language business question into a correct PostgreSQL SELECT query.

IMPORTANT: Users are non-technical business people. They use everyday words, not database terms.
You must intelligently map their intent to the correct tables and columns.

━━━ DATABASE SCHEMA (use ONLY these columns — never invent column names) ━━━
{schema}

━━━ LIVE DATABASE METADATA (actual values — use exactly as shown) ━━━
{metadata}

━━━ BUSINESS VOCABULARY GUIDE (user says → database means) ━━━

DEALS:
  "categories" / "types" / "groups" / "classification"  → GROUP BY deals.stage
  "deal stages" / "pipeline stages" / "deal phases"     → GROUP BY deals.stage
  "type of deals" / "deal types"                        → GROUP BY deals.stage
  "won" / "successful" / "closed" / "converted deals"   → stage='Closed Won' OR "dealWonAt" IS NOT NULL
  "lost" / "failed" / "dropped" / "rejected deals"      → stage='Closed Lost' OR "dealLostAt" IS NOT NULL
  "open" / "active" / "ongoing" / "in progress" deals   → stage NOT IN ('Closed Won','Closed Lost')
  "at risk" / "stuck" / "not moving" deals              → open deals WHERE "closeDate"::timestamptz < NOW()
  "pipeline" / "deal pipeline"                          → deals table, GROUP BY stage
  "negotiable deals" / "deals in negotiation"           → stage ILIKE '%Negotiation%'
  "proposal stage" / "deals in proposal"                → stage ILIKE '%Quotation%' OR stage ILIKE '%Proposal%'
  "deal value" / "potential revenue" / "pipeline value" → SUM(deals.grand_total_in_usd)
  "win rate" / "conversion rate" (deals)
    → ROUND(100.0*COUNT(CASE WHEN stage='Closed Won' THEN 1 END)/NULLIF(COUNT(*),0),2) AS win_rate_pct
  "average deal size"                                   → AVG(grand_total_in_usd) GROUP BY stage
  "deals by year and stage"                             → GROUP BY EXTRACT(year FROM NULLIF("closeDate",'')::timestamptz), stage
  "year wise stages" / "yearly breakdown"               → GROUP BY year, stage ORDER BY year DESC

REVENUE & INVOICES:
  "revenue" / "income" / "earnings" / "money received" / "billing" / "collection"
    → SUM(invoices.grandtotal_in_usd) WHERE payment_status='paid'
  "revenue in 2025" / "revenue of 2025" / "2025 income" / "total revenue generated in 2025"
    → WHERE payment_status='paid' AND EXTRACT(year FROM NULLIF("invoice_date",'')::timestamptz)=2025
  "total revenue generated till date" / "total collection"
    → SELECT ROUND(SUM(grandtotal_in_usd)::numeric,2) FROM invoices WHERE payment_status='paid'
  "total sales for last month" / "revenue last month"
    → paid invoices WHERE DATE_TRUNC('month',NULLIF("invoice_date",'')::timestamptz)=DATE_TRUNC('month',NOW()-INTERVAL '1 month')
  "revenue this financial year" (India FY Apr–Mar)
    → WHERE payment_status='paid' AND NULLIF("invoice_date",'')::timestamptz >= DATE_TRUNC('year', NOW() - INTERVAL '3 months') + INTERVAL '3 months'
  "revenue by company" / "customer-wise revenue"
    → JOIN companies, GROUP BY "companyName", SUM(grandtotal_in_usd) WHERE payment_status='paid'
  "revenue summary for last 6 months" / "monthly trend"
    → GROUP BY DATE_TRUNC('month', NULLIF("invoice_date",'')::timestamptz), SUM paid ORDER BY month
  "pending bills" / "unpaid" / "outstanding" / "pending invoices"
    → payment_status NOT IN ('paid','cancelled')
  "overdue" / "past due" / "overdue payments"
    → NULLIF("due_date",'')::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')
  "invoices by status" / "invoice status"               → GROUP BY payment_status ORDER BY COUNT(*) DESC
  "first invoice" / "1st invoice"                       → ORDER BY NULLIF("invoice_date",'')::timestamptz ASC NULLS LAST LIMIT 1
  "invoices due next month"
    → DATE_TRUNC('month',NULLIF("due_date",'')::timestamptz)=DATE_TRUNC('month',NOW()+INTERVAL '1 month')
  "INR invoices" → WHERE currency='INR' | "USD invoices" → WHERE currency='USD'
  "top 10 invoices by amount"                           → ORDER BY grandtotal_in_usd DESC LIMIT 10
  "paid invoices"                                       → WHERE payment_status='paid'

CUSTOMERS & COMPANIES:
  "customers" / "clients" / "accounts" / "businesses"  → companies table
  "top customers by revenue"
    → JOIN invoices, GROUP BY "companyName", SUM(grandtotal_in_usd) WHERE paid, ORDER BY revenue DESC
  "customers with pending invoices"
    → JOIN invoices WHERE payment_status NOT IN ('paid','cancelled')
  "customers with no business in N months"
    → companies LEFT JOIN invoices ON i.company=c._id AND invoice_date >= NOW()-INTERVAL 'N months' WHERE i._id IS NULL
  "companies by region / source"                        → GROUP BY region or source column
  "new leads this month"
    → contacts WHERE "lifecycleStage"='Lead' AND DATE_TRUNC('month',NULLIF("createdAt",'')::timestamptz)=DATE_TRUNC('month',NOW())
  "leads not contacted in 7 days"
    → contacts WHERE "lifecycleStage"='Lead' AND NULLIF("updatedAt",'')::timestamptz < NOW()-INTERVAL '7 days'
  "which contacts have no deals"
    → contacts c LEFT JOIN deals d ON d.contact=c._id WHERE d._id IS NULL

TASKS:
  "tasks" / "follow-ups" / "to-dos" / "action items"   → createtasks table ("Task" column = capital T)
  "pending tasks" / "open tasks"                        → WHERE status='Pending'
  "completed tasks"                                     → WHERE status='Completed'
  "overdue tasks"                                       → NULLIF("due_date",'')::timestamptz < NOW() AND status!='Completed'
  "who has pending tasks" / "pending tasks per user"
    → GROUP BY u.name (JOIN users ON u._id=t."createdBy"), COUNT(*) WHERE t.status='Pending'
  "tasks for sales team"                                → JOIN users WHERE department ILIKE '%sales%'
  "[name]'s tasks" / "tasks by [name]" / "tasks for [name]"
    → t."createdBy" IN (SELECT _id FROM users WHERE name ILIKE '%name%')
  "follow-ups overdue this week"
    → due_date < NOW() AND status!='Completed' AND due_date >= DATE_TRUNC('week',NOW())

PEOPLE & USERS:
  "who is [name]" / "details of [name]" / "find [name]"
    → contacts WHERE "firstName" ILIKE '%name%' OR "lastName" ILIKE '%name%' OR email ILIKE '%name%'
    → UNION users WHERE name ILIKE '%name%'
  "summary of [person]" / "profile of [person]"        → full detail columns from contacts or users
  "which sales rep closed most"                         → JOIN deals, GROUP BY u.name, COUNT closed won, ORDER BY DESC
  "target vs achieved"                                  → targets table JOIN users

MEETINGS:
  "meetings for today"  → DATE(NULLIF("createdAt",'')::timestamptz)=CURRENT_DATE
  "meetings for [date]" → DATE(date_field)='YYYY-MM-DD'

ANALYSIS:
  "KPI report" / "dashboard" / "today's status" / "give me KPI"
    → SELECT (SELECT COUNT(*) FROM deals WHERE not deleted) AS total_deals,
             (SELECT COUNT(*) FROM deals WHERE stage NOT IN ('Closed Won','Closed Lost') AND (deleted=false OR deleted IS NULL)) AS open_deals,
             (SELECT ROUND(SUM(grandtotal_in_usd)::numeric,2) FROM invoices WHERE payment_status='paid') AS total_revenue,
             (SELECT COUNT(*) FROM createtasks WHERE status='Pending' AND (deleted=false OR deleted IS NULL)) AS pending_tasks,
             (SELECT COUNT(*) FROM invoices WHERE payment_status NOT IN ('paid','cancelled') AND (deleted=false OR deleted IS NULL)) AS pending_invoices
  "funnel / conversion rate lead to customer"
    → companies: COUNT where lifecycleStage='Lead' vs 'Customer', percentage
  "high activity companies with weak payment"
    → companies JOIN invoices, GROUP BY company, COUNT invoices, SUM paid vs total ratio
  "pipeline by stage" / "pipeline distribution"
    → GROUP BY stage, COUNT(*), SUM(grand_total_in_usd) ORDER BY COUNT DESC
  "compare this vs last month/year"
    → use CTEs: WITH this_period AS (...), last_period AS (...) SELECT both

TIME EXPRESSIONS:
  "in 2025" / "of 2025" / "year 2025"    → EXTRACT(year FROM NULLIF("field",'')::timestamptz)=2025
  "this year"                             → EXTRACT(year FROM NULLIF("field",'')::timestamptz)=EXTRACT(year FROM NOW())
  "last year"                             → EXTRACT(year FROM NULLIF("field",'')::timestamptz)=EXTRACT(year FROM NOW())-1
  "this month"                            → DATE_TRUNC('month',NULLIF("field",'')::timestamptz)=DATE_TRUNC('month',NOW())
  "last month"                            → DATE_TRUNC('month',...)=DATE_TRUNC('month',NOW()-INTERVAL '1 month')
  "this week"                             → DATE_TRUNC('week',...)=DATE_TRUNC('week',NOW())
  "last 3 months" / "last 6 months"      → NULLIF("field",'')::timestamptz >= NOW()-INTERVAL 'N months'
  "last quarter"                          → DATE_TRUNC('quarter',...)=DATE_TRUNC('quarter',NOW()-INTERVAL '3 months')
  "Q1/Q2/Q3/Q4"                          → EXTRACT(quarter FROM NULLIF("field",'')::timestamptz)=N
  "today"                                 → DATE(NULLIF("field",'')::timestamptz)=CURRENT_DATE
  "yesterday"                             → DATE(NULLIF("field",'')::timestamptz)=CURRENT_DATE-1
  "financial year" India (Apr 1 – Mar 31) → >= DATE_TRUNC('year',NOW()-INTERVAL '3 months')+INTERVAL '3 months'

━━━ CURRENT DATE/TIME: {now} ━━━

━━━ SQL RULES (follow ALL) ━━━
1.  Output ONLY raw SQL — no markdown fences, no explanation, no comments
2.  Only SELECT — NEVER INSERT/UPDATE/DELETE/DROP/CREATE/TRUNCATE/ALTER
3.  ANTI-HALLUCINATION: use ONLY column names visible in the SCHEMA section above.
    Never invent or guess column names. Check schema first.
4.  Soft-delete filter on EVERY table used:
    • All tables:  (deleted = false OR deleted IS NULL)
    • outreaches:  ("isDeleted" = false OR "isDeleted" IS NULL)
5.  LEFT JOIN users/companies to replace ObjectId hex strings with real names
6.  All date fields are TEXT — cast: NULLIF("field_name", '')::timestamptz
7.  Monetary amounts: ROUND(value::numeric, 2)
8.  Text search: ILIKE '%term%'  (case-insensitive)
9.  Default LIMIT 100 unless user asks for a specific count
10. Table aliases: d=deals, i=invoices, u=users, c=companies, t=createtasks, s=sales
11. All computed/joined columns need meaningful AS aliases
12. GROUP BY queries: ORDER BY the aggregate metric DESC
13. COALESCE nulls: COALESCE(field, 0) for numbers, COALESCE(field, 'Unknown') for text
14. Comparison queries (this vs last): use CTEs
15. KPI/dashboard: pack all metrics in one SELECT with subqueries
"""

_RETRY_SUFFIX = """

─── PREVIOUS ATTEMPT FAILED ───
Failed SQL:
{failed_sql}

Error message:
{error}

Write a CORRECTED SQL query. Output ONLY the fixed SQL — nothing else.
"""

_SYNTHESIS_SYSTEM = """\
You are a senior CRM business analyst presenting database query results to a non-technical business user.

━━━ ZERO HALLUCINATION RULES (most important) ━━━
• ONLY use numbers, names, and values that appear in the SQL results below
• NEVER invent data, assume trends, or add information not in the results
• NEVER round numbers differently from what the data shows
• If 0 rows returned: say clearly what was searched and that no data was found — do NOT guess why
• Do not mention percentages or ratios unless the SQL result contains them

━━━ RESPONSE FORMAT ━━━
1. Lead with a direct 1-2 sentence answer (e.g. "There are **23 open deals** in your pipeline.")
2. Always mention the total count or sum where relevant
3. For 1 number result: state it clearly in bold, add context (what period, what filter)
4. For list results (>3 rows): show as a clean markdown table with column headers
5. For group-by / breakdown results: summarize each category, highlight the top 3
6. Add 2-3 sentences of plain-English business insight based on what the data actually shows
7. Use **bold** for key numbers, totals, and important names
8. Use plain English — no SQL terms (no "WHERE clause", "JOIN", "NULL", "column name" etc.)
9. For time-filtered results: always mention the time period (e.g. "for March 2025")
10. For comparison results (this vs last month): show both values and the difference clearly

━━━ STYLE ━━━
• Speak like a helpful business analyst, not a programmer
• Keep it concise but complete — cover all the data returned
• Use bullet points for multiple items when a table isn't needed
• End with one actionable insight or observation if relevant
"""

_SYNTHESIS_USER = """\
User's question: "{query}"

SQL that was executed:
{sql}

Query results — {row_count} rows returned:
{data}

Using ONLY the data shown above (no invented facts), write a clear and helpful response to the user's question.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SQL CLEANING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _clean_sql(raw: str) -> str:
    """Strip markdown fences and leading/trailing non-SQL text from LLM output."""
    text = raw.strip()

    # Remove ```sql ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:sql|postgres|postgresql)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # If LLM prefixed with explanation, extract the SELECT/WITH block
    lines = text.splitlines()
    sql_lines: List[str] = []
    found = False
    for line in lines:
        stripped = line.strip().upper()
        if not found and (stripped.startswith("SELECT") or stripped.startswith("WITH ")):
            found = True
        if found:
            sql_lines.append(line)

    return "\n".join(sql_lines).strip() if sql_lines else text.strip()


_BLOCKED_KEYWORDS = [
    "INSERT ", "UPDATE ", "DELETE ", "DROP ", "CREATE ",
    "TRUNCATE ", "ALTER ", "GRANT ", "REVOKE ", "EXECUTE ",
    "CALL ", "COPY ", "VACUUM ", "REINDEX ",
]


def _validate_sql(sql: str) -> Tuple[bool, str]:
    """Ensure SQL is a safe SELECT. Returns (is_valid, reason)."""
    if not sql:
        return False, "empty SQL"
    upper = sql.upper().strip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH ")):
        return False, f"must start with SELECT or WITH, got: {sql[:50]!r}"
    for kw in _BLOCKED_KEYWORDS:
        if kw in upper:
            return False, f"blocked keyword detected: {kw.strip()}"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _format_for_synthesis(rows: List, limit: int = _MAX_ROWS_FOR_SYNTHESIS) -> str:
    """Format SQL rows into readable text for the synthesis LLM."""
    if not rows:
        return "(no rows returned)"
    subset = rows[:limit]
    table_md = format_rows_as_markdown_table(subset, max_rows=limit)
    if len(rows) > limit:
        table_md += f"\n\n*(showing first {limit} of {len(rows)} total rows)*"
    return table_md


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════

def generate_sql(
    query: str,
    schema_context: str,
    metadata_context: str = "",
    memory_context: str = "",
    previous_error: str = "",
    failed_sql: str = "",
) -> Optional[str]:
    """Ask the LLM to generate a PostgreSQL SELECT for the user's query.

    Uses the 'sql' provider chain (Groq 70b → Gemini → OpenRouter → Ollama 7b)
    for maximum SQL quality and accuracy.

    Returns raw SQL string, or None if all providers failed.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system = _SQL_SYSTEM.format(
        schema=schema_context,
        metadata=metadata_context or "(metadata not available)",
        now=now,
    )

    mem_block = (
        f"\nCONVERSATION CONTEXT (use this to resolve pronouns/references):\n{memory_context}\n"
        if memory_context else ""
    )
    user_msg = f"{mem_block}User question: {query}"

    if previous_error and failed_sql:
        user_msg += _RETRY_SUFFIX.format(failed_sql=failed_sql, error=previous_error)

    raw = llm_call("sql", system, user_msg, max_tokens=600)
    if not raw:
        LOGGER.warning("sql_agent.generate_sql: all LLM providers failed")
        return None

    return _clean_sql(raw)


def synthesize_response(
    query: str,
    sql: str,
    rows: List,
    memory_context: str = "",
) -> str:
    """Ask the LLM to turn SQL results into a rich business-language response.

    Returns markdown-formatted response string.
    Falls back to a plain table if all LLM providers fail.
    """
    row_count = len(rows)
    data_str = _format_for_synthesis(rows)

    user_msg = _SYNTHESIS_USER.format(
        query=query,
        sql=sql,
        row_count=row_count,
        data=data_str,
    )

    response = llm_call("synthesize", _SYNTHESIS_SYSTEM, user_msg, max_tokens=1500)
    if response:
        return response.strip()

    # Fallback: plain answer without LLM narration
    LOGGER.warning("sql_agent.synthesize_response: all LLM providers failed — using raw table")
    if not rows:
        return "No data found matching your query. Please try rephrasing or check your filters."
    return f"**Found {row_count} results:**\n\n{data_str}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    query: str,
    agent,
    memory_context: str = "",
) -> Optional[Dict[str, Any]]:
    """Full pipeline: query → SQL → execute → synthesize → result dict.

    Args:
        query:          User's natural language question
        agent:          LangChain AgentExecutor with sql_db_query tool
        memory_context: Optional conversation context string from ChatMemory

    Returns:
        Dict with: answer, tables_used, sql_queries, confidence, layer, _rows
        Returns None on complete failure (all retries exhausted, LLM unavailable).
    """
    t0 = time.perf_counter()

    # ── Schema + metadata context (both cached after first call) ─────────────
    schema_ctx = build_text2sql_schema(agent)
    if not schema_ctx:
        LOGGER.error("sql_agent: no schema context — cannot generate SQL")
        return None

    # Live DB metadata: actual enum values, row counts, column name reminders
    try:
        metadata_ctx = build_db_metadata(agent)
    except Exception as exc:
        LOGGER.warning("sql_agent: metadata build failed (non-fatal): %s", exc)
        metadata_ctx = ""

    table_names = get_table_names(agent)

    # ── SQL generation with retry on execution error ──────────────────────────
    sql: Optional[str] = None
    rows: Optional[List] = None
    last_error = ""
    last_sql = ""

    for attempt in range(_MAX_RETRIES + 1):
        # Generate SQL (includes error feedback on retries)
        candidate_sql = generate_sql(
            query=query,
            schema_context=schema_ctx,
            metadata_context=metadata_ctx,
            memory_context=memory_context,
            previous_error=last_error if attempt > 0 else "",
            failed_sql=last_sql if attempt > 0 else "",
        )

        if not candidate_sql:
            LOGGER.warning("sql_agent: no SQL from LLM (attempt %d/%d)", attempt + 1, _MAX_RETRIES + 1)
            continue

        # Safety validation
        valid, reason = _validate_sql(candidate_sql)
        if not valid:
            LOGGER.warning(
                "sql_agent: unsafe SQL on attempt %d: %s | sql=%.80s",
                attempt + 1, reason, candidate_sql,
            )
            last_error = f"SQL validation failed: {reason}"
            last_sql = candidate_sql
            continue

        LOGGER.info(
            "sql_agent: executing SQL (attempt %d) | sql=%.80s",
            attempt + 1, candidate_sql,
        )

        # Execute
        result = run_sql(agent, candidate_sql)
        if result.error:
            LOGGER.warning(
                "sql_agent: SQL error (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES + 1, result.error,
            )
            last_error = result.error
            last_sql = candidate_sql
            continue

        # Success
        sql = candidate_sql
        rows = result.rows
        LOGGER.info(
            "sql_agent: SUCCESS (attempt %d) | %d rows | %.0fms",
            attempt + 1, len(rows), (time.perf_counter() - t0) * 1000,
        )
        break

    if sql is None or rows is None:
        LOGGER.error(
            "sql_agent: FAILED after %d attempts | last_error=%s | last_sql=%.60s",
            _MAX_RETRIES + 1, last_error, last_sql,
        )
        return None

    # ── Response synthesis ────────────────────────────────────────────────────
    answer = synthesize_response(query, sql, rows, memory_context)

    # ── Extract referenced table names from SQL ───────────────────────────────
    sql_upper = sql.upper()
    used_tables = []
    for tbl in table_names:
        pattern = rf'(?:FROM|JOIN)\s+"{re.escape(tbl)}"'
        pattern2 = rf'(?:FROM|JOIN)\s+{re.escape(tbl)}\b'
        if re.search(pattern, sql, re.I) or re.search(pattern2, sql, re.I):
            used_tables.append(tbl)

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    LOGGER.info(
        "sql_agent: DONE | %.0fms | tables=%s | rows=%d",
        elapsed, used_tables, len(rows),
    )

    return {
        "answer":      answer,
        "tables_used": used_tables,
        "sql_queries": [sql],
        "confidence":  0.92,
        "layer":       "sql_agent",
        "_rows":       rows,
    }
