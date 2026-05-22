"""medium_agent.py — LangGraph StateGraph for MEDIUM complexity CRM queries.

Handles: multi-table joins, analysis/calculation on data, simple period comparisons.
Target latency: < 15 seconds.

Graph flow (7 nodes):
  fetch_schema → decompose → generate_sql → execute_parallel
    → analyze → synthesize → save_examples

All LLM calls route through pipeline/llm_router.py (Groq→Gemini→OpenRouter→Ollama).
All SQL execution uses direct psycopg2 via schema._run_sql_direct().
Live schema always from pipeline/db_schema.py (zero hardcoded column names).
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as futures_wait
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from config import settings
from pipeline.utils import check_hallucinated_tables, check_mandatory_filters

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# DATE CONTEXT HELPER  (shared by medium + complex agents)
# ══════════════════════════════════════════════════════════════════════════════

def _date_context() -> str:
    """Return a compact current date/time context block for LLM prompts.

    Gives agents awareness of today's date so they can correctly resolve
    relative expressions like 'last month', 'last 7 days', 'this year'.
    """
    now        = datetime.now()
    today      = now.strftime("%Y-%m-%d")
    today_full = now.strftime("%A, %B %d, %Y")

    # Last month
    first_this_month  = now.replace(day=1)
    last_month_end    = first_this_month - timedelta(days=1)
    last_month_start  = last_month_end.replace(day=1)
    last_month_num    = last_month_start.month
    last_month_year   = last_month_start.year
    last_month_name   = last_month_start.strftime("%B %Y")

    # This month
    this_month_num  = now.month
    this_month_year = now.year
    this_month_name = now.strftime("%B %Y")

    # Common intervals
    days_7    = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    days_30   = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    months_3  = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    months_6  = (now - timedelta(days=180)).strftime("%Y-%m-%d")
    months_12 = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    # Last quarter (the previous completed calendar quarter)
    current_q   = (now.month - 1) // 3 + 1          # 1..4
    last_q      = current_q - 1 if current_q > 1 else 4
    last_q_year = this_month_year if current_q > 1 else this_month_year - 1
    last_q_months = {1: (1,2,3), 2: (4,5,6), 3: (7,8,9), 4: (10,11,12)}[last_q]
    lq_in_list  = ",".join(str(m) for m in last_q_months)

    return (
        f"-- CURRENT DATE CONTEXT (MANDATORY: apply these exact values for any time-based query):\n"
        f"-- TODAY          : {today} ({today_full})\n"
        f"-- THIS MONTH     : {this_month_name}  → month={this_month_num}, year={this_month_year}\n"
        f"-- LAST MONTH     : {last_month_name}  → month={last_month_num}, year={last_month_year}\n"
        f"--                  SQL: EXTRACT(MONTH FROM <date_col>)={last_month_num} AND EXTRACT(YEAR FROM <date_col>)={last_month_year}\n"
        f"-- LAST QUARTER   : Q{last_q} {last_q_year} (months {lq_in_list}) — the PREVIOUS completed quarter\n"
        f"--                  SQL: EXTRACT(MONTH FROM <date_col>) IN ({lq_in_list}) AND EXTRACT(YEAR FROM <date_col>)={last_q_year}\n"
        f"--                  ✗ NEVER use EXTRACT(QUARTER FROM ...) — always use month IN list above\n"
        f"-- LAST 7 DAYS    : {days_7} to {today}   → SQL: <date_col> >= '{days_7}'\n"
        f"-- LAST 30 DAYS   : {days_30} to {today}  → SQL: <date_col> >= '{days_30}'\n"
        f"-- LAST 3 MONTHS  : {months_3} to {today} → SQL: <date_col> >= '{months_3}'\n"
        f"-- LAST 6 MONTHS  : {months_6} to {today} → SQL: <date_col> >= '{months_6}'\n"
        f"-- LAST 12 MONTHS : {months_12} to {today}→ SQL: <date_col> >= '{months_12}'\n"
        f"-- THIS YEAR      : year={this_month_year}  → SQL: EXTRACT(YEAR FROM <date_col>)={this_month_year}\n"
        f"-- LAST YEAR      : year={this_month_year-1}\n"
        f"-- DATE COLUMN PER TABLE (replace <date_col> with correct column):\n"
        f"--   invoices revenue  → payment_date   (ALWAYS use payment_date for revenue time filters)\n"
        f"--   invoices created  → createdAt\n"
        f"--   deals             → createdAt (created), dealWonAt (won), dealLostAt (lost)\n"
        f"--   sales             → sales_date\n"
        f"--   contacts/companies→ createdAt\n"
        f"--   tasks             → due_date (due), createdAt (created)\n"
        f"-- RULE: If the query mentions ANY time period (last month, this year, last 7 days etc.)\n"
        f"--       you MUST add the corresponding date filter from above — NEVER omit it.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class MediumState(TypedDict):
    query:          str
    classification: Dict[str, Any]
    schema:         Dict[str, Any]        # from MCP / db_schema
    sub_queries:    List[Dict[str, str]]  # [{sql_intent, tables_needed, expected_output_type}]
    sql_results:    List[Dict[str, Any]]  # per-sub-query results
    analysis:       str                   # LLM analysis of the combined results
    final_response: str
    attempts:       int
    errors:         List[str]
    t_start:        float


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SQL HELPERS  (also used by complex_agent)
# ══════════════════════════════════════════════════════════════════════════════

def extract_sql(raw: str) -> Optional[str]:
    """Extract a SELECT statement from LLM output."""
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"(SELECT\b.+?)(?:;|\n{3,}|$)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip().rstrip(";")
        if len(sql) > 10 and sql.upper().startswith("SELECT"):
            if not re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE)\b", sql, re.IGNORECASE):
                return sql
    return None


_SQL_PSEUDO_TABLES = {
    "nullif", "coalesce", "current_date", "current_timestamp", "now",
    "extract", "date_trunc", "date_part", "interval", "rank", "row_number",
    "dense_rank", "lateral", "unnest", "generate_series", "values", "dual",
    "information_schema", "pg_tables",
}

def extract_tables(sql: str) -> List[str]:
    matches = re.findall(r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', sql, re.IGNORECASE)
    return sorted(set(m for m in matches if m and m.lower() not in _SQL_PSEUDO_TABLES))


def extract_bad_column(error: str) -> Optional[str]:
    m = re.search(
        r'column\s+"?([^"\s]+)"?\s+(?:does not exist|of relation)',
        error, re.IGNORECASE,
    )
    return m.group(1).strip().strip('"') if m else None


def run_sql_with_selfheal(
    sub_query_text: str,
    schema_str:     str,
    examples_str:   str,
    all_tables:     List[str],
    max_attempts:   int = 3,
) -> Dict[str, Any]:
    """Generate SQL for a sub-query and execute it with self-healing (max 3 attempts).

    Returns: {success, rows, columns, row_count, sql_used, tables_used, attempts, error}
    """
    from pipeline.schema_compact import COMPACT_SYSTEM_PROMPT
    from pipeline.llm_router import call as llm_call
    from pipeline.db_schema import find_correct_column
    from mcp_server.crm_mcp import execute_sql

    last_error: Optional[str] = None
    last_sql:   Optional[str] = None
    heal_hint:  str           = ""

    for attempt in range(1, max_attempts + 1):
        parts = [
            f"-- Few-shot examples:\n{examples_str}",
            f"\n-- Schema:\n{schema_str}",
            f"\nQuestion: {sub_query_text}",
        ]
        if heal_hint:
            parts.append(f"\n-- Fix required: {heal_hint}")
        parts.append("\nSQL:")

        raw = llm_call("sql", COMPACT_SYSTEM_PROMPT, "\n".join(parts))
        if not raw:
            last_error = f"LLM returned no output (attempt {attempt})"
            continue

        sql = extract_sql(raw)
        if not sql:
            last_error = f"Could not extract SQL from LLM output (attempt {attempt})"
            continue

        # ── Hallucination guard ───────────────────────────────────────────────
        halluc_err = check_hallucinated_tables(sql)
        if halluc_err:
            last_error = halluc_err
            heal_hint  = halluc_err
            LOGGER.warning("Medium agent attempt %d HALLUCINATED TABLE: %s", attempt, halluc_err[:120])
            continue

        # ── Mandatory filter guard ────────────────────────────────────────────
        filter_err = check_mandatory_filters(sql, sub_query_text)
        if filter_err:
            last_error = filter_err
            heal_hint  = filter_err
            LOGGER.warning("Medium agent attempt %d MISSING FILTER: %s", attempt, filter_err[:120])
            continue

        last_sql = sql
        result   = execute_sql(sql)

        if result.get("error"):
            last_error = result["error"]
            err_lower  = last_error.lower()

            if "column" in err_lower and "exist" in err_lower:
                bad = extract_bad_column(last_error)
                if bad:
                    correct = find_correct_column(bad, all_tables)
                    heal_hint = (
                        f"Column '{bad}' does not exist. "
                        + (f"Use '{correct}' instead." if correct
                           else "Check the schema above for the real column name.")
                    )
                else:
                    heal_hint = f"Fix: {last_error[:200]}"

            elif "aggregate functions are not allowed in where" in err_lower:
                heal_hint = (
                    "CRITICAL FIX: You used SUM()/COUNT() inside a WHERE clause — this is illegal in SQL. "
                    "NEVER put aggregate functions in WHERE. Use one of these patterns instead:\n"
                    "PATTERN 1 — HAVING (preferred):\n"
                    "  SELECT u.name FROM \"users\" u\n"
                    "  JOIN \"targets\" t ON t.\"userId\" = u._id\n"
                    "  LEFT JOIN \"sales\" s ON s.\"salesOwner\" = u._id\n"
                    "    AND EXTRACT(MONTH FROM s.sales_date) = t.month\n"
                    "    AND EXTRACT(YEAR FROM s.sales_date) = t.year\n"
                    "    AND s.status = 'Confirm' AND NOT s.deleted\n"
                    "  WHERE t.month = 4 AND t.year = 2026\n"
                    "  GROUP BY u.name, t.\"targetInUSD\"\n"
                    "  HAVING COALESCE(SUM(s.grand_total_in_usd), 0) >= t.\"targetInUSD\"\n"
                    "PATTERN 2 — Correlated subquery:\n"
                    "  WHERE (SELECT COALESCE(SUM(s.grand_total_in_usd),0) FROM \"sales\" s\n"
                    "         WHERE s.\"salesOwner\"=u._id AND s.status='Confirm'\n"
                    "         AND NOT s.deleted) >= t.\"targetInUSD\"\n"
                    "Rewrite the SQL using PATTERN 1 or PATTERN 2."
                )

            elif "syntax error" in err_lower and "<=" in last_error:
                heal_hint = (
                    "SQL syntax error with range comparison. "
                    "PostgreSQL does NOT support chained comparisons like 'a <= x <= b'. "
                    "Use: x BETWEEN a AND b  OR  (x >= a AND x <= b). "
                    "For month range: use t.month IN (2, 3, 4) or t.month BETWEEN 2 AND 4. "
                    "Rewrite the SQL fixing this syntax."
                )

            elif "missing from-clause entry" in err_lower:
                heal_hint = (
                    "Subquery alias scope error. The outer SELECT cannot use ANY table alias "
                    "(c., d., u., s., i., t. etc.) that is defined INSIDE the subquery. "
                    "Fix: remove ALL table-alias prefixes from the outer SELECT and use plain column aliases.\n"
                    "WRONG: SELECT u.name, cnt FROM (SELECT u.name, COUNT(*) AS cnt ...) sub\n"
                    "RIGHT: SELECT user_name, cnt FROM (SELECT u.name AS user_name, COUNT(*) AS cnt ...) sub\n"
                    "WRONG: SELECT c.\"companyName\", deal_count FROM (SELECT c.\"companyName\" ...) ranked\n"
                    "RIGHT: SELECT company_name, deal_count FROM (SELECT c.\"companyName\" AS company_name ...) ranked"
                )

            elif "ambiguous" in err_lower:
                heal_hint = (
                    "Column is ambiguous. Prefix every column with its table alias "
                    "(e.g., d.name, u.name — not just name)."
                )
            else:
                heal_hint = f"Fix this error: {last_error[:200]}"

            LOGGER.warning("Sub-query attempt %d failed: %s | sql: %.80s", attempt, last_error, sql)
            continue

        # Success
        rows    = result.get("rows", [])
        columns = result.get("columns", [])
        return {
            "success":     True,
            "rows":        rows,
            "columns":     columns,
            "row_count":   len(rows),
            "sql_used":    sql,
            "tables_used": extract_tables(sql),
            "attempts":    attempt,
            "error":       None,
        }

    return {
        "success":     False,
        "rows":        [],
        "columns":     [],
        "row_count":   0,
        "sql_used":    last_sql or "",
        "tables_used": [],
        "attempts":    max_attempts,
        "error":       str(last_error),
    }


def format_results_for_llm(sql_results: List[Dict]) -> str:
    """Format all SQL results into a concise text block for LLM analysis.

    For money/revenue results: applies currency conversion to USD first,
    then passes the USD-converted view to the LLM (no hallucination risk —
    numbers come from Frankfurter live rates applied to real DB values).
    """
    from agents.currency_format import apply_currency_conversion, is_money_query

    blocks = []
    for i, res in enumerate(sql_results, 1):
        sq   = res.get("sub_query", f"Sub-query {i}")
        data = res.get("result_data", {})
        if not data.get("success"):
            blocks.append(f"--- Part {i}: {sq}\nError: {data.get('error', 'unknown')}\n")
            continue

        rows    = data.get("rows", [])
        columns = data.get("columns", [])
        sql_used = data.get("sql_used", "")

        if not rows:
            blocks.append(f"--- Part {i}: {sq}\nResult: No records found.\n")
            continue

        # ── Currency-aware formatting for money results ────────────────────────
        if sql_used and is_money_query(sql_used):
            currency_block = apply_currency_conversion(rows, columns, sql_used)
            if currency_block:
                blocks.append(f"--- Part {i}: {sq}\n{currency_block}\n")
                continue

        # ── Standard compact table (max 30 rows for LLM token budget) ─────────
        header = " | ".join(str(c) for c in (columns or [f"col{j}" for j in range(len(rows[0]))]))
        row_lines = []
        for row in rows[:30]:
            row_lines.append(" | ".join("" if v is None else str(v) for v in row))

        total  = data.get("row_count", len(rows))
        suffix = f"\n[...{total - 30} more rows]" if total > 30 else ""
        blocks.append(f"--- Part {i}: {sq}\n{header}\n" + "\n".join(row_lines) + suffix + "\n")

    return "\n".join(blocks)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_schema_node(state: MediumState) -> Dict:
    """Node 1: Fetch live schema + inject current date context."""
    from pipeline.db_schema import build_schema_for_query, get_all_crm_tables

    tables_hint = state["classification"].get("tables_needed", [])
    schema_str  = build_schema_for_query(state["query"], hint_tables=tables_hint, max_tables=8)

    # Prepend current date context so every downstream node knows today's date
    schema_with_dates = _date_context() + "\n" + schema_str

    return {
        "schema": {
            "schema_str":  schema_with_dates,
            "all_tables":  get_all_crm_tables(),
        }
    }


def decompose_node(state: MediumState) -> Dict:
    """Node 2: Intent-preserving decomposition into 2-4 atomic sub-queries.
    Bypassed when DECOMPOSER_ENABLED=false — passes raw query as single sub-query.
    """
    from config import settings
    if not settings.decomposer_enabled:
        LOGGER.info("Medium agent: decomposer DISABLED — passing raw query directly")
        return {"sub_queries": [{"sub_query": state["query"], "intent": "general"}]}

    from pipeline.llm_router import call as llm_call

    all_tables  = state["schema"].get("all_tables", [])
    tables_str  = ", ".join(all_tables[:30])
    schema_str  = state["schema"].get("schema_str", "")
    query       = state["query"]

    system = (
        "You are a CRM query decomposer with deep SQL and domain knowledge.\n\n"
        "YOUR JOB:\n"
        "1. Understand what the user TRULY wants (don't change their intent)\n"
        "2. Break it into the MINIMUM atomic sub-queries (max 4) each answerable by 1 SQL\n"
        "3. Each sub-query must use the CORRECT tables for what it's measuring\n\n"

        "OUTPUT: JSON array ONLY. No text outside the array.\n"
        '[{"sub_query": "...", "intent": "count|list|sum|compare|rank|lookup"}]\n\n'

        "INTENT PRESERVATION RULES (never break these):\n"
        "- Keep the user's exact meaning — never substitute one metric for another\n"
        "- 'achieve target' = compare targets.targetInUSD vs SUM(confirmed sales) — NEVER use deals\n"
        "- 'revenue' = invoices.grandtotal_in_usd WHERE payment_status='paid' AND date filter on payment_date (NOT invoice_date)\n"
        "- 'confirmed sales' = sales.grand_total_in_usd WHERE status='Confirm'\n"
        "- 'last month' = the calendar month before today (see date context in schema)\n"
        "- 'last 7 days' = from 7 days ago to today (see date context in schema)\n"
        "- 'which rep/user' queries need: JOIN users table, GROUP BY user, ORDER BY metric\n\n"

        "CRM TABLE ROUTING (use correct tables):\n"
        "- Target achievement : targets (targetInUSD, userId, month, year) + sales (salesOwner, grand_total_in_usd) + users\n"
        "- Revenue / income   : invoices (grandtotal_in_usd, payment_status='paid', date on payment_date) — NOT deals\n"
        "- Sales orders       : sales (grand_total_in_usd, status='Confirm', salesOwner)\n"
        "- Won/lost deals     : deals (dealWonAt, dealLostAt, owner, stage)\n"
        "- Pipeline health    : deals (lastActivity, stage, createdAt)\n"
        "- Tasks              : createtasks (status, priority, due_date, createdBy)\n"
        "- Users/reps         : users (name, isActive, department)\n"
        "- Companies at risk  : companies + invoices/sales JOIN to check last activity\n\n"

        "DATE RULE: The schema above has a CURRENT DATE CONTEXT block — use those exact\n"
        "dates/months for any date filtering. Never guess dates.\n\n"

        "SIZE RULE: Use minimum sub-queries needed:\n"
        "- 1 sub-query: single metric with filter\n"
        "- 2 sub-queries: compare two things OR one lookup + one calculation\n"
        "- 3-4 sub-queries: multiple independent metrics needed\n"
        "- NEVER create sub-queries that aren't needed to answer the query"
    )

    user = (
        f"Schema (includes current date context):\n{schema_str[:800]}\n\n"
        f"Available tables: {tables_str}\n\n"
        f'Query: "{query}"\n\n'
        "Decompose into sub-queries (JSON array only):"
    )

    raw = llm_call("decompose", system, user, max_tokens=500)
    sub_queries = _parse_sub_queries(raw)

    if not sub_queries:
        sub_queries = [{"sub_query": query, "intent": "general"}]

    LOGGER.info("Medium agent decomposed into %d sub-queries", len(sub_queries))
    return {"sub_queries": sub_queries}


def _parse_sub_queries(raw: Optional[str]) -> List[Dict[str, str]]:
    """Parse JSON array of sub-queries from LLM output."""
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()

    # Find JSON array
    start = cleaned.find("[")
    end   = cleaned.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    try:
        items = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return []

    valid = []
    allowed_intents = {"count", "list", "sum", "average", "filter",
                       "compare", "lookup", "rank", "trend", "general"}
    for item in items:
        if not isinstance(item, dict):
            continue
        sq = str(item.get("sub_query", "")).strip()
        if not sq or len(sq) < 4:
            continue
        intent = str(item.get("intent", "general")).strip().lower()
        if intent not in allowed_intents:
            intent = "general"
        valid.append({"sub_query": sq, "intent": intent})

    return valid[:4]  # max 4 sub-queries for medium


def generate_sql_node(state: MediumState) -> Dict:
    """Node 3: Prepare schema + examples for parallel SQL execution.

    (SQL is actually generated inside execute_parallel_node for true parallelism)
    """
    from mcp_server.crm_mcp import mcp_get_examples

    examples    = mcp_get_examples()[:4]
    examples_str = "\n".join(
        f"-- Q: {ex['natural_query']}\n{ex['sql']}" for ex in examples
    )

    return {
        "schema": {
            **state["schema"],
            "examples_str": examples_str,
        }
    }


def execute_parallel_node(state: MediumState) -> Dict:
    """Node 4: Run all sub-queries in parallel with self-healing (max 3 attempts each)."""
    sub_queries  = state["sub_queries"]
    schema_str   = state["schema"].get("schema_str", "")
    examples_str = state["schema"].get("examples_str", "")
    all_tables   = state["schema"].get("all_tables", [])

    sql_results:  List[Dict] = []
    errors:       List[str]  = list(state.get("errors", []))

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 6),
                            thread_name_prefix="med_exec") as pool:
        future_map = {
            pool.submit(
                run_sql_with_selfheal,
                sq["sub_query"], schema_str, examples_str, all_tables,
            ): sq
            for sq in sub_queries
        }

        done, not_done = futures_wait(future_map, timeout=80)

        for future in done:
            sq = future_map[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {"success": False, "rows": [], "error": str(exc)}
            sql_results.append({"sub_query": sq["sub_query"], "intent": sq["intent"],
                                 "result_data": res})

        for future in not_done:
            sq = future_map[future]
            future.cancel()
            errors.append(f"Timeout on sub-query: {sq['sub_query'][:60]}")
            sql_results.append({"sub_query": sq["sub_query"], "intent": sq["intent"],
                                 "result_data": {"success": False, "error": "timeout"}})

    LOGGER.info("Medium agent: %d/%d sub-queries succeeded",
                sum(1 for r in sql_results if r["result_data"].get("success")), len(sql_results))
    return {"sql_results": sql_results, "errors": errors}


def analyze_node(state: MediumState) -> Dict:
    """Node 5: LLM analysis of combined SQL results (Groq 70b → Gemini → Ollama)."""
    from pipeline.llm_router import call as llm_call

    data_block = format_results_for_llm(state["sql_results"])

    system = (
        "You are a CRM data analyst. Analyze the database query results below and provide "
        "a clear, data-driven analysis.\n\n"
        "STRICT RULES:\n"
        "1. ONLY reference facts present in the data provided below.\n"
        "2. NEVER invent numbers, names, or dates not in the data.\n"
        "3. If a sub-query returned an error, acknowledge it as 'data unavailable'.\n"
        "4. Calculate derived metrics (percentages, growth rates, rankings) where the "
        "   source numbers are present in the data.\n"
        "5. Keep analysis concise (under 300 words) — synthesize_node will format it."
    )
    user = (
        f'User asked: "{state["query"]}"\n\n'
        f"Database results:\n{data_block}\n\n"
        "Provide a concise analysis of this data. Only state facts from the data above."
    )

    analysis = llm_call("synthesize", system, user, max_tokens=600)
    return {"analysis": analysis or "Analysis unavailable — proceeding with raw data."}


def synthesize_node(state: MediumState) -> Dict:
    """Node 6: Build the final structured response for the user."""
    from pipeline.llm_router import call as llm_call

    data_block = format_results_for_llm(state["sql_results"])
    analysis   = state.get("analysis", "")

    system = (
        "You are a CRM business analyst producing a professional response.\n\n"
        "ABSOLUTE ZERO-HALLUCINATION RULES:\n"
        "1. Use ONLY numbers/names/dates that appear in 'Database Results' below\n"
        "2. NEVER invent, estimate, or assume any value not in the results\n"
        "3. If a sub-query returned 0 rows → say 'No records found for [topic]', don't guess\n"
        "4. If a sub-query errored → say 'Data unavailable for [topic]', don't substitute\n"
        "5. NEVER say a user achieved/failed a target unless targets+sales data confirms it\n"
        "6. If same person appears in two conflicting results → flag the inconsistency\n"
        "7. Bold ALL key numbers: **176 deals**, **$2.4M**, **42%**\n"
        "8. Start directly with the finding — no 'Based on the data' preamble\n"
        "9. Structure: 2-3 sentence summary + data table/list\n"
        "10. If the data shows NO results for the period → clearly state that"
    )
    user = (
        f'User asked: "{state["query"]}"\n\n'
        f"Database Results:\n{data_block}\n\n"
        f"Analysis Notes:\n{analysis[:400]}\n\n"
        "Write the professional business response (only use facts from Database Results above):"
    )

    response = llm_call("synthesize", system, user, max_tokens=1200)
    return {"final_response": response or _fallback_format(state["sql_results"])}


def save_examples_node(state: MediumState) -> Dict:
    """Node 7: Save successful queries for future few-shot use."""
    from mcp_server.crm_mcp import mcp_save_query

    for res_item in state["sql_results"]:
        data = res_item.get("result_data", {})
        if data.get("success") and data.get("sql_used"):
            tables = data.get("tables_used", [])
            mcp_save_query(res_item["sub_query"], data["sql_used"], tables)

    return {}  # no state change needed


def _fallback_format(sql_results: List[Dict]) -> str:
    """Format result without LLM when synthesis fails."""
    parts = []
    for item in sql_results:
        sq   = item.get("sub_query", "")
        data = item.get("result_data", {})
        if not data.get("success"):
            parts.append(f"**{sq}**: Data unavailable.")
            continue
        rows    = data.get("rows", [])
        columns = data.get("columns", [])
        if not rows:
            parts.append(f"**{sq}**: No records found.")
            continue
        if len(rows) == 1 and len(rows[0]) == 1:
            parts.append(f"**{sq}**: **{rows[0][0]}**")
        else:
            header = " | ".join(str(c) for c in (columns or [f"col{i}" for i in range(len(rows[0]))]))
            rows_str = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows[:20])
            parts.append(f"**{sq}**\n{header}\n{rows_str}")
    return "\n\n".join(parts) if parts else "No matching records were found for this query in the current data."


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_medium_graph():
    """Build and compile the LangGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(MediumState)

    graph.add_node("fetch_schema",      fetch_schema_node)
    graph.add_node("decompose",         decompose_node)
    graph.add_node("generate_sql",      generate_sql_node)
    graph.add_node("execute_parallel",  execute_parallel_node)
    graph.add_node("analyze",           analyze_node)
    graph.add_node("synthesize",        synthesize_node)
    graph.add_node("save_examples",     save_examples_node)

    graph.set_entry_point("fetch_schema")
    graph.add_edge("fetch_schema",     "decompose")
    graph.add_edge("decompose",        "generate_sql")
    graph.add_edge("generate_sql",     "execute_parallel")
    graph.add_edge("execute_parallel", "analyze")
    graph.add_edge("analyze",          "synthesize")
    graph.add_edge("synthesize",       "save_examples")
    graph.add_edge("save_examples",    END)

    return graph.compile()


# Cache the compiled graph (thread-safe lazy init)
_graph_instance = None
_graph_lock     = __import__("threading").Lock()


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        with _graph_lock:
            if _graph_instance is None:
                _graph_instance = _build_medium_graph()
    return _graph_instance


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_medium_agent(query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a MEDIUM query through the LangGraph 7-node pipeline.

    Args:
        query:          Raw user query string
        classification: Output from classifier.classify()

    Returns:
        Standard pipeline response dict compatible with api.py
    """
    t_start = time.monotonic()
    LOGGER.info("Medium agent START | query: %.80s", query)

    initial_state: MediumState = {
        "query":          query,
        "classification": classification,
        "schema":         {},
        "sub_queries":    [],
        "sql_results":    [],
        "analysis":       "",
        "final_response": "",
        "attempts":       0,
        "errors":         [],
        "t_start":        t_start,
    }

    try:
        graph       = _get_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        LOGGER.error("Medium agent graph error: %s", exc, exc_info=True)
        elapsed = int((time.monotonic() - t_start) * 1000)
        return {
            "answer":                "I wasn't able to complete the analysis for this query. Please try rephrasing or breaking it into a simpler question.",
            "data":                  None,
            "sql_queries":           [],
            "tables_used":           [],
            "confidence":            0.0,
            "agent_type":            "medium",
            "latency_ms":            elapsed,
            "attempts":              0,
            "classification_reason": classification.get("reason", ""),
            "layer":                 "medium_agent_error",
        }

    # Collect metadata from all sub-query results
    all_sql:    List[str] = []
    all_tables: set       = set()
    for res_item in final_state.get("sql_results", []):
        data = res_item.get("result_data", {})
        if data.get("sql_used"):
            all_sql.append(data["sql_used"])
        all_tables.update(data.get("tables_used", []))

    elapsed = int((time.monotonic() - t_start) * 1000)
    LOGGER.info("Medium agent DONE | %dms | tables=%s", elapsed, sorted(all_tables))

    return {
        "answer":                final_state.get("final_response") or "No response generated.",
        "data":                  None,
        "sql_queries":           all_sql,
        "tables_used":           sorted(all_tables),
        "confidence":            0.85,
        "agent_type":            "medium",
        "latency_ms":            elapsed,
        "attempts":              len(final_state.get("sql_results", [])),
        "classification_reason": classification.get("reason", ""),
        "layer":                 "medium_agent",
    }
