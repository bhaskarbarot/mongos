"""text2sql.py — Ollama Text2SQL engine with parallel model execution.

Production-grade NL→SQL conversion pipeline:
  • Parallel primary + fallback model calls (first-success-wins)
  • Multi-stage SQL validation (syntax, JSONB compliance, table existence)
  • Smart table detection from generated SQL
  • Retry with prompt refinement on validation failure
  • Result formatting with column header extraction

Model priority:
  1. PRIMARY:  Qwen2.5-Coder 3B  (fast, 8s timeout)
  2. FALLBACK: Arctic Text2SQL 7B (accurate, 20s timeout)
  Both run in parallel — first valid SQL wins.

Public API:
    run(query, agent) -> Optional[Dict]
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
    get_document_fields,
    get_table_names,
    resolve_entity_table,
    run_sql,
)
from pipeline.utils import (
    Timer,
    coerce_number,
    fmt_number,
    format_rows_as_markdown_table,
    normalize_text,
)

LOGGER = logging.getLogger("sql_chatbot")

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_SQL_RETRIES      = 1       # retry once with refined prompt on failure
_PARALLEL_TIMEOUT_S   = 25      # max wait for parallel model pool
_MAX_RESULT_ROWS      = 50      # cap rows returned to user
_SQL_EXEC_TIMEOUT_S   = 30      # SQL execution timeout


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA MODEL CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama(
    prompt: str,
    model: str,
    timeout: int,
) -> Optional[str]:
    """Send prompt to Ollama and return raw response string.

    Args:
        prompt:  Full prompt text (schema + question)
        model:   Ollama model name
        timeout: Request timeout in seconds

    Returns:
        Raw model response string, or None on any failure
    """
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": 0,
            "num_predict": 500,
            "stop": [
                "Question:", "Explanation:", "Note:",
                "\n\n\n", "```\n\n", "Here is",
            ],
        },
    }
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            result = data.get("response", "").strip()
            if result:
                LOGGER.debug("Ollama [%s] responded (%d chars)", model, len(result))
            return result or None
    except Exception as exc:
        LOGGER.debug("Ollama [%s] error: %s", model, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(query: str, schema: str, retry_hint: str = "") -> str:
    """Build Text2SQL prompt with schema context.

    Args:
        query:      User's natural language question
        schema:     Schema context from build_text2sql_schema()
        retry_hint: Optional hint for retry (e.g., previous error)
    """
    hint_block = f"\nIMPORTANT: {retry_hint}\n" if retry_hint else ""

    return (
        f"{schema}\n\n"
        f"{hint_block}"
        f"Question: {query}\n\n"
        "Write ONLY the SQL query. No explanation, no markdown, no text before or after.\n"
        "Rules:\n"
        "- Use DIRECT column names — NOT document->>'field'\n"
        "- Numbers (NUMERIC): WHERE grand_total_in_usd > 1000\n"
        "- Booleans (BOOLEAN): WHERE deleted = false OR deleted IS NULL\n"
        "- Date strings (TEXT): WHERE \"closeDate\" > '2025-01-01'\n"
        "- Mixed-case columns: always double-quote: \"closeDate\", \"createdAt\"\n"
        "- Table names in double quotes: FROM \"tableName\"\n"
        "- Soft-delete: WHERE deleted = false OR deleted IS NULL\n"
        "- Use ILIKE for text matching\n"
        "- Output ONLY a SELECT statement\n\n"
        "SQL:"
    )


def _build_retry_prompt(
    query: str,
    schema: str,
    prev_sql: str,
    error: str,
) -> str:
    """Build a refined prompt for retry after first attempt failed."""
    return (
        f"{schema}\n\n"
        f"Question: {query}\n\n"
        f"Previous SQL attempt (FAILED):\n{prev_sql}\n\n"
        f"Error: {error}\n\n"
        "Fix the SQL. Output ONLY the corrected SELECT statement. "
        "Use direct column names (NOT document->>'field').\n\n"
        "SQL:"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SQL EXTRACTION AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sql(raw: str) -> Optional[str]:
    """Extract a clean SELECT statement from model output.

    Handles: ```sql``` blocks, <execute> blocks, bare SELECT, and mixed output.
    """
    if not raw:
        return None

    # Strip leading/trailing whitespace and common prefixes
    cleaned = raw.strip()
    cleaned = re.sub(r"^(?:Here is|The SQL|SQL query|Answer:)\s*:?\s*", "", cleaned, flags=re.I)

    # ```sql ... ``` code blocks (prefer last one — models sometimes iterate)
    code_blocks = re.findall(
        r"```(?:sql|SQL|postgresql)?\s*(SELECT.+?)```",
        cleaned, re.DOTALL,
    )
    if code_blocks:
        sql = code_blocks[-1].strip()
        return _clean_sql(sql)

    # <execute>...</execute> blocks
    exec_m = re.search(
        r"<execute>\s*(SELECT.+?)(?:</execute>|```|$)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if exec_m:
        return _clean_sql(exec_m.group(1))

    # Bare SELECT statement
    m = re.search(
        r"(SELECT\b.+?)(?:;|\n\n\n|Explanation:|Note:|Question:|$)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return _clean_sql(m.group(1))

    return None


def _clean_sql(sql: str) -> str:
    """Normalize extracted SQL: strip semicolons, extra whitespace."""
    sql = sql.strip().rstrip(";").strip()
    # Remove trailing incomplete lines
    sql = re.sub(r"\n--.*$", "", sql)
    # Collapse multiple spaces/newlines
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def _validate_sql(sql: str, table_names: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate SQL for JSONB correctness and safety.

    Returns:
        (is_valid, error_message) — error_message is empty if valid
    """
    if not sql:
        return False, "empty SQL"

    sql_upper = sql.upper().strip()

    # Must be a SELECT
    if not sql_upper.startswith("SELECT"):
        return False, "not a SELECT statement"

    # Block destructive operations
    if re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|GRANT)\b", sql, re.I):
        return False, "destructive operation detected"

    # Wrong dialect functions
    if re.search(r"\bDATE\(['\"]?now['\"]?\)", sql, re.I):
        return False, "wrong dialect: DATE('now') — use NOW()"
    if re.search(r"\bIFNULL\b|\bNVL\b|\bIF\(|\bDATEDIFF\b", sql, re.I):
        return False, "wrong dialect: MySQL/Oracle function used"
    if re.search(r"\bSTRFTIME\b|\bSQLITE\b", sql, re.I):
        return False, "wrong dialect: SQLite function used"

    # Block old JSONB syntax — direct column access is now correct
    if re.search(r"document\s*->>'", sql, re.I):
        return False, "old JSONB syntax detected: use direct column names instead of document->>'field'"

    # Table name validation (if table list provided)
    if table_names:
        used_tables = re.findall(
            r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            sql, re.IGNORECASE,
        )
        unknown = [t for t in used_tables if t not in table_names]
        if unknown:
            return False, f"unknown tables: {unknown}"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# SQL COLUMN HEADER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_column_headers(sql: str) -> List[str]:
    """Extract column aliases from SQL SELECT clause for table rendering.

    Parses: SELECT ... AS alias patterns and document->>'field' patterns.
    """
    # Get the SELECT clause (between SELECT and FROM)
    select_m = re.search(r"SELECT\s+(.+?)\s+FROM\b", sql, re.DOTALL | re.IGNORECASE)
    if not select_m:
        return []

    select_clause = select_m.group(1)
    headers = []

    # Split by top-level commas (not inside parentheses)
    depth = 0
    current = ""
    for ch in select_clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            headers.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        headers.append(current.strip())

    # Extract alias from each column expression
    clean_headers = []
    for expr in headers:
        # Explicit AS alias
        as_m = re.search(r"\bAS\s+\"?(\w+)\"?\s*$", expr, re.I)
        if as_m:
            clean_headers.append(as_m.group(1))
            continue
        # bare column name "field" or field without alias
        bare_m = re.search(r'^"?(\w+)"?\s*$', expr.strip())
        if bare_m:
            clean_headers.append(bare_m.group(1))
            continue
        # COUNT/SUM/AVG etc.
        agg_m = re.search(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, re.I)
        if agg_m:
            clean_headers.append(agg_m.group(1).lower())
            continue
        # Fallback
        clean_headers.append(expr[:20].strip())

    return clean_headers


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL SQL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _generate_with_model(
    prompt: str,
    model: str,
    timeout: int,
    table_names: List[str],
) -> Optional[str]:
    """Generate and validate SQL with a single model. Used as thread target."""
    raw = _call_ollama(prompt, model, timeout)
    if not raw:
        return None

    sql = _extract_sql(raw)
    if not sql:
        LOGGER.debug("[%s] no SQL extracted from output", model)
        return None

    is_valid, err = _validate_sql(sql, table_names)
    if not is_valid:
        LOGGER.debug("[%s] SQL validation failed: %s | SQL: %.80s", model, err, sql)
        return None

    return sql


def generate_sql(query: str, agent) -> Optional[str]:
    """Generate SQL for a query using parallel primary + fallback models.

    Strategy:
      1. Fire both models in parallel (ThreadPoolExecutor)
      2. Return the first valid SQL (primary model preferred if both succeed fast)
      3. If both fail, retry once with refined prompt using primary model
      4. Return None if all attempts fail

    Args:
        query: Natural language question
        agent: DB agent for schema access

    Returns:
        Validated SQL string, or None
    """
    schema      = build_text2sql_schema(agent)
    prompt      = _build_prompt(query, schema)
    table_names = get_table_names(agent)

    # ── Parallel execution: both models simultaneously ─────────────────────
    with Timer("text2sql_parallel") as timer:
        valid_sql: Optional[str] = None
        primary_sql: Optional[str] = None
        fallback_sql: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2s") as pool:
            future_primary = pool.submit(
                _generate_with_model, prompt,
                settings.ollama_primary_model,
                settings.ollama_primary_timeout,
                table_names,
            )
            future_fallback = pool.submit(
                _generate_with_model, prompt,
                settings.ollama_fallback_model,
                settings.ollama_fallback_timeout,
                table_names,
            )

            futures = {
                future_primary:  "primary",
                future_fallback: "fallback",
            }

            for future in as_completed(futures, timeout=_PARALLEL_TIMEOUT_S):
                label = futures[future]
                try:
                    sql = future.result(timeout=2)
                    if sql:
                        if label == "primary":
                            primary_sql = sql
                        else:
                            fallback_sql = sql
                        # First valid result — accept it immediately
                        if valid_sql is None:
                            valid_sql = sql
                            LOGGER.info(
                                "Text2SQL: %s model won (%.0fms): %.80s",
                                label, timer.elapsed_ms, sql,
                            )
                except Exception as exc:
                    LOGGER.debug("Text2SQL %s future error: %s", label, exc)

    # Prefer primary model's output if both succeeded
    if primary_sql:
        valid_sql = primary_sql
    elif fallback_sql:
        valid_sql = fallback_sql

    if valid_sql:
        LOGGER.info("Text2SQL generated: %.100s", valid_sql)
        return valid_sql

    # ── Retry with refined prompt (primary model only) ─────────────────────
    LOGGER.debug("Text2SQL: parallel failed, attempting retry with refined prompt")
    raw = _call_ollama(prompt, settings.ollama_primary_model, settings.ollama_primary_timeout)
    if raw:
        sql = _extract_sql(raw)
        if sql:
            is_valid, err = _validate_sql(sql, table_names)
            if not is_valid and _MAX_SQL_RETRIES > 0:
                retry_prompt = _build_retry_prompt(query, schema, sql, err)
                raw2 = _call_ollama(
                    retry_prompt,
                    settings.ollama_primary_model,
                    settings.ollama_primary_timeout,
                )
                if raw2:
                    sql2 = _extract_sql(raw2)
                    if sql2:
                        ok, _ = _validate_sql(sql2, table_names)
                        if ok:
                            LOGGER.info("Text2SQL retry success: %.80s", sql2)
                            return sql2
            elif is_valid:
                return sql

    LOGGER.warning("Text2SQL: all attempts failed for query: %.60s", query)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names used in a SQL query."""
    return sorted({
        m for m in re.findall(
            r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            sql, re.IGNORECASE,
        ) if m
    })


def _format_result(rows: List[Any], sql: str, query: str) -> str:
    """Format SQL result rows into user-friendly output.

    Handles:
      • Empty results
      • Single scalar values (count, sum, etc.)
      • Single row results
      • Multi-row tables with extracted column headers
    """
    if not rows:
        return "No data found for this query."

    # All-null check
    first = rows[0]
    if isinstance(first, (list, tuple)):
        if len(rows) == 1 and all(v is None for v in first):
            return "No data found for this query."
    elif first is None:
        return "No data found for this query."

    # Single scalar value
    if len(rows) == 1 and not isinstance(first, (list, tuple)):
        return f"Result: **{fmt_number(first)}**"

    # Single row with single column
    if len(rows) == 1 and isinstance(first, (list, tuple)) and len(first) == 1:
        val = first[0]
        n = coerce_number(val)
        if isinstance(n, (int, float)):
            return f"Result: **{fmt_number(n)}**"
        return f"Result: **{val}**"

    # Extract column headers from SQL
    headers = _extract_column_headers(sql)

    # Limit rows
    display_rows = rows[:_MAX_RESULT_ROWS]

    # Multi-row result → markdown table
    table = format_rows_as_markdown_table(
        display_rows,
        headers=headers if headers else None,
        max_rows=_MAX_RESULT_ROWS,
        show_overflow=len(rows) > _MAX_RESULT_ROWS,
    )

    count_str = f"**{len(rows)}**" if len(rows) <= _MAX_RESULT_ROWS else (
        f"**{_MAX_RESULT_ROWS}** of **{len(rows)}**"
    )
    return f"Found {count_str} result(s):\n\n{table}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """Generate SQL → execute → return formatted result dict.

    Used for SIMPLE queries that fast-path missed.

    Args:
        query: Natural language question
        agent: DB agent with SQL execution tools

    Returns:
        Result dict: {answer, tables_used, confidence, sql_queries}
        None if SQL generation failed entirely.
    """
    with Timer("text2sql_total") as timer:
        LOGGER.info("Text2SQL: processing query: %.60s", query)

        sql = generate_sql(query, agent)
        if not sql:
            LOGGER.warning("Text2SQL: no valid SQL for: %.60s", query)
            return None

        # Execute the SQL
        _res = run_sql(agent, sql)
        if _res.error:
            LOGGER.warning("Text2SQL exec failed: %s | SQL: %.80s", _res.error, sql)
            return None

        rows        = _res.rows
        tables_used = _extract_tables_from_sql(sql)
        body        = _format_result(rows, sql, query)

        # Confidence scoring
        confidence = 0.90
        if not rows or "No data found" in body:
            confidence = 0.70
        elif len(rows) == 1:
            confidence = 0.92
        elif len(rows) > 5:
            confidence = 0.88

    LOGGER.info(
        "Text2SQL: done in %.0fms | rows=%d | tables=%s",
        timer.elapsed_ms, len(rows) if rows else 0, tables_used,
    )

    return {
        "answer":      body,
        "tables_used": tables_used,
        "confidence":  confidence,
        "sql_queries": [sql],
    }