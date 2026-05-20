"""simple_agent.py — CrewAI-based agent for SIMPLE CRM queries.

Handles: single-table lookups, basic counts, filters, aggregations.
Target latency: < 5 seconds.

Architecture:
  1. Tools defined as CrewAI BaseTool subclasses (wrap MCP functions + llm_router)
  2. _generate_execute_selfheal() — deterministic 3-attempt self-heal loop
     (SQL generation with our specialized Text2SQL models, not general-purpose LLM)
  3. CrewAI Crew narrates the raw result into professional business language
  4. run_simple_agent() — public entry point called from pipeline/main.py

Zero-hallucination rules enforced:
  • SQL generated from live schema only (db_schema.py)
  • Empty result → "No records found" (never invented data)
  • Column errors auto-corrected via find_column tool
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from config import settings
from agents.medium_agent import _date_context

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# SQL HELPERS  (shared with self-heal loop)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sql(raw: str) -> Optional[str]:
    """Extract the first valid SELECT statement from LLM output."""
    if not raw:
        return None
    cleaned = raw.strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    m = re.search(r"(SELECT\b.+?)(?:;|\n{3,}|$)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip().rstrip(";")
        if len(sql) > 10 and sql.upper().startswith("SELECT"):
            # Basic safety: reject destructive statements
            if not re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE)\b", sql, re.IGNORECASE):
                return sql
    return None


_SQL_PSEUDO_TABLES = {
    "nullif", "coalesce", "current_date", "current_timestamp", "now",
    "extract", "date_trunc", "date_part", "interval", "rank", "row_number",
    "dense_rank", "lateral", "unnest", "generate_series", "values", "dual",
    "information_schema", "pg_tables", "json_array_elements", "jsonb_array_elements",
}

def _extract_tables_from_sql(sql: str) -> List[str]:
    """Extract referenced table names from a SQL statement."""
    matches = re.findall(
        r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', sql, re.IGNORECASE
    )
    return sorted(set(
        m for m in matches if m and m.lower() not in _SQL_PSEUDO_TABLES
    ))


def _extract_bad_column(error: str) -> Optional[str]:
    """Pull the bad column name out of a PostgreSQL 'column X does not exist' error."""
    m = re.search(
        r'column\s+"?([^"\s]+)"?\s+(?:does not exist|of relation)',
        error, re.IGNORECASE,
    )
    return m.group(1).strip().strip('"') if m else None


def _format_rows_as_text(rows: List[Any], columns: List[str], sql: str = "") -> str:
    """Convert SQL result rows into readable markdown.

    Tries currency-aware USD conversion first (same logic as fast_path.py).
    Falls back to standard table if no currency column detected.
    """
    if not rows:
        return "No records found for this query."

    # ── Currency-aware formatting (Frankfurter live rates) ────────────────────
    if sql and any(kw in sql.lower() for kw in [
        "grand_total", "grandtotal", "amount", "revenue", "netpayable", "subtotal",
    ]):
        from agents.currency_format import apply_currency_conversion
        currency_result = apply_currency_conversion(rows, columns, sql)
        if currency_result:
            return currency_result

    # ── Standard formatting ───────────────────────────────────────────────────
    if len(rows) == 1 and len(rows[0]) == 1:
        val = rows[0][0]
        if val is None:
            return "No records found for this query."
        try:
            n = float(str(val).replace(",", ""))
            if n == int(n):
                return f"**{int(n):,}**"
            return f"**{n:,.2f}**"
        except (ValueError, TypeError):
            return f"**{val}**"

    if not columns:
        columns = [f"col{i+1}" for i in range(len(rows[0]) if rows else 1)]

    header = "| " + " | ".join(str(c).replace("_", " ").title() for c in columns) + " |"
    sep    = "| " + " | ".join("---" for _ in columns) + " |"
    data_rows = []
    for row in rows[:200]:
        cells = []
        for v in row:
            if v is None:
                cells.append("—")
            else:
                try:
                    n = float(str(v).replace(",", ""))
                    if n == int(n):
                        cells.append(f"{int(n):,}")
                    else:
                        cells.append(f"{n:,.2f}")
                except (ValueError, TypeError):
                    cells.append(str(v))
        data_rows.append("| " + " | ".join(cells) + " |")

    total  = len(rows)
    shown  = min(total, 200)
    suffix = f"\n\n_Showing {shown} of {total} records._" if total > 200 else ""
    return f"Found **{total}** record(s):\n\n{header}\n{sep}\n" + "\n".join(data_rows) + suffix


# ══════════════════════════════════════════════════════════════════════════════
# CORE SELF-HEAL LOOP  (deterministic — not agentic)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_execute_selfheal(
    query: str,
    tables_hint: List[str],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Generate SQL and execute it with up to max_attempts self-healing retries.

    Attempt 1: Generate SQL from live schema + examples
    Attempt 2: If column error → look up correct column name, rebuild prompt
    Attempt 3: Wider schema context + explicit column list in prompt

    Returns dict: {success, rows, columns, row_count, sql_used, tables_used, attempts, error}
    """
    from pipeline.db_schema import build_schema_for_query, find_correct_column, get_all_crm_tables
    from pipeline.schema_compact import COMPACT_SYSTEM_PROMPT
    from pipeline.llm_router import call as llm_call, sql_cache_get, sql_cache_set
    from mcp_server.crm_mcp import mcp_get_examples, mcp_save_query, execute_sql

    all_tables = get_all_crm_tables()

    # ── Check SQL cache first ─────────────────────────────────────────────────
    cached_sql = sql_cache_get(query)

    # ── Build few-shot examples string ────────────────────────────────────────
    examples    = mcp_get_examples()[:4]
    examples_str = "\n".join(
        f"-- Q: {ex['natural_query']}\n{ex['sql']}"
        for ex in examples
    )

    last_error:  Optional[str] = None
    last_sql:    Optional[str] = None
    heal_hint:   str           = ""

    for attempt in range(1, max_attempts + 1):

        # ── Build schema context (wider on attempt 3) ─────────────────────────
        extra = list(tables_hint) if tables_hint else []
        schema_str = build_schema_for_query(
            query,
            hint_tables=extra,
            max_tables=6 if attempt < 3 else 10,
        )
        # Prepend current date context so the LLM knows today's date for filters
        schema_str = _date_context() + "\n" + schema_str

        # ── Build user prompt ─────────────────────────────────────────────────
        if attempt == 1 and cached_sql:
            sql = cached_sql
            LOGGER.info("Simple agent: using cached SQL for attempt 1")
        else:
            parts = [f"-- Few-shot examples:\n{examples_str}",
                     f"\n-- Schema (includes current date):\n{schema_str}",
                     f"\nQuestion: {query}"]
            if heal_hint:
                parts.append(f"\n-- Self-heal hint: {heal_hint}")
            parts.append("\nSQL:")
            user_prompt = "\n".join(parts)

            raw_sql = llm_call("sql", COMPACT_SYSTEM_PROMPT, user_prompt)
            if not raw_sql:
                last_error = f"LLM returned no output on attempt {attempt}"
                LOGGER.warning("Simple agent: %s", last_error)
                continue

            sql = _extract_sql(raw_sql)
            if not sql:
                last_error = f"Could not extract valid SQL from LLM output on attempt {attempt}"
                LOGGER.warning("Simple agent: %s | raw: %.80s", last_error, raw_sql)
                continue

        last_sql = sql
        LOGGER.info("Simple agent attempt %d SQL: %.120s", attempt, sql)

        # ── Execute ───────────────────────────────────────────────────────────
        exec_result = execute_sql(sql)

        if exec_result.get("error"):
            last_error = exec_result["error"]
            LOGGER.warning("Simple agent attempt %d exec error: %s", attempt, last_error)

            # ── Self-heal strategy ────────────────────────────────────────────
            if "column" in last_error.lower() and "exist" in last_error.lower():
                bad_col = _extract_bad_column(last_error)
                if bad_col:
                    correct = find_correct_column(bad_col, all_tables)
                    if correct:
                        heal_hint = (
                            f"Column '{bad_col}' does not exist. "
                            f"Use '{correct}' instead. Re-write the SQL with the correct column name."
                        )
                    else:
                        heal_hint = (
                            f"Column '{bad_col}' does not exist in this DB. "
                            f"Check the schema above for the real column name."
                        )
                else:
                    heal_hint = f"Fix this error: {last_error[:200]}"

            elif "ambiguous" in last_error.lower():
                heal_hint = (
                    "Column reference is ambiguous. Add a table alias prefix to every "
                    "column in the SELECT and WHERE clauses (e.g., d.name, u.name, not just name)."
                )

            elif "syntax" in last_error.lower():
                heal_hint = (
                    f"SQL syntax error: {last_error[:200]}. "
                    "Simplify the query, double-check quoted table/column names, "
                    "and ensure NULLIF casts are correct."
                )

            else:
                heal_hint = f"Previous SQL had this error: {last_error[:200]}. Write a corrected SQL."

            continue  # retry

        # ── Success ───────────────────────────────────────────────────────────
        rows      = exec_result.get("rows", [])
        columns   = exec_result.get("columns", [])
        tables_used = _extract_tables_from_sql(sql)

        if attempt == 1 and not cached_sql:
            sql_cache_set(query, sql)

        mcp_save_query(query, sql, tables_used)

        return {
            "success":     True,
            "rows":        rows,
            "columns":     columns,
            "row_count":   exec_result.get("row_count", len(rows)),
            "sql_used":    sql,
            "tables_used": tables_used,
            "attempts":    attempt,
            "error":       None,
        }

    # All attempts exhausted
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


# ══════════════════════════════════════════════════════════════════════════════
# CREWAI TOOLS  (wrap backend functions for the narration agent)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _NarrateInput(BaseModel):
        query:       str = Field(description="Original user query")
        data_summary: str = Field(description="JSON-encoded SQL result summary")

    class _NarrateTool(BaseTool):
        """Format raw SQL result data into a professional business narrative."""
        name:        str = "format_data_result"
        description: str = (
            "Given raw SQL result data and the original query, produce a professional "
            "2-4 sentence business summary. Bold all key numbers."
        )
        args_schema: type[BaseModel] = _NarrateInput

        def _run(self, query: str, data_summary: str) -> str:
            from pipeline.llm_router import call as llm_call
            system = (
                "You are a senior CRM business analyst. Given raw query result data, "
                "write a professional 2-4 sentence business summary.\n"
                "RULES:\n"
                "1. Start DIRECTLY with the finding — no 'Based on the data' preamble\n"
                "2. Bold ALL key numbers: **176 deals**, **$58,296**, **42%**\n"
                "3. Use business language: pipeline, revenue, conversion, performance\n"
                "4. If the result is 0 or empty, state clearly that none were found\n"
                "5. NEVER invent numbers not present in the data provided\n"
                "6. Maximum 80 words — be concise and actionable"
            )
            user = (
                f'User asked: "{query}"\n'
                f"Database result:\n{data_summary[:600]}\n\n"
                "Write the 2-4 sentence professional summary:"
            )
            result = llm_call("narrate", system, user, max_tokens=180)
            return result or data_summary

    _CREWAI_AVAILABLE = True

except ImportError:
    _CREWAI_AVAILABLE = False
    LOGGER.warning("crewai not installed — narration will use llm_router directly")


def _narrate_result(query: str, result_data: Dict) -> str:
    """Narrate a SQL result into professional business language.

    Uses CrewAI narration crew if available, otherwise falls back to
    llm_router directly. Either way: Groq 8b → Gemini → Ollama.
    """
    rows    = result_data.get("rows", [])
    columns = result_data.get("columns", [])
    sql     = result_data.get("sql_used", "")

    # Build formatted data string — currency-aware (converts to USD if needed)
    raw_text = _format_rows_as_text(rows, columns, sql)

    if _CREWAI_AVAILABLE and settings.groq_api_key:
        try:
            from crewai import Agent, Task, Crew, Process

            narrator = Agent(
                role="CRM Business Analyst",
                goal="Summarize CRM data results in clear, professional business language",
                backstory=(
                    "You are a senior CRM analyst who translates raw database results "
                    "into actionable business insights. You bold all key numbers and "
                    "never invent data that wasn't in the result."
                ),
                tools=[_NarrateTool()],
                llm=_get_agent_llm(),
                verbose=False,
                max_iter=3,
            )

            narrate_task = Task(
                description=(
                    f'The user asked: "{query}"\n\n'
                    f"Database returned this data:\n{raw_text[:800]}\n\n"
                    "Use the format_data_result tool to produce a professional 2-4 sentence summary. "
                    "Bold all key numbers. NEVER invent numbers not in the data."
                ),
                expected_output=(
                    "A 2-4 sentence professional business summary with all numbers bolded. "
                    "Followed by the formatted data table."
                ),
                agent=narrator,
            )

            crew   = Crew(agents=[narrator], tasks=[narrate_task],
                          process=Process.sequential, verbose=False)
            output = crew.kickoff()
            summary = str(output).strip() if output else ""

            if summary and len(summary) > 15:
                return f"## Summary\n\n{summary}\n\n---\n\n{raw_text}"

        except Exception as exc:
            LOGGER.debug("CrewAI narration error: %s — falling back to direct call", exc)

    # Direct narration fallback via llm_router
    from pipeline.llm_router import call as llm_call
    system = (
        "You are a senior CRM business analyst. Write a professional 2-4 sentence "
        "business summary from raw database data.\n"
        "RULES:\n"
        "1. Start directly with the finding — no preamble\n"
        "2. Bold ALL key numbers: **176 deals**, **$2.4M**\n"
        "3. NEVER invent numbers not in the data provided\n"
        "4. If result is 0 or empty → state clearly that none were found\n"
        "5. If data shows 'No records found' → do NOT make up an answer\n"
        "6. Max 80 words — concise and factual"
    )
    user = f'User asked: "{query}"\nData:\n{raw_text[:600]}\n\nWrite factual summary (only use numbers from data above):'
    narrated = llm_call("narrate", system, user, max_tokens=180)

    if narrated and len(narrated.strip()) > 15:
        return f"## Summary\n\n{narrated.strip()}\n\n---\n\n{raw_text}"

    return raw_text


def _get_agent_llm():
    """Return the best available LLM for CrewAI agent reasoning."""
    if settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_classify_model,
                temperature=0,
            )
        except ImportError:
            pass

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_classify_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    except ImportError:
        pass

    # CrewAI string format (uses litellm + env vars)
    if settings.groq_api_key:
        return f"groq/{settings.groq_classify_model}"
    return f"ollama/{settings.ollama_classify_model}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_simple_agent(query: str, classification: Dict) -> Dict[str, Any]:
    """Execute a SIMPLE query using the self-heal SQL loop + CrewAI narration.

    Args:
        query:          Raw user query string
        classification: Output from classifier.classify()

    Returns:
        Standard pipeline response dict compatible with api.py
    """
    t_start     = time.monotonic()
    tables_hint = classification.get("tables_needed", [])

    LOGGER.info("Simple agent START | query: %.80s | tables_hint=%s", query, tables_hint)

    # ── Step 1: SQL generation + execution with 3-attempt self-heal ───────────
    result_data = _generate_execute_selfheal(query, tables_hint)

    if not result_data["success"]:
        elapsed = int((time.monotonic() - t_start) * 1000)
        LOGGER.warning(
            "Simple agent FAILED after %d attempts | error: %s",
            result_data["attempts"], result_data["error"],
        )
        answer = (
            "I wasn't able to retrieve data for this query after several attempts. "
            f"Last error: {result_data['error'] or 'unknown'}. "
            "Please try rephrasing your question."
        )
        return {
            "answer":                 answer,
            "data":                   None,
            "sql_queries":            [result_data["sql_used"]] if result_data["sql_used"] else [],
            "tables_used":            result_data["tables_used"],
            "confidence":             0.0,
            "agent_type":             "simple",
            "latency_ms":             elapsed,
            "attempts":               result_data["attempts"],
            "classification_reason":  classification.get("reason", ""),
            "layer":                  "simple_agent_failed",
        }

    # ── Step 2: Narrate the result via CrewAI ─────────────────────────────────
    final_answer = _narrate_result(query, result_data)

    elapsed = int((time.monotonic() - t_start) * 1000)
    LOGGER.info(
        "Simple agent DONE | %dms | attempts=%d | tables=%s",
        elapsed, result_data["attempts"], result_data["tables_used"],
    )

    return {
        "answer": final_answer,
        "data": {
            "rows":    result_data["rows"][:50],   # cap for API response size
            "columns": result_data["columns"],
        } if result_data["rows"] else None,
        "sql_queries":            [result_data["sql_used"]],
        "tables_used":            result_data["tables_used"],
        "confidence":             0.90 if result_data["rows"] else 0.85,
        "agent_type":             "simple",
        "latency_ms":             elapsed,
        "attempts":               result_data["attempts"],
        "classification_reason":  classification.get("reason", ""),
        "layer":                  "simple_agent",
    }
