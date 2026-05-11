"""intent_router.py — LLM-based semantic intent router (the fast-path brain).

Problem solved:
    fast_path.py has 26 handlers with rigid regex patterns. Adding a new query
    variant requires editing code. This module replaces that with a tiny LLM
    that understands ANY phrasing and maps it to a SQL template.

    "tell me all deals"          → same as "give me all deals"
    "show closed won"            → same as "list all closed won deals"
    "how many open invoices do we have" → count + invoices + status=unpaid

Position in pipeline:
    fast_path (regex, <50ms) → intent_router (LLM, ~400ms) → classifier → ...

How it works:
    1. A tiny local LLM (qwen2.5:1.5b via classify chain) extracts structured intent:
       { action, entity, filters, confidence }
    2. A SQL template is built from the intent — no LLM-generated SQL (reliable)
    3. SQL executes → result formatted with business-friendly labels
    4. Returns None (< confidence threshold) so full pipeline takes over for
       complex / multi-entity / analytical queries

Public API:
    run(query, agent, memory_context="") -> Optional[Dict]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pipeline.llm import call as llm_call
from pipeline.schema import (
    REGISTRY,
    build_revenue_coalesce,
    find_revenue_table,
    get_document_fields,
    get_table_names,
    resolve_entity_table,
    run_sql,
)
from pipeline.utils import Timer, fmt_number, format_rows_as_markdown_table, normalize_text

LOGGER = logging.getLogger("sql_chatbot")

_CONFIDENCE_MIN = 0.55   # below this → let full pipeline handle
_LIST_LIMIT     = 50


# ══════════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = """CRM intent extractor. Output ONLY valid JSON.

{"action":"count|list|sum|find|top_n","entity":"deals|invoices|contacts|companies|users|tasks|targets|outreaches|products|regions","filters":{"stage":"closed_won|closed_lost|open","status":"paid|unpaid|overdue|pending|completed|draft|cancelled","owner":null,"time_period":"this_month|last_month|this_year|last_year|this_quarter|last_quarter|today|null","search":null,"limit":10},"confidence":0.0}

ACTIONS: count=how many/count, list=show/give/tell/display/all, sum=revenue/total/amount, find=search specific record, top_n=top N/best/highest
CONFIDENCE: 0.95=clear single-entity, <0.40=needs 2+ entities/comparison/analysis (full pipeline handles those)
ENTITIES: deal/opportunity→deals, invoice/bill→invoices, contact/person/lead→contacts, company/account/customer→companies, user/employee/rep→users, task/todo→tasks, target/goal→targets
STAGES: won/closed won→closed_won, lost/closed lost→closed_lost, open→open
STATUS: paid→paid, unpaid/outstanding→unpaid, overdue→overdue, pending→pending, completed→completed

EXAMPLES:
"tell me all deals"→{"action":"list","entity":"deals","filters":{},"confidence":0.95}
"show closed won"→{"action":"list","entity":"deals","filters":{"stage":"closed_won"},"confidence":0.95}
"list closed lost"→{"action":"list","entity":"deals","filters":{"stage":"closed_lost"},"confidence":0.95}
"total revenue this month"→{"action":"sum","entity":"invoices","filters":{"time_period":"this_month"},"confidence":0.95}
"overdue invoices"→{"action":"list","entity":"invoices","filters":{"status":"overdue"},"confidence":0.95}
"pending tasks"→{"action":"list","entity":"tasks","filters":{"status":"pending"},"confidence":0.95}
"find John Smith"→{"action":"find","entity":"contacts","filters":{"search":"John Smith"},"confidence":0.90}
"top 5 by revenue"→{"action":"top_n","entity":"invoices","filters":{"limit":5},"confidence":0.85}
"compare this vs last month"→{"action":"compare","entity":"invoices","filters":{},"confidence":0.10}
"target vs achieved"→{"action":"compare","entity":"targets","filters":{},"confidence":0.10}"""


# ══════════════════════════════════════════════════════════════════════════════
# TIME PERIOD → SQL CLAUSE
# ══════════════════════════════════════════════════════════════════════════════

_ENTITY_DATE_FIELD: Dict[str, str] = {
    "invoices":  "invoice_date",
    "deals":     "closeDate",
    "tasks":     "dueDate",
    "contacts":  "createdAt",
    "companies": "createdAt",
    "targets":   "month",
    "users":     "createdAt",
}


def _time_clause(period: str, date_field: str) -> str:
    """Convert a time_period string to a SQL WHERE fragment (no f-string curly confusion)."""
    df = "NULLIF(document->>'" + date_field + "','')::timestamptz"
    if period == "this_month":
        return "DATE_TRUNC('month', " + df + ") = DATE_TRUNC('month', NOW())"
    if period == "last_month":
        return "DATE_TRUNC('month', " + df + ") = DATE_TRUNC('month', NOW() - INTERVAL '1 month')"
    if period == "this_year":
        return "DATE_PART('year', " + df + ") = DATE_PART('year', NOW())"
    if period == "last_year":
        return "DATE_PART('year', " + df + ") = DATE_PART('year', NOW()) - 1"
    if period == "this_quarter":
        return "DATE_TRUNC('quarter', " + df + ") = DATE_TRUNC('quarter', NOW())"
    if period == "last_quarter":
        return "DATE_TRUNC('quarter', " + df + ") = DATE_TRUNC('quarter', NOW() - INTERVAL '3 months')"
    if period == "today":
        return "DATE(" + df + ") = CURRENT_DATE"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# SQL WHERE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_where(entity: str, table: str, filters: Dict, fields: List[str]) -> str:
    """Build a SQL WHERE clause from extracted filters. Always safe — no raw user SQL."""
    clauses = ["COALESCE(document->>'deleted','false') != 'true'"]

    stage = (filters.get("stage") or "").lower().replace(" ", "_")
    if stage and entity == "deals":
        if stage == "open":
            clauses.append("document->>'dealWonAt' IS NULL AND document->>'dealLostAt' IS NULL")
        elif "won" in stage:
            clauses.append("document->>'stage' ILIKE 'Closed Won'")
        elif "lost" in stage:
            clauses.append("document->>'stage' ILIKE 'Closed Lost'")
        else:
            stage_esc = stage.replace("'", "''")
            clauses.append("document->>'stage' ILIKE '%" + stage_esc + "%'")

    status = (filters.get("status") or "").lower()
    if status:
        if entity == "invoices":
            if status in ("unpaid", "outstanding", "open"):
                clauses.append("document->>'payment_status' NOT IN ('paid','cancelled')")
            elif status == "overdue":
                clauses.append(
                    "NULLIF(document->>'due_date','')::timestamptz < NOW() "
                    "AND document->>'payment_status' NOT IN ('paid','cancelled')"
                )
            elif status == "paid":
                clauses.append("document->>'payment_status' = 'paid'")
            elif status in ("draft", "cancelled", "approved", "submitted"):
                clauses.append("document->>'payment_status' = '" + status + "'")
        elif entity == "tasks":
            if status == "pending":
                clauses.append("document->>'status' = 'Pending'")
            elif status in ("completed", "done"):
                clauses.append("document->>'status' = 'Completed'")
            elif status == "overdue":
                clauses.append(
                    "NULLIF(document->>'dueDate','')::timestamptz < NOW() "
                    "AND document->>'status' != 'Completed'"
                )
            elif status == "open":
                clauses.append("document->>'status' = 'Open'")
        else:
            status_esc = status.replace("'", "''")
            clauses.append("document->>'status' ILIKE '" + status_esc + "'")

    owner = filters.get("owner")
    if owner:
        o = str(owner).replace("'", "''")
        owner_field = REGISTRY.get(table, "owner", fields)
        if owner_field:
            clauses.append("document->>'" + owner_field + "' ILIKE '%" + o + "%'")
        else:
            clauses.append(
                "(document->>'owner' ILIKE '%" + o + "%' "
                "OR document->>'salesOwner' ILIKE '%" + o + "%')"
            )

    search = filters.get("search")
    if search:
        s = str(search).replace("'", "''")
        name_f  = REGISTRY.get(table, "name", fields)
        first_f = REGISTRY.get(table, "first_name", fields)
        email_f = REGISTRY.get(table, "email", fields)
        id_f    = REGISTRY.get(table, "identifier", fields)
        parts   = []
        if name_f:
            parts.append("document->>'" + name_f + "' ILIKE '%" + s + "%'")
        if first_f and first_f != name_f:
            parts.append("document->>'" + first_f + "' ILIKE '%" + s + "%'")
        if email_f:
            parts.append("document->>'" + email_f + "' ILIKE '%" + s + "%'")
        if id_f and id_f not in (name_f, first_f):
            parts.append("document->>'" + id_f + "' ILIKE '%" + s + "%'")
        if not parts:
            parts = ["document->>'name' ILIKE '%" + s + "%'"]
        clauses.append("(" + " OR ".join(parts) + ")")

    return " AND ".join(clauses)


# ══════════════════════════════════════════════════════════════════════════════
# SQL TEMPLATE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_sql(
    action: str,
    entity: str,
    table: str,
    filters: Dict,
    fields: List[str],
) -> Optional[str]:
    """Build a validated SQL query from structured intent. Returns None if unsupported."""
    where = _build_where(entity, table, filters, fields)
    limit = int(filters.get("limit") or _LIST_LIMIT)

    # Append time-period clause
    period = (filters.get("time_period") or "").lower()
    if period:
        date_f = _ENTITY_DATE_FIELD.get(entity, "createdAt")
        tc = _time_clause(period, date_f)
        if tc:
            where = where + " AND " + tc

    name_expr = REGISTRY.display_name_expr(table, fields)

    # ── count ─────────────────────────────────────────────────────────────────
    if action == "count":
        return 'SELECT COUNT(*)::int FROM "' + table + '" WHERE ' + where

    # ── list ──────────────────────────────────────────────────────────────────
    if action == "list":
        status_f = REGISTRY.get(table, "status", fields)
        amount_f = REGISTRY.get(table, "amount", fields)
        date_f   = REGISTRY.get(table, "date",   fields)
        owner_f  = REGISTRY.get(table, "owner",  fields)

        selects = [name_expr + " AS name"]
        if status_f:
            selects.append("document->>'" + status_f + "' AS status")
        if amount_f and entity in ("deals", "invoices", "sales"):
            selects.append("document->>'" + amount_f + "' AS amount")
        if date_f:
            selects.append("document->>'" + date_f + "' AS date")
        if owner_f:
            selects.append("document->>'" + owner_f + "' AS owner")

        return (
            'SELECT ' + ", ".join(selects)
            + ' FROM "' + table + '" WHERE ' + where
            + ' ORDER BY ' + name_expr
            + ' LIMIT ' + str(limit)
        )

    # ── sum ───────────────────────────────────────────────────────────────────
    if action == "sum":
        amount_f = REGISTRY.get(table, "amount", fields) or "grandtotal_in_usd"
        return (
            "SELECT SUM(NULLIF(document->>'" + amount_f + "','')::numeric) AS total"
            + ' FROM "' + table + '" WHERE ' + where
        )

    # ── find / lookup ─────────────────────────────────────────────────────────
    if action in ("find", "lookup"):
        return 'SELECT * FROM "' + table + '" WHERE ' + where + ' LIMIT 10'

    # ── top_n ─────────────────────────────────────────────────────────────────
    if action == "top_n":
        # For top_n on invoices: group by company name, sum revenue
        if entity in ("invoices", "sales"):
            amount_f = REGISTRY.get(table, "amount", fields) or "grandtotal_in_usd"
            return (
                'SELECT ' + name_expr + ' AS name, '
                + "SUM(NULLIF(document->>'" + amount_f + "','')::numeric) AS total"
                + ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY ' + name_expr
                + ' ORDER BY total DESC NULLS LAST'
                + ' LIMIT ' + str(limit)
            )
        # For top_n on deals: group by owner, sum deal value
        amount_f = REGISTRY.get(table, "amount", fields)
        if amount_f:
            return (
                'SELECT ' + name_expr + ' AS name, '
                + "SUM(NULLIF(document->>'" + amount_f + "','')::numeric) AS total"
                + ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY ' + name_expr
                + ' ORDER BY total DESC NULLS LAST'
                + ' LIMIT ' + str(limit)
            )
        return (
            'SELECT ' + name_expr + ' AS name'
            + ' FROM "' + table + '" WHERE ' + where
            + ' LIMIT ' + str(limit)
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM INTENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_intent(
    query: str,
    table_names: List[str],
    memory_context: str = "",
) -> Optional[Dict]:
    """Call the tiny LLM to extract structured intent JSON. Returns dict or None."""
    ctx_block = ("\nSession context: " + memory_context) if memory_context else ""
    user_msg = (
        "Available tables: " + ", ".join(table_names[:15])
        + ctx_block
        + '\n\nUser query: "' + query + '"'
        + "\n\nExtract intent as JSON:"
    )

    raw = llm_call("classify", _SYSTEM, user_msg, max_tokens=200)
    if not raw:
        return None

    content = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    content = re.sub(r"\s*```$", "", content).strip()

    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return None

    try:
        parsed     = json.loads(m.group(0))
        action     = str(parsed.get("action", "")).lower().strip()
        entity     = str(parsed.get("entity", "")).lower().strip()
        confidence = float(parsed.get("confidence", 0.0))
        if not action or not entity:
            return None
        return {
            "action":     action,
            "entity":     entity,
            "filters":    parsed.get("filters") or {},
            "confidence": confidence,
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        LOGGER.debug("IntentRouter JSON parse failed: %s | raw: %.100s", exc, raw)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def _format_result(rows: Any, action: str, entity: str, filters: Dict) -> str:
    """Produce a clean, human-readable answer from SQL rows."""
    if not rows:
        stage  = filters.get("stage", "")
        status = filters.get("status", "")
        search = filters.get("search", "")
        if search:
            return f"No {entity} found matching **'{search}'**."
        qualifier = ""
        if stage:
            qualifier = " " + stage.replace("_", " ")
        elif status:
            qualifier = " " + status
        return f"No{qualifier} {entity} found matching your criteria."

    if action == "count":
        val = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
        n   = int(val) if val is not None else 0

        stage  = filters.get("stage", "")
        status = filters.get("status", "")
        period = filters.get("time_period", "")

        qualifier = ""
        if stage:
            qualifier = " " + stage.replace("_", " ")
        elif status:
            qualifier = " " + status
        period_desc = (" (" + period.replace("_", " ") + ")") if period else ""

        return f"There are **{fmt_number(n)}**{qualifier} {entity}{period_desc}."

    if action == "sum":
        val = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
        if val is None:
            return "No revenue data found for the selected criteria."
        period = filters.get("time_period", "")
        period_desc = period.replace("_", " ") if period else "all time"
        return f"**Total revenue ({period_desc}): ${float(val):,.2f}**"

    if action in ("list", "find", "top_n", "lookup"):
        n = len(rows)
        table = format_rows_as_markdown_table(rows, max_rows=_LIST_LIMIT)
        count_str = f"**{n}**" if n <= _LIST_LIMIT else f"**{_LIST_LIMIT}** of **{n}**"
        return f"Found {count_str} {entity}:\n\n{table}"

    return format_rows_as_markdown_table(rows, max_rows=_LIST_LIMIT)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, agent, memory_context: str = "") -> Optional[Dict[str, Any]]:
    """
    Attempt to handle a user query via LLM intent extraction + SQL template.

    Returns a result dict on success; returns None to signal the full pipeline
    should handle the query (low confidence, multi-entity, complex analysis).

    Args:
        query:          User's natural-language query
        agent:          LangChain AgentExecutor with SQL tools
        memory_context: Optional context string from ChatMemory.get_context()

    Returns:
        Result dict {answer, tables_used, confidence, sql_queries, _entity, _count}
        or None.
    """
    with Timer("intent_router") as timer:
        table_names = get_table_names(agent)
        if not table_names:
            return None

        # Step 1: LLM intent extraction
        intent = _extract_intent(query, table_names, memory_context)
        if intent is None:
            LOGGER.debug("IntentRouter: no intent extracted for: %.60s", query)
            return None

        confidence = intent["confidence"]
        action     = intent["action"]
        filters    = intent.get("filters") or {}

        # Normalise entity — LLM sometimes uses aliases or actual table names
        # instead of the canonical set defined in the system prompt.
        _ENTITY_NORM: Dict[str, str] = {
            "bill": "invoices", "bills": "invoices", "billing": "invoices",
            "opportunity": "deals", "opportunities": "deals", "pipeline": "deals",
            "lead": "contacts", "leads": "contacts", "person": "contacts",
            "people": "contacts",
            "account": "companies", "accounts": "companies",
            "customer": "companies", "customers": "companies", "client": "companies",
            "employee": "users", "employees": "users", "rep": "users",
            "salesperson": "users", "salespeople": "users",
            "todo": "tasks", "todos": "tasks", "followup": "tasks",
            "goal": "targets", "quota": "targets",
            "campaign": "outreaches",
            "sale": "sales", "order": "sales",
        }
        entity = _ENTITY_NORM.get(intent["entity"], intent["entity"])

        # Low confidence or complex action → let full pipeline handle
        if confidence < _CONFIDENCE_MIN or action in (
            "compare", "report", "trend", "analysis", "kpi", "summary"
        ):
            LOGGER.info(
                "IntentRouter: skip (conf=%.2f, action=%s) → escalate: %.60s",
                confidence, action, query,
            )
            return None

        # Step 2: Resolve entity → actual DB table
        # Special case: revenue queries go to the invoice/sales table
        if action == "sum" and entity in ("deals", "pipeline"):
            rev_table = find_revenue_table(table_names)
            if rev_table:
                entity = "invoices"
                table  = rev_table
            else:
                return None
        else:
            table = resolve_entity_table(entity, table_names, query)
        if not table:
            LOGGER.debug("IntentRouter: could not resolve entity '%s'", entity)
            return None

        # Step 3: Get fields for this table
        fields = get_document_fields(agent, table)
        if not fields:
            return None

        # Step 4: Build SQL from template
        sql = _build_sql(action, entity, table, filters, fields)
        if not sql:
            LOGGER.debug("IntentRouter: no template for action=%s entity=%s", action, entity)
            return None

        # Step 5: Execute
        _res = run_sql(agent, sql)
        if _res.error:
            LOGGER.warning("IntentRouter SQL error: %s | sql: %.80s", _res.error, sql)
            return None

    LOGGER.info(
        "IntentRouter HIT | conf=%.2f action=%s entity=%s | %.0fms | sql: %.60s",
        confidence, action, entity, timer.elapsed_ms, sql,
    )

    answer = _format_result(_res.rows, action, entity, filters)

    return {
        "answer":      answer,
        "tables_used": [table],
        "confidence":  round(confidence * 0.92, 3),
        "sql_queries": [sql],
        "_entity":     entity,
        "_count":      (
            _res.rows[0][0]
            if action == "count" and _res.rows and isinstance(_res.rows[0], (list, tuple))
            else None
        ),
    }
