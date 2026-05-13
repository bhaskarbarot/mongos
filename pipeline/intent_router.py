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

_SYSTEM = """CRM intent extractor. Output ONLY valid JSON. No explanation, no markdown.

SCHEMA:
{"action":"count|list|sum|find|top_n|detail|group_by","entity":"deals|invoices|contacts|companies|users|createtasks|targets|outreaches|sales|meetings","filters":{"stage":"closed_won|closed_lost|open","status":"paid|unpaid|overdue|pending|completed|draft|cancelled","owner":"person_name_string_or_null","time_period":"this_month|last_month|this_year|last_year|this_quarter|last_quarter|today|tomorrow|next_month|null","search":"search_term_or_null","limit":null,"sort":"asc|desc","group_by":"owner|stage|status|currency|priority|jobTitle|null","currency":"USD|INR|null","detail":false},"confidence":0.0}

ACTIONS:
  count = "how many / count / total count / number of"
  list = "show / give / tell / display / all / list / get / share / fetch / what are"
  sum = "total / revenue / amount / billing / income / earned / collected"
  find = "find / search / lookup / locate / show specific record"
  top_n = "top N / best / highest / most / biggest / largest"
  detail = "detail / full info / complete info / everything about / profile / summary of"
  group_by = "by owner / per person / grouped by / breakdown / each user / who has / who have / by stage / by status"

PERSON/OWNER FILTER — CRITICAL: extract ANY person name into "owner" filter for task queries:
  "Ketul's pending tasks" → owner="Ketul"
  "tasks by ketul" → owner="Ketul"
  "pending tasks for ketul" → owner="Ketul"
  "which task is pending by ketul" → owner="Ketul"
  "tasks assigned to yash bhide" → owner="Yash Bhide"
  "what kartik needs to do" → owner="Kartik"
  "share me kartik's task status" → owner="Kartik"
  "pending tasks for the sales team" → owner="sales"
  "tasks assigned to me" (user=current) → owner=null

ENTITY MAP:
  task/todo/follow-up/assignment/to-do → createtasks
  deal/opportunity/pipeline/bid/opportunity → deals
  invoice/bill/payment/receipt/billing → invoices
  contact/person/lead/prospect/client-person → contacts
  company/account/client/customer/business → companies
  user/employee/rep/member/staff/person → users
  meeting/appointment/schedule/call/calendar → meetings
  sale/order/SO/sales-order → sales

STATUS MAP:
  pending/open/active → pending (tasks) | unpaid (invoices)
  completed/done/finished/closed → completed (tasks) | paid (invoices)
  overdue/late/past-due/outstanding → overdue
  paid/cleared/settled → paid

EXAMPLES — every variation of same intent maps to same JSON:
"ketul's pending tasks"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul","status":"pending"},"confidence":0.95}
"give me ketul's pending tasks"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul","status":"pending"},"confidence":0.95}
"pending tasks by ketul"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul","status":"pending"},"confidence":0.95}
"which task is pending for ketul"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul","status":"pending"},"confidence":0.95}
"what tasks does ketul have"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul"},"confidence":0.95}
"tasks assigned to ketul trivedi"→{"action":"list","entity":"createtasks","filters":{"owner":"ketul trivedi"},"confidence":0.95}
"show yash bhide's tasks"→{"action":"list","entity":"createtasks","filters":{"owner":"yash bhide"},"confidence":0.95}
"kartik's task status"→{"action":"list","entity":"createtasks","filters":{"owner":"kartik"},"confidence":0.95}
"share me kartik's task status"→{"action":"list","entity":"createtasks","filters":{"owner":"kartik"},"confidence":0.95}
"who has pending tasks"→{"action":"group_by","entity":"createtasks","filters":{"group_by":"owner","status":"pending"},"confidence":0.95}
"who have pending tasks"→{"action":"group_by","entity":"createtasks","filters":{"group_by":"owner","status":"pending"},"confidence":0.95}
"pending tasks per user"→{"action":"group_by","entity":"createtasks","filters":{"group_by":"owner","status":"pending"},"confidence":0.95}
"tasks breakdown by person"→{"action":"group_by","entity":"createtasks","filters":{"group_by":"owner"},"confidence":0.90}
"tell me all deals"→{"action":"list","entity":"deals","filters":{},"confidence":0.95}
"show closed won"→{"action":"list","entity":"deals","filters":{"stage":"closed_won"},"confidence":0.95}
"open deals"→{"action":"list","entity":"deals","filters":{"stage":"open"},"confidence":0.95}
"deals by stage"→{"action":"group_by","entity":"deals","filters":{"group_by":"stage"},"confidence":0.90}
"deals by owner"→{"action":"group_by","entity":"deals","filters":{"group_by":"owner"},"confidence":0.90}
"total revenue this month"→{"action":"sum","entity":"invoices","filters":{"time_period":"this_month","status":"paid"},"confidence":0.95}
"overdue invoices"→{"action":"list","entity":"invoices","filters":{"status":"overdue"},"confidence":0.95}
"pending invoices"→{"action":"list","entity":"invoices","filters":{"status":"unpaid"},"confidence":0.95}
"invoices due next month"→{"action":"list","entity":"invoices","filters":{"status":"unpaid","time_period":"next_month"},"confidence":0.90}
"USD invoices"→{"action":"list","entity":"invoices","filters":{"currency":"USD"},"confidence":0.90}
"invoices in USD currency"→{"action":"list","entity":"invoices","filters":{"currency":"USD"},"confidence":0.90}
"customers with pending invoices"→{"action":"list","entity":"invoices","filters":{"status":"unpaid"},"confidence":0.90}
"contacts by job title"→{"action":"group_by","entity":"contacts","filters":{"group_by":"jobTitle"},"confidence":0.90}
"find John Smith"→{"action":"find","entity":"contacts","filters":{"search":"John Smith"},"confidence":0.90}
"search Automation Systems"→{"action":"find","entity":"companies","filters":{"search":"Automation Systems"},"confidence":0.90}
"summary of Kartik Trivedi"→{"action":"detail","entity":"contacts","filters":{"search":"Kartik Trivedi"},"confidence":0.90}
"details of ketul"→{"action":"detail","entity":"contacts","filters":{"search":"ketul"},"confidence":0.90}
"meetings for today"→{"action":"list","entity":"meetings","filters":{"time_period":"today"},"confidence":0.95}
"meetings tomorrow"→{"action":"list","entity":"meetings","filters":{"time_period":"tomorrow"},"confidence":0.95}
"today's meetings"→{"action":"list","entity":"meetings","filters":{"time_period":"today"},"confidence":0.95}
"top 5 invoices by amount"→{"action":"top_n","entity":"invoices","filters":{"limit":5,"sort":"desc"},"confidence":0.95}
"first invoice"→{"action":"detail","entity":"invoices","filters":{"sort":"asc","limit":1},"confidence":0.95}
"last 5 contacts"→{"action":"list","entity":"contacts","filters":{"sort":"desc","limit":5},"confidence":0.95}
"compare this vs last month"→{"action":"compare","entity":"invoices","filters":{},"confidence":0.10}"""


# ══════════════════════════════════════════════════════════════════════════════
# TIME PERIOD → SQL CLAUSE
# ══════════════════════════════════════════════════════════════════════════════

_ENTITY_DATE_FIELD: Dict[str, str] = {
    "invoices":    "invoice_date",
    "deals":       "closeDate",
    "tasks":       "due_date",
    "createtasks": "due_date",
    "contacts":    "createdAt",
    "companies":   "createdAt",
    "targets":     "month",
    "users":       "createdAt",
    "sales":       "sales_date",
    "meetings":    "createdAt",
}


def _time_clause(period: str, date_field: str) -> str:
    """Convert a time_period string to a SQL WHERE fragment.
    Date fields are now TEXT columns storing ISO strings — cast via ::timestamptz.
    """
    # NULLIF handles the rare empty-string edge case
    df = 'NULLIF("' + date_field + '", \'\')::timestamptz'
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
    if period == "tomorrow":
        return "DATE(" + df + ") = CURRENT_DATE + 1"
    if period == "next_month":
        return "DATE_TRUNC('month', " + df + ") = DATE_TRUNC('month', NOW() + INTERVAL '1 month')"
    if period == "next_week":
        return (
            "DATE_TRUNC('week', " + df + ") = "
            "DATE_TRUNC('week', NOW() + INTERVAL '1 week')"
        )
    # ISO date string like "2026-05-12"
    import re as _re
    if _re.match(r"^\d{4}-\d{2}-\d{2}$", period):
        return "DATE(" + df + ") = '" + period + "'"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# SQL WHERE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_where(entity: str, table: str, filters: Dict, fields: List[str]) -> str:
    """Build a SQL WHERE clause from extracted filters — direct column access, no JSONB."""
    # Per-table soft delete — outreaches uses isDeleted, everything else uses deleted
    if entity == "outreaches" or table == "outreaches":
        clauses = ['("isDeleted" = false OR "isDeleted" IS NULL)']
    else:
        clauses = ["(deleted = false OR deleted IS NULL)"]

    stage = (filters.get("stage") or "").lower().replace(" ", "_")
    if stage and entity == "deals":
        if stage == "open":
            # Open = not won and not lost
            clauses.append("stage NOT IN ('Closed Won', 'Closed Lost')")
        elif "won" in stage:
            clauses.append("stage = 'Closed Won'")
        elif "lost" in stage:
            clauses.append("stage = 'Closed Lost'")
        else:
            stage_esc = stage.replace("'", "''")
            clauses.append("stage ILIKE '%" + stage_esc + "%'")

    status = (filters.get("status") or "").lower()
    if status:
        if entity == "invoices":
            if status in ("unpaid", "outstanding", "open"):
                clauses.append("payment_status NOT IN ('paid','cancelled')")
            elif status == "overdue":
                clauses.append(
                    "NULLIF(\"due_date\", '')::timestamptz < NOW() "
                    "AND payment_status NOT IN ('paid','cancelled')"
                )
            elif status == "paid":
                clauses.append("payment_status = 'paid'")
            elif status in ("draft", "cancelled", "approved", "submitted"):
                clauses.append("payment_status = '" + status + "'")
        elif entity in ("tasks", "createtasks"):
            if status == "pending":
                clauses.append("status = 'Pending'")
            elif status in ("completed", "done"):
                clauses.append("status = 'Completed'")
            elif status == "overdue":
                clauses.append(
                    'NULLIF("due_date", \'\')::timestamptz < NOW() '
                    "AND status != 'Completed'"
                )
            elif status == "open":
                clauses.append("status = 'Open'")
        else:
            status_esc = status.replace("'", "''")
            clauses.append("status ILIKE '" + status_esc + "'")

    # Currency filter (for invoices)
    currency = (filters.get("currency") or "").upper()
    if currency:
        clauses.append("currency = '" + currency + "'")

    owner = filters.get("owner")
    if owner:
        o = str(owner).replace("'", "''")
        # owner/createdBy stores user._id (ObjectId hex) — must resolve via name subquery
        if entity in ("createtasks", "tasks"):
            clauses.append(
                '"createdBy"::text IN '
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10)"
            )
        elif entity == "deals":
            clauses.append(
                "owner::text IN "
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10)"
            )
        elif entity in ("invoices", "billing"):
            clauses.append(
                '"invoiceOwner"::text IN '
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10)"
            )
        elif entity == "sales":
            clauses.append(
                '"salesOwner"::text IN '
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10)"
            )
        elif entity == "contacts":
            clauses.append(
                '"contactOwner"::text IN '
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10)"
            )
        else:
            # Generic fallback — try owner and salesOwner
            clauses.append(
                "(owner::text IN "
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10) "
                "OR \"salesOwner\"::text IN "
                "(SELECT _id FROM \"users\" WHERE name ILIKE '%" + o + "%' LIMIT 10))"
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
            parts.append('"' + name_f + '"::text ILIKE \'%' + s + "%'")
        if first_f and first_f != name_f:
            parts.append('"' + first_f + '"::text ILIKE \'%' + s + "%'")
        if email_f:
            parts.append('"' + email_f + '"::text ILIKE \'%' + s + "%'")
        if id_f and id_f not in (name_f, first_f):
            parts.append('"' + id_f + '"::text ILIKE \'%' + s + "%'")
        if not parts:
            parts = ["name::text ILIKE '%" + s + "%'"]
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
    sort  = (filters.get("sort") or "desc").lower()
    is_detail = filters.get("detail") or action == "detail"

    # Append time-period clause
    period = (filters.get("time_period") or "").lower()
    if period:
        date_f = _ENTITY_DATE_FIELD.get(entity, "createdAt")
        tc = _time_clause(period, date_f)
        if tc:
            where = where + " AND " + tc

    name_expr = REGISTRY.display_name_expr(table, fields)

    # Date field for ORDER BY
    _ORDER_DATE = _ENTITY_DATE_FIELD.get(entity, "createdAt")
    if sort == "asc":
        order_clause = f'ORDER BY NULLIF("{_ORDER_DATE}", \'\')::timestamptz ASC NULLS LAST'
    else:
        order_clause = f'ORDER BY NULLIF("{_ORDER_DATE}", \'\')::timestamptz DESC NULLS LAST'

    # ── group_by — aggregation by a field (owner, stage, status, jobTitle) ──────
    if action == "group_by":
        group_dimension = (filters.get("group_by") or "").lower()

        if group_dimension in ("owner", "person", "user", "rep"):
            if entity in ("createtasks", "tasks"):
                # Tasks grouped by user — JOIN users to get readable names
                status_filter = ""
                stat = (filters.get("status") or "").lower()
                if stat == "pending":
                    status_filter = "AND t.status = 'Pending'"
                elif stat in ("completed", "done"):
                    status_filter = "AND t.status = 'Completed'"
                return f"""SELECT
  COALESCE(u.name, 'Unassigned') AS user_name,
  COUNT(t._id)::int AS task_count
FROM "{table}" t
LEFT JOIN "users" u ON u._id = t."createdBy"::text
WHERE (t.deleted = false OR t.deleted IS NULL)
  {status_filter}
GROUP BY u._id, u.name
ORDER BY task_count DESC"""

            elif entity == "deals":
                return f"""SELECT
  COALESCE(u.name, d.owner, 'Unassigned') AS owner_name,
  COUNT(d._id)::int AS deal_count,
  ROUND(SUM(d.grand_total_in_usd)::numeric, 2) AS total_value
FROM "deals" d
LEFT JOIN "users" u ON u._id = d.owner
WHERE {where}
GROUP BY u._id, COALESCE(u.name, d.owner)
ORDER BY deal_count DESC"""

            elif entity == "invoices":
                return f"""SELECT
  COALESCE(c."companyName", i."companyName", 'Unknown') AS company,
  COUNT(i._id)::int AS invoice_count,
  ROUND(SUM(i.grandtotal_in_usd)::numeric, 2) AS total_usd
FROM "invoices" i
LEFT JOIN "companies" c ON c._id = i.company
WHERE {where}
GROUP BY COALESCE(c."companyName", i."companyName")
ORDER BY total_usd DESC"""

        elif group_dimension in ("stage",):
            return (
                'SELECT COALESCE(stage, \'Unknown\') AS stage, COUNT(*)::int AS count'
                ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY stage ORDER BY count DESC'
            )

        elif group_dimension in ("status",):
            status_field = "payment_status" if entity == "invoices" else "status"
            return (
                'SELECT COALESCE("' + status_field + '", \'Unknown\') AS status, COUNT(*)::int AS count'
                ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY "' + status_field + '" ORDER BY count DESC'
            )

        elif group_dimension in ("jobtitle", "job_title", "jobtitle"):
            return (
                'SELECT COALESCE("jobTitle", \'Unknown\') AS job_title, COUNT(*)::int AS count'
                ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY "jobTitle" ORDER BY count DESC'
            )

        elif group_dimension in ("currency",):
            return (
                'SELECT COALESCE(currency, \'Unknown\') AS currency,'
                ' COUNT(*)::int AS count,'
                ' ROUND(SUM(grandtotal_in_usd)::numeric, 2) AS total_usd'
                ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY currency ORDER BY total_usd DESC'
            )

        elif group_dimension in ("priority",):
            return (
                'SELECT COALESCE(priority, \'Unknown\') AS priority, COUNT(*)::int AS count'
                ' FROM "' + table + '" WHERE ' + where
                + ' GROUP BY priority ORDER BY count DESC'
            )

        # Generic group_by fallback
        gf = group_dimension or "status"
        return (
            'SELECT COALESCE("' + gf + '", \'Unknown\') AS ' + gf + ', COUNT(*)::int AS count'
            ' FROM "' + table + '" WHERE ' + where
            + ' GROUP BY "' + gf + '" ORDER BY count DESC'
        )

    # ── count ─────────────────────────────────────────────────────────────────
    if action == "count":
        return 'SELECT COUNT(*)::int FROM "' + table + '" WHERE ' + where

    # ── detail — show all key columns for one record ───────────────────────────
    if action == "detail" or is_detail:
        # Build a rich SELECT with all meaningful columns
        _DETAIL_COLS: Dict[str, List[str]] = {
            "invoices":    ["invoice_number", "payment_status", "grandtotal_in_usd",
                            "currency", "invoice_date", "due_date", "payment_date",
                            "companyName", "approval_status", "notes"],
            "deals":       ["name", "stage", "grand_total_in_usd", "currency",
                            "closeDate", "dealWonAt", "dealLostAt", "createdAt"],
            "companies":   ["companyName", "lifecycleStage", "leadStatus",
                            "country", "industry", "email", "phoneNumber", "createdAt"],
            "contacts":    ["firstName", "lastName", "email", "phoneNumber",
                            "jobTitle", "lifecycleStage", "leadStatus", "createdAt"],
            "sales":       ["sales_number", "status", "grand_total_in_usd",
                            "currency", "sales_date", "isRecurring"],
            "createtasks": ["Task", "status", "priority", "due_date", "createdAt"],
        }
        detail_cols = _DETAIL_COLS.get(table, [])
        selects = []
        for col in detail_cols:
            if col in fields:
                selects.append('"' + col + '"')
        if not selects:
            selects = ["*"]
        return (
            'SELECT ' + ", ".join(selects)
            + ' FROM "' + table + '" WHERE ' + where
            + ' ' + order_clause
            + ' LIMIT ' + str(max(1, limit))
        )

    # ── list ──────────────────────────────────────────────────────────────────
    if action == "list":
        status_f = REGISTRY.get(table, "status", fields)
        amount_f = REGISTRY.get(table, "amount", fields)
        date_f   = REGISTRY.get(table, "date",   fields)
        owner_f  = REGISTRY.get(table, "owner",  fields)

        # Build enriched query with company name + owner name JOINs where applicable
        if entity == "deals":
            sql = (
                f'SELECT d.name, d.stage, d.grand_total_in_usd AS amount,'
                f' d."closeDate" AS close_date,'
                f' COALESCE(c."companyName", d.company) AS company,'
                f' COALESCE(u.name, d.owner) AS owner'
                f' FROM "deals" d'
                f' LEFT JOIN "companies" c ON c._id = d.company'
                f' LEFT JOIN "users" u ON u._id = d.owner'
                f' WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql
        if entity == "invoices":
            sql = (
                f'SELECT i.invoice_number, i.payment_status AS status,'
                f' i."grandtotal_in_usd" AS amount_usd, i.invoice_date,'
                f' i."due_date" AS due_date,'
                f' COALESCE(c."companyName", i."companyName") AS company'
                f' FROM "invoices" i'
                f' LEFT JOIN "companies" c ON c._id = i.company'
                f' WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql
        if entity == "sales":
            sql = (
                f'SELECT s.sales_number, s.status, s.grand_total_in_usd AS amount_usd,'
                f' s.sales_date,'
                f' COALESCE(c."companyName", s.company) AS company,'
                f' COALESCE(u.name, s."salesOwner") AS sales_rep'
                f' FROM "sales" s'
                f' LEFT JOIN "companies" c ON c._id = s.company'
                f' LEFT JOIN "users" u ON u._id = s."salesOwner"'
                f' WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql
        if entity == "contacts":
            sql = (
                f'SELECT TRIM(CONCAT(COALESCE("firstName",\'\'),\' \',COALESCE("lastName",\'\'))) AS name,'
                f' email, "jobTitle" AS job_title, "lifecycleStage" AS stage, "leadStatus" AS status'
                f' FROM "contacts" WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql
        if entity in ("users",):
            sql = (
                f'SELECT name, email, department FROM "users"'
                f' WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql

        if entity in ("createtasks", "tasks"):
            # Tasks list — JOIN users to show owner name
            sql = (
                f'SELECT t."Task" AS task, t.status, t.priority,'
                f' t."due_date" AS due_date,'
                f' COALESCE(u.name, \'Unassigned\') AS assigned_to'
                f' FROM "{table}" t'
                f' LEFT JOIN "users" u ON u._id = t."createdBy"::text'
                f' WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql

        if entity in ("meetings",):
            meeting_table = table  # resolved by resolve_entity_table
            sql = (
                f'SELECT * FROM "{meeting_table}" WHERE {where} {order_clause}'
                + (f' LIMIT {limit}' if filters.get("limit") else '')
            )
            return sql

        # Generic fallback
        selects = [name_expr + " AS name"]
        if status_f:
            selects.append('"' + status_f + '" AS status')
        if amount_f and entity in ("deals", "invoices", "sales"):
            selects.append('"' + amount_f + '" AS amount')
        if date_f:
            selects.append('"' + date_f + '" AS date')

        return (
            'SELECT ' + ", ".join(selects)
            + ' FROM "' + table + '" WHERE ' + where
            + ' ' + order_clause
            + (f' LIMIT {limit}' if filters.get("limit") else '')
        )

    # ── sum ───────────────────────────────────────────────────────────────────
    if action == "sum":
        amount_f = REGISTRY.get(table, "amount", fields) or "grandtotal_in_usd"
        return (
            'SELECT COALESCE(SUM("' + amount_f + '"), 0) AS total'
            + ' FROM "' + table + '" WHERE ' + where
        )

    # ── find / lookup ─────────────────────────────────────────────────────────
    if action in ("find", "lookup"):
        name_f  = REGISTRY.get(table, "name",       fields)
        first_f = REGISTRY.get(table, "first_name",  fields)
        last_f  = REGISTRY.get(table, "last_name",   fields)
        email_f = REGISTRY.get(table, "email",       fields)
        phone_f = REGISTRY.get(table, "phone",       fields)
        title_f = REGISTRY.get(table, "title",       fields)
        status_f= REGISTRY.get(table, "status",      fields)
        owner_f = REGISTRY.get(table, "owner",       fields)

        selects = []
        if name_f:
            selects.append(name_expr + " AS name")
        elif first_f and last_f:
            selects.append(
                'TRIM(CONCAT(COALESCE("' + first_f + '"::text,\'\'),\' \','
                'COALESCE("' + last_f + '"::text,\'\'))) AS name'
            )
        elif first_f:
            selects.append('"' + first_f + '" AS name')
        if email_f:
            selects.append('"' + email_f + '" AS email')
        if phone_f:
            selects.append('"' + phone_f + '" AS phone')
        if title_f:
            selects.append('"' + title_f + '" AS title')
        if status_f:
            selects.append('"' + status_f + '" AS status')
        if owner_f:
            selects.append('"' + owner_f + '"::text AS owner')
        if not selects:
            selects = ['"_id"']

        return (
            'SELECT ' + ", ".join(selects)
            + ' FROM "' + table + '" WHERE ' + where
            + ' LIMIT 10'
        )

    # ── top_n ─────────────────────────────────────────────────────────────────
    if action == "top_n":
        amount_f = REGISTRY.get(table, "amount", fields)
        # Fallback to known field names per table
        if not amount_f:
            if table == "invoices":
                amount_f = "grandtotal_in_usd"
            elif table in ("deals", "sales"):
                amount_f = "grand_total_in_usd"

        top_order = "DESC" if sort != "asc" else "ASC"

        # For invoices/sales: list individual records sorted by amount
        if entity in ("invoices", "sales") and amount_f:
            status_f = REGISTRY.get(table, "status", fields)
            date_f   = REGISTRY.get(table, "date",   fields)
            selects  = [name_expr + " AS name"]
            if status_f:
                selects.append('"' + status_f + '" AS status')
            selects.append('"' + amount_f + '" AS amount')
            if date_f:
                selects.append('"' + date_f + '" AS date')
            return (
                'SELECT ' + ", ".join(selects)
                + ' FROM "' + table + '" WHERE ' + where
                + ' ORDER BY "' + amount_f + '" ' + top_order + ' NULLS LAST'
                + ' LIMIT ' + str(limit)
            )
        # For deals: sort by deal value
        if amount_f:
            status_f = REGISTRY.get(table, "status", fields)
            date_f   = REGISTRY.get(table, "date",   fields)
            selects  = [name_expr + " AS name"]
            if status_f:
                selects.append('"' + status_f + '" AS status')
            selects.append('"' + amount_f + '" AS amount')
            if date_f:
                selects.append('"' + date_f + '" AS date')
            return (
                'SELECT ' + ", ".join(selects)
                + ' FROM "' + table + '" WHERE ' + where
                + ' ORDER BY "' + amount_f + '" ' + top_order + ' NULLS LAST'
                + ' LIMIT ' + str(limit)
            )
        return (
            'SELECT ' + name_expr + ' AS name'
            + ' FROM "' + table + '" WHERE ' + where
            + ' ' + order_clause
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

def _headers_from_sql(sql: str) -> List[str]:
    """Extract readable column names from a SQL SELECT clause."""
    m = re.search(r"SELECT\s+(.+?)\s+FROM\b", sql, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    clause = m.group(1)
    cols: List[str] = []
    depth, cur = 0, ""
    for ch in clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cols.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        cols.append(cur.strip())

    headers = []
    for col in cols:
        # Explicit AS alias
        as_m = re.search(r'\bAS\s+"?(\w+)"?\s*$', col, re.I)
        if as_m:
            headers.append(as_m.group(1))
            continue
        # Bare "colName" or colName
        bare_m = re.match(r'^"?(\w+)"?\s*$', col.strip())
        if bare_m:
            headers.append(bare_m.group(1))
            continue
        # Fallback: last token
        headers.append(col.split()[-1].strip('"') if col.split() else col[:15])
    return headers


def _format_result(rows: Any, action: str, entity: str, filters: Dict,
                   sql: str = "") -> str:
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

    # Extract column headers from SQL for better table display
    headers = _headers_from_sql(sql) if sql else []

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
        return f"**Total revenue ({period_desc}): USD {float(val):,.2f}**"

    # ── Detail action: show as rich key-value card ─────────────────────────────
    if action == "detail" or filters.get("detail"):
        row = rows[0] if isinstance(rows[0], (list, tuple)) else [rows[0]]
        lines = [f"**{entity.title()} Details:**\n"]
        for i, val in enumerate(row):
            col_name = headers[i] if i < len(headers) else f"Field {i+1}"
            # Format the column name nicely
            label = col_name.replace("_", " ").replace("At", " Date").title()
            # Format the value
            if val is None or val == "" or val == "None":
                continue  # skip empty fields
            # Try to format numbers as USD if they look like amounts
            if col_name.lower() in ("grandtotal_in_usd", "grand_total_in_usd",
                                    "subtotal_in_usd", "amount"):
                try:
                    lines.append(f"- **{label}**: USD {float(val):,.2f}")
                    continue
                except (ValueError, TypeError):
                    pass
            # Truncate very long values
            str_val = str(val)
            if len(str_val) > 100:
                str_val = str_val[:100] + "…"
            lines.append(f"- **{label}**: {str_val}")
        return "\n".join(lines)

    # ── Group by — aggregation result ─────────────────────────────────────────
    if action == "group_by":
        n = len(rows)
        tbl = format_rows_as_markdown_table(rows, headers=headers or None, max_rows=500)
        group_dim = filters.get("group_by", "group")
        status = filters.get("status", "")
        qualifier = f" {status}" if status else ""
        return f"**{entity.title()}{qualifier} by {group_dim} ({n} groups):**\n\n{tbl}"

    # ── List / find / top_n ────────────────────────────────────────────────────
    if action in ("list", "find", "top_n", "lookup"):
        n = len(rows)
        tbl = format_rows_as_markdown_table(rows, headers=headers or None, max_rows=500)
        owner = filters.get("owner", "")
        qualifier = f" for **{owner}**" if owner else ""
        status = filters.get("status", "")
        status_q = f" ({status})" if status else ""
        return f"**{n}{status_q} {entity}{qualifier}:**\n\n{tbl}"

    return format_rows_as_markdown_table(rows, headers=headers or None, max_rows=500)


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
            "todo": "createtasks", "todos": "createtasks", "followup": "createtasks",
            "task": "createtasks", "tasks": "createtasks",
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

    answer = _format_result(_res.rows, action, entity, filters, sql=sql)

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
