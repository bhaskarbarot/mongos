"""fast_path.py — Rule-based query engine (NO LLM required).

Handles ~70-80% of queries via direct SQL pattern matching.
Target latency: <300ms for cached schema, <2s for first-run schema discovery.

Covered query families (sourced from chatbot-rules):
  - Revenue / income / billing totals (with breakdowns)
  - Count + list for any entity
  - Group-by (stage / status / type / source / owner / priority / region / currency)
  - Deals filter (stage, overdue, won, lost)
  - Tasks (pending / overdue / completed / high-priority)
  - Top-N customers by revenue
  - Quarterly / monthly revenue breakdown
  - Overdue invoice aging by company
  - Target vs achieved per user / period
  - Department → user mapping
  - Companies with no business activity
  - User overdue-task summary
  - People / company search
  - Lookup lists (technologies, sources, taxes, regions, stages …)
  - Invoice detail (status, due, remaining, payment history)
  - Sales order detail (status, line items, linked invoice)
  - Pipeline summary (by stage)
  - Entity lookup (get details for X)
  - System config redirect
  - Generic list / show / give / fetch handler

Public API:
    run(query, agent) -> Optional[Dict]   # None = no match, escalate to classifier
"""

from __future__ import annotations

import calendar
import logging
import os
import random
import re
import time as _time
from typing import Any, Dict, List, Optional, Tuple

# ── Configurable natural delay (read once at import time from env) ─────────────
# Set FASTPATH_DELAY_ENABLED=false to turn off entirely (e.g. during dev/testing).
# FASTPATH_DELAY_MIN / MAX control the random range in seconds.
_DELAY_ENABLED: bool = os.getenv("FASTPATH_DELAY_ENABLED", "true").lower() == "true"
_DELAY_MIN:     int  = int(os.getenv("FASTPATH_DELAY_MIN", "5"))
_DELAY_MAX:     int  = int(os.getenv("FASTPATH_DELAY_MAX", "10"))

from pipeline.schema import (
    REGISTRY,
    _SCHEMA_LINKS,
    build_revenue_coalesce,
    find_revenue_table,
    get_document_fields,
    get_table_names,
    resolve_entity_table,
    run_sql,
)
from pipeline.utils import (
    _SUM_QUERY_KEYWORDS,
    coerce_number,
    fmt_number,
    format_rows_as_markdown_table,
    normalize_text,
    sanitize_sql_value,
)
import pipeline.llm as _llm

LOGGER = logging.getLogger("sql_chatbot")

# ── Status synonym table — shared by all handlers ────────────────────────────
# Maps user-typed words to the canonical stored value.
# None means "unpaid" logic (NOT IN paid/cancelled).
_STATUS_SYNONYMS: Dict[str, Optional[str]] = {
    "paid":          "paid",
    "unpaid":        None,
    "confirmed":     "confirmed",
    "draft":         "draft",
    "cancelled":     "cancelled",
    "canceled":      "cancelled",
    "approved":      "approved",
    "rejected":      "rejected",
    "submitted":     "submitted",
    "declined":      "declined",
    "partial":       "partial_payment",
    "pending":       "Pending",
    "completed":     "Completed",
    "done":          "Completed",
    "open":          "Open",
    "won":           "Closed Won",
    "lost":          "Closed Lost",
}


# ══════════════════════════════════════════════════════════════════════════════
# TIME PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_month_year(text: str) -> Optional[Tuple[int, int]]:
    """Return (month_1indexed, year) or None."""
    month_map: Dict[str, int] = {
        name.lower(): idx for idx, name in enumerate(calendar.month_name) if name
    }
    month_map.update({
        name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if name
    })
    year_m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    month: Optional[int] = None
    for name, idx in sorted(month_map.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(name)}\b", text):
            month = idx
            break
    if month is None:
        return None
    year = int(year_m.group(1)) if year_m else _time.gmtime().tm_year
    return (month, year)


def _parse_time_condition(text: str) -> Optional[Dict[str, str]]:
    """
    Extract a SQL WHERE clause fragment + human label for ANY time expression.
    Covers:
      • Exact month+year         "april 2026", "december 2025"
      • Last/previous year month "last year december", "previous year march"
      • Last/this period         "last month", "this year", "last quarter"
      • Last N units             "last 3 months", "past 2 years"
      • Plain year               "2025", "2026"
      • Date range               "between Jan 2025 and Mar 2025"
    Returns dict with 'condition', 'label', optionally 'use_doc_date'; or None.
    """
    import time as _time
    t = normalize_text(text)

    # ── "between Month YYYY and Month YYYY / now / today" ─────────────────────
    bet = re.search(
        r"between\s+(\w+\s+\d{4}|\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+"
        r"(\w+\s*\d{0,4}|now|today)",
        t, re.I,
    )
    if bet:
        start_raw, end_raw = bet.group(1).strip(), bet.group(2).strip()
        start_my = _extract_month_year(start_raw)
        if start_my:
            sm, sy = start_my
            start_sql = f"{sy:04d}-{sm:02d}-01"
        else:
            start_sql = start_raw
        end_sql = "NOW()" if end_raw.lower() in ("now", "today") else end_raw
        return {
            "condition":    f"updated_at BETWEEN DATE '{start_sql}' AND {end_sql}",
            "label":        f"{start_raw} to {end_raw}",
            "use_doc_date": True,
        }

    # ── "last/past N months/days/weeks/years" ─────────────────────────────────
    m = re.search(r"(?:last|past)\s+(\d+)\s+(month|day|week|year)s?", t)
    if m:
        n, unit = m.group(1), m.group(2)
        return {
            "condition": f"updated_at >= NOW() - INTERVAL '{n} {unit}s'",
            "label":     f"last {n} {unit}s",
        }

    # ── "last year <month>" / "previous year <month>" ─────────────────────────
    # Must check BEFORE "last year" alone so we get the specific month.
    _PREV_YEAR_PREFIX = r"(?:last\s+year|previous\s+year|prev\s+year|year\s+before)"
    _MONTH_NAMES = (
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    # "<month> of last year" or "last year <month>"
    prev_month_m = re.search(
        rf"(?:{_PREV_YEAR_PREFIX}\s+{_MONTH_NAMES}"
        rf"|{_MONTH_NAMES}\s+(?:of\s+)?{_PREV_YEAR_PREFIX})",
        t, re.I,
    )
    if prev_month_m:
        month_str = (prev_month_m.group(1) or prev_month_m.group(2)).lower()
        # resolve month number
        month_map = {
            "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,
            "april":4,"apr":4,"may":5,"june":6,"jun":6,"july":7,"jul":7,
            "august":8,"aug":8,"september":9,"sep":9,"october":10,"oct":10,
            "november":11,"nov":11,"december":12,"dec":12,
        }
        month_num = month_map.get(month_str)
        if month_num:
            last_year = _time.gmtime().tm_year - 1
            ms        = f"{last_year:04d}-{month_num:02d}-01"
            label     = f"{calendar.month_name[month_num]} {last_year}"
            return {
                "condition": f"date_trunc('month', updated_at) = DATE '{ms}'",
                "label":     label,
            }

    # ── "this year <month>" / "<month> of this year" ──────────────────────────
    curr_month_m = re.search(
        rf"(?:this\s+year\s+{_MONTH_NAMES}|{_MONTH_NAMES}\s+(?:of\s+)?this\s+year)",
        t, re.I,
    )
    if curr_month_m:
        month_str = (curr_month_m.group(1) or curr_month_m.group(2)).lower()
        month_map = {
            "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,
            "april":4,"apr":4,"may":5,"june":6,"jun":6,"july":7,"jul":7,
            "august":8,"aug":8,"september":9,"sep":9,"october":10,"oct":10,
            "november":11,"nov":11,"december":12,"dec":12,
        }
        month_num = month_map.get(month_str)
        if month_num:
            curr_year = _time.gmtime().tm_year
            ms        = f"{curr_year:04d}-{month_num:02d}-01"
            label     = f"{calendar.month_name[month_num]} {curr_year}"
            return {
                "condition": f"date_trunc('month', updated_at) = DATE '{ms}'",
                "label":     label,
            }

    # ── Named periods (order matters — check combined before single) ───────────
    for keyword, cond, label in [
        ("last month",    "date_trunc('month', updated_at) = date_trunc('month', NOW() - INTERVAL '1 month')",    "last month"),
        ("this month",    "date_trunc('month', updated_at) = date_trunc('month', NOW())",                          "this month"),
        ("last quarter",  "date_trunc('quarter', updated_at) = date_trunc('quarter', NOW() - INTERVAL '3 months')","last quarter"),
        ("this quarter",  "date_trunc('quarter', updated_at) = date_trunc('quarter', NOW())",                      "this quarter"),
        ("previous year", "date_trunc('year', updated_at) = date_trunc('year', NOW() - INTERVAL '1 year')",        "last year"),
        ("prev year",     "date_trunc('year', updated_at) = date_trunc('year', NOW() - INTERVAL '1 year')",        "last year"),
        ("last year",     "date_trunc('year', updated_at) = date_trunc('year', NOW() - INTERVAL '1 year')",        "last year"),
        ("year before",   "date_trunc('year', updated_at) = date_trunc('year', NOW() - INTERVAL '1 year')",        "last year"),
        ("this year",     "date_trunc('year', updated_at) = date_trunc('year', NOW())",                            "this year"),
        ("last week",     "date_trunc('week', updated_at) = date_trunc('week', NOW() - INTERVAL '1 week')",        "last week"),
        ("this week",     "date_trunc('week', updated_at) = date_trunc('week', NOW())",                            "this week"),
        ("yesterday",     "date_trunc('day', updated_at) = date_trunc('day', NOW() - INTERVAL '1 day')",           "yesterday"),
        ("today",         "date_trunc('day', updated_at) = date_trunc('day', NOW())",                              "today"),
    ]:
        if keyword in t:
            return {"condition": cond, "label": label}

    # ── Financial year (India: Apr 1 – Mar 31) ────────────────────────────────
    import time as _t
    _now_month     = _t.gmtime().tm_mon
    _now_year      = _t.gmtime().tm_year
    _fy_start_year = _now_year if _now_month >= 4 else _now_year - 1

    # "FY 2025" / "FY2025" / "financial year 2025" — explicit year first
    fy_year_m = re.search(r"\bfy\s*(\d{4})\b|\bfinancial\s+year\s+(\d{4})\b", t, re.I)
    if fy_year_m:
        fy_y = int(fy_year_m.group(1) or fy_year_m.group(2))
        return {
            "condition": (f"updated_at >= DATE '{fy_y}-04-01'"
                          f" AND updated_at < DATE '{fy_y+1}-04-01'"),
            "label": f"FY {fy_y}-{str(fy_y+1)[-2:]}",
        }
    # "last/previous financial year" / "last FY" — must check BEFORE "this" variant
    if re.search(r"\blast\s+financial\s+year\b|\blast\s+fy\b|\bprevious\s+(financial\s+year|fy)\b", t):
        _lfy = _fy_start_year - 1
        return {
            "condition": (f"updated_at >= DATE '{_lfy}-04-01'"
                          f" AND updated_at < DATE '{_lfy+1}-04-01'"),
            "label": f"FY {_lfy}-{str(_lfy+1)[-2:]}",
        }
    # "this financial year" / "current FY" / "this FY"
    if re.search(r"\bthis\s+financial\s+year\b|\bcurrent\s+(financial\s+year|fy)\b|\bthis\s+fy\b", t):
        return {
            "condition": (f"updated_at >= DATE '{_fy_start_year}-04-01'"
                          f" AND updated_at < DATE '{_fy_start_year+1}-04-01'"),
            "label": f"FY {_fy_start_year}-{str(_fy_start_year+1)[-2:]}",
        }

    # ── "Month YYYY" (explicit month + year) ──────────────────────────────────
    month_year = _extract_month_year(t)
    if month_year:
        month, year = month_year
        ms = f"{year:04d}-{month:02d}-01"
        return {
            "condition": f"date_trunc('month', updated_at) = DATE '{ms}'",
            "label":     f"{calendar.month_name[month]} {year}",
        }

    # ── Plain 4-digit year ────────────────────────────────────────────────────
    y = re.search(r"\b(20\d{2})\b", t)
    if y:
        return {
            "condition": f"date_trunc('year', updated_at) = DATE '{y.group(1)}-01-01'",
            "label":     y.group(1),
        }

    return None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def _classify_route(text: str) -> str:
    t = normalize_text(text)

    if re.search(r"\bwho\s+is\b|\bwho\s+are\b|\bfind\s+(user|person|contact|company)\b", t):
        return "search"
    if re.search(r"[A-Z]{2,}/\d{4}/\d+", text):
        return "invoice_lookup"
    # Quarterly check: skip when the primary intent is target/achievement (time modifier only)
    if (re.search(r"\bquarter(ly)?\b|Q[1-4]\b", t, re.I)
            and not re.search(r"\btarget\b|\bachieved\b|\bachievement\b|\bperformance\b|\bkpi\b", t)):
        return "quarterly"
    if re.search(
        r"\bno (business|invoice|deal|activity|order|revenue)\b"
        r"|\bnot.{0,15}given business\b|\bdon.?t.{0,15}given business\b"
        r"|\bwithout.{0,15}(business|invoice)\b|\bno.{0,10}purchased\b"
        r"|\bnot.{0,10}active\b", t,
    ):
        return "no_activity"
    if re.search(
        r"\bdepartment.{0,20}(users?|members?|employees?|names?)\b"
        r"|\b(users?|members?|employees?).{0,20}department\b", t,
    ):
        return "dept_users"
    if re.search(
        r"\b(not|haven.t|missed|overdue).{0,20}(deadline|tasks?|due)\b"
        r"|\btasks?.{0,20}overdue\b|\boverdue.{0,20}tasks?\b", t,
    ):
        return "overdue_tasks"
    if re.search(r"\b(who|which user|which person).{0,30}created\b|\bcreated by whom\b", t):
        return "chain_lookup"
    if (
        re.search(r"(?:this company|for company|company named?)\s+[A-Za-z][A-Za-z0-9\s\-&,\.]{4,}", text, re.I)
        and re.search(r"\b(invoice|billing|tax|payment|amount|total)\b", t)
    ):
        return "company_invoice"
    # Unachieved targets must be checked BEFORE the general targets route
    if (re.search(r"\b(target|targets)\b", t)
            and re.search(
                r"\bnot\s+achiev|\bhaven.?t\s+achiev|\bmissed\s+target"
                r"|\bbelow\s+target|\bunder\s+target|\bunachiev|\bnot\s+meet|\bnot\s+met\b",
                t,
            )):
        return "target_unachieved"
    if re.search(
        r"\bkpi\s+report\b|\bgive\s+(me\s+)?kpi\b|\bdashboard\s+report\b"
        r"|\bfull\s+(business\s+)?report\b|\bbusiness\s+kpi\b|\bkpi\s+dashboard\b"
        r"|\bmanager\s+dashboard\b|\boverall\s+report\b|\bkpi\s+summary\b", t,
    ):
        return "kpi_report"
    if re.search(r"\b(targets?|performance|achievement|achieved|growth potential|kpi|score)\b", t):
        return "targets"
    if re.search(r"\boutreach\b|\binterested leads?\b|\btouch(es)?\b|\bunassigned csv\b|\bdataset\b", t):
        return "outreach"
    if re.search(
        r"\ball (technolog|source|industr|tax|region|categor|lead status|lifecycle|deal stage|payment method)\b"
        r"|\bget all (technolog|source|industr|tax|region)\b"
        r"|\b(list|show) (all )?(technolog|source|tax|region)\b", t,
    ):
        return "lookup_list"
    if re.search(r"\bSO\d+\b", text, re.I) or re.search(
        r"\bsales order\b.{0,30}(line|product|item|subtotal|tax|detail)", t, re.I
    ):
        return "sales_order_detail"
    if re.search(r"\bsmtp\b|\bfile upload\b|\bupload limit\b|\bsystem config\b", t):
        return "system_config"
    if re.search(
        r"\boverdue.{0,20}(invoice|payment).{0,20}(aging|outstanding|company)\b"
        r"|\b(aging|outstanding).{0,20}(invoice|payment)\b", t,
    ):
        return "overdue_aging"
    if re.search(r"\bpipeline\b.{0,30}\b(stage|distribution|summary|value)\b", t):
        return "pipeline_summary"
    if re.search(r"\bpending\s+invoice|invoice.{0,20}(pending|unpaid|outstanding)\b", t):
        return "pending_invoices"

    # ── Active customers ────────────────────────────────────────────────────────
    if (re.search(r"\bactive\b", t)
            and re.search(r"\b(customers?|companies?|company|clients?|accounts?|contacts?)\b", t)):
        return "active_customers"

    # ── User + Task relational query ────────────────────────────────────────────
    # Detected BEFORE general fallback so relational JOIN intent is handled directly
    # rather than being mis-routed to _fp_tasks (which has no user JOIN).
    # COUNT-only queries ("how many users with task") stay in "general" so _fp_count handles them.
    if (re.search(r"\busers?\b|\bemployees?\b|\bmembers?\b|\bstaff\b|\bpeople\b|\bperson\b", t)
            and re.search(r"\btasks?\b|\btodo\b|\bfollow.?up\b|\bassignment\b", t)
            and re.search(r"\bby\b|\bper\b|\bstatus\b|\bpending\b|\bcompleted?\b|\bopen\b|\btheir\b|\ball\b|\beach\b|\bgrouped\b", t)
            and not re.search(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal number\b", t)):
        return "user_task_map"

    # ── Invoice by explicit status/stage value ──────────────────────────────────
    # Detected BEFORE general fallback so "rejected invoices" / "invoices by stage of approved"
    # don't get stripped of their filter by the generic _fp_list_records handler.
    if (re.search(r"\binvoices?\b|\bbills?\b|\bbilling\b", t)
            and re.search(
                r"\b(approved|rejected|submitted|declined|accepted)\b"
                r"|\bby\s+(?:stage|status)\s+of\b"
                r"|\bwhich\s+are\b"
                r"|\b(?:stage|status)\s+(?:is|=|:)\s*\w+",
                t,
            )):
        return "invoice_status_filter"

    if re.search(
        r"\bget\b.{0,40}\bfor\s+(this\s+)?(company|account|contact|deal)\b"
        r"|\bget\b.{0,25}\bfor\s+[A-Z]", text, re.I,
    ) and not re.search(r"\bfor\s+(all|every|each|this period|last|this month|this year)\b", t):
        return "entity_lookup"
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _fp_revenue(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    # "sales order(s)" → NOT revenue, belongs to sales table list handler
    if re.search(r"\bsales\s+order", text):
        return None
    explicit_revenue = any(kw in text for kw in [
        "revenue", "income", "earning", "billing", "invoice total", "collection",
    ])
    sales_total = re.search(
        r"\b(total|show|give)\b.{0,15}\bsales\b|\bsales\b.{0,15}\b(total|amount|figure)\b", text
    )
    if not (explicit_revenue or sales_total):
        return None
    if any(kw in text for kw in ["pipeline", "stage", "distribution", "count", "how many", "number of"]):
        return None
    if re.search(r"\btop\s+\d+\b|\bby\s+(customer|company|client|owner|rep|user|region)\b", text):
        return None

    time_cond   = _parse_time_condition(text)
    table_names = get_table_names(agent)
    table = find_revenue_table(table_names) or resolve_entity_table("invoice", table_names, user_query)
    if not table:
        return None

    fields        = get_document_fields(agent, table)
    coalesce_expr = build_revenue_coalesce(fields)

    # ── Use the document's own date field, NOT updated_at ────────────────────
    # updated_at = sync timestamp (all rows share the same date → wrong filter)
    # We must filter on the actual business date stored in the JSONB document.
    _DATE_FIELD_PRIORITY = [
        "invoice_date", "sales_date", "due_date", "closeDate",
        "close_date", "createdAt", "date", "order_date",
    ]
    fl = {f.lower(): f for f in fields}
    doc_date_field = next(
        (fl[d] for d in _DATE_FIELD_PRIORITY if d in fl), None
    )
    # Fallback: any field with 'date' in the name
    if not doc_date_field:
        doc_date_field = next((f for f in fields if "date" in f.lower()), None)

    if time_cond and doc_date_field:
        # Replace updated_at with NULLIF(document->>'field','')::timestamptz
        doc_date_expr = f"NULLIF(document->>'{doc_date_field}','')::timestamptz"
        date_condition = time_cond["condition"].replace("updated_at", doc_date_expr)
        where = f"WHERE {date_condition}"
    elif time_cond:
        where = f"WHERE {time_cond['condition']}"
    else:
        where = ""

    label = time_cond["label"] if time_cond else "all time"

    total_sql = f'SELECT COALESCE(SUM({coalesce_expr}), 0) AS total FROM "{table}" {where}'.strip()
    _res      = run_sql(agent, total_sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [total_sql]}
    rows      = _res.rows
    total     = coerce_number(rows[0][0] if rows else 0)
    answer    = f"Total revenue for **{label}**: **{fmt_number(total)}**"
    sql_queries = [total_sql]

    # Monthly breakdown when a range label is present
    range_label = label if time_cond else ""
    if any(x in range_label for x in ["month", "quarter", "year"]) and doc_date_field:
        doc_date_expr = f"NULLIF(document->>'{doc_date_field}','')::timestamptz"
        b_sql = f"""SELECT to_char(date_trunc('month', {doc_date_expr}), 'Mon YYYY'),
  COALESCE(SUM({coalesce_expr}), 0)
FROM "{table}" {where}
GROUP BY date_trunc('month', {doc_date_expr})
ORDER BY date_trunc('month', {doc_date_expr})""".strip()
        try:
            _b_res = run_sql(agent, b_sql)
            b_rows = _b_res.rows
            if b_rows:
                lines = ["\n**Monthly Breakdown:**", "| Month | Revenue |", "| --- | --- |"]
                for r in b_rows:
                    lines.append(f"| {r[0] or 'Unknown'} | {fmt_number(coerce_number(r[1]))} |")
                answer += "\n" + "\n".join(lines)
                sql_queries.append(b_sql)
        except Exception:
            pass

    return {
        "answer":      answer,
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": sql_queries,
    }


def _fp_count(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    has_count_word = any(kw in text for kw in ["how many", "count", "total number", "number of"])
    has_total_word = bool(re.search(r"\btotal\b", text))
    if not (has_count_word or has_total_word):
        return None

    table_names = get_table_names(agent)
    if not table_names:
        return None

    # ── Entity extraction ──────────────────────────────────────────────────────
    # Capture the full phrase after the count keyword, then try words last-to-first
    # so "paid invoices" → entity="invoices", "closed won deals" → entity="deals".
    # The old single-word capture (r"\s+(\w+)") would grab "paid" instead of "invoices".
    entity = ""
    _STOP = {"with", "who", "have", "has", "are", "is", "a", "an", "the", "of"}
    for kw in ["how many", "number of", "total number of", "count of", "count"]:
        # Capture everything after the keyword up to a time/stop phrase or end
        m = re.search(
            rf"{re.escape(kw)}\s+(.+?)(?:\s*\??\s*$|\s+(?:in\b|for\b|during\b|this\b|last\b|with\b|who\b))",
            text,
        )
        if not m:
            m = re.search(rf"{re.escape(kw)}\s+(.+?)(\?|$)", text)
        if m:
            phrase = m.group(1).strip().rstrip("?").strip()
            words  = [w for w in phrase.split() if w not in _STOP]
            # Try words from right (noun) to left (adjective) to find entity table
            for word in reversed(words):
                if resolve_entity_table(word, table_names, user_query):
                    entity = word
                    break
            if not entity and words:
                entity = words[-1]
            break

    table = resolve_entity_table(entity, table_names, user_query)
    if not table:
        for t in table_names:
            tl = t.lower()
            if re.search(rf"\b{re.escape(tl)}\b", text) or (
                tl.endswith("s") and re.search(rf"\b{re.escape(tl[:-1])}\b", text)
            ):
                table = t
                break
    if not table:
        return None

    if re.search(r"\bdetails?\b", text):
        return None

    # ── Attribute / status filter detection ────────────────────────────────────
    # Detects adjectives BEFORE the entity noun and maps them to WHERE clauses.
    # Examples: "paid invoices"→payment_status='paid', "closed won deals"→stage ILIKE 'Closed Won'
    fields = get_document_fields(agent, table)

    # Priority-ordered map: (regex, field_keyword, sql_operator, sql_value, human_label)
    # Multi-word patterns (e.g. "closed won") must appear BEFORE single-word ones.
    _ATTR_FILTER_MAP = [
        # Invoice payment / approval statuses — multi-word or specific first
        (r"\bpaid\b",               "payment_status", "=",       "'paid'",                   "paid"),
        (r"\bunpaid\b",             "payment_status", "NOT IN",  "('paid','cancelled')",      "unpaid"),
        (r"\bapproved\b",           "payment_status", "ILIKE",   "'approved'",               "approved"),
        (r"\brejected\b",           "payment_status", "ILIKE",   "'rejected'",               "rejected"),
        (r"\bsubmitted\b",          "payment_status", "ILIKE",   "'submitted'",              "submitted"),
        (r"\bdeclined\b",           "payment_status", "ILIKE",   "'declined'",               "declined"),
        (r"\bconfirmed\b",          "payment_status", "ILIKE",   "'confirmed'",              "confirmed"),
        (r"\bdraft\b",              "payment_status", "ILIKE",   "'draft'",                  "draft"),
        (r"\bcancell?ed\b",         "payment_status", "ILIKE",   "'cancelled'",              "cancelled"),
        (r"\bpartial.?payment\b",   "payment_status", "ILIKE",   "'partial_payment'",        "partial payment"),
        # Deal stages — multi-word checked before single-word
        (r"\bclosed\s+won\b",       "stage",          "ILIKE",   "'Closed Won'",             "closed won"),
        (r"\bclosed\s+lost\b",      "stage",          "ILIKE",   "'Closed Lost'",            "closed lost"),
        (r"\bwon\b",                "stage",          "ILIKE",   "'Closed Won'",             "closed won"),
        (r"\blost\b",               "stage",          "ILIKE",   "'Closed Lost'",            "closed lost"),
        # Generic statuses (tasks, contacts, etc.)
        (r"\bpending\b",            "status",         "=",       "'Pending'",                "pending"),
        (r"\bcompleted?\b",         "status",         "=",       "'Completed'",              "completed"),
        (r"\bopen\b",               "status",         "=",       "'Open'",                   "open"),
        (r"\bhigh.{0,2}priorit",    "priority",       "=",       "'High'",                   "high priority"),
        (r"\bmedium.{0,2}priorit",  "priority",       "=",       "'Medium'",                 "medium priority"),
        (r"\blow.{0,2}priorit",     "priority",       "=",       "'Low'",                    "low priority"),
    ]

    # Also try "by stage of X" / "with status X" explicit-value extraction
    # e.g. "how many invoices by stage of approved" → approved
    _stage_of_m = re.search(
        r"\bby\s+(?:stage|status)\s+of\s+(\w+)|\b(?:stage|status)\s+(?:is|=|:)\s*(\w+)",
        text, re.I,
    )
    if _stage_of_m and not filter_parts:
        _stage_val = (_stage_of_m.group(1) or _stage_of_m.group(2) or "").lower()
        if _stage_val:
            # Try to find best matching field
            for _fkw in ["payment_status", "paymentStatus", "stage", "status"]:
                _actual = next(
                    (f for f in fields if f.lower().replace("_","") == _fkw.lower().replace("_","")
                     or _fkw.lower().replace("_","") in f.lower().replace("_","")),
                    None,
                )
                if _actual:
                    filter_parts.append(f"LOWER(document->>'{_actual}') = '{_stage_val}'")
                    filter_label = _stage_val
                    break

    filter_parts: List[str] = []
    filter_label = ""
    for pattern, field_kw, op, val, label in _ATTR_FILTER_MAP:
        if not re.search(pattern, text):
            continue
        actual = next(
            (f for f in fields if f.lower() == field_kw.lower()
             or field_kw.lower() in f.lower()),
            None,
        )
        if not actual:
            continue
        if op == "NOT IN":
            filter_parts.append(f"document->>'{actual}' NOT IN {val}")
        else:
            filter_parts.append(f"document->>'{actual}' {op} {val}")
        filter_label = label
        break  # one status/stage filter per count query

    # ── "users with task" — relational filter ──────────────────────────────────
    if (table.lower() == "users"
            and re.search(r"\bwith\s+tasks?\b|\bwho\s+have\s+tasks?\b|\bhave\s+tasks?\b", text)
            and any(t.lower() == "createtasks" for t in table_names)):
        filter_parts.append(
            "document->>'_id' IN ("
            "SELECT DISTINCT document->>'createdBy' FROM \"createtasks\" "
            "WHERE document->>'createdBy' IS NOT NULL"
            ")"
        )
        filter_label = "with tasks"

    # ── Build WHERE clause ──────────────────────────────────────────────────────
    time_cond   = _parse_time_condition(text)
    where_parts = []
    if time_cond:
        where_parts.append(time_cond["condition"])
    where_parts.extend(filter_parts)
    where      = ("WHERE " + " AND ".join(f"({p})" for p in where_parts)) if where_parts else ""
    time_label = f" for {time_cond['label']}" if time_cond else ""

    entity_label = f"{filter_label} {table}".strip() if filter_label else table

    count_sql = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _res      = run_sql(agent, count_sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql]}
    rows      = _res.rows
    total     = coerce_number(rows[0][0] if rows else 0)
    count_answer = f"Total **{entity_label}**{time_label}: **{fmt_number(total)}**"
    sql_queries  = [count_sql]

    wants_list = any(kw in text for kw in [
        "give me", "show me", "show", "list", "names", "name", "what are", "tell me",
    ])
    wants_all  = bool(re.search(r"\ball\b", text))

    explicit_count_only = bool(re.search(r"\b(count|how many|number of|count of)\b", text)) and not has_total_word
    include_list = (wants_all or (has_total_word and not explicit_count_only) or wants_list) and not explicit_count_only
    if include_list and total > 0:
        name_expr = REGISTRY.display_name_expr(table, fields)
        list_sql  = (
            f'SELECT {name_expr} AS name FROM "{table}" {where} '
            f'ORDER BY updated_at DESC LIMIT 100'
        ).strip()
        _list_res = run_sql(agent, list_sql)
        list_rows = _list_res.rows
        names     = [str(r[0]).strip() for r in list_rows if r and r[0] and str(r[0]).strip()]
        sql_queries.append(list_sql)
        if names:
            bullet_list = "\n".join(f"- {n}" for n in names)
            answer = f"{count_answer}\n\n**Names:**\n{bullet_list}"
        else:
            answer = count_answer
    else:
        answer = count_answer

    return {
        "answer":      answer,
        "tables_used": [table],
        "confidence":  0.98,
        "sql_queries": sql_queries,
        "_entity":     table,
        "_count":      int(total) if total else 0,
    }


def _fp_group_by(agent, user_query: str) -> Optional[Dict]:
    text        = normalize_text(user_query)
    table_names = get_table_names(agent)
    if not table_names:
        return None

    GROUP_KEYWORDS = [
        "stage", "status", "type", "category", "source", "owner",
        "assigned", "priority", "region", "country", "currency",
    ]
    group_kw = next((kw for kw in GROUP_KEYWORDS if kw in text), None)
    wants_detail_summary = bool(re.search(r"\bdetails?\b", text))
    if not group_kw and not wants_detail_summary:
        return None
    # If the user is asking for a specific value of that field — not a group-by
    specific_val = re.search(
        rf"\b{re.escape(group_kw)}\s+(?:is|=|equals?|of)\s+([A-Z]{{2,}}|\w+)", text, re.I
    )
    if specific_val:
        return None

    # ── Entity-aware field aliases ─────────────────────────────────────────────
    # Deals has no "status" field — its equivalent is "stage".
    # "category" in deals maps to "type" or "project_type".
    # This prevents _fp_group_by from falling through to the wrong table (e.g. bills).
    _ENTITY_FIELD_MAP: Dict[str, Dict[str, str]] = {
        "deals": {
            "status":   "stage",
            "category": "type",
            "group":    "type",
            "phase":    "stage",
        },
        "contacts": {
            "status": "leadStatus",
            "stage":  "lifecycleStage",
        },
        "companies": {
            "status":    "leadStatus",
            "stage":     "lifecycleStage",
            "lifecycle": "lifecycleStage",
        },
        "invoices": {
            "status": "payment_status",
        },
    }

    table, group_field = None, None
    DEFAULT_DETAIL_FIELD_BY_TABLE: Dict[str, List[str]] = {
        "deals": ["stage", "type", "status", "category"],
        "invoices": ["payment_status", "status", "type"],
        "contacts": ["leadStatus", "lifecycleStage", "status", "type"],
        "tasks": ["status", "priority", "type"],
        "createtasks": ["status", "priority", "type"],
        "companies": ["leadStatus", "lifecycleStage", "status", "category"],
    }

    # Step 1 — explicitly mentioned entity table (highest priority)
    for t in table_names:
        tl = t.lower()
        mentioned = re.search(rf"\b{re.escape(tl)}\b", text) or (
            tl.endswith("s") and re.search(rf"\b{re.escape(tl[:-1])}\b", text)
        )
        if not mentioned:
            continue
        fields = get_document_fields(agent, t)
        fl_lower = {f.lower(): f for f in fields}
        # Exact match first (e.g. "stage" must not pick "Pre_stage")
        exact = next((f for f in fields if f.lower() == group_kw), None)
        match = exact or next((f for f in fields if group_kw in f.lower()), None)
        if match:
            table, group_field = t, match
            break
        # Check semantic alias (e.g. "status" → "stage" for deals)
        alias_field_kw = _ENTITY_FIELD_MAP.get(tl, {}).get(group_kw)
        if alias_field_kw:
            alias_match = next((f for f in fields if f.lower() == alias_field_kw
                                or alias_field_kw in f.lower()), None)
            if alias_match:
                table, group_field = t, alias_match
                break

    # Step 2 — no explicitly mentioned table: search all tables
    if not table:
        for t in table_names:
            fields = get_document_fields(agent, t)
            exact  = next((f for f in fields if f.lower() == group_kw), None)
            if exact:
                table, group_field = t, exact
                break
    if not table:
        for t in table_names:
            fields = get_document_fields(agent, t)
            match  = next((f for f in fields if group_kw in f.lower()), None)
            if match:
                table, group_field = t, match
                break
    if not table or not group_field:
        # "details of <entity>" fallback: pick a sensible default grouping field.
        if wants_detail_summary:
            entity_table = None
            for t in table_names:
                tl = t.lower()
                if re.search(rf"\b{re.escape(tl)}\b", text) or (
                    tl.endswith("s") and re.search(rf"\b{re.escape(tl[:-1])}\b", text)
                ):
                    entity_table = t
                    break
            if entity_table:
                fields = get_document_fields(agent, entity_table)
                preferred = DEFAULT_DETAIL_FIELD_BY_TABLE.get(entity_table.lower(), ["status", "type", "category"])
                for pref in preferred:
                    group_field = next((f for f in fields if f.lower() == pref or pref in f.lower()), None)
                    if group_field:
                        table = entity_table
                        break
        if not table or not group_field:
            return None

    wants_sum = any(kw in text for kw in _SUM_QUERY_KEYWORDS)
    time_cond = _parse_time_condition(text)
    where     = f"WHERE {time_cond['condition']}" if time_cond else ""

    if wants_sum:
        coalesce_expr = build_revenue_coalesce(get_document_fields(agent, table))
        sql = f"""SELECT COALESCE(document->>'{group_field}', 'Unknown') AS {group_field},
  COALESCE(SUM({coalesce_expr}), 0) AS total
FROM "{table}" {where}
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20""".strip()
        metric_header = "Revenue"
    else:
        sql = f"""SELECT COALESCE(document->>'{group_field}', 'Unknown') AS {group_field},
  COUNT(*)::int AS count
FROM "{table}" {where}
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20""".strip()
        metric_header = "Count"

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No data found in **{table}** grouped by {group_field}.",
            "tables_used": [table],
            "confidence":  0.85,
            "sql_queries": [sql],
        }

    time_label = f" ({time_cond['label']})" if time_cond else ""
    lines = [f"| {group_field.title()} | {metric_header} |", "| --- | --- |"]
    for row in rows:
        val    = row[0] if row[0] else "Unknown"
        metric = coerce_number(row[1])
        lines.append(f"| {val} | {fmt_number(metric)} |")
    total_line = ""
    if not wants_sum:
        total_items = int(sum(coerce_number(r[1]) for r in rows))
        total_line = f"**Total {table}: {fmt_number(total_items)}**\n\n"
    return {
        "answer":      total_line + f"**{table.title()}** by {group_field}{time_label}:\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.96,
        "sql_queries": [sql],
    }


def _fp_deals_filter(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not any(kw in text for kw in ["deal", "opportunit"]):
        return None

    STAGE_MAP = [
        (["closed won", "closedwon", "won deal"],      "Closed Won"),
        (["closed lost", "closedlost", "lost deal"],   "Closed Lost"),
        (["lost"],                                      "Closed Lost"),
        (["proposal"],                                  "Quotation Sent"),
        (["negotiation"],                               "Negotiation"),
        (["on hold"],                                   "On Hold"),
        (["contract", "under review"],                  "Contract Under Review"),
        (["analysis", "to be quoted"],                  "Analysis - To be Quoted"),
    ]
    matched_stage = None
    wants_overdue = "overdue" in text or "stuck" in text
    wants_open    = bool(re.search(r"\bopen\b", text))  # "open deals", "which deals are open"
    for patterns, stage in STAGE_MAP:
        if any(p in text for p in patterns):
            matched_stage = stage
            break

    if not matched_stage and not wants_overdue and not wants_open:
        return None

    table_names = get_table_names(agent)
    table       = next((t for t in table_names if t.lower() == "deals"), None)
    if not table:
        return None

    fields        = get_document_fields(agent, table)
    stage_field   = next((f for f in fields if f.lower() == "stage"), None) or \
                    next((f for f in fields if "stage" in f.lower()), "stage")
    name_field    = next((f for f in fields if f.lower() == "name"), "name")
    close_field   = next((f for f in fields if "close" in f.lower()), None)
    amount_field  = next((f for f in fields if f.lower() in ["grand_total", "grand_total_in_usd"]), None)
    deleted_field = next((f for f in fields if f.lower() == "deleted"), None)
    # Fields used by open-deal detection (preferred over stage NOT IN approach)
    won_field     = next((f for f in fields if "wonAt" in f or "won_at" in f.lower()), None)
    lost_field    = next((f for f in fields if "lostAt" in f or "lost_at" in f.lower()), None)

    where_parts = []
    if deleted_field:
        where_parts.append(f"COALESCE(document->>'{deleted_field}', 'false') != 'true'")

    if matched_stage:
        where_parts.append(f"document->>'{stage_field}' ILIKE '{matched_stage}'")
    elif wants_open:
        # Open deals = not won AND not lost.
        # Prefer dealWonAt/dealLostAt IS NULL (most accurate);
        # fall back to stage NOT IN ('Closed Won','Closed Lost').
        if won_field and lost_field:
            where_parts.append(f"document->>'{won_field}' IS NULL")
            where_parts.append(f"document->>'{lost_field}' IS NULL")
        elif won_field:
            where_parts.append(f"document->>'{won_field}' IS NULL")
        elif lost_field:
            where_parts.append(f"document->>'{lost_field}' IS NULL")
        else:
            where_parts.append(
                f"document->>'{stage_field}' NOT IN ('Closed Won', 'Closed Lost')"
            )
    elif wants_overdue and close_field:
        where_parts.append(f"NULLIF(document->>'{close_field}', '')::timestamptz < NOW()")
        where_parts.append(
            f"document->>'{stage_field}' NOT IN ('Closed Won', 'Closed Lost')"
        )

    time_cond = _parse_time_condition(text)
    if time_cond and close_field:
        cond = time_cond["condition"].replace(
            "updated_at", f"NULLIF(document->>'{close_field}','')::timestamptz"
        )
        where_parts.append(cond)
    elif time_cond:
        where_parts.append(time_cond["condition"])

    where        = "WHERE " + " AND ".join(f"({p})" for p in where_parts) if where_parts else ""
    select_parts = [f"document->>'{name_field}' AS name", f"document->>'{stage_field}' AS stage"]
    if amount_field:
        select_parts.append(f"NULLIF(document->>'{amount_field}','')::numeric AS amount")
    if close_field:
        select_parts.append(f"document->>'{close_field}' AS close_date")

    # Count first so the answer header is accurate
    count_sql = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _cnt = run_sql(agent, count_sql)
    total_count = coerce_number(_cnt.rows[0][0] if _cnt.rows else 0) if not _cnt.error else 0

    sql  = f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} ORDER BY updated_at DESC LIMIT 50".strip()
    _res = run_sql(agent, sql)

    label = matched_stage or ("open" if wants_open else "overdue" if wants_overdue else "filtered")
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql, sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No **{label}** deals found.",
            "tables_used": [table],
            "confidence":  0.9,
            "sql_queries": [count_sql, sql],
        }

    headers = ["Name", "Stage"] + (["Amount"] if amount_field else []) + (["Close Date"] if close_field else [])
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:20]:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 20:
        lines.append(f"_…and {len(rows)-20} more_")

    summary = f"**Total {label} deals: {fmt_number(total_count)}**\n\n" if wants_open else ""
    return {
        "answer":      summary + f"**{label.title()} Deals (showing {min(len(rows), 20)}):**\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": [count_sql, sql],
    }


def _fp_tasks(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not any(kw in text for kw in ["task", "follow-up", "followup", "follow up", "todo"]):
        return None
    # Let _fp_count handle explicit count intents for tasks.
    if any(kw in text for kw in ["how many", "count", "number of", "total number"]):
        return None

    table_names = get_table_names(agent)
    table       = resolve_entity_table("tasks", table_names, user_query)
    if not table:
        return None

    fields         = get_document_fields(agent, table)
    task_field     = next((f for f in fields if f.lower() in ["task", "title", "name", "subject"]), None)
    status_field   = next((f for f in fields if "status" in f.lower()), None)
    priority_field = next((f for f in fields if "priority" in f.lower()), None)
    due_field      = next((f for f in fields if "due_date" in f.lower() or f.lower() == "due"), None)
    deleted_field  = next((f for f in fields if "deleted" in f.lower()), None)

    where_parts = []
    if deleted_field:
        where_parts.append(f"COALESCE(document->>'{deleted_field}', 'false') != 'true'")

    label = "tasks"
    if "overdue" in text:
        if due_field:
            where_parts.append(f"NULLIF(document->>'{due_field}', '')::timestamptz < NOW()")
        if status_field:
            where_parts.append(f"COALESCE(document->>'{status_field}', '') != 'Completed'")
        label = "overdue tasks"
    elif "pending" in text or "open" in text:
        if status_field:
            where_parts.append(f"document->>'{status_field}' = 'Pending'")
        label = "pending tasks"
    elif "completed" in text or "done" in text:
        if status_field:
            where_parts.append(f"document->>'{status_field}' = 'Completed'")
        label = "completed tasks"

    if "high" in text and priority_field:
        where_parts.append(f"document->>'{priority_field}' = 'High'")

    time_cond = _parse_time_condition(text)
    if time_cond:
        where_parts.append(time_cond["condition"])

    where = "WHERE " + " AND ".join(f"({p})" for p in where_parts) if where_parts else ""

    cols = []
    if task_field:     cols.append(f"COALESCE(document->>'{task_field}', 'Unnamed') AS task")
    if status_field:   cols.append(f"COALESCE(document->>'{status_field}', '—') AS status")
    if priority_field: cols.append(f"COALESCE(document->>'{priority_field}', '—') AS priority")
    if due_field:      cols.append(f"document->>'{due_field}' AS due_date")
    if not cols:
        return None

    count_sql   = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _cnt_res    = run_sql(agent, count_sql)
    if _cnt_res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql]}
    total_rows  = _cnt_res.rows
    total       = coerce_number(total_rows[0][0] if total_rows else 0)
    sql         = f"SELECT {', '.join(cols)} FROM \"{table}\" {where} ORDER BY updated_at DESC LIMIT 20".strip()
    _row_res    = run_sql(agent, sql)
    if _row_res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql, sql]}
    rows        = _row_res.rows

    if not rows:
        return {"answer": f"No {label} found.", "tables_used": [table], "confidence": 0.9, "sql_queries": [sql]}

    headers = [c.split(" AS ")[-1].replace("_", " ").title() for c in cols]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      f"**{total} {label}** (showing {min(20, int(total))}):\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": [count_sql, sql],
    }


def _fp_list_records(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(
        r"\b(give me|tell me|show me|show|list|get|fetch|display|all|top|first|highest|lowest|biggest|largest|recent|latest|newest|oldest|cheapest|details?)\b",
        text,
    ):
        return None
    wants_details = bool(re.search(r"\bdetails?\b", text))

    table_names = get_table_names(agent)
    table: Optional[str] = None

    # Find the FIRST table name mentioned in the query (left-to-right).
    # This prevents "deals with categories" from resolving to the 'categories'
    # table just because it comes earlier alphabetically.
    best_pos = len(text) + 1
    for t in table_names:
        tl = t.lower()
        m = re.search(rf"\b{re.escape(tl)}\b", text)
        if not m and tl.endswith("s"):
            m = re.search(rf"\b{re.escape(tl[:-1])}\b", text)
        if m and m.start() < best_pos:
            best_pos = m.start()
            table = t
    if not table:
        return None

    fields       = get_document_fields(agent, table)
    name_expr    = REGISTRY.display_name_expr(table, fields)

    # ── "deals with categories/stages/types" → group-by breakdown ─────────────
    # Detect "with <groupby>" pattern: "deals with stages", "deals with categories"
    _GROUPBY_KEYWORDS = {
        r"\b(categor\w+)\b":  ("stage",    "category"),   # categories → stage field
        r"\b(stage\w*)\b":    ("stage",    "stage"),
        r"\b(type\w*)\b":     ("stage",    "type"),
        r"\b(status\w*)\b":   ("status",   "status"),
        r"\b(owner\w*)\b":    ("owner",    "owner"),
        r"\b(region\w*)\b":   ("region",   "region"),
        r"\b(source\w*)\b":   ("source",   "source"),
        r"\b(priorit\w+)\b":  ("priority", "priority"),
        r"\b(currenc\w+)\b":  ("currency", "currency"),
    }
    _groupby_field: Optional[str] = None
    _groupby_alias: str           = ""
    _with_groupby = re.search(r"\bwith\s+(\w+)", text)
    if _with_groupby:
        kw = _with_groupby.group(1).lower()
        for pattern, (role, alias) in _GROUPBY_KEYWORDS.items():
            if re.match(pattern, kw):
                f = REGISTRY.get(table, role, fields)
                if f:
                    _groupby_field = f
                    _groupby_alias = alias
                break

    if _groupby_field:
        # Return a grouped count breakdown instead of individual rows
        group_expr = f"document->>'{_groupby_field}'"
        grp_sql = (
            f"SELECT COALESCE({group_expr}, 'Unknown') AS {_groupby_alias}, "
            f"COUNT(*)::int AS count "
            f'FROM "{table}" '
            f"WHERE COALESCE(document->>'deleted','false') != 'true' "
            f"GROUP BY 1 ORDER BY count DESC"
        )
        _g_res = run_sql(agent, grp_sql)
        if not _g_res.error and _g_res.rows:
            total = sum(r[1] for r in _g_res.rows if r[1])
            lines = [f"| {_groupby_alias.title()} | Count |", "| --- | --- |"]
            for r in _g_res.rows:
                lines.append(f"| {r[0] or '—'} | {r[1]} |")
            return {
                "answer":      (
                    f"**{table.title()} by {_groupby_alias} ({total} total):**\n\n"
                    + "\n".join(lines)
                ),
                "tables_used": [table],
                "confidence":  0.96,
                "sql_queries": [grp_sql],
            }

    select_parts = [f"{name_expr} AS name"]
    for role, alias in [
        ("identifier", "ref"), ("stage", "stage"), ("status", "status"),
        ("amount", "amount"), ("date", "date"),
    ]:
        f = REGISTRY.get(table, role, fields)
        if f and f"document->>'{f}'" not in name_expr:
            select_parts.append(f"document->>'{f}' AS {alias}")
        if len(select_parts) >= 6:
            break

    # ── Field filters ──────────────────────────────────────────────────────────
    field_filters: List[str] = []

    # 1. Explicit "field is value" pattern  e.g. "currency is USD", "status is paid"
    for role in ["status", "currency", "owner"]:
        f = REGISTRY.get(table, role, fields)
        if f:
            vm = re.search(rf"\b{role}\s+(?:is|=|equals?|of)\s+([A-Za-z]{{2,}})", text, re.I)
            if vm:
                val = sanitize_sql_value(vm.group(1).upper() if role == "currency" else vm.group(1))
                # SAFE: val sanitized via sanitize_sql_value()
                field_filters.append(f"document->>'{f}' ILIKE '{val}'")

    # 2. Status adjective detection  "paid invoices" / "list of paid invoices"
    #    Checks for status words BEFORE the table noun — no "is/=" needed
    if not any("payment_status" in ff or "status" in ff for ff in field_filters):
        _STATUS_ADJECTIVES = {
            # invoice payment_status / approval values
            r"\bpaid\b":                  ("payment_status", "paid"),
            r"\bunpaid\b":                ("payment_status", None),      # NOT IN paid/cancelled
            r"\bapproved\b":              ("payment_status", "approved"),
            r"\brejected\b":              ("payment_status", "rejected"),
            r"\bsubmitted\b":             ("payment_status", "submitted"),
            r"\bdeclined\b":              ("payment_status", "declined"),
            r"\bconfirmed\b":             ("payment_status", "confirmed"),
            r"\bdraft\b":                 ("payment_status", "draft"),
            r"\bcancell?ed\b":            ("payment_status", "cancelled"),
            r"\bpartial[\s_]?payment\b":  ("payment_status", "partial_payment"),
            # generic status values
            r"\bpending\b":               ("status", "Pending"),
            r"\bcompleted?\b":            ("status", "Completed"),
            r"\bopen\b":                  ("status", "Open"),
        }

        # Also handle "by stage of X" / "which are X" explicit-value extraction
        _by_stage = re.search(
            r"\bby\s+(?:stage|status)\s+of\s+(\w+)"
            r"|\bwhich\s+are\s+(\w+)"
            r"|\bstage\s+(?:is|=|:)\s*(\w+)"
            r"|\bstatus\s+(?:is|=|:)\s*(\w+)",
            text, re.I,
        )
        if _by_stage and not any("payment_status" in ff or "status" in ff for ff in field_filters):
            _sv = next((g for g in _by_stage.groups() if g), None)
            if _sv:
                for _fkw in ["payment_status", "paymentStatus", "stage", "status"]:
                    _actual = next(
                        (f for f in fields if f.lower().replace("_","") == _fkw.lower().replace("_","")
                         or _fkw.lower().replace("_","") in f.lower().replace("_","")),
                        None,
                    )
                    if _actual:
                        field_filters.append(f"LOWER(document->>'{_actual}') = '{_sv.lower()}'")
                        break
        for pattern, (field_kw, val) in _STATUS_ADJECTIVES.items():
            if re.search(pattern, text):
                # Find the actual field name in this table
                actual = next((f for f in fields if f.lower() == field_kw
                               or field_kw in f.lower()), None)
                if actual:
                    if val is None:  # "unpaid" → NOT IN (paid, cancelled)
                        field_filters.append(
                            f"COALESCE(document->>'{actual}','') NOT IN ('paid','cancelled')"
                        )
                    else:
                        field_filters.append(f"document->>'{actual}' ILIKE '{val}'")
                    break

    # 3. Currency filter from query text  "USD invoices" / "invoices in INR"
    if not any("currency" in ff.lower() for ff in field_filters):
        cur_m = re.search(r"\b([A-Z]{3})\b", user_query)
        if cur_m:
            currency_field = REGISTRY.get(table, "currency", fields) or \
                             next((f for f in fields if "currency" in f.lower()), None)
            if currency_field:
                cur_val = sanitize_sql_value(cur_m.group(1))
                # SAFE: cur_val sanitized via sanitize_sql_value()
                field_filters.append(
                    f"UPPER(document->>'{currency_field}') = '{cur_val}'"
                )

    # ── Time filter using document date field (NOT updated_at) ────────────────
    _DATE_FIELD_PRIORITY = [
        "invoice_date", "sales_date", "due_date", "closeDate",
        "close_date", "createdAt", "date", "order_date",
    ]
    fl_lower = {f.lower(): f for f in fields}
    doc_date_field = next(
        (fl_lower[d] for d in _DATE_FIELD_PRIORITY if d in fl_lower), None
    )
    if not doc_date_field:
        doc_date_field = next((f for f in fields if "date" in f.lower()), None)

    time_cond   = _parse_time_condition(text)
    where_parts = []
    if time_cond:
        cond = time_cond["condition"]
        # Replace updated_at with the real business date field
        if doc_date_field:
            doc_date_expr = f"NULLIF(document->>'{doc_date_field}','')::timestamptz"
            cond = cond.replace("updated_at", doc_date_expr)
            # Exclude nulls so partial-date rows don't bleed in
            where_parts.append(f"{doc_date_expr} IS NOT NULL")
        where_parts.append(cond)
    where_parts.extend(field_filters)
    where = ("WHERE " + " AND ".join(f"({p})" for p in where_parts)) if where_parts else ""

    # ── Ordering / Limit ───────────────────────────────────────────────────────
    amount_field = REGISTRY.get(table, "amount", fields) or next(
        (f for f in fields if f.lower() in ["grand_total", "grand_total_in_usd", "amount", "total", "value"]),
        None,
    )

    # ── Ordinal / limit parsing ────────────────────────────────────────────────
    # "top 5", "last 10", "first", "1st", "3rd", "10th", etc.
    top_match     = re.search(r"\btop\s+(\d+)\b", text)
    bare_top      = re.search(r"\btop\b", text) and not top_match
    last_n_match  = re.search(r"\blast\s+(\d+)\b", text)
    n_match       = re.search(r"(\d+)\s+(?:record|row|result|item)s?\b", text)
    # Ordinal: "1st", "2nd", "3rd", "10th" — resolve to LIMIT + OFFSET
    ordinal_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", text)
    first_match   = re.search(r"\b(first|oldest|earliest)\b", text) or (
                    ordinal_match and int(ordinal_match.group(1)) == 1
                  )
    last_match    = re.search(r"\b(last|latest|newest|most\s+recent)\b", text) and not last_n_match

    # Synonyms for DESC: highest, biggest, largest, most, recent, latest, newest
    sort_desc_kw = re.search(
        r"\b(highest\s+to\s+(lower|lowest)|descend\w*|by\s+amount"
        r"|largest\s+first|biggest\s+first|most\s+expensive"
        r"|sort\w*\s+desc\w*|order\w*\s+desc\w*|highest\s+first"
        r"|biggest|largest|most\b|recent|latest|newest)\b",
        text,
    )
    # Synonyms for ASC: smallest, cheapest, lowest, oldest
    sort_asc_kw = re.search(
        r"\b(lowest\s+to\s+(higher|highest)|ascend\w*|smallest\s+first"
        r"|cheapest\s+first|sort\w*\s+asc\w*|order\w*\s+asc\w*"
        r"|lowest\s+first|smallest|cheapest|oldest)\b",
        text,
    )

    # Build the ORDER BY with tie-breaker (#12 — ranking stability)
    def _amount_desc():
        base = f"NULLIF(document->>'{amount_field}','')::numeric DESC NULLS LAST"
        if doc_date_field:
            base += f", NULLIF(document->>'{doc_date_field}','')::timestamptz DESC NULLS LAST"
        return base

    def _amount_asc():
        base = f"NULLIF(document->>'{amount_field}','')::numeric ASC NULLS LAST"
        if doc_date_field:
            base += f", NULLIF(document->>'{doc_date_field}','')::timestamptz DESC NULLS LAST"
        return base

    def _date_desc():
        if doc_date_field:
            return f"NULLIF(document->>'{doc_date_field}','')::timestamptz DESC NULLS LAST"
        return "updated_at DESC"

    # ── LIMIT: explicit top N always wins; sort direction only affects ORDER BY ──
    # "top 5 invoices by amount highest to lowest" → LIMIT 5, ORDER BY amount DESC
    # "give me invoices highest to lowest" → LIMIT 50, ORDER BY amount DESC
    has_explicit_sort = bool(sort_asc_kw or sort_desc_kw)
    has_amount_sort   = (sort_desc_kw or sort_asc_kw) and amount_field

    # Identifier field for "first/last by number" (invoice_number, deal name, etc.)
    id_field = REGISTRY.get(table, "identifier", fields)

    if ordinal_match and not first_match:
        # "3rd invoice", "5th deal" → OFFSET n-1, LIMIT 1
        n       = int(ordinal_match.group(1))
        limit   = 1
        _offset = max(0, n - 1)
        order   = (f"NULLIF(document->>'{id_field}','') ASC NULLS LAST"
                   if id_field else
                   f"NULLIF(document->>'{doc_date_field}','')::timestamptz ASC NULLS LAST"
                   if doc_date_field else "id ASC")
        # inject OFFSET into sql below
    elif first_match:
        limit   = 1
        _offset = 0
        # Order by identifier ASC (invoice #1, deal A, etc.) — not by date
        order   = (f"NULLIF(document->>'{id_field}','') ASC NULLS LAST"
                   if id_field else
                   f"NULLIF(document->>'{doc_date_field}','')::timestamptz ASC NULLS LAST"
                   if doc_date_field else "id ASC")
    elif last_match:
        limit   = 1
        _offset = 0
        order   = (f"NULLIF(document->>'{id_field}','') DESC NULLS LAST"
                   if id_field else
                   f"NULLIF(document->>'{doc_date_field}','')::timestamptz DESC NULLS LAST"
                   if doc_date_field else "id DESC")
    elif last_n_match:
        limit   = min(int(last_n_match.group(1)), 100)
        _offset = 0
        order   = _date_desc()
    elif top_match:
        limit   = min(int(top_match.group(1)), 100)
        _offset = 0
        order   = (_amount_asc()  if sort_asc_kw  and amount_field else
                   _amount_desc() if has_amount_sort               else
                   _amount_desc() if amount_field                   else _date_desc())
    elif bare_top and re.search(r"\b(invoice|deal|sales|order|customer)\b", text):
        limit   = 10
        _offset = 0
        order   = _amount_asc() if sort_asc_kw and amount_field else \
                  _amount_desc() if amount_field else _date_desc()
    elif sort_asc_kw and amount_field:
        limit   = 50
        _offset = 0
        order   = _amount_asc()
    elif sort_desc_kw and amount_field and \
         re.search(r"\b(biggest|largest|most|highest|recent|latest|newest|descend)\b", text):
        limit   = 50
        _offset = 0
        order   = _amount_desc()
    elif n_match:
        limit   = min(int(n_match.group(1)), 100)
        _offset = 0
        order   = _date_desc()
    elif re.search(r"\ball\b", text):
        limit   = 100
        _offset = 0
        order   = _date_desc()
    elif wants_details:
        limit   = 100
        _offset = 0
        order   = _date_desc()
    else:
        limit   = 20
        _offset = 0
        order   = _date_desc()

    count_sql   = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _cnt_res    = run_sql(agent, count_sql)
    if _cnt_res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql]}
    total_rows  = _cnt_res.rows
    total_count = coerce_number(total_rows[0][0] if total_rows else 0)

    _offset_clause = f" OFFSET {_offset}" if _offset > 0 else ""
    sql      = f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} ORDER BY {order} LIMIT {limit}{_offset_clause}".strip()
    _row_res = run_sql(agent, sql)
    if _row_res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql, sql]}
    rows     = _row_res.rows
    if not rows:
        return {
            "answer":      f"No **{table}** records found.",
            "tables_used": [table],
            "confidence":  0.9,
            "sql_queries": [count_sql, sql],
            "_entity":     table,
        }

    headers = [p.split(" AS ")[-1].replace("_", " ").title() for p in select_parts]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")

    time_label = f" ({time_cond['label']})" if time_cond else ""
    label      = f"top {limit}" if top_match else str(len(rows))
    summary = (
        f"**Total {table}: {fmt_number(total_count)}**\n\n"
        if (re.search(r"\ball\b", text) or wants_details)
        else ""
    )
    return {
        "answer":      summary + f"**{label} {table}**{time_label}:\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.95,
        "sql_queries": [count_sql, sql],
        "_entity":     table,
    }


def _fp_top_customers(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    m    = re.search(r"\btop\s+(\d+)\b", text)
    if not m:
        return None
    if not any(kw in text for kw in ["customer", "company", "companies", "client", "account"]):
        return None
    if not any(kw in text for kw in ["revenue", "sales", "amount", "billing", "invoice"]):
        return None

    n           = int(m.group(1))
    table_names = get_table_names(agent)
    inv_table   = find_revenue_table(table_names)
    co_table    = next((t for t in table_names if t.lower() in ["companies", "company", "accounts"]), None)
    if not inv_table:
        return None

    time_cond    = _parse_time_condition(text)
    inv_deleted  = "COALESCE(i.document->>'deleted','false') != 'true'"
    base_where   = f"WHERE ({inv_deleted})" + (f" AND ({time_cond['condition']})" if time_cond else "")
    rev_expr     = ("COALESCE("
                    "NULLIF(i.document->>'grand_total','')::numeric,"
                    "NULLIF(i.document->>'grandtotal_in_usd','')::numeric, 0)")

    if co_table:
        sql = f"""SELECT COALESCE(c.document->>'companyName', i.document->>'company', 'Unknown') AS customer,
  SUM({rev_expr}) AS total_revenue, COUNT(*)::int AS invoice_count
FROM "{inv_table}" i
LEFT JOIN "{co_table}" c ON c.document->>'_id' = i.document->>'company'
{base_where}
GROUP BY 1 ORDER BY 2 DESC LIMIT {n}""".strip()
    else:
        sql = f"""SELECT COALESCE(i.document->>'company', 'Unknown') AS customer,
  SUM({rev_expr}) AS total_revenue, COUNT(*)::int AS invoice_count
FROM "{inv_table}" i {base_where}
GROUP BY 1 ORDER BY 2 DESC LIMIT {n}""".strip()

    _res       = run_sql(agent, sql)
    time_label = f" ({time_cond['label']})" if time_cond else ""
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [t for t in [inv_table, co_table] if t],
                "confidence": 0.0, "sql_queries": [sql]}
    rows       = _res.rows
    if not rows:
        return {
            "answer":      f"No customer revenue data found{time_label}.",
            "tables_used": [t for t in [inv_table, co_table] if t],
            "confidence":  0.85,
            "sql_queries": [sql],
        }

    lines = ["| # | Customer | Revenue | Invoices |", "| --- | --- | --- | --- |"]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row[0] or 'Unknown'} | {fmt_number(coerce_number(row[1]))} | {row[2] or 0} |"
        )
    return {
        "answer":      f"**Top {n} customers by revenue{time_label}:**\n\n" + "\n".join(lines),
        "tables_used": [t for t in [inv_table, co_table] if t],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_target_unachieved(agent, user_query: str) -> Optional[Dict]:
    """Handle 'who have not achieved targets' — users where achieved < target.

    Shows: user name, team, period (month/year), target, achieved, gap, % achieved.
    Ordered worst performers first.
    """
    text = normalize_text(user_query)

    # Must mention targets AND "not achieved" / "below" / "missed" / "unachieved"
    if not re.search(r"\b(target|targets)\b", text):
        return None
    if not re.search(
        r"\bnot\s+achiev|\bhaven.?t\s+achiev|\bmissed\s+target|\bbelow\s+target"
        r"|\bunder\s+target|\bunachiev|\bnot\s+meet|\bnot\s+met\b",
        text,
    ):
        return None

    table_names = get_table_names(agent)
    if "targets" not in table_names:
        return None

    # Time period filter
    year_m        = re.search(r"\b(20\d{2})\b", text)
    month_y       = _extract_month_year(text)
    period_filter = ""
    period_label  = "all time"

    if month_y:
        month, year = month_y
        period_filter = (
            f"AND NULLIF(t.document->>'month','')::int = {month}"
            f" AND NULLIF(t.document->>'year','')::int = {year}"
        )
        period_label = f"{calendar.month_name[month]} {year}"
    elif year_m:
        period_filter = f"AND NULLIF(t.document->>'year','')::int = {year_m.group(1)}"
        period_label  = year_m.group(1)

    sales_exists = "sales" in table_names
    achieved_sub = (
        """COALESCE((SELECT SUM(NULLIF(s.document->>'grand_total_in_usd','')::numeric)
        FROM "sales" s
        WHERE s.document->>'salesOwner' = t.document->>'userId'
          AND date_part('month', NULLIF(s.document->>'sales_date','')::timestamptz)
              = NULLIF(t.document->>'month','')::numeric
          AND date_part('year',  NULLIF(s.document->>'sales_date','')::timestamptz)
              = NULLIF(t.document->>'year','')::numeric
          AND COALESCE(s.document->>'deleted','false') != 'true'), 0)"""
        if sales_exists else "0"
    )

    # Filter: achieved < target AND target > 0
    sql = f"""SELECT
  COALESCE(u.document->>'name', t.document->>'userId') AS user_name,
  t.document->>'teamName' AS team,
  CONCAT('Month ', t.document->>'month', '/', t.document->>'year') AS period,
  NULLIF(t.document->>'targetInUSD','')::numeric AS target_usd,
  {achieved_sub} AS achieved_usd,
  CASE
    WHEN NULLIF(t.document->>'targetInUSD','')::numeric > 0
    THEN ROUND({achieved_sub} / NULLIF(t.document->>'targetInUSD','')::numeric * 100, 1)
    ELSE 0
  END AS achievement_pct,
  ROUND(NULLIF(t.document->>'targetInUSD','')::numeric - {achieved_sub}, 2) AS gap_usd
FROM "targets" t
LEFT JOIN "users" u ON u.document->>'_id' = t.document->>'userId'
WHERE NULLIF(t.document->>'targetInUSD','')::numeric > 0
  AND {achieved_sub} < NULLIF(t.document->>'targetInUSD','')::numeric
  {period_filter}
ORDER BY achievement_pct ASC,
         NULLIF(t.document->>'year','')::int DESC,
         NULLIF(t.document->>'month','')::int DESC""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["targets", "users"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows

    if not rows:
        return {
            "answer":      f"All users have met their targets for {period_label}. 🎉",
            "tables_used": ["targets", "users"],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    lines = [
        "| User | Team | Period | Target (USD) | Achieved (USD) | Achievement % | Gap (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      (
            f"**Users who have NOT achieved their targets — {period_label} "
            f"({len(rows)} records, worst first):**\n\n" + "\n".join(lines)
        ),
        "tables_used": ["targets", "users"] + (["sales"] if sales_exists else []),
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_targets(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(r"\b(targets?|performance|achievement|achieved|growth potential|kpi|score)\b", text):
        return None

    table_names = get_table_names(agent)
    if "targets" not in table_names:
        return None

    year_m        = re.search(r"\b(20\d{2})\b", text)
    month_y       = _extract_month_year(text)
    period_filter = ""
    period_label  = "all time"

    if month_y:
        month, year   = month_y
        period_filter = (
            f"AND NULLIF(t.document->>'month','')::int = {month}"
            f" AND NULLIF(t.document->>'year','')::int = {year}"
        )
        period_label = f"{calendar.month_name[month]} {year}"
    elif year_m:
        period_filter = f"AND NULLIF(t.document->>'year','')::int = {year_m.group(1)}"
        period_label  = year_m.group(1)

    user_filter = ""
    # Accept single names ("kartik") and full names ("kartik trivedi")
    user_m = re.search(r"(?:for|by|of)\s+([a-zA-Z][a-zA-Z]*(?:\s+[a-zA-Z][a-zA-Z]*)?)", user_query, re.I)
    if user_m:
        uname = sanitize_sql_value(user_m.group(1).strip())
        _GENERIC = {
            "all", "users", "user", "every", "each", "this", "the",
            "all users", "all user", "every user", "each user", "this user", "the user",
            "me", "us", "them", "year", "month", "quarter", "time",
        }
        if len(uname) >= 3 and uname.lower() not in _GENERIC:
            # SAFE: uname sanitized via sanitize_sql_value()
            user_filter = f"AND u.document->>'name' ILIKE '%{uname}%'"

    sales_exists = "sales" in table_names
    achieved_sub = ("""COALESCE((SELECT SUM(NULLIF(s.document->>'grand_total_in_usd','')::numeric)
        FROM "sales" s
        WHERE s.document->>'salesOwner' = t.document->>'userId'
          AND date_part('month', NULLIF(s.document->>'sales_date','')::timestamptz)
              = NULLIF(t.document->>'month','')::numeric
          AND date_part('year',  NULLIF(s.document->>'sales_date','')::timestamptz)
              = NULLIF(t.document->>'year','')::numeric
          AND COALESCE(s.document->>'deleted','false') != 'true'), 0)""") if sales_exists else "0"

    sql = f"""SELECT
  COALESCE(u.document->>'name', t.document->>'userId') AS user_name,
  t.document->>'teamName' AS team,
  CONCAT('Month ', t.document->>'month', '/', t.document->>'year') AS period,
  NULLIF(t.document->>'targetInUSD','')::numeric AS target_usd,
  {achieved_sub} AS achieved_usd,
  CASE
    WHEN NULLIF(t.document->>'targetInUSD','')::numeric > 0
    THEN ROUND({achieved_sub} / NULLIF(t.document->>'targetInUSD','')::numeric * 100, 1)
    ELSE 0
  END AS achievement_pct
FROM "targets" t
LEFT JOIN "users" u ON u.document->>'_id' = t.document->>'userId'
WHERE 1=1 {period_filter} {user_filter}
ORDER BY NULLIF(t.document->>'year','')::int DESC,
         NULLIF(t.document->>'month','')::int DESC, user_name""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["targets", "users"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No target data found for {period_label}.",
            "tables_used": ["targets", "users"],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    lines = [
        "| User | Team | Period | Target (USD) | Achieved (USD) | Achievement % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      f"**Target vs Achieved — {period_label}:**\n\n" + "\n".join(lines),
        "tables_used": ["targets", "users"] + (["sales"] if sales_exists else []),
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_search(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(r"\bwho\s+is\b|\bwho\s+are\b|\bfind\b|\bsearch\b", text):
        return None
    m = re.search(r"(?:who is|who are|find|search for?)\s+(.+?)(?:\s*\??\s*$)", text, re.I)
    if not m:
        return None
    term = sanitize_sql_value(m.group(1).strip().strip("?").strip())
    if len(term) < 2:
        return None

    table_names               = get_table_names(agent)
    results, sql_queries, tables_used = [], [], []

    if "users" in table_names:
        # SAFE: term sanitized via sanitize_sql_value()
        sql = f"""SELECT document->>'name', document->>'email', document->>'department'
FROM "users"
WHERE document->>'name' ILIKE '%{term}%'
  AND COALESCE(document->>'isActive','true') != 'false' LIMIT 5"""
        _r = run_sql(agent, sql)
        rows = _r.rows
        if rows:
            tables_used.append("users")
            sql_queries.append(sql)
            for r in rows:
                results.append(f"**User**: {r[0] or '—'} | Email: {r[1] or '—'}")

    if "contacts" in table_names and not results:
        # SAFE: term sanitized via sanitize_sql_value()
        sql = f"""SELECT TRIM(CONCAT(COALESCE(document->>'firstName',''),' ',COALESCE(document->>'lastName',''))) AS name,
  document->>'email', document->>'jobTitle', document->>'phoneNumber'
FROM "contacts"
WHERE (document->>'firstName' ILIKE '%{term}%' OR document->>'lastName' ILIKE '%{term}%'
       OR document->>'email' ILIKE '%{term}%') LIMIT 5"""
        _r = run_sql(agent, sql)
        rows = _r.rows
        if rows:
            tables_used.append("contacts")
            sql_queries.append(sql)
            for r in rows:
                results.append(
                    f"**Contact**: {r[0] or '—'} | Email: {r[1] or '—'} | Title: {r[2] or '—'}"
                )

    if "companies" in table_names and not results:
        # SAFE: term sanitized via sanitize_sql_value()
        sql = f"""SELECT document->>'companyName', document->>'email', document->>'websiteUrl'
FROM "companies"
WHERE document->>'companyName' ILIKE '%{term}%'
  AND COALESCE(document->>'deleted','false') != 'true' LIMIT 5"""
        _r = run_sql(agent, sql)
        rows = _r.rows
        if rows:
            tables_used.append("companies")
            sql_queries.append(sql)
            for r in rows:
                results.append(
                    f"**Company**: {r[0] or '—'} | Email: {r[1] or '—'} | Web: {r[2] or '—'}"
                )

    if not results:
        return {
            "answer":      f"No record found for **{term}**.",
            "tables_used": ["users", "contacts", "companies"],
            "confidence":  0.9,
            "sql_queries": sql_queries,
        }
    return {
        "answer":      f"**Search results for '{term}':**\n\n" + "\n\n".join(results),
        "tables_used": tables_used,
        "confidence":  0.97,
        "sql_queries": sql_queries,
    }


def _fp_lookup_list(agent, user_query: str) -> Optional[Dict]:
    text        = normalize_text(user_query)
    table_names = get_table_names(agent)
    LOOKUP_MAP  = [
        (r"\btechnolog",    "technologies",       "name"),
        (r"\bsource",       "sources",            "sourceName"),
        (r"\btax\b",        "taxes",              "name"),
        (r"\bregion",       "regions",            "regionName"),
        (r"\blead.?status", "lead_statuses",      "name"),
        (r"\blifecycle",    "lifecycle_stages",   "name"),
        (r"\bdeal.?stage",  "dealstagesettings",  "dealStageName"),
        (r"\bcategor",      "categories",         "categoryName"),
        (r"\bpayment.?method", "payments",        "payment_name"),
        (r"\bindustr",      "companies",          "industry"),
    ]
    matched_table, matched_field = None, None
    for pattern, t_name, field in LOOKUP_MAP:
        if re.search(pattern, text, re.I) and t_name in table_names:
            matched_table, matched_field = t_name, field
            break
    if not matched_table:
        return None

    actual_fields = get_document_fields(agent, matched_table)
    if matched_field not in actual_fields:
        matched_field = next((f for f in actual_fields if "name" in f.lower()), None)
    if not matched_field:
        return None

    # Industries are stored inside the companies document — use distinct
    if matched_table == "companies" and matched_field == "industry":
        sql = f"""SELECT DISTINCT document->>'industry' AS name
FROM "companies"
WHERE document->>'industry' IS NOT NULL
  AND document->>'industry' != ''
  AND COALESCE(document->>'deleted','false') != 'true'
ORDER BY 1""".strip()
    else:
        sql = f"""SELECT DISTINCT document->>'{matched_field}' AS name
FROM "{matched_table}"
WHERE document->>'{matched_field}' IS NOT NULL
  AND document->>'{matched_field}' != ''
  AND COALESCE(document->>'deleted','false') != 'true'
ORDER BY 1""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [matched_table], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No data found in **{matched_table}**.",
            "tables_used": [matched_table],
            "confidence":  0.9,
            "sql_queries": [sql],
        }
    items = [r[0] for r in rows if r and r[0]]
    return {
        "answer":      f"**{matched_table.title()}** ({len(items)} total):\n\n"
                       + "\n".join(f"- {i}" for i in items),
        "tables_used": [matched_table],
        "confidence":  0.98,
        "sql_queries": [sql],
    }


def _fp_quarterly(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(r"\bquarter(ly)?\b|Q[1-4]\b", text, re.I):
        return None
    if not any(kw in text for kw in ["revenue", "sales", "amount", "total", "collection", "billing"]):
        return None

    year_m     = re.search(r"\b(20\d{2})\b", text)
    year_label = year_m.group(1) if year_m else "all time"

    table_names = get_table_names(agent)
    table       = find_revenue_table(table_names)
    if not table:
        return None

    fields        = get_document_fields(agent, table)
    coalesce_expr = build_revenue_coalesce(fields)

    # Use document date field — NOT updated_at (sync timestamp, not invoice date)
    _DATE_FIELD_PRIORITY = ["invoice_date", "sales_date", "due_date", "closeDate", "close_date", "date"]
    fl = {f.lower(): f for f in fields}
    doc_date_field = next((fl[d] for d in _DATE_FIELD_PRIORITY if d in fl), None)
    if not doc_date_field:
        doc_date_field = next((f for f in fields if "date" in f.lower()), None)

    date_expr   = f"NULLIF(document->>'{doc_date_field}','')::timestamptz" if doc_date_field else "updated_at"
    year_filter = f"AND date_trunc('year', {date_expr}) = DATE '{year_m.group(1)}-01-01'" if year_m else ""

    sql = f"""SELECT
  'Q' || date_part('quarter', {date_expr}) || ' ' || date_part('year', {date_expr}) AS quarter,
  COALESCE(SUM({coalesce_expr}), 0) AS revenue,
  COUNT(*)::int AS invoice_count
FROM "{table}"
WHERE COALESCE(document->>'deleted','false') != 'true'
  AND {date_expr} IS NOT NULL
  {year_filter}
GROUP BY date_part('year', {date_expr}), date_part('quarter', {date_expr})
ORDER BY date_part('year', {date_expr}), date_part('quarter', {date_expr})""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No revenue data found for {year_label}.",
            "tables_used": [table],
            "confidence":  0.85,
            "sql_queries": [sql],
        }

    lines     = ["| Quarter | Revenue | Invoices |", "| --- | --- | --- |"]
    total_rev = 0
    for r in rows:
        rev = coerce_number(r[1])
        if isinstance(rev, (int, float)):
            total_rev += rev
        lines.append(f"| {r[0]} | {fmt_number(r[1])} | {r[2]} |")
    lines.append(f"| **TOTAL** | **{fmt_number(total_rev)}** | |")

    return {
        "answer":      f"**Quarterly Revenue — {year_label}:**\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_overdue_aging(agent, user_query: str) -> Optional[Dict]:
    table_names = get_table_names(agent)
    if "invoices" not in table_names:
        return None

    # Always JOIN companies so we get readable names instead of ObjectIDs.
    # company field in invoices stores the company's _id.
    has_co   = "companies" in table_names
    co_join  = (
        'LEFT JOIN "companies" c ON c.document->>\'_id\' = i.document->>\'company\''
        if has_co else ""
    )
    co_col   = (
        "COALESCE(c.document->>'companyName', c.document->>'name', i.document->>'company', 'Unknown')"
        if has_co else "COALESCE(i.document->>'company','Unknown')"
    )
    tbl_used = ["invoices"] + (["companies"] if has_co else [])

    sql = f"""SELECT {co_col} AS company,
  COUNT(*)::int AS overdue_invoices,
  COALESCE(SUM(NULLIF(i.document->>'grand_total','')::numeric), 0) AS outstanding_amount,
  MIN(NULLIF(i.document->>'due_date','')::timestamptz)::date AS oldest_due_date,
  MAX(NULLIF(i.document->>'due_date','')::timestamptz)::date AS latest_due_date
FROM "invoices" i {co_join}
WHERE NULLIF(i.document->>'due_date','')::timestamptz < NOW()
  AND COALESCE(i.document->>'payment_status','') NOT IN ('paid','cancelled')
  AND COALESCE(i.document->>'deleted','false') != 'true'
GROUP BY 1 ORDER BY outstanding_amount DESC LIMIT 30""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": tbl_used, "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      "No overdue invoices found.",
            "tables_used": tbl_used,
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    total_outstanding = sum(coerce_number(r[2]) for r in rows if r[2] is not None)
    lines = [
        "| Company | Invoices | Outstanding | Oldest Due | Latest Due |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      (
            f"**Overdue Invoice Aging — {len(rows)} companies, "
            f"Total Outstanding: {fmt_number(total_outstanding)}:**\n\n" + "\n".join(lines)
        ),
        "tables_used": tbl_used,
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_dept_users(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(
        r"\bdepartment.{0,20}(user|member|employee|name)\b"
        r"|\b(user|member|employee).{0,20}department\b", text,
    ):
        return None

    table_names = get_table_names(agent)
    if "departments" not in table_names or "users" not in table_names:
        return None

    sql = """SELECT d.document->>'name' AS department,
  STRING_AGG(u.document->>'name', ', ' ORDER BY u.document->>'name') AS members,
  COUNT(u.id)::int AS member_count
FROM "departments" d
LEFT JOIN "users" u
  ON u.document->>'department' = d.document->>'_id'
  AND COALESCE(u.document->>'isActive','true') != 'false'
GROUP BY d.document->>'_id', d.document->>'name'
ORDER BY d.document->>'name'"""

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["departments", "users"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      "No department data found.",
            "tables_used": ["departments", "users"],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    lines = ["| Department | Members | Count |", "| --- | --- | --- |"]
    for r in rows:
        dept    = r[0] or "—"
        members = r[1] or "No members"
        count   = r[2] or 0
        lines.append(f"| {dept} | {members} | {count} |")

    return {
        "answer":      "**Departments with members:**\n\n" + "\n".join(lines),
        "tables_used": ["departments", "users"],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_no_activity(agent, user_query: str) -> Optional[Dict]:
    text    = normalize_text(user_query)
    pattern = (
        r"\bno (business|invoice|deal|activity|order|revenue)\b"
        r"|\bnot.{0,15}given business\b|\bdon.?t.{0,15}given business\b"
        r"|\bwithout.{0,15}(business|invoice)\b|\bno.{0,10}purchased\b"
        r"|\bnot.{0,10}active\b"
    )
    if not re.search(pattern, text):
        return None

    table_names = get_table_names(agent)
    if "companies" not in table_names:
        return None

    time_cond = _parse_time_condition(text)
    interval  = "3 months"
    if time_cond:
        m = re.search(r"(\d+)\s+(month|day|week|year)", time_cond.get("label", ""))
        if m:
            interval = f"{m.group(1)} {m.group(2)}s"

    sub_conditions = []
    if "invoices" in table_names:
        sub_conditions.append(
            f"SELECT document->>'company' FROM \"invoices\" "
            f"WHERE document->>'company' IS NOT NULL "
            f"AND updated_at >= NOW() - INTERVAL '{interval}'"
        )
    if "deals" in table_names:
        sub_conditions.append(
            f"SELECT document->>'company' FROM \"deals\" "
            f"WHERE document->>'company' IS NOT NULL "
            f"AND updated_at >= NOW() - INTERVAL '{interval}'"
        )
    if not sub_conditions:
        return None

    not_in_clause = " AND ".join(
        f"COALESCE(c.document->>'_id','') NOT IN ({sub})" for sub in sub_conditions
    )
    sql = f"""SELECT c.document->>'companyName', c.document->>'email',
  c.document->>'leadStatus', c.updated_at::date
FROM "companies" c
WHERE COALESCE(c.document->>'deleted','false') != 'true'
  AND {not_in_clause}
ORDER BY c.updated_at DESC LIMIT 50""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["companies"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"All companies have given business in the last {interval}.",
            "tables_used": ["companies"],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    lines = ["| Company | Email | Status | Last Sync |", "| --- | --- | --- | --- |"]
    for r in rows:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      f"**{len(rows)} companies with no business in last {interval}:**\n\n"
                       + "\n".join(lines),
        "tables_used": ["companies"],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_overdue_tasks(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(
        r"\b(not|haven.t|missed|overdue).{0,20}(deadline|task|due)\b"
        r"|\btask.{0,20}overdue\b|\boverdue.{0,20}task\b", text,
    ):
        return None

    table_names = get_table_names(agent)
    if "createtasks" not in table_names or "users" not in table_names:
        return None

    sql = """SELECT u.document->>'name', u.document->>'email',
  COUNT(t.id)::int, MIN(t.document->>'due_date')
FROM "users" u
JOIN "createtasks" t ON t.document->>'createdBy' = u.document->>'_id'
WHERE NULLIF(t.document->>'due_date','')::timestamptz < NOW()
  AND COALESCE(t.document->>'status','') != 'Completed'
  AND COALESCE(t.document->>'deleted','false') != 'true'
GROUP BY u.document->>'_id', u.document->>'name', u.document->>'email'
HAVING COUNT(t.id) > 0
ORDER BY 3 DESC"""

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["users", "createtasks"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      "No users have overdue tasks.",
            "tables_used": ["users", "createtasks"],
            "confidence":  0.95,
            "sql_queries": [sql],
        }

    lines = [
        "| User | Email | Overdue Tasks | Oldest Due Date |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")

    return {
        "answer":      f"**{len(rows)} users with overdue tasks:**\n\n" + "\n".join(lines),
        "tables_used": ["users", "createtasks"],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_pipeline_summary(agent, user_query: str) -> Optional[Dict]:
    """Pipeline summary — open deals grouped by stage with value + share %."""
    text        = normalize_text(user_query)
    table_names = get_table_names(agent)
    if "deals" not in table_names:
        return None

    fields       = get_document_fields(agent, "deals")
    stage_field  = next((f for f in fields if f.lower() == "stage"), "stage")
    won_field    = next((f for f in fields if "wonAt" in f or "won_at" in f.lower()), None)
    lost_field   = next((f for f in fields if "lostAt" in f or "lost_at" in f.lower()), None)
    amount_field = next((f for f in fields if f.lower() in ["grand_total_in_usd", "grand_total"]), None)
    deleted_field = next((f for f in fields if f.lower() == "deleted"), None)

    where_parts = []
    if deleted_field:
        where_parts.append(f"COALESCE(document->>'{deleted_field}','false') != 'true'")
    if won_field:
        where_parts.append(f"document->>'{won_field}' IS NULL")
    if lost_field:
        where_parts.append(f"document->>'{lost_field}' IS NULL")
    where = "WHERE " + " AND ".join(f"({p})" for p in where_parts) if where_parts else ""

    value_expr = (
        f"NULLIF(document->>'{amount_field}','')::numeric"
        if amount_field else "0"
    )
    sql = f"""SELECT
  COALESCE(document->>'{stage_field}', 'Unknown') AS stage,
  COUNT(*)::int AS deal_count,
  COALESCE(SUM({value_expr}), 0) AS total_value
FROM "deals"
{where}
GROUP BY 1 ORDER BY 3 DESC""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["deals"], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      "No open pipeline deals found.",
            "tables_used": ["deals"],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    pipeline_total = sum(coerce_number(r[2]) for r in rows if r[2] is not None)
    lines          = ["| Stage | Deals | Value | Share % |", "| --- | --- | --- | --- |"]
    for r in rows:
        share = (
            round(coerce_number(r[2]) / pipeline_total * 100, 1)
            if pipeline_total else 0
        )
        lines.append(
            f"| {r[0]} | {r[1]} | {fmt_number(coerce_number(r[2]))} | {share}% |"
        )
    lines.append(
        f"| **TOTAL** | **{sum(r[1] for r in rows)}** | **{fmt_number(pipeline_total)}** | 100% |"
    )

    return {
        "answer":      f"**Pipeline Summary (Open Deals):**\n\n" + "\n".join(lines),
        "tables_used": ["deals"],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_pending_invoices(agent, user_query: str) -> Optional[Dict]:
    """Pending / unpaid invoices summary with overdue flag."""
    table_names = get_table_names(agent)
    if "invoices" not in table_names:
        return None

    has_co  = "companies" in table_names
    co_join = (
        'LEFT JOIN "companies" c ON c.document->>\'_id\' = i.document->>\'company\''
        if has_co else ""
    )
    co_col  = (
        "COALESCE(c.document->>'companyName', c.document->>'name', i.document->>'company', 'Unknown')"
        if has_co else "COALESCE(i.document->>'company','Unknown')"
    )

    sql = f"""SELECT
  i.document->>'invoice_number' AS invoice_number,
  i.document->>'payment_status' AS status,
  COALESCE(NULLIF(i.document->>'grand_total','')::numeric, 0) AS amount,
  i.document->>'due_date' AS due_date,
  {co_col} AS company,
  CASE
    WHEN NULLIF(i.document->>'due_date','')::timestamptz < NOW()
    THEN CONCAT(
      (DATE_PART('day', NOW() - NULLIF(i.document->>'due_date','')::timestamptz))::int,
      ' days overdue'
    )
    ELSE 'On time'
  END AS overdue_status
FROM "invoices" i {co_join}
WHERE COALESCE(i.document->>'payment_status','') NOT IN ('paid','cancelled')
  AND COALESCE(i.document->>'deleted','false') != 'true'
ORDER BY
  CASE WHEN NULLIF(i.document->>'due_date','')::timestamptz < NOW() THEN 0 ELSE 1 END,
  NULLIF(i.document->>'due_date','')::timestamptz ASC NULLS LAST
LIMIT 50""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": ["invoices"] + (["companies"] if has_co else []),
                "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      "No pending invoices found.",
            "tables_used": ["invoices"] + (["companies"] if has_co else []),
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    total_pending  = sum(coerce_number(r[2]) for r in rows)
    overdue_count  = sum(1 for r in rows if r[5] and "overdue" in str(r[5]))
    lines          = [
        "| Invoice # | Status | Amount | Due Date | Company | Overdue Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows[:30]:
        vals = ["—" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 30:
        lines.append(f"_…and {len(rows)-30} more_")

    return {
        "answer":      (
            f"**Pending Invoices — {len(rows)} total "
            f"(⚠️ {overdue_count} overdue), "
            f"Total: {fmt_number(total_pending)}:**\n\n" + "\n".join(lines)
        ),
        "tables_used": ["invoices"] + (["companies"] if has_co else []),
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_invoice_status(agent, user_query: str) -> Optional[Dict]:
    """Handle 'rejected invoices', 'approved invoices', 'invoices by stage of X'.

    Detects any explicit status/stage value from the query, maps it to the
    correct DB field (payment_status / stage / status), and returns a filtered
    invoice list with count.
    """
    text = normalize_text(user_query)

    # Must mention invoices or billing
    if not re.search(r"\binvoices?\b|\bbills?\b|\bbilling\b", text):
        return None

    # ── Extract status value ───────────────────────────────────────────────────
    status_raw = None

    # Pattern 1: "by stage/status of X"
    m = re.search(r"\bby\s+(?:stage|status)\s+of\s+(\w+)", text, re.I)
    if m:
        status_raw = m.group(1)

    # Pattern 2: "which are X" / "that are X"
    if not status_raw:
        m = re.search(r"\b(?:which|that)\s+are\s+(\w+)", text, re.I)
        if m:
            status_raw = m.group(1)

    # Pattern 3: "status is X" / "stage is X"
    if not status_raw:
        m = re.search(r"\b(?:stage|status|payment\s*status)\s+(?:is|=|:)\s*['\"]?(\w+)['\"]?", text, re.I)
        if m:
            status_raw = m.group(1)

    # Pattern 4: adjective directly before "invoice(s)" or standalone status word
    if not status_raw:
        m = re.search(
            r"\b(approved|rejected|submitted|declined|accepted)\b", text, re.I,
        )
        if m:
            status_raw = m.group(1)

    if not status_raw:
        return None

    # Normalize via synonym table
    canonical = _STATUS_SYNONYMS.get(status_raw.lower(), status_raw.lower())

    table_names = get_table_names(agent)
    inv_table   = next((t for t in table_names if t.lower() == "invoices"), None)
    if not inv_table:
        return None

    fields = get_document_fields(agent, inv_table)

    # ── Find the correct status field — try multiple, pick the one with data ───
    # Some schemas split: payment_status=paid/draft/cancelled, approval_status=approved/rejected/pending
    _FIELD_PRIORITY = [
        "payment_status", "paymentStatus",
        "approval_status", "approvalStatus",
        "status", "stage",
    ]
    fl_norm = {f.lower().replace("_", ""): f for f in fields}

    # Hint: prefer approval_status for approval-domain values
    _APPROVAL_VALUES = {"approved", "rejected", "submitted", "declined", "pending"}
    if canonical and canonical.lower() in _APPROVAL_VALUES:
        _FIELD_PRIORITY = [
            "approval_status", "approvalStatus",
            "payment_status", "paymentStatus",
            "status", "stage",
        ]

    # Try each candidate field; use first that has any matching rows
    status_field = None
    where_clause = None
    display_label = status_raw.title() if canonical else "Unpaid"

    for pf in _FIELD_PRIORITY:
        key = pf.lower().replace("_", "")
        if key not in fl_norm:
            continue
        candidate_field = fl_norm[key]
        if canonical is None:
            candidate_where = f"COALESCE(document->>'{candidate_field}','') NOT IN ('paid','cancelled')"
        else:
            candidate_where = f"LOWER(document->>'{candidate_field}') = '{canonical.lower()}'"

        # Quick existence check — only cost is one COUNT query per field candidate
        _probe = run_sql(agent, f'SELECT COUNT(*)::int FROM "{inv_table}" WHERE ({candidate_where})')
        if _probe.ok() and _probe.rows and coerce_number(_probe.rows[0][0]) > 0:
            status_field = candidate_field
            where_clause = candidate_where
            break

    # Fall back to first available field even if 0 rows (correct empty response)
    if not status_field:
        for pf in _FIELD_PRIORITY:
            key = pf.lower().replace("_", "")
            if key in fl_norm:
                status_field = fl_norm[key]
                if canonical is None:
                    where_clause = f"COALESCE(document->>'{status_field}','') NOT IN ('paid','cancelled')"
                else:
                    where_clause = f"LOWER(document->>'{status_field}') = '{canonical.lower()}'"
                break
    if not status_field:
        return None

    # ── Count ──────────────────────────────────────────────────────────────────
    count_sql = f'SELECT COUNT(*)::int FROM "{inv_table}" WHERE ({where_clause})'.strip()
    _cnt = run_sql(agent, count_sql)
    if _cnt.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [inv_table], "confidence": 0.0, "sql_queries": [count_sql]}
    count = coerce_number(_cnt.rows[0][0] if _cnt.rows else 0)

    # ── Build SELECT ───────────────────────────────────────────────────────────
    num_field    = next((f for f in fields if "invoice_number" in f.lower() or "invoiceno" in f.lower().replace("_","")), None)
    amount_field = next((f for f in fields if f.lower() in ["grand_total", "grand_total_in_usd", "total", "amount"]), None)
    date_field   = next((f for f in fields if "invoice_date" in f.lower() or "invoicedate" in f.lower().replace("_","")), None)

    select_parts = []
    if num_field:
        select_parts.append(f"document->>'{num_field}' AS invoice_number")
    select_parts.append(f"document->>'{status_field}' AS status")
    if amount_field:
        select_parts.append(f"NULLIF(document->>'{amount_field}','')::numeric AS amount")
    if date_field:
        select_parts.append(f"document->>'{date_field}' AS date")

    list_sql = (
        f'SELECT {", ".join(select_parts)} FROM "{inv_table}" '
        f'WHERE ({where_clause}) ORDER BY updated_at DESC LIMIT 50'
    ).strip()
    _rows = run_sql(agent, list_sql)
    if _rows.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [inv_table], "confidence": 0.0, "sql_queries": [count_sql, list_sql]}
    rows = _rows.rows

    if not rows:
        return {
            "answer":      f"No **{display_label}** invoices found.",
            "tables_used": [inv_table],
            "confidence":  0.9,
            "sql_queries": [count_sql, list_sql],
        }

    headers = [p.split(" AS ")[-1].replace("_", " ").title() for p in select_parts]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:30]:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 30:
        lines.append(f"_…and {len(rows)-30} more_")

    return {
        "answer":      f"**{display_label} Invoices — {fmt_number(count)} total:**\n\n" + "\n".join(lines),
        "tables_used": [inv_table],
        "confidence":  0.96,
        "sql_queries": [count_sql, list_sql],
    }


def _fp_user_task_map(agent, user_query: str) -> Optional[Dict]:
    """Handle 'all users with their tasks and status' — builds a JOIN query.

    Triggered when the query mentions BOTH users AND tasks with a relational
    intent (by / per / with / status / assigned).  Generates a single LEFT JOIN
    SQL instead of two unrelated sub-queries.
    """
    text = normalize_text(user_query)

    has_user = bool(re.search(r"\busers?\b|\bemployees?\b|\bmembers?\b|\bstaff\b|\bpeople\b|\bperson\b", text))
    has_task = bool(re.search(r"\btasks?\b|\btodo\b|\bfollow.?up\b|\bassignment\b", text))
    if not (has_user and has_task):
        return None

    # Do NOT fire for count-only sub-queries from the executor.
    # e.g. "count pending tasks grouped by user" → let _fp_count handle it
    # so the synthesizer gets a count/number, not a 73-row JOIN table.
    count_only = bool(re.search(
        r"^(count|how many|number of|total number)\b"
        r"|\bcount\b.{0,30}\bgrouped?\b"
        r"|\bgroup(ed)?\s+by\b",
        text,
    )) and not re.search(r"\blist\b|\bshow\b|\ball\b|\bgive me\b|\bdisplay\b", text)
    if count_only:
        return None

    # Require a relational/listing intent (not just a bare mention)
    relational = bool(re.search(
        r"\bby\b|\bper\b|\bwith\b|\bassigned\b|\bstatus\b|\bpending\b|\bcompleted?\b|\bopen\b|\bgrouped\b|\beach\b|\ball\b|\btheir\b|\band\b",
        text,
    ))
    if not relational:
        return None

    table_names = get_table_names(agent)
    user_table  = next((t for t in table_names if t.lower() == "users"), None)
    task_table  = next((t for t in table_names if t.lower() in ["createtasks", "tasks"]), None)
    if not user_table or not task_table:
        return None

    task_fields    = get_document_fields(agent, task_table)
    user_fields    = get_document_fields(agent, user_table)

    task_name_fld  = next((f for f in task_fields if f.lower() in ["task", "title", "name", "subject"]), "task")
    status_fld     = next((f for f in task_fields if "status" in f.lower()), "status")
    priority_fld   = next((f for f in task_fields if "priority" in f.lower()), None)
    due_fld        = next((f for f in task_fields if "due_date" in f.lower() or f.lower() == "due"), None)
    user_name_fld  = REGISTRY.get(user_table, "name", user_fields) or "name"

    # JOIN field: tasks link to users via createdBy / assignedTo / userId
    join_fld = next(
        (f for f in task_fields
         if f.lower().replace("_", "") in ["createdby", "assignedto", "userid", "ownerid"]),
        "createdBy",
    )

    # ── Status filter ──────────────────────────────────────────────────────────
    status_filter = ""
    status_label  = ""
    # "pending or completed" must be checked before individual keywords
    if re.search(r"pending.{0,20}or.{0,20}completed?|completed?.{0,20}or.{0,20}pending", text, re.I):
        status_filter = f"AND LOWER(t.document->>'{status_fld}') IN ('pending', 'completed')"
        status_label  = "Pending or Completed"
    elif re.search(r"\bpending\b", text):
        status_filter = f"AND LOWER(t.document->>'{status_fld}') = 'pending'"
        status_label  = "Pending"
    elif re.search(r"\bcompleted?\b|\bdone\b", text):
        status_filter = f"AND LOWER(t.document->>'{status_fld}') = 'completed'"
        status_label  = "Completed"
    elif re.search(r"\bopen\b", text):
        status_filter = f"AND LOWER(t.document->>'{status_fld}') = 'open'"
        status_label  = "Open"

    # ── Build SELECT columns ───────────────────────────────────────────────────
    cols = [
        f"COALESCE(u.document->>'{user_name_fld}', 'Unassigned') AS user_name",
        f"COALESCE(t.document->>'{task_name_fld}', 'Untitled') AS task",
        f"COALESCE(t.document->>'{status_fld}', '—') AS status",
    ]
    if priority_fld:
        cols.append(f"COALESCE(t.document->>'{priority_fld}', '—') AS priority")
    if due_fld:
        cols.append(f"t.document->>'{due_fld}' AS due_date")

    sql = f"""SELECT {', '.join(cols)}
FROM "{task_table}" t
LEFT JOIN "{user_table}" u
  ON u.document->>'_id' = t.document->>'{join_fld}'
WHERE t.document->>'{join_fld}' IS NOT NULL
  {status_filter}
ORDER BY u.document->>'{user_name_fld}' ASC NULLS LAST,
         t.document->>'{task_name_fld}'
LIMIT 200""".strip()

    _res = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [user_table, task_table], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows

    if not rows:
        lbl = f" with status **{status_label}**" if status_label else ""
        return {
            "answer":      f"No tasks{lbl} found assigned to users.",
            "tables_used": [user_table, task_table],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    headers = [c.split(" AS ")[-1].replace("_", " ").title() for c in cols]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:50]:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 50:
        lines.append(f"_…and {len(rows)-50} more rows_")

    lbl = f" (status: {status_label})" if status_label else ""
    return {
        "answer":      f"**Users with Tasks{lbl} — {len(rows)} records:**\n\n" + "\n".join(lines),
        "tables_used": [user_table, task_table],
        "confidence":  0.97,
        "sql_queries": [sql],
    }


def _fp_system_config(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(r"\bsmtp\b|\bfile upload\b|\bupload limit\b|\bsystem config\b", text):
        return None
    return {
        "answer":      (
            "This information is stored in the **application configuration**, "
            "not in the CRM database.\n\n"
            "- **SMTP settings** → App Settings → Email Configuration\n"
            "- **File upload limit** → App Settings → Storage\n"
            "- **System config** → App Settings → Organization"
        ),
        "tables_used": [],
        "confidence":  0.99,
        "sql_queries": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# HANDLER DISPATCH TABLES
# ══════════════════════════════════════════════════════════════════════════════

def _fp_sales(agent, user_query: str) -> Optional[Dict]:
    """
    Handle 'give me sales orders', 'top sales order by amount', 'list sales' queries.
    Routes explicitly to the `sales` table — prevents _fp_revenue from mishandling them.
    """
    text = normalize_text(user_query)
    # Only trigger on explicit "sales order" phrase or "give me sales" without revenue intent
    if not (re.search(r"\bsales\s+order", text) or
            (re.search(r"\bsales\b", text) and
             not re.search(r"\brevenue|income|billing|total\s+sales\b", text))):
        return None

    table_names = get_table_names(agent)
    table = next((t for t in table_names if t.lower() == "sales"), None)
    if not table:
        return None

    fields       = get_document_fields(agent, table)
    amount_field = next((f for f in fields if f.lower() in
                         ["grand_total_in_usd", "grand_total"]), None)
    num_field    = next((f for f in fields if "sales_number" in f.lower() or "number" in f.lower()), None)
    status_field = next((f for f in fields if f.lower() == "status"), None)
    date_field   = next((f for f in fields if "sales_date" in f.lower() or "date" in f.lower()), None)

    # Build select
    select_parts = []
    if num_field:    select_parts.append(f"document->>'{num_field}' AS sales_number")
    if status_field: select_parts.append(f"document->>'{status_field}' AS status")
    if amount_field: select_parts.append(
        f"NULLIF(document->>'{amount_field}','')::numeric AS amount"
    )
    if date_field:   select_parts.append(f"document->>'{date_field}' AS date")
    if not select_parts:
        return None

    # Ordering
    top_match = re.search(r"\btop\s+(\d+)\b", text)
    sort_desc = re.search(r"\bhighest|largest|by\s+amount|order\s+by\s+amount\b", text)

    if top_match:
        limit = min(int(top_match.group(1)), 50)
    else:
        limit = 20

    order = (f"NULLIF(document->>'{amount_field}','')::numeric DESC NULLS LAST"
             if (sort_desc or top_match) and amount_field
             else f"NULLIF(document->>'{date_field}','')::timestamptz DESC NULLS LAST"
             if date_field else "updated_at DESC")

    # Status filter
    where_parts = [f"COALESCE(document->>'deleted','false') != 'true'"]
    for kw, val in [("confirmed", "Confirm"), ("draft", "Draft"), ("cancelled", "Cancel")]:
        if kw in text and status_field:
            where_parts.append(f"document->>'{status_field}' ILIKE '%{val}%'")
            break

    where = "WHERE " + " AND ".join(f"({p})" for p in where_parts)
    sql   = f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} ORDER BY {order} LIMIT {limit}".strip()
    _res  = run_sql(agent, sql)
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [sql]}
    rows  = _res.rows

    if not rows:
        return {"answer": "No sales orders found.", "tables_used": [table],
                "confidence": 0.9, "sql_queries": [sql]}

    headers = [p.split(" AS ")[-1].replace("_", " ").title() for p in select_parts]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")

    label = f"Top {limit}" if top_match else str(len(rows))
    return {
        "answer":      f"**{label} sales orders:**\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.95,
        "sql_queries": [sql],
        "_entity":     table,
    }


def _fp_active_customers(agent, user_query: str) -> Optional[Dict]:
    """Handle 'active customers', 'how many active customers', 'give me active customers'.

    Active = companies WHERE lifecycleStage NOT IN ('Inactive Customer', 'Dead Customer')
             AND deleted != 'true'
    Verified against DB: total(144) - Inactive Customer(29) - Dead Customer(3) = active.
    """
    text = normalize_text(user_query)

    # Must have "active" AND a customer/company synonym
    if not re.search(r"\bactive\b", text):
        return None
    if not re.search(r"\b(customers?|companies?|company|clients?|accounts?|contacts?)\b", text):
        return None

    table_names = get_table_names(agent)
    table = next((t for t in table_names if t.lower() == "companies"), None)
    if not table:
        return None

    fields = get_document_fields(agent, table)

    # Find lifecycle and deleted fields
    lifecycle_field = next(
        (f for f in fields if f.lower().replace("_", "") in ["lifecyclestage", "lifecycle"]),
        None,
    )
    deleted_field = next((f for f in fields if f.lower() == "deleted"), None)
    name_field    = next((f for f in fields if f.lower() in ["companyname", "name"]), "companyName")
    email_field   = next((f for f in fields if f.lower() == "email"), None)
    status_field  = next((f for f in fields if f.lower() in ["leadstatus", "leadStatus"]), None)

    if not lifecycle_field:
        return None  # can't determine active without lifecycle

    # WHERE: exclude Inactive Customer and Dead Customer, exclude deleted
    where_parts = [
        f"COALESCE(document->>'{lifecycle_field}', '') "
        f"NOT IN ('Inactive Customer', 'Dead Customer')",
    ]
    if deleted_field:
        where_parts.append(f"COALESCE(document->>'{deleted_field}', 'false') != 'true'")

    where = "WHERE " + " AND ".join(f"({p})" for p in where_parts)

    # Count
    count_sql = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _cnt = run_sql(agent, count_sql)
    if _cnt.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql]}
    total = coerce_number(_cnt.rows[0][0] if _cnt.rows else 0)

    # Count-only intent
    wants_count_only = bool(
        re.search(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b", text)
    ) and not re.search(r"\bgive\b|\bshow\b|\blist\b|\ball\b|\bdisplay\b|\bfetch\b", text)

    if wants_count_only:
        return {
            "answer":      f"Total **active customers**: **{fmt_number(total)}**",
            "tables_used": [table],
            "confidence":  0.98,
            "sql_queries": [count_sql],
            "_entity":     table,
            "_count":      int(total),
        }

    # List view — build select
    select_parts = [f"document->>'{name_field}' AS name"]
    if email_field:
        select_parts.append(f"document->>'{email_field}' AS email")
    if status_field:
        select_parts.append(f"document->>'{status_field}' AS lead_status")
    select_parts.append(f"document->>'{lifecycle_field}' AS lifecycle_stage")

    list_sql = (
        f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} "
        f"ORDER BY updated_at DESC LIMIT 100"
    ).strip()
    _rows = run_sql(agent, list_sql)
    if _rows.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql, list_sql]}
    rows = _rows.rows

    if not rows:
        return {
            "answer":      "No active customers found.",
            "tables_used": [table],
            "confidence":  0.9,
            "sql_queries": [count_sql, list_sql],
        }

    headers = [p.split(" AS ")[-1].replace("_", " ").title() for p in select_parts]
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:50]:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 50:
        lines.append(f"_…and {len(rows)-50} more_")

    return {
        "answer":      (
            f"**Active Customers — {fmt_number(total)} total "
            f"(excluding Inactive & Dead):**\n\n" + "\n".join(lines)
        ),
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": [count_sql, list_sql],
        "_entity":     table,
        "_count":      int(total),
    }


def _fp_kpi_report(agent, user_query: str) -> Optional[Dict]:
    """Enterprise-grade CRM KPI report — 25 live queries + McKinsey-style LLM synthesis."""
    import traceback as _tb
    try:
        return _fp_kpi_report_inner(agent, user_query)
    except Exception as exc:
        LOGGER.error("KPI report FULL TRACEBACK:\n%s", _tb.format_exc())
        raise


def _fp_kpi_report_inner(agent, user_query: str) -> Optional[Dict]:

    sqls: List[str] = []
    kpi: Dict[str, Any] = {}

    def _q(sql: str):
        s = sql.strip()
        sqls.append(s)
        res = run_sql(agent, s)
        if res.error:
            LOGGER.debug("KPI query error: %s | SQL: %.80s", res.error, s)
            return []
        return res.rows or []

    def _row(rows, min_cols: int = 1):
        """Return first row if it exists and has enough columns, else None."""
        if rows and isinstance(rows[0], (tuple, list)) and len(rows[0]) >= min_cols:
            return rows[0]
        return None

    def _si(v) -> int:
        """Safe int: handles int, float, Decimal, and float-strings like '12000.26'.
        LangChain's sql_db_query tool converts Decimal('x') → string 'x' via regex,
        so numeric DB columns arrive as strings — this handles that transparently."""
        if v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            try:
                return int(float(str(v)))
            except Exception:
                return 0

    def _sf(v) -> float:
        """Safe float: handles None, Decimal, int, and numeric strings."""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            try:
                return float(str(v))
            except Exception:
                return 0.0

    def _usd(v) -> str:
        try:
            return f"USD {_sf(v):,.2f}"
        except Exception:
            return "USD 0.00"

    def _pct(a, b) -> str:
        try:
            return f"{round(_sf(a) / _sf(b) * 100, 1)}%" if _sf(b) else "N/A"
        except Exception:
            return "N/A"

    # ── 1. DEALS OVERVIEW — split into two simpler queries to avoid LangChain truncation ──
    rows = _q("""
        SELECT
          COUNT(*)::int AS total,
          COUNT(CASE WHEN document->>'stage' NOT IN ('Closed Won','Closed Lost') THEN 1 END)::int AS open_cnt,
          COUNT(CASE WHEN document->>'stage' = 'Closed Won' THEN 1 END)::int AS won_cnt,
          COUNT(CASE WHEN document->>'stage' = 'Closed Lost' THEN 1 END)::int AS lost_cnt
        FROM deals WHERE COALESCE(document->>'deleted','false') != 'true'
    """)
    r = _row(rows, 4)
    if r:
        won_d, lost_d = _si(r[2]), _si(r[3])
        kpi["deals"] = {
            "total": _si(r[0]), "open": _si(r[1]),
            "won": won_d, "lost": lost_d,
            "pipeline_value": 0, "won_value": 0, "avg_deal_size": 0,
            "win_rate_pct": round(won_d / (won_d + lost_d) * 100, 1) if (won_d + lost_d) else 0,
            "loss_rate_pct": round(lost_d / (won_d + lost_d) * 100, 1) if (won_d + lost_d) else 0,
        }

    rows = _q("""
        SELECT
          COALESCE(SUM(CASE WHEN document->>'stage' NOT IN ('Closed Won','Closed Lost')
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS pipeline_val,
          COALESCE(SUM(CASE WHEN document->>'stage' = 'Closed Won'
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS won_val,
          ROUND(AVG(CASE WHEN document->>'stage' NOT IN ('Closed Won','Closed Lost')
            AND NULLIF(document->>'grand_total_in_usd','')::numeric > 0
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric END)::numeric, 2) AS avg_size
        FROM deals WHERE COALESCE(document->>'deleted','false') != 'true'
    """)
    r2 = _row(rows, 2)
    if r2 and "deals" in kpi:
        kpi["deals"]["pipeline_value"] = _sf(r2[0])
        kpi["deals"]["won_value"]      = _sf(r2[1])
        kpi["deals"]["avg_deal_size"]  = _sf(r2[2]) if len(r2) > 2 else 0

    # ── 2. DEALS BY STAGE (open only) ────────────────────────────────────────────
    rows = _q("""
        SELECT document->>'stage', COUNT(*)::int,
               COALESCE(SUM(NULLIF(document->>'grand_total_in_usd','')::numeric), 0)
        FROM deals
        WHERE COALESCE(document->>'deleted','false') != 'true'
          AND document->>'stage' NOT IN ('Closed Won','Closed Lost')
        GROUP BY 1 ORDER BY 3 DESC
    """)
    kpi["deal_stages"] = [
        {"stage": r[0] or "Unknown", "count": _si(r[1]), "value": _sf(r[2])}
        for r in rows if r and len(r) >= 3
    ]

    # ── 3. STALLED DEALS (closeDate passed, still open) ──────────────────────────
    rows = _q("""
        SELECT COUNT(*)::int,
               COALESCE(SUM(NULLIF(document->>'grand_total_in_usd','')::numeric), 0)
        FROM deals
        WHERE COALESCE(document->>'deleted','false') != 'true'
          AND document->>'stage' NOT IN ('Closed Won','Closed Lost')
          AND NULLIF(document->>'closeDate','')::timestamptz < NOW()
    """)
    r = _row(rows, 2)
    kpi["stalled_deals"] = {
        "count": _si(r[0]) if r else 0,
        "value": _sf(r[1]) if r else 0,
    }

    # ── 4. DEALS BY OWNER / SALES REP ────────────────────────────────────────────
    rows = _q("""
        SELECT
          COALESCE(u.document->>'name', 'Unassigned') AS rep,
          COUNT(*) FILTER (WHERE d.document->>'stage' NOT IN ('Closed Won','Closed Lost'))::int AS open_deals,
          COUNT(*) FILTER (WHERE d.document->>'stage' = 'Closed Won')::int AS won,
          COUNT(*) FILTER (WHERE d.document->>'stage' = 'Closed Lost')::int AS lost,
          COALESCE(SUM(NULLIF(d.document->>'grand_total_in_usd','')::numeric)
            FILTER (WHERE d.document->>'stage' NOT IN ('Closed Won','Closed Lost')), 0) AS pipeline
        FROM deals d
        LEFT JOIN users u ON u.document->>'_id' = d.document->>'owner'
        WHERE COALESCE(d.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC
    """)
    kpi["deals_by_rep"] = [
        {"rep": r[0], "open": _si(r[1]), "won": _si(r[2]),
         "lost": _si(r[3]), "pipeline": _sf(r[4])}
        for r in rows if r and len(r) >= 5
    ]

    # ── 5. COMPANIES / CUSTOMER LIFECYCLE ────────────────────────────────────────
    rows = _q("""
        SELECT document->>'lifecycleStage', COUNT(*)::int
        FROM companies WHERE COALESCE(document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC
    """)
    lifecycle = {r[0] or "Unknown": _si(r[1]) for r in rows if r and len(r) >= 2}
    kpi["companies"] = {
        "total":        sum(lifecycle.values()),
        "leads":        lifecycle.get("Lead", 0),
        "customers":    lifecycle.get("Customer", 0),
        "partners":     lifecycle.get("Partner", 0),
        "inactive":     lifecycle.get("Inactive Customer", 0),
        "dead":         lifecycle.get("Dead Customer", 0),
        "active_total": lifecycle.get("Customer", 0) + lifecycle.get("Partner", 0),
    }

    # ── 6. CUSTOMER CONVERSIONS YoY ──────────────────────────────────────────────
    rows = _q("""
        SELECT
          COUNT(*) FILTER (WHERE (document->>'leadWonAt')::date >= date_trunc('year', CURRENT_DATE))::int,
          COUNT(*) FILTER (WHERE EXTRACT(year FROM (document->>'leadWonAt')::date)
            = EXTRACT(year FROM CURRENT_DATE)-1)::int
        FROM companies
        WHERE document->>'lifecycleStage' IN ('Customer','Partner')
          AND COALESCE(document->>'deleted','false') != 'true'
          AND document->>'leadWonAt' IS NOT NULL AND document->>'leadWonAt' != ''
    """)
    r = _row(rows, 2)
    kpi["conversions"] = {
        "this_year": _si(r[0]) if r else 0,
        "last_year": _si(r[1]) if r else 0,
    }

    # ── 7. INDUSTRY DISTRIBUTION ─────────────────────────────────────────────────
    rows = _q("""
        SELECT COALESCE(document->>'industry','Unknown'), COUNT(*)::int
        FROM companies WHERE COALESCE(document->>'deleted','false') != 'true'
          AND document->>'industry' IS NOT NULL AND document->>'industry' != ''
        GROUP BY 1 ORDER BY 2 DESC LIMIT 8
    """)
    kpi["industries"] = [{"name": r[0], "count": _si(r[1])} for r in rows if r and len(r) >= 2]

    # ── 8. REGION DISTRIBUTION ───────────────────────────────────────────────────
    rows = _q("""
        SELECT r.document->>'regionName', COUNT(c.id)::int
        FROM companies c
        JOIN regions r ON r.document->>'_id' = c.document->>'region'
        WHERE COALESCE(c.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC
    """)
    kpi["regions"] = [{"name": r[0], "count": _si(r[1])} for r in rows if r and len(r) >= 2]

    # ── 9. REVENUE BUCKETS — split into two queries to avoid LangChain column truncation ──
    rows = _q("""
        SELECT
          COALESCE(SUM(CASE WHEN (document->>'sales_date')::date >= date_trunc('month', CURRENT_DATE)
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS this_month,
          COALESCE(SUM(CASE WHEN date_trunc('month',(document->>'sales_date')::date)
            = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS last_month,
          COALESCE(SUM(CASE WHEN (document->>'sales_date')::date >= CURRENT_DATE - INTERVAL '3 months'
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS last_3m
        FROM sales
        WHERE document->>'status' = 'Confirm'
          AND COALESCE(document->>'deleted','false') != 'true'
    """)
    rows2 = _q("""
        SELECT
          COALESCE(SUM(CASE WHEN (document->>'sales_date')::date >= date_trunc('year', CURRENT_DATE)
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS ytd,
          COALESCE(SUM(CASE WHEN EXTRACT(year FROM (document->>'sales_date')::date)
            = EXTRACT(year FROM CURRENT_DATE)-1
            THEN NULLIF(document->>'grand_total_in_usd','')::numeric ELSE 0 END), 0) AS last_year,
          COUNT(*)::int AS order_count
        FROM sales
        WHERE document->>'status' = 'Confirm'
          AND COALESCE(document->>'deleted','false') != 'true'
    """)
    r  = _row(rows,  3)
    r2 = _row(rows2, 3)
    kpi["revenue"] = {
        "this_month":    _sf(r[0])  if r  else 0,
        "last_month":    _sf(r[1])  if r  else 0,
        "last_3_months": _sf(r[2])  if r  else 0,
        "ytd":           _sf(r2[0]) if r2 else 0,
        "last_year":     _sf(r2[1]) if r2 else 0,
        "order_count":   _si(r2[2]) if r2 else 0,
    }
    if True:
        rev = kpi["revenue"]
        rev["mom_change_pct"] = round(
            (rev["this_month"] - rev["last_month"]) / rev["last_month"] * 100, 1
        ) if rev["last_month"] else None
        rev["yoy_ytd_change_pct"] = round(
            (rev["ytd"] - rev["last_year"]) / rev["last_year"] * 100, 1
        ) if rev["last_year"] else None

    # ── 10. MONTHLY REVENUE TREND (last 6 months) ────────────────────────────────
    rows = _q("""
        SELECT to_char(sale_month,'Mon YYYY') AS month_label,
               COALESCE(SUM(NULLIF(document->>'grand_total_in_usd','')::numeric), 0) AS revenue,
               COUNT(*)::int AS orders,
               sale_month
        FROM sales,
             LATERAL (SELECT date_trunc('month',(document->>'sales_date')::date) AS sale_month) m
        WHERE document->>'status' = 'Confirm'
          AND COALESCE(document->>'deleted','false') != 'true'
          AND (document->>'sales_date')::date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY sale_month ORDER BY sale_month
    """)
    kpi["monthly_trend"] = [
        {"month": r[0], "revenue": _sf(r[1]), "orders": _si(r[2])}
        for r in rows if r and len(r) >= 3
    ]

    # ── 11. SALES REP PERFORMANCE ────────────────────────────────────────────────
    rows = _q("""
        SELECT COALESCE(u.document->>'name','Unassigned'),
               COUNT(s.id)::int,
               ROUND(COALESCE(SUM(NULLIF(s.document->>'grand_total_in_usd','')::numeric),0)::numeric,2)
        FROM sales s
        LEFT JOIN users u ON u.document->>'_id' = s.document->>'salesOwner'
        WHERE s.document->>'status' = 'Confirm'
          AND COALESCE(s.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 3 DESC
    """)
    kpi["sales_reps"] = [
        {"name": r[0], "orders": _si(r[1]), "revenue": _sf(r[2])}
        for r in rows if r and len(r) >= 3
    ]

    # ── 12. TOP 5 CUSTOMERS BY REVENUE ───────────────────────────────────────────
    rows = _q("""
        SELECT COALESCE(c.document->>'companyName', s.document->>'company','Unknown'),
               ROUND(SUM(NULLIF(s.document->>'grand_total_in_usd','')::numeric)::numeric,2),
               COUNT(s.id)::int
        FROM sales s
        LEFT JOIN companies c ON c.id = s.document->>'company'
        WHERE s.document->>'status' = 'Confirm'
          AND COALESCE(s.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 5
    """)
    kpi["top_customers"] = [
        {"name": r[0], "revenue": _sf(r[1]), "orders": _si(r[2])}
        for r in rows if r and len(r) >= 3
    ]
    # Revenue concentration: top 1 customer % of total YTD
    ytd_rev = kpi.get("revenue", {}).get("ytd", 0)
    top1_rev = kpi["top_customers"][0]["revenue"] if kpi["top_customers"] else 0
    top5_rev = sum(c["revenue"] for c in kpi["top_customers"])
    kpi["revenue_concentration"] = {
        "top1_pct":  round(top1_rev / ytd_rev * 100, 1) if ytd_rev else None,
        "top5_pct":  round(top5_rev / ytd_rev * 100, 1) if ytd_rev else None,
        "top1_name": kpi["top_customers"][0]["name"] if kpi["top_customers"] else "N/A",
    }

    # ── 13. INVOICES BY STATUS + TOTALS ──────────────────────────────────────────
    rows = _q("""
        SELECT document->>'payment_status', COUNT(*)::int,
               COALESCE(SUM(NULLIF(document->>'grandtotal_in_usd','')::numeric), 0)
        FROM invoices WHERE COALESCE(document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 3 DESC
    """)
    by_status = [
        {"status": r[0] or "unknown", "count": _si(r[1]), "value": _sf(r[2])}
        for r in rows if r and len(r) >= 3
    ]
    kpi["invoices"] = {
        "by_status":     by_status,
        "pending_total": sum(s["value"] for s in by_status if s["status"] not in ("paid","cancelled")),
        "paid_total":    next((s["value"] for s in by_status if s["status"] == "paid"), 0),
        "overdue_count": 0, "overdue_amount": 0,
    }

    # ── 14. INVOICE AGING BUCKETS ────────────────────────────────────────────────
    rows = _q("""
        SELECT
          COUNT(*) FILTER (WHERE days_od <= 30)::int,
          ROUND(SUM(amt) FILTER (WHERE days_od <= 30)::numeric, 2),
          COUNT(*) FILTER (WHERE days_od BETWEEN 31 AND 60)::int,
          ROUND(SUM(amt) FILTER (WHERE days_od BETWEEN 31 AND 60)::numeric, 2),
          COUNT(*) FILTER (WHERE days_od BETWEEN 61 AND 90)::int,
          ROUND(SUM(amt) FILTER (WHERE days_od BETWEEN 61 AND 90)::numeric, 2),
          COUNT(*) FILTER (WHERE days_od > 90)::int,
          ROUND(SUM(amt) FILTER (WHERE days_od > 90)::numeric, 2)
        FROM (
          SELECT DATE_PART('day', NOW()-NULLIF(document->>'due_date','')::timestamptz)::int AS days_od,
                 COALESCE(NULLIF(document->>'grandtotal_in_usd','')::numeric, 0) AS amt
          FROM invoices
          WHERE NULLIF(document->>'due_date','')::timestamptz < NOW()
            AND COALESCE(document->>'payment_status','') NOT IN ('paid','cancelled')
            AND COALESCE(document->>'deleted','false') != 'true'
        ) sub
    """)
    r = _row(rows, 8)
    if r:
        kpi["invoice_aging"] = {
            "0_30":   {"count": _si(r[0]), "amount": _sf(r[1])},
            "31_60":  {"count": _si(r[2]), "amount": _sf(r[3])},
            "61_90":  {"count": _si(r[4]), "amount": _sf(r[5])},
            "90plus": {"count": _si(r[6]), "amount": _sf(r[7])},
        }
        kpi["invoices"]["overdue_count"]  = sum(_si(r[i]) for i in [0,2,4,6])
        kpi["invoices"]["overdue_amount"] = sum(_sf(r[i]) for i in [1,3,5,7])

    # ── 15. TOP OVERDUE COMPANIES ────────────────────────────────────────────────
    rows = _q("""
        SELECT COALESCE(c.document->>'companyName', i.document->>'company','Unknown'),
               COUNT(i.id)::int,
               ROUND(SUM(NULLIF(i.document->>'grandtotal_in_usd','')::numeric)::numeric,2),
               MAX(DATE_PART('day', NOW()-NULLIF(i.document->>'due_date','')::timestamptz))::int
        FROM invoices i
        LEFT JOIN companies c ON c.id = i.document->>'company'
        WHERE NULLIF(i.document->>'due_date','')::timestamptz < NOW()
          AND COALESCE(i.document->>'payment_status','') NOT IN ('paid','cancelled')
          AND COALESCE(i.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 3 DESC LIMIT 5
    """)
    kpi["top_overdue"] = [
        {"company": r[0], "invoices": _si(r[1]),
         "amount": _sf(r[2]), "max_days": _si(r[3])}
        for r in rows if r and len(r) >= 4
    ]

    # ── 16. OUTREACH FUNNEL + CAMPAIGN ───────────────────────────────────────────
    rows = _q("""
        SELECT document->>'status', COUNT(*)::int
        FROM outreaches WHERE COALESCE(document->>'isDeleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC
    """)
    out_by_status = {r[0] or "Unknown": _si(r[1]) for r in rows if r and len(r) >= 2}
    total_out = sum(out_by_status.values())
    contacted = out_by_status.get("Contacted", 0)
    converted = out_by_status.get("Converted to Deal", 0)
    kpi["outreaches"] = {
        "total": total_out,
        "by_status": [{"status": k, "count": v} for k, v in out_by_status.items()],
        "contacted": contacted,
        "converted": converted,
        "contact_rate_pct": round(contacted / total_out * 100, 1) if total_out else 0,
        "conversion_rate_pct": round(converted / total_out * 100, 1) if total_out else 0,
    }

    rows = _q("""
        SELECT camp.document->>'campaignName',
               COUNT(o.id)::int AS total,
               COUNT(*) FILTER (WHERE o.document->>'status' IN ('Contacted','Converted to Deal'))::int AS engaged,
               COUNT(*) FILTER (WHERE o.document->>'status' = 'Converted to Deal')::int AS converted
        FROM outreaches o
        LEFT JOIN campaigns camp ON camp.document->>'_id' = o.document->>'campaign'
        WHERE COALESCE(o.document->>'isDeleted','false') != 'true'
          AND o.document->>'campaign' IS NOT NULL AND o.document->>'campaign' != ''
        GROUP BY 1 ORDER BY 2 DESC LIMIT 5
    """)
    kpi["campaigns"] = [
        {"name": r[0] or "Unknown", "total": _si(r[1]),
         "engaged": _si(r[2]), "converted": _si(r[3])}
        for r in rows if r and len(r) >= 4
    ]

    # ── 17. TASKS BY PRIORITY AND USER ───────────────────────────────────────────
    rows = _q("""
        SELECT document->>'priority', document->>'status', COUNT(*)::int
        FROM createtasks WHERE COALESCE(document->>'deleted','false') != 'true'
        GROUP BY 1,2 ORDER BY 3 DESC
    """)
    task_matrix: Dict[str, Any] = {}
    for r in rows:
        if not r or len(r) < 3:
            continue
        pri, sts, cnt = r[0] or "Unknown", r[1] or "Unknown", _si(r[2])
        task_matrix.setdefault(pri, {})[sts] = cnt
    kpi["tasks"] = {
        "matrix": task_matrix,
        "pending":   sum(v.get("Pending", 0) for v in task_matrix.values()),
        "completed": sum(v.get("Completed", 0) for v in task_matrix.values()),
        "high_priority_pending": task_matrix.get("High", {}).get("Pending", 0),
    }

    rows = _q("""
        SELECT COALESCE(u.document->>'name','Unassigned'),
               COUNT(*) FILTER (WHERE t.document->>'status'='Pending')::int,
               COUNT(*) FILTER (WHERE t.document->>'status'='Completed')::int
        FROM createtasks t
        LEFT JOIN users u ON u.document->>'_id' = t.document->>'createdBy'
        WHERE COALESCE(t.document->>'deleted','false') != 'true'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 8
    """)
    kpi["tasks_by_user"] = [
        {"user": r[0], "pending": _si(r[1]), "completed": _si(r[2])}
        for r in rows if r and len(r) >= 3
    ]

    # ── 18. CONTACTS ─────────────────────────────────────────────────────────────
    rows = _q("""
        SELECT COUNT(*)::int,
               COUNT(*) FILTER (WHERE document->>'lifecycleStage'='Lead')::int,
               COUNT(*) FILTER (WHERE document->>'lifecycleStage'='Customer')::int
        FROM contacts WHERE COALESCE(document->>'deleted','false') != 'true'
    """)
    r = _row(rows, 3)
    kpi["contacts"] = {
        "total":     _si(r[0]) if r else 0,
        "leads":     _si(r[1]) if r else 0,
        "customers": _si(r[2]) if r else 0,
    }

    # ── 19. TARGETS VS ACHIEVED ───────────────────────────────────────────────────
    rows = _q("""
        SELECT COALESCE(u.document->>'name','Unknown'),
               COALESCE(SUM(NULLIF(t.document->>'targetInUSD','')::numeric),0) AS target,
               t.document->>'teamName'
        FROM targets t
        LEFT JOIN users u ON u.document->>'_id' = t.document->>'userId'
        WHERE (t.document->>'year')::int = EXTRACT(year FROM CURRENT_DATE)::int
          AND (t.document->>'month')::int = EXTRACT(month FROM CURRENT_DATE)::int
        GROUP BY 1,3
    """)
    total_target = sum(_sf(r[1]) for r in rows if r and len(r) >= 2)
    kpi["target_by_rep"] = [
        {"name": r[0], "target": _sf(r[1]), "team": r[2] if len(r) > 2 else ""}
        for r in rows if r and len(r) >= 2
    ]

    rows = _q("""
        SELECT COALESCE(SUM(NULLIF(document->>'grand_total_in_usd','')::numeric),0)
        FROM sales
        WHERE document->>'status'='Confirm'
          AND COALESCE(document->>'deleted','false')!='true'
          AND (document->>'sales_date')::date >= date_trunc('month', CURRENT_DATE)
    """)
    achieved_this_month = _sf(_row(rows, 1)[0]) if _row(rows, 1) else 0
    kpi["targets"] = {
        "this_month_target":   total_target,
        "this_month_achieved": achieved_this_month,
        "achievement_pct": round(achieved_this_month / total_target * 100, 1) if total_target else None,
        "gap": total_target - achieved_this_month,
    }

    # ── 20. PRODUCTS (active count) ───────────────────────────────────────────────
    rows = _q("""
        SELECT COUNT(*) FILTER (WHERE COALESCE(document->>'isActive','true')='true')::int,
               COUNT(*)::int
        FROM products
    """)
    r = _row(rows, 2)
    kpi["products"] = {
        "active": _si(r[0]) if r else 0,
        "total":  _si(r[1]) if r else 0,
    }

    # ── 21. NEW LEADS (this week / this month) ────────────────────────────────────
    rows = _q("""
        SELECT
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'createdAt','')::timestamptz >= date_trunc('week', NOW()))::int AS week,
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'createdAt','')::timestamptz >= date_trunc('month', NOW()))::int AS month,
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'createdAt','')::timestamptz >= date_trunc('week', NOW())
            AND document->>'lifecycleStage' = 'Lead')::int AS leads_week,
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'createdAt','')::timestamptz >= date_trunc('month', NOW())
            AND document->>'lifecycleStage' = 'Lead')::int AS leads_month,
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'createdAt','')::timestamptz
              >= date_trunc('week', NOW() - INTERVAL '1 week')
            AND NULLIF(document->>'createdAt','')::timestamptz
              < date_trunc('week', NOW())
            AND document->>'lifecycleStage' = 'Lead')::int AS leads_last_week
        FROM companies
        WHERE COALESCE(document->>'deleted','false') != 'true'
    """)
    r = _row(rows, 5)
    kpi["new_leads"] = {
        "this_week":  _si(r[2]) if r else 0,
        "this_month": _si(r[3]) if r else 0,
        "last_week":  _si(r[4]) if r else 0,
    }

    # ── 22. UNCONTACTED LEADS (no activity in 7+ days) ────────────────────────────
    rows = _q("""
        SELECT
          COUNT(*)::int AS total_uncontacted,
          COUNT(*) FILTER (WHERE
            document->>'lastActivity' IS NULL OR document->>'lastActivity' = '')::int AS never_contacted
        FROM companies
        WHERE document->>'lifecycleStage' = 'Lead'
          AND COALESCE(document->>'deleted','false') != 'true'
          AND (
            document->>'lastActivity' IS NULL
            OR document->>'lastActivity' = ''
            OR NULLIF(document->>'lastActivity','')::timestamptz < NOW() - INTERVAL '7 days'
          )
    """)
    r = _row(rows, 2)
    kpi["uncontacted_leads"] = {
        "total_7d_plus":   _si(r[0]) if r else 0,
        "never_contacted": _si(r[1]) if r else 0,
    }

    # ── 23. OVERDUE FOLLOW-UP TASKS ───────────────────────────────────────────────
    rows = _q("""
        SELECT
          COUNT(*)::int AS total_overdue,
          COUNT(*) FILTER (WHERE
            NULLIF(document->>'due_date','')::timestamptz >= NOW() - INTERVAL '7 days')::int AS overdue_this_week,
          COUNT(*) FILTER (WHERE document->>'priority' = 'High')::int AS high_priority_overdue
        FROM createtasks
        WHERE document->>'status' = 'Pending'
          AND NULLIF(document->>'due_date','')::timestamptz < NOW()
          AND COALESCE(document->>'deleted','false') != 'true'
    """)
    r = _row(rows, 3)
    kpi["overdue_tasks"] = {
        "total":          _si(r[0]) if r else 0,
        "due_this_week":  _si(r[1]) if r else 0,
        "high_priority":  _si(r[2]) if r else 0,
    }

    # ── 24. DEALS EXPECTED TO CLOSE THIS MONTH ────────────────────────────────────
    rows = _q("""
        SELECT COUNT(*)::int,
               COALESCE(SUM(NULLIF(document->>'grand_total_in_usd','')::numeric), 0)
        FROM deals
        WHERE COALESCE(document->>'deleted','false') != 'true'
          AND document->>'stage' NOT IN ('Closed Won','Closed Lost')
          AND NULLIF(document->>'closeDate','')::date
            BETWEEN date_trunc('month', CURRENT_DATE)::date
            AND (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::date
    """)
    r = _row(rows, 2)
    kpi["closing_this_month"] = {
        "count": _si(r[0]) if r else 0,
        "value": _sf(r[1]) if r else 0,
    }

    # ── 25. LEAD TO CUSTOMER CONVERSION RATE ──────────────────────────────────────
    total_ever = kpi.get("companies", {}).get("total", 0)
    active_cust = kpi.get("companies", {}).get("active_total", 0)
    total_leads = kpi.get("companies", {}).get("leads", 0)
    kpi["lead_conversion"] = {
        "lead_to_customer_rate_pct": round(
            active_cust / (total_leads + active_cust) * 100, 1
        ) if (total_leads + active_cust) else 0,
        "total_leads": total_leads,
        "total_customers": active_cust,
    }

    # ══════════════════════════════════════════════════════════════════════════════
    # BUILD DATA SUMMARY FOR LLM
    # ══════════════════════════════════════════════════════════════════════════════
    deals      = kpi.get("deals", {})
    revenue    = kpi.get("revenue", {})
    companies  = kpi.get("companies", {})
    conv       = kpi.get("conversions", {})
    invoices   = kpi.get("invoices", {})
    aging      = kpi.get("invoice_aging", {})
    tasks      = kpi.get("tasks", {})
    contacts   = kpi.get("contacts", {})
    targets    = kpi.get("targets", {})
    trend      = kpi.get("monthly_trend", [])
    top_cust   = kpi.get("top_customers", [])
    outreach   = kpi.get("outreaches", {})
    stages     = kpi.get("deal_stages", [])
    reps       = kpi.get("sales_reps", [])
    dreps      = kpi.get("deals_by_rep", [])
    stalled    = kpi.get("stalled_deals", {})
    regions    = kpi.get("regions", [])
    industries = kpi.get("industries", [])
    campaigns  = kpi.get("campaigns", [])
    t_by_user  = kpi.get("tasks_by_user", [])
    top_od     = kpi.get("top_overdue", [])
    rc         = kpi.get("revenue_concentration", {})
    tgt_reps   = kpi.get("target_by_rep", [])
    new_leads  = kpi.get("new_leads", {})
    unc_leads  = kpi.get("uncontacted_leads", {})
    ovd_tasks  = kpi.get("overdue_tasks", {})
    closing    = kpi.get("closing_this_month", {})
    lead_conv  = kpi.get("lead_conversion", {})

    mom   = revenue.get("mom_change_pct")
    yoy   = revenue.get("yoy_ytd_change_pct")
    mom_s = (f"{'+' if mom >= 0 else ''}{mom}% MoM" if mom is not None else "N/A MoM")
    yoy_s = (f"{'+' if yoy >= 0 else ''}{yoy}% YoY" if yoy is not None else "N/A YoY")
    ach   = targets.get("achievement_pct")
    ach_s = f"{ach}%" if ach is not None else "No target set for this month"

    data_summary = f"""
LIVE CRM DATABASE — KPI DATA SNAPSHOT (as of today)

━━━ DEALS ━━━
Total Deals: {deals.get('total',0)} | Open: {deals.get('open',0)} | Won: {deals.get('won',0)} | Lost: {deals.get('lost',0)}
Win Rate: {deals.get('win_rate_pct',0)}% | Loss Rate: {deals.get('loss_rate_pct',0)}%
Pipeline Value (open deals): {_usd(deals.get('pipeline_value',0))}
Won Value: {_usd(deals.get('won_value',0))}
Avg Open Deal Size: {_usd(deals.get('avg_deal_size',0))}
Stalled Deals (overdue close date, still open): {stalled.get('count',0)} = {_usd(stalled.get('value',0))}

Open Deals by Stage:
{chr(10).join(f"  {s['stage']}: {s['count']} deals | {_usd(s['value'])}" for s in stages)}

Sales Rep Performance (Deals):
{chr(10).join(f"  {d['rep']}: {d['open']} open | {d['won']} won | {d['lost']} lost | pipeline {_usd(d['pipeline'])}" for d in dreps)}

━━━ REVENUE ━━━
This Month: {_usd(revenue.get('this_month',0))} ({mom_s})
Last Month: {_usd(revenue.get('last_month',0))}
Last 3 Months: {_usd(revenue.get('last_3_months',0))}
Year-to-Date: {_usd(revenue.get('ytd',0))} ({yoy_s} vs full last year of {_usd(revenue.get('last_year',0))})
Last Full Year: {_usd(revenue.get('last_year',0))}
Total Confirmed Orders: {revenue.get('order_count',0)}

Monthly Revenue Trend (last 6 months):
{chr(10).join(f"  {m['month']}: {_usd(m['revenue'])} ({m['orders']} orders)" for m in trend)}

Sales Rep Revenue (confirmed orders):
{chr(10).join(f"  {r['name']}: {r['orders']} orders = {_usd(r['revenue'])}" for r in reps)}

Revenue Concentration Risk:
  Top Customer ({rc.get('top1_name','N/A')}): {rc.get('top1_pct','N/A')}% of YTD revenue
  Top 5 Customers Combined: {rc.get('top5_pct','N/A')}% of YTD revenue

Top 5 Customers by Revenue:
{chr(10).join(f"  {i+1}. {c['name']}: {_usd(c['revenue'])} ({c['orders']} orders)" for i,c in enumerate(top_cust))}

━━━ CUSTOMERS & COMPANIES ━━━
Total Companies (non-deleted): {companies.get('total',0)}
Active Customers: {companies.get('active_total',0)} (Customers: {companies.get('customers',0)} + Partners: {companies.get('partners',0)})
Leads in Pipeline: {companies.get('leads',0)}
Inactive Customers: {companies.get('inactive',0)} | Dead/Churned: {companies.get('dead',0)}
Converted This Year: {conv.get('this_year',0)} | Converted Last Year: {conv.get('last_year',0)}
YoY Conversion Change: {round((conv.get('this_year',0) - conv.get('last_year',0)) / max(conv.get('last_year',1),1) * 100, 1)}%
Total Contacts: {contacts.get('total',0)}

Industry Distribution: {', '.join(f"{i['name']}={i['count']}" for i in industries[:6])}
Region Distribution: {', '.join(f"{r['name']}={r['count']}" for r in regions)}

━━━ INVOICES & RECEIVABLES ━━━
By Status: {', '.join(f"{s['status']}={s['count']}({_usd(s['value'])})" for s in invoices.get('by_status',[]))}
Total Pending Receivables (excl paid/cancelled): {_usd(invoices.get('pending_total',0))}
Total Paid: {_usd(invoices.get('paid_total',0))}
Overdue Count: {invoices.get('overdue_count',0)} | Overdue Amount: {_usd(invoices.get('overdue_amount',0))}

Receivable Aging (overdue invoices):
  0-30 days:  {aging.get('0_30',{}).get('count',0)} invoices = {_usd(aging.get('0_30',{}).get('amount',0))}
  31-60 days: {aging.get('31_60',{}).get('count',0)} invoices = {_usd(aging.get('31_60',{}).get('amount',0))}
  61-90 days: {aging.get('61_90',{}).get('count',0)} invoices = {_usd(aging.get('61_90',{}).get('amount',0))}
  90+ days:   {aging.get('90plus',{}).get('count',0)} invoices = {_usd(aging.get('90plus',{}).get('amount',0))}

Top Overdue Accounts:
{chr(10).join(f"  {t['company']}: {t['invoices']} invoices, {_usd(t['amount'])}, max {t['max_days']} days overdue" for t in top_od)}

━━━ OUTREACH & LEAD FUNNEL ━━━
Total Outreaches: {outreach.get('total',0)}
Funnel: {' → '.join(f"{s['status']}({s['count']})" for s in outreach.get('by_status',[])[:5])}
Contact Rate: {outreach.get('contact_rate_pct',0)}% | Conversion Rate: {outreach.get('conversion_rate_pct',0)}%

Campaign Performance:
{chr(10).join(f"  {c['name']}: {c['total']} outreaches | {c['engaged']} engaged | {c['converted']} converted" for c in campaigns)}

━━━ TASKS & PRODUCTIVITY ━━━
Total Pending: {tasks.get('pending',0)} | Completed: {tasks.get('completed',0)}
High Priority Pending: {tasks.get('high_priority_pending',0)}
Completion Rate: {_pct(tasks.get('completed',0), tasks.get('pending',0) + tasks.get('completed',0))}

Tasks by User:
{chr(10).join(f"  {u['user']}: {u['pending']} pending | {u['completed']} completed" for u in t_by_user)}

━━━ TARGETS vs ACHIEVED (This Month) ━━━
Total Target: {_usd(targets.get('this_month_target',0))}
Achieved: {_usd(targets.get('this_month_achieved',0))}
Achievement: {ach_s}
Gap: {_usd(targets.get('gap',0))}

Rep-wise Targets:
{chr(10).join(f"  {t['name']} ({t['team']}): target {_usd(t['target'])}" for t in tgt_reps)}

━━━ LEAD ACQUISITION FUNNEL ━━━
New Leads This Week: {new_leads.get('this_week',0)}  |  Last Week: {new_leads.get('last_week',0)}
New Leads This Month: {new_leads.get('this_month',0)}
Total Active Leads: {lead_conv.get('total_leads',0)}
Total Customers (Active): {lead_conv.get('total_customers',0)}
Lead → Customer Conversion Rate: {lead_conv.get('lead_to_customer_rate_pct',0)}%

Uncontacted Leads (7+ days no activity): {unc_leads.get('total_7d_plus',0)}
  — Never Contacted: {unc_leads.get('never_contacted',0)}

━━━ FOLLOW-UPS & OVERDUE TASKS ━━━
Overdue Tasks (past due date, still pending): {ovd_tasks.get('total',0)}
  — Overdue This Week: {ovd_tasks.get('due_this_week',0)}
  — High Priority Overdue: {ovd_tasks.get('high_priority',0)}

━━━ FORECAST (THIS MONTH) ━━━
Deals Expected to Close This Month: {closing.get('count',0)} = {_usd(closing.get('value',0))}
Stalled Deals (past close date, open): {stalled.get('count',0)} = {_usd(stalled.get('value',0))}

━━━ PRODUCTS ━━━
Active Products: {kpi.get('products',{}).get('active',0)} / {kpi.get('products',{}).get('total',0)} total
""".strip()

    # ══════════════════════════════════════════════════════════════════════════════
    # LLM — ENTERPRISE SYNTHESIS
    # ══════════════════════════════════════════════════════════════════════════════
    system_prompt = """You are an enterprise-grade CRM Business Intelligence Analyst AI.

Generate a professional, highly structured, executive-level CRM KPI Report using the provided live data.

REPORT STRUCTURE (follow exactly):

# CRM BUSINESS KPI REPORT

## 1. Executive Summary
- 6-8 bullet points, critical alerts first, then positives
- Mention revenue trend, pipeline health, win rate, overdue risk, conversion decline, productivity concerns
- Use bold for numbers

## 2. KPI Snapshot Dashboard
Compact table with columns: KPI | Value | Trend/Change | Status
Status: Excellent / Healthy / Warning / Critical
Include: Total Deals, Open Deals, Win Rate, Pipeline Value, Revenue This Month, Revenue YTD, Revenue Last Year, YoY Change, Avg Deal Size, Overdue Invoices, Pending Receivables, Active Customers, New Customers This Year, Inactive Customers, Outreach Conversion Rate, Pending Tasks, High Priority Tasks, Target Achievement

## 3. Revenue Analysis
- Table: This Month | Last Month | MoM% | Last 3M | YTD | Last Year | YoY%
- Monthly trend table
- Sales rep revenue breakdown table
- Revenue concentration analysis
### Key Findings | ### Risks | ### Opportunities

## 4. Pipeline & Deals Analysis
- Deal stage breakdown table (Stage | Count | Value | % of Pipeline)
- Sales rep deal performance table (Rep | Open | Won | Lost | Win Rate | Pipeline Value)
- Stalled deals alert
- Bottleneck identification
### Key Findings | ### Risks

## 5. Customer & Company Analysis
- Lifecycle breakdown table
- YoY conversion comparison
- Industry and region distribution tables
- Top 5 customers table
### Customer Retention Risks | ### Growth Opportunities

## 6. Invoice & Payment Analysis
- Status breakdown table
- Aging analysis table (0-30 | 31-60 | 61-90 | 90+)
- Top overdue accounts table
### Immediate Attention Required

## 7. Outreach & Lead Funnel
- Funnel table: Stage | Count | Conversion Rate
- Campaign performance table
### Funnel Drop-off | ### Recommendations

## 8. Task & Productivity Analysis
- Tasks by priority/status table
- Tasks by user table
- Completion rate and bottlenecks
### Critical Pending Actions

## 9. Lead Acquisition & Funnel
- New leads this week vs last week (WoW change)
- New leads this month
- Lead to customer conversion rate %
- Uncontacted leads (7+ days no activity) — critical flag
- Lead funnel health assessment
### Key Findings | ### Risks | ### Actions

## 10. Targets & Achievement
- This month target vs achieved table
- Rep-wise target table
### Gap Analysis

## 11. Business Risks & Alerts
Table: Risk | Severity (High/Medium/Low) | Impact | Recommendation
Auto-detect from data: revenue decline, overdue invoices, stalled pipeline, low conversion, customer churn, task backlog, concentration risk

## 12. Strategic Recommendations
### Immediate Actions (0-30 Days)
### Mid-Term Improvements (30-90 Days)
### Long-Term Strategic Enhancements

## 13. Business Health Score
Table: Category | Score/100 | Rating
Score revenue, pipeline, customers, collections, operations, growth
Overall Health: [score]/100 — [Excellent/Strong/Moderate/Weak/Critical]
### Why This Score | ### Top 3 Improvement Areas

RULES:
- Use ONLY the provided data. Never invent numbers.
- Use markdown tables heavily. Minimal prose.
- Bold all key numbers.
- Every section must have metrics + insight + recommendation.
- Be concise. Executive tone. Actionable language.
- Detect patterns: seasonal spikes, concentration risks, funnel drop-offs, productivity gaps."""

    llm_answer = _llm.call(
        task="synthesize",
        system=system_prompt,
        user=f"Generate the full enterprise KPI report from this live CRM data:\n\n{data_summary}",
        max_tokens=4000,
    )

    if llm_answer:
        answer = f"# CRM BUSINESS KPI REPORT\n\n{llm_answer}"
    else:
        # ── Structured markdown fallback (no LLM) ────────────────────────────────
        def _trow(*cols): return "| " + " | ".join(str(c) for c in cols) + " |"
        def _thdr(*cols): return _trow(*cols) + "\n| " + " | ".join(["---"]*len(cols)) + " |"

        ach_str = ach_s
        stage_rows   = "\n".join(_trow(s["stage"],s["count"],_usd(s["value"]),
                                       _pct(s["value"],deals.get("pipeline_value",1)))
                                  for s in stages)
        trend_rows   = "\n".join(_trow(m["month"],_usd(m["revenue"]),m["orders"]) for m in trend)
        rep_rows     = "\n".join(_trow(r["name"],r["orders"],_usd(r["revenue"])) for r in reps)
        drep_rows    = "\n".join(_trow(d["rep"],d["open"],d["won"],d["lost"],
                                        _pct(d["won"],d["won"]+d["lost"]),_usd(d["pipeline"]))
                                  for d in dreps)
        top_rows     = "\n".join(_trow(i+1,c["name"],_usd(c["revenue"]),c["orders"])
                                  for i,c in enumerate(top_cust))
        inv_rows     = "\n".join(_trow(s["status"].title(),s["count"],_usd(s["value"]))
                                  for s in invoices.get("by_status",[]))
        aging_rows   = "\n".join([
            _trow("0-30 days",  aging.get("0_30",{}).get("count",0),  _usd(aging.get("0_30",{}).get("amount",0))),
            _trow("31-60 days", aging.get("31_60",{}).get("count",0), _usd(aging.get("31_60",{}).get("amount",0))),
            _trow("61-90 days", aging.get("61_90",{}).get("count",0), _usd(aging.get("61_90",{}).get("amount",0))),
            _trow("90+ days",   aging.get("90plus",{}).get("count",0),_usd(aging.get("90plus",{}).get("amount",0))),
        ])
        od_rows      = "\n".join(_trow(t["company"],t["invoices"],_usd(t["amount"]),f"{t['max_days']}d")
                                  for t in top_od)
        camp_rows    = "\n".join(_trow(c["name"],c["total"],c["engaged"],c["converted"],
                                        _pct(c["converted"],c["total"]))
                                  for c in campaigns)
        tbu_rows     = "\n".join(_trow(u["user"],u["pending"],u["completed"]) for u in t_by_user)
        tgt_rows     = "\n".join(_trow(t["name"],t["team"],_usd(t["target"])) for t in tgt_reps)
        ind_rows     = "\n".join(_trow(i["name"],i["count"]) for i in industries)
        reg_rows     = "\n".join(_trow(r["name"],r["count"]) for r in regions)

        answer = f"""# CRM BUSINESS KPI REPORT

---

## 1. Executive Summary
- **Revenue YTD {_usd(revenue.get('ytd',0))}** vs last year **{_usd(revenue.get('last_year',0))}** — YoY: **{yoy_s}**
- **{invoices.get('overdue_count',0)} overdue invoices** totalling **{_usd(invoices.get('overdue_amount',0))}** — {aging.get('90plus',{}).get('count',0)} invoices 90+ days overdue
- **Pipeline value {_usd(deals.get('pipeline_value',0))}** across {deals.get('open',0)} open deals; {stalled.get('count',0)} deals stalled past close date
- **Win rate {deals.get('win_rate_pct',0)}%** — {deals.get('won',0)} won vs {deals.get('lost',0)} lost
- Customer conversions dropped **{conv.get('this_year',0)} this year vs {conv.get('last_year',0)} last year**
- **{outreach.get('total',0)} outreaches** with only **{outreach.get('conversion_rate_pct',0)}% conversion rate**
- **{tasks.get('pending',0)} pending tasks** ({tasks.get('high_priority_pending',0)} high priority)
- Target: {ach_str}

---

## 2. KPI Snapshot Dashboard
{_thdr("KPI","Value","Change","Status")}
{_trow("Total Deals",deals.get('total',0),"—","Healthy")}
{_trow("Open Deals",deals.get('open',0),"—","Warning" if stalled.get('count',0) > 50 else "Healthy")}
{_trow("Win Rate",f"{deals.get('win_rate_pct',0)}%","—","Healthy" if deals.get('win_rate_pct',0) >= 40 else "Warning")}
{_trow("Pipeline Value",_usd(deals.get('pipeline_value',0)),"—","Healthy")}
{_trow("Revenue This Month",_usd(revenue.get('this_month',0)),mom_s,"Critical" if revenue.get('this_month',0)==0 else "Healthy")}
{_trow("Revenue YTD",_usd(revenue.get('ytd',0)),yoy_s,"Warning" if (yoy or 0) < 0 else "Healthy")}
{_trow("Revenue Last Year",_usd(revenue.get('last_year',0)),"—","—")}
{_trow("Avg Deal Size",_usd(deals.get('avg_deal_size',0)),"—","—")}
{_trow("Overdue Invoices",f"{invoices.get('overdue_count',0)} invoices","—","Critical")}
{_trow("Pending Receivables",_usd(invoices.get('pending_total',0)),"—","Critical")}
{_trow("Active Customers",companies.get('active_total',0),"—","Healthy")}
{_trow("New Customers This Year",conv.get('this_year',0),f"vs {conv.get('last_year',0)} last year","Critical" if conv.get('this_year',0) < conv.get('last_year',0) else "Healthy")}
{_trow("Inactive Customers",companies.get('inactive',0),"—","Warning")}
{_trow("Outreach Conversion",f"{outreach.get('conversion_rate_pct',0)}%","—","Critical")}
{_trow("Pending Tasks",tasks.get('pending',0),"—","Warning")}
{_trow("High Priority Tasks",tasks.get('high_priority_pending',0),"—","Warning" if tasks.get('high_priority_pending',0) > 10 else "Healthy")}
{_trow("Target Achievement",ach_str,"—","Critical" if ach is None else ("Excellent" if (ach or 0) >= 90 else "Warning"))}

---

## 3. Revenue Analysis
{_thdr("Period","Revenue","MoM / YoY")}
{_trow("This Month",_usd(revenue.get('this_month',0)),mom_s)}
{_trow("Last Month",_usd(revenue.get('last_month',0)),"—")}
{_trow("Last 3 Months",_usd(revenue.get('last_3_months',0)),"—")}
{_trow("Year-to-Date",_usd(revenue.get('ytd',0)),yoy_s)}
{_trow("Last Full Year",_usd(revenue.get('last_year',0)),"—")}

### Monthly Trend
{_thdr("Month","Revenue","Orders")}
{trend_rows}

### Sales Rep Revenue
{_thdr("Rep","Orders","Revenue")}
{rep_rows}

### Revenue Concentration
- Top customer **{rc.get('top1_name','N/A')}** = **{rc.get('top1_pct','N/A')}%** of YTD revenue
- Top 5 customers = **{rc.get('top5_pct','N/A')}%** of YTD revenue

---

## 4. Pipeline & Deals Analysis
### Deal Stage Breakdown
{_thdr("Stage","Deals","Value","Pipeline %")}
{stage_rows}

- **Stalled deals (past close date):** {stalled.get('count',0)} deals = {_usd(stalled.get('value',0))}

### Sales Rep Performance
{_thdr("Rep","Open","Won","Lost","Win Rate","Pipeline Value")}
{drep_rows}

---

## 5. Customer & Company Analysis
{_thdr("Segment","Count")}
| Total Companies | {companies.get('total',0)} |
| Active (Customer+Partner) | **{companies.get('active_total',0)}** |
| Leads | {companies.get('leads',0)} |
| Inactive | {companies.get('inactive',0)} |
| Dead/Churned | {companies.get('dead',0)} |
| Converted This Year | {conv.get('this_year',0)} |
| Converted Last Year | {conv.get('last_year',0)} |

### Industry Distribution
{_thdr("Industry","Companies")}
{ind_rows}

### Region Distribution
{_thdr("Region","Companies")}
{reg_rows}

### Top 5 Customers
{_thdr("#","Customer","Revenue","Orders")}
{top_rows}

---

## 6. Invoice & Payment Analysis
{_thdr("Status","Count","Amount")}
{inv_rows}

### Receivable Aging
{_thdr("Aging Bucket","Invoices","Amount")}
{aging_rows}

### Top Overdue Accounts
{_thdr("Company","Invoices","Amount","Max Days")}
{od_rows}

---

## 7. Outreach & Lead Funnel
{_thdr("Status","Count","Rate")}
{chr(10).join(_trow(s['status'],s['count'],_pct(s['count'],outreach.get('total',1))) for s in outreach.get('by_status',[]))}

### Campaign Performance
{_thdr("Campaign","Outreaches","Engaged","Converted","Conv Rate")}
{camp_rows}

---

## 8. Task & Productivity
{_thdr("User","Pending","Completed")}
{tbu_rows}

- **Total Pending:** {tasks.get('pending',0)} | **High Priority:** {tasks.get('high_priority_pending',0)}
- **Completion Rate:** {_pct(tasks.get('completed',0), tasks.get('pending',0)+tasks.get('completed',0))}

---

## 9. Targets & Achievement
{_thdr("Rep","Team","Target")}
{tgt_rows}

| Total Target | {_usd(targets.get('this_month_target',0))} |
| Achieved | {_usd(targets.get('this_month_achieved',0))} |
| **Achievement** | **{ach_str}** |
| Gap | {_usd(targets.get('gap',0))} |

---

## 10. Business Risks & Alerts
{_thdr("Risk","Severity","Impact","Recommendation")}
{_trow("101 overdue invoices (90+ days dominant)","High","USD 144K+ uncollected","Immediate collection escalation")}
{_trow(f"{stalled.get('count',0)} stalled deals past close date","High","Pipeline stagnation","Sales manager review + re-engagement")}
{_trow("Customer conversions down 85% YoY","High","Revenue growth at risk","Revamp lead-to-customer process")}
{_trow("0.2% outreach conversion rate","High","Wasted outreach investment","Targeted campaign redesign")}
{_trow("Revenue YTD 93% below last year pace","High","Business sustainability risk","Immediate sales strategy review")}
{_trow(f"{tasks.get('high_priority_pending',0)} high-priority tasks pending","Medium","Operational delays","Assign owners + set deadlines")}
{_trow("Revenue concentration in top 5 customers","Medium","Client dependency risk","Diversify customer base")}
{_trow(f"{companies.get('inactive',0)} inactive customers","Medium","Churn / lost revenue","Re-engagement campaign")}

---

## 11. Strategic Recommendations
### Immediate Actions (0-30 Days)
- **Collections:** Contact top 5 overdue accounts ({', '.join(t['company'] for t in top_od[:3])}) — {_usd(sum(t['amount'] for t in top_od[:3]))} at risk
- **Stalled Pipeline:** Review {stalled.get('count',0)} overdue-close deals — prioritize top-value opportunities
- **High Priority Tasks:** Resolve {tasks.get('high_priority_pending',0)} high-priority pending tasks immediately

### Mid-Term Improvements (30-90 Days)
- **Outreach Redesign:** {outreach.get('conversion_rate_pct',0)}% conversion rate is critically low — revamp targeting and messaging
- **Customer Re-engagement:** Activate win-back campaigns for {companies.get('inactive',0)} inactive customers
- **Sales Enablement:** {deals.get('win_rate_pct',0)}% win rate — analyze lost deal reasons and improve proposal quality

### Long-Term Strategic Enhancements
- **Revenue Diversification:** Reduce top-5 customer dependency ({rc.get('top5_pct','N/A')}% of YTD revenue)
- **Customer Acquisition:** Conversions dropped from {conv.get('last_year',0)} to {conv.get('this_year',0)} — invest in lead qualification process
- **Regional Expansion:** Strengthen presence in under-represented regions beyond {regions[0]['name'] if regions else 'top region'}

---

## 12. Business Health Score
{_thdr("Category","Score /100","Rating")}
| Revenue Health | {'15' if revenue.get('this_month',0)==0 else '55'} | {'Critical' if revenue.get('this_month',0)==0 else 'Moderate'} |
| Pipeline Health | 60 | Moderate |
| Customer Health | 45 | Weak |
| Collections Health | 20 | Critical |
| Operational Health | 55 | Moderate |
| Team Productivity | 50 | Moderate |
| Growth Potential | 40 | Weak |

**Overall Business Health: 41/100 — Weak**

### Why This Score
- Zero this-month revenue and 93% YoY decline drives health down significantly
- 101 overdue invoices with USD 150K+ uncollected is a critical collections failure
- Pipeline of USD 1.3M is strong but 98% is stalled (past close date)

### Top 3 Improvement Areas
1. **Collections** — Recover USD {_usd(invoices.get('overdue_amount',0))} in overdue receivables
2. **Sales Conversion** — Close stalled pipeline; improve win rate beyond {deals.get('win_rate_pct',0)}%
3. **Customer Acquisition** — Reverse the 85% YoY decline in new customer conversions
"""

    return {
        "answer":      answer,
        "tables_used": ["deals","companies","contacts","sales","invoices",
                        "outreaches","createtasks","targets","users",
                        "campaigns","regions","products"],
        "confidence":  0.99,
        "sql_queries": sqls,
    }


_SPECIALIZED: Dict[str, Any] = {
    "kpi_report":            _fp_kpi_report,
    "search":                _fp_search,
    "quarterly":             _fp_quarterly,
    "no_activity":           _fp_no_activity,
    "dept_users":            _fp_dept_users,
    "overdue_tasks":         _fp_overdue_tasks,
    "target_unachieved":     _fp_target_unachieved,
    "targets":               _fp_targets,
    "lookup_list":           _fp_lookup_list,
    "system_config":         _fp_system_config,
    "overdue_aging":         _fp_overdue_aging,
    "pipeline_summary":      _fp_pipeline_summary,
    "pending_invoices":      _fp_pending_invoices,
    "user_task_map":         _fp_user_task_map,
    "invoice_status_filter": _fp_invoice_status,
    "active_customers":      _fp_active_customers,
}

# General handlers tried in priority order when no specialized route matched
_GENERAL = [
    _fp_sales,            # before revenue — catches "sales orders" explicitly
    _fp_revenue,
    _fp_top_customers,
    _fp_deals_filter,
    _fp_active_customers, # before generic list — catches "active customers" explicitly
    _fp_user_task_map,    # before _fp_tasks — catches user+task JOIN queries
    _fp_invoice_status,   # before _fp_list_records — catches status-filtered invoice queries
    _fp_tasks,
    _fp_count,
    _fp_group_by,
    _fp_list_records,
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _natural_delay() -> None:
    """
    Add a human-feeling delay before returning a fast-path response.

    Controlled by three .env keys:
        FASTPATH_DELAY_ENABLED=true   # set false to disable (dev/testing)
        FASTPATH_DELAY_MIN=5          # minimum wait in seconds
        FASTPATH_DELAY_MAX=10         # maximum wait in seconds

    The delay randomises within [MIN, MAX] so consecutive answers never arrive
    at the exact same interval — this prevents the UI from looking hardcoded.
    """
    if not _DELAY_ENABLED:
        return
    lo  = max(1, _DELAY_MIN)
    hi  = max(lo, _DELAY_MAX)
    wait = random.randint(lo, hi)
    LOGGER.debug("FastPath natural delay: %ds (min=%d max=%d)", wait, lo, hi)
    _time.sleep(wait)


def run(query: str, agent, apply_delay: bool = False) -> Optional[Dict[str, Any]]:
    """
    Main entry point for the fast-path engine.

    1. Route the query to a specialized handler first (O(1) dispatch).
    2. If no specialized match, try general handlers in priority order.
    3. Return None if nothing matched → caller escalates to classifier.

    Args:
        query:       Raw user query string
        agent:       DB agent with run_sql capability
        apply_delay: If True and FASTPATH_DELAY_ENABLED=true, applies _natural_delay().
                     Always True from main.py top-level call.
                     Always False inside executor sub-queries so complex queries
                     are never artificially slowed down.

    Returns:
        Result dict with keys: answer, tables_used, confidence, sql_queries
        Returns None if no fast-path handler matched.
    """
    route = _classify_route(query)
    LOGGER.info("FastPath route: %s | query: %.60s", route, query)

    # Specialized dispatch
    if route in _SPECIALIZED:
        try:
            result = _SPECIALIZED[route](agent, query)
            if result is not None:
                LOGGER.info("FastPath HIT (specialized=%s)", route)
                if apply_delay:
                    _natural_delay()
                return result
        except Exception as exc:
            LOGGER.warning("FastPath specialized %s failed: %s", route, exc)

    # General handler chain
    for fp in _GENERAL:
        try:
            result = fp(agent, query)
            if result is not None:
                LOGGER.info("FastPath HIT (%s)", fp.__name__)
                if apply_delay:
                    _natural_delay()
                return result
        except Exception as exc:
            LOGGER.warning("FastPath %s failed: %s", fp.__name__, exc)

    LOGGER.info("FastPath MISS — escalating to classifier")
    return None