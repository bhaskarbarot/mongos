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
import re
import time as _time
from typing import Any, Dict, List, Optional, Tuple

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

LOGGER = logging.getLogger("sql_chatbot")


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
    if re.search(r"\bquarter(ly)?\b|Q[1-4]\b", t, re.I):
        return "quarterly"
    if re.search(
        r"\bno (business|invoice|deal|activity|order|revenue)\b"
        r"|\bnot.{0,15}given business\b|\bdon.?t.{0,15}given business\b"
        r"|\bwithout.{0,15}(business|invoice)\b|\bno.{0,10}purchased\b"
        r"|\bnot.{0,10}active\b", t,
    ):
        return "no_activity"
    if re.search(
        r"\bdepartment.{0,20}(user|member|employee|name)\b"
        r"|\b(user|member|employee).{0,20}department\b", t,
    ):
        return "dept_users"
    if re.search(
        r"\b(not|haven.t|missed|overdue).{0,20}(deadline|task|due)\b"
        r"|\btask.{0,20}overdue\b|\boverdue.{0,20}task\b", t,
    ):
        return "overdue_tasks"
    if re.search(r"\b(who|which user|which person).{0,30}created\b|\bcreated by whom\b", t):
        return "chain_lookup"
    if (
        re.search(r"(?:this company|for company|company named?)\s+[A-Za-z][A-Za-z0-9\s\-&,\.]{4,}", text, re.I)
        and re.search(r"\b(invoice|billing|tax|payment|amount|total)\b", t)
    ):
        return "company_invoice"
    if re.search(r"\b(target|performance|achievement|achieved|growth potential|kpi|score)\b", t):
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
        # Invoice payment statuses
        (r"\bpaid\b",               "payment_status", "=",       "'paid'",                  "paid"),
        (r"\bunpaid\b",             "payment_status", "NOT IN",  "('paid','cancelled')",     "unpaid"),
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
    matched_stage   = None
    wants_overdue   = "overdue" in text or "stuck" in text
    for patterns, stage in STAGE_MAP:
        if any(p in text for p in patterns):
            matched_stage = stage
            break

    if not matched_stage and not wants_overdue:
        return None

    table_names = get_table_names(agent)
    table       = next((t for t in table_names if t.lower() == "deals"), None)
    if not table:
        return None

    fields       = get_document_fields(agent, table)
    stage_field  = next((f for f in fields if f.lower() == "stage"), None) or \
                   next((f for f in fields if "stage" in f.lower()), "stage")
    name_field   = next((f for f in fields if f.lower() == "name"), "name")
    close_field  = next((f for f in fields if "close" in f.lower()), None)
    amount_field = next((f for f in fields if f.lower() in ["grand_total", "grand_total_in_usd"]), None)
    deleted_field = next((f for f in fields if f.lower() == "deleted"), None)

    where_parts = []
    if deleted_field:
        where_parts.append(f"COALESCE(document->>'{deleted_field}', 'false') != 'true'")
    if matched_stage:
        where_parts.append(f"document->>'{stage_field}' ILIKE '{matched_stage}'")
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

    where          = "WHERE " + " AND ".join(f"({p})" for p in where_parts) if where_parts else ""
    select_parts   = [f"document->>'{name_field}' AS name", f"document->>'{stage_field}' AS stage"]
    if amount_field:
        select_parts.append(f"NULLIF(document->>'{amount_field}','')::numeric AS amount")
    if close_field:
        select_parts.append(f"document->>'{close_field}' AS close_date")

    sql  = f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} ORDER BY updated_at DESC LIMIT 50".strip()
    _res = run_sql(agent, sql)
    label = matched_stage or ("overdue" if wants_overdue else "filtered")
    if _res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [sql]}
    rows = _res.rows
    if not rows:
        return {
            "answer":      f"No **{label}** deals found.",
            "tables_used": [table],
            "confidence":  0.9,
            "sql_queries": [sql],
        }

    headers = ["Name", "Stage"] + (["Amount"] if amount_field else []) + (["Close Date"] if close_field else [])
    lines   = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:20]:
        vals = ["—" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > 20:
        lines.append(f"_…and {len(rows)-20} more_")

    return {
        "answer":      f"**{len(rows)} {label} deals:**\n\n" + "\n".join(lines),
        "tables_used": [table],
        "confidence":  0.97,
        "sql_queries": [sql],
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
    for t in table_names:
        tl = t.lower()
        if re.search(rf"\b{re.escape(tl)}\b", text) or (
            tl.endswith("s") and re.search(rf"\b{re.escape(tl[:-1])}\b", text)
        ):
            table = t
            break
    if not table:
        return None

    fields       = get_document_fields(agent, table)
    name_expr    = REGISTRY.display_name_expr(table, fields)
    select_parts = [f"{name_expr} AS name"]
    for role, alias in [
        ("identifier", "ref"), ("status", "status"),
        ("amount", "amount"), ("date", "date"), ("email", "email"),
    ]:
        f = REGISTRY.get(table, role, fields)
        if f and f"document->>'{f}'" not in name_expr:
            select_parts.append(f"document->>'{f}' AS {alias}")
        if len(select_parts) >= 5:
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
            # invoice payment_status values
            r"\bpaid\b":                  ("payment_status", "paid"),
            r"\bunpaid\b":                ("payment_status", None),      # NOT IN paid/cancelled
            r"\bconfirmed\b":             ("payment_status", "confirmed"),
            r"\bdraft\b":                 ("payment_status", "draft"),
            r"\bcancell?ed\b":            ("payment_status", "cancelled"),
            r"\bpartial[\s_]?payment\b":  ("payment_status", "partial_payment"),
            # generic status values
            r"\bpending\b":               ("status", "Pending"),
            r"\bcompleted?\b":            ("status", "Completed"),
            r"\bopen\b":                  ("status", "Open"),
        }
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

    # top N (explicit number) or bare "top" (→ 10) or singular noun (→ 1)
    top_match   = re.search(r"\btop\s+(\d+)\b", text)
    bare_top    = re.search(r"\btop\b", text) and not top_match       # "top invoice" / "top deal"
    singular    = re.search(r"\b(the\s+)?(top|best|highest|biggest)\s+\w+\b", text) and \
                  not re.search(r"\b(top|best)\s+\d+\b|\bplural\b", text)
    n_match     = re.search(r"\blast\s+(\d+)\b|(\d+)\s+(?:record|row|result|item)s?\b", text)
    first_match = re.search(r"\b(1st|first|oldest|earliest)\b", text)

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

    if first_match:
        limit = 1
        order = (f"NULLIF(document->>'{doc_date_field}','')::timestamptz ASC NULLS LAST"
                 if doc_date_field else "updated_at ASC")
    elif top_match:
        # "top N …" — N always sets the limit; sort direction sets order
        limit = min(int(top_match.group(1)), 100)
        order = (_amount_asc()  if sort_asc_kw  and amount_field else
                 _amount_desc() if has_amount_sort               else
                 _amount_desc() if amount_field                   else _date_desc())
    elif bare_top and re.search(r"\b(invoice|deal|sales|order|customer)\b", text):
        limit = 10
        order = _amount_asc() if sort_asc_kw and amount_field else \
                _amount_desc() if amount_field else _date_desc()
    elif sort_asc_kw and amount_field:
        limit = 50
        order = _amount_asc()
    elif sort_desc_kw and amount_field and \
         re.search(r"\b(biggest|largest|most|highest|recent|latest|newest|descend)\b", text):
        limit = 50
        order = _amount_desc()
    elif n_match:
        num   = n_match.group(1) or n_match.group(2)
        limit = min(int(num), 100) if num else 20
        order = _date_desc()
    elif re.search(r"\ball\b", text):
        limit = 100
        order = _date_desc()
    elif wants_details:
        limit = 100
        order = _date_desc()
    else:
        limit = 20
        order = _date_desc()

    count_sql   = f'SELECT COUNT(*)::int FROM "{table}" {where}'.strip()
    _cnt_res    = run_sql(agent, count_sql)
    if _cnt_res.error:
        return {"answer": "Unable to retrieve data at this time. Please try again.",
                "tables_used": [table], "confidence": 0.0, "sql_queries": [count_sql]}
    total_rows  = _cnt_res.rows
    total_count = coerce_number(total_rows[0][0] if total_rows else 0)

    sql      = f"SELECT {', '.join(select_parts)} FROM \"{table}\" {where} ORDER BY {order} LIMIT {limit}".strip()
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


def _fp_targets(agent, user_query: str) -> Optional[Dict]:
    text = normalize_text(user_query)
    if not re.search(r"\b(target|performance|achievement|achieved|growth potential|kpi|score)\b", text):
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
    user_m = re.search(r"(?:for|by|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", user_query, re.I)
    if user_m:
        uname   = sanitize_sql_value(user_m.group(1).strip())
        _GENERIC = {"all users", "all user", "every user", "each user", "this user", "the user"}
        if uname.lower() not in _GENERIC:
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

    links    = _SCHEMA_LINKS
    has_co   = links.get("invoices", {}).get("company") == "companies"
    co_join  = (
        'LEFT JOIN "companies" c ON c.document->>\'_id\' = i.document->>\'company\''
        if has_co else ""
    )
    co_col   = (
        "COALESCE(c.document->>'companyName', i.document->>'company', 'Unknown')"
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

    links   = _SCHEMA_LINKS
    has_co  = links.get("invoices", {}).get("company") == "companies"
    co_join = (
        'LEFT JOIN "companies" c ON c.document->>\'_id\' = i.document->>\'company\''
        if has_co else ""
    )
    co_col  = (
        "COALESCE(c.document->>'companyName', i.document->>'company', 'Unknown')"
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


_SPECIALIZED: Dict[str, Any] = {
    "search":           _fp_search,
    "quarterly":        _fp_quarterly,
    "no_activity":      _fp_no_activity,
    "dept_users":       _fp_dept_users,
    "overdue_tasks":    _fp_overdue_tasks,
    "targets":          _fp_targets,
    "lookup_list":      _fp_lookup_list,
    "system_config":    _fp_system_config,
    "overdue_aging":    _fp_overdue_aging,
    "pipeline_summary": _fp_pipeline_summary,
    "pending_invoices": _fp_pending_invoices,
}

# General handlers tried in priority order when no specialized route matched
_GENERAL = [
    _fp_sales,          # before revenue — catches "sales orders" explicitly
    _fp_revenue,
    _fp_top_customers,
    _fp_deals_filter,
    _fp_tasks,
    _fp_count,
    _fp_group_by,
    _fp_list_records,
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """
    Main entry point for the fast-path engine.

    1. Route the query to a specialized handler first (O(1) dispatch).
    2. If no specialized match, try general handlers in priority order.
    3. Return None if nothing matched → caller escalates to classifier.

    Target latency: <300ms for cached schema, <2s on first run.
    Never raises — exceptions in individual handlers are caught and logged.

    Args:
        query: Raw user query string
        agent: DB agent with run_sql capability

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
                return result
        except Exception as exc:
            LOGGER.warning("FastPath specialized %s failed: %s", route, exc)

    # General handler chain
    for fp in _GENERAL:
        try:
            result = fp(agent, query)
            if result is not None:
                LOGGER.info("FastPath HIT (%s)", fp.__name__)
                return result
        except Exception as exc:
            LOGGER.warning("FastPath %s failed: %s", fp.__name__, exc)

    LOGGER.info("FastPath MISS — escalating to classifier")
    return None