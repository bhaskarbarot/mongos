"""db_schema.py — Live database schema introspection, zero hardcoding.

Reads real table/column metadata directly from PostgreSQL information_schema.
Cached per-table for 10 minutes. Adapts automatically when the DB changes.

Key features:
  - Real column names with correct casing (no guessing)
  - Auto-detects soft-delete column (deleted vs isDeleted vs none)
  - Discovers enum values from actual data for text columns
  - Keyword-based table relevance scoring for dynamic schema selection
  - Column name correction: given a bad column name → finds the real one
  - FK inference from column name patterns

Public API:
  build_schema_for_query(query, hint_tables=[]) -> str
  get_column_map(table_name) -> {lowercase_name: actual_name}
  get_all_crm_tables() -> list[str]
  score_tables(query) -> list[(table, score)]
"""
from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Optional, Tuple

LOGGER_NAME = "sql_chatbot"

import logging
LOGGER = logging.getLogger(LOGGER_NAME)

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_TTL   = 600   # 10 minutes
_cache_lock  = threading.Lock()
_col_cache:  Dict[str, Tuple[list, float]] = {}   # table → (columns, ts)
_enum_cache: Dict[str, Tuple[list, float]] = {}   # "table.col" → (values, ts)
_table_cache: Tuple[Optional[list], float] = (None, 0)

# ── FK inference: column name → referenced table ──────────────────────────────
_FK_MAP = {
    "owner": "users", "userid": "users", "createdby": "users",
    "updatedby": "users", "salesowner": "users", "assignedto": "users",
    "companyowner": "users", "contactowner": "users", "invoiceowner": "users",
    "business_analyst": "users", "reporting_manager": "users",
    "company": "companies", "companyid": "companies",
    "department": "departments",
    "region": "regions",
    "campaign": "campaigns",
    "vendor": "vendors",
    "contact": "contacts",
    "source": "sources",
    "product_type": "projecttypes",
    "outreachid": "outreaches",
    "dealsid": "deals", "salesid": "sales",
    "invoiceid": "invoices", "contactid": "contacts",
    "dealid": "deals",
}

# ── Tables that have business data (exclude system/utility tables) ─────────────
_CRM_TABLES = [
    "deals", "invoices", "sales", "companies", "contacts", "users",
    "createtasks", "targets", "outreaches", "vendors", "bills",
    "departments", "regions", "products", "sources", "technologies",
    "taxes", "categories", "campaigns", "lead_statuses", "lifecycle_stages",
    "dealstagesettings", "payments", "emails", "commonnotes", "activitylogs",
    "countryregions", "projecttypes", "notes", "publicleads", "meetings",
]

# ── Keyword → table relevance map (used for dynamic selection) ────────────────
_KEYWORDS: Dict[str, List[str]] = {
    "deals":             ["deal","deals","pipeline","stage","won","lost","close","open",
                          "negotiation","contract","opportunity","proposal","quote","bid"],
    "invoices":          ["invoice","invoices","payment","revenue","paid","unpaid","overdue",
                          "billing","amount","inr","usd","gbp","currency","due","aging","receipt"],
    "sales":             ["sales","order","confirm","so","sale","salesowner","confirmed","order"],
    "companies":         ["company","companies","customer","client","account","business","firm",
                          "organization","industry","lifecycle","health","active","partner"],
    "contacts":          ["contact","contacts","person","people","firstname","lastname",
                          "job","title","mobile","phone","email"],
    "users":             ["user","users","rep","owner","employee","staff","assigned","manager",
                          "team","active","admin","people","salesperson","sales rep"],
    "createtasks":       ["task","tasks","pending","overdue","priority","to-do","todo",
                          "due","assign","high priority","medium","productivity"],
    "targets":           ["target","targets","achievement","quota","kpi","goal","achieved",
                          "performance","gap","percentage","attain"],
    "meetings":          ["meeting","meetings","scheduled","calendar","call","appointment","event"],
    "outreaches":        ["outreach","outreaches","prospect","contacted","campaign","cold",
                          "not contacted","funnel","conversion","converted"],
    "departments":       ["department","departments","team","division","group","dept"],
    "regions":           ["region","regions","geography","location","area","zone"],
    "vendors":           ["vendor","vendors","supplier","suppliers"],
    "bills":             ["bill","bills","payable","expense","vendor bill","vendor invoice"],
    "products":          ["product","products","item","service","sku","price","active product"],
    "sources":           ["source","sources","lead source","channel","origin","where from"],
    "technologies":      ["technology","technologies","tech","stack","platform","framework"],
    "taxes":             ["tax","taxes","gst","vat","rate","percent"],
    "categories":        ["category","categories","type","classification"],
    "campaigns":         ["campaign","campaigns","marketing"],
    "lead_statuses":     ["lead status","leadstatus","qualified","unqualified"],
    "lifecycle_stages":  ["lifecycle","lifecycle stage","life cycle"],
    "dealstagesettings": ["deal stage","pipeline stage"],
    "payments":          ["payment method","payment mode","bank","wire","transfer"],
    "emails":            ["email","emails","message","subject","sent","inbox","mail"],
    "commonnotes":       ["note","notes","comment","pin","pinned","remark","activity note"],
    "activitylogs":      ["activity","log","history","audit","track","change","who did"],
    "countryregions":    ["country","countries"],
    "projecttypes":      ["project type","engagement","dedicated","fixed price","t&m"],
    "notes":             ["outreach note","outreach activity"],
    "publicleads":       ["public lead","web lead","form lead","inbound","website lead"],
    "meetings":          ["meeting","meetings","calendar","scheduled","call"],
}

# ── Columns to always exclude from schema (internal/noisy) ───────────────────
_EXCLUDE_COLS = {
    "_synced_at", "__v", "updated_at", "updatedAt", "document",
    "tokens", "password", "googleAccessToken", "googleAccessEmail",
}


def _get_conn():
    from config import settings
    import psycopg2
    return psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db, connect_timeout=5,
    )


def _run(sql: str, params=None) -> list:
    """Run a quick introspection query, return list of rows."""
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as exc:
        LOGGER.debug("db_schema query error: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TABLE LIST
# ══════════════════════════════════════════════════════════════════════════════

def get_all_crm_tables() -> List[str]:
    """Return the known CRM table list (filtered to only those that exist in DB)."""
    global _table_cache
    with _cache_lock:
        tables, ts = _table_cache
        if tables and (time.monotonic() - ts) < _CACHE_TTL:
            return tables

    rows = _run("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' AND table_type='BASE TABLE'
    """)
    existing = {r[0] for r in rows}
    result   = [t for t in _CRM_TABLES if t in existing]

    with _cache_lock:
        _table_cache = (result, time.monotonic())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN METADATA
# ══════════════════════════════════════════════════════════════════════════════

def _get_columns(table: str) -> List[Tuple[str, str, str]]:
    """Return [(col_name, data_type, is_nullable)] from information_schema."""
    with _cache_lock:
        entry = _col_cache.get(table)
        if entry and (time.monotonic() - entry[1]) < _CACHE_TTL:
            return entry[0]

    rows = _run("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position
    """, (table,))

    cols = [
        (r[0], r[1], r[2])
        for r in rows
        if r[0] not in _EXCLUDE_COLS
    ]
    with _cache_lock:
        _col_cache[table] = (cols, time.monotonic())
    return cols


def _needs_quote(col_name: str) -> bool:
    """True if the column name contains uppercase letters (needs double quotes in SQL)."""
    return any(c.isupper() for c in col_name)


def _short_type(pg_type: str) -> str:
    if "int"  in pg_type: return "INT"
    if "numeric" in pg_type or "float" in pg_type or "double" in pg_type: return "NUM"
    if "bool" in pg_type:  return "BOOL"
    if "json" in pg_type:  return "JSONB"
    if "timestamp" in pg_type or "date" in pg_type: return "TIMESTAMP"
    return "TEXT"


def get_column_map(table: str) -> Dict[str, str]:
    """Return {lowercase_col_name: actual_col_name} for correcting wrong column references."""
    cols = _get_columns(table)
    return {c[0].lower(): c[0] for c in cols}


# ══════════════════════════════════════════════════════════════════════════════
# ENUM VALUE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

_ENUM_COLUMNS = {
    "deals":             ["stage", "type", "currency"],
    "invoices":          ["payment_status", "approval_status", "currency"],
    "sales":             ["status", "currency"],
    "companies":         ["lifecycleStage", "leadStatus", "industry"],
    "contacts":          ["lifecycleStage", "leadStatus"],
    "createtasks":       ["status", "priority"],
    "outreaches":        ["status", "leadStatus", "priority"],
    "vendors":           ["stage"],
    "bills":             ["status", "billType"],
    "dealstagesettings": ["dealStageName"],
    "lead_statuses":     ["name"],
    "lifecycle_stages":  ["name"],
    "departments":       ["name"],
    "regions":           ["regionName"],
    "sources":           ["sourceName"],
    "projecttypes":      ["name"],
    "payments":          ["payment_name"],
}


def _get_enum_values(table: str, col: str, limit: int = 20) -> List[str]:
    key = f"{table}.{col}"
    with _cache_lock:
        entry = _enum_cache.get(key)
        if entry and (time.monotonic() - entry[1]) < _CACHE_TTL:
            return entry[0]

    # Need double quotes for mixed-case columns
    col_ref = f'"{col}"' if _needs_quote(col) else col
    rows = _run(f"""
        SELECT DISTINCT {col_ref}
        FROM "{table}"
        WHERE {col_ref} IS NOT NULL AND {col_ref} != ''
        LIMIT {limit}
    """)
    vals = sorted([r[0] for r in rows if r[0]])[:12]  # cap at 12
    with _cache_lock:
        _enum_cache[key] = (vals, time.monotonic())
    return vals


# ══════════════════════════════════════════════════════════════════════════════
# TABLE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_tables(query: str) -> List[Tuple[str, float]]:
    """Return list of (table, score) sorted by relevance to the query."""
    q = query.lower()
    all_tables = get_all_crm_tables()
    scored = []
    for tbl in all_tables:
        kws = _KEYWORDS.get(tbl, [])
        score = sum(1.5 if kw in q else 0 for kw in kws)
        scored.append((tbl, score))
    return sorted(scored, key=lambda x: -x[1])


# ══════════════════════════════════════════════════════════════════════════════
# COMPACT SCHEMA BUILDER  (dynamic, from live DB)
# ══════════════════════════════════════════════════════════════════════════════

# Notes added per table when building schema
_TABLE_NOTES = {
    "deals": [
        "-- Open: \"dealWonAt\" IS NULL AND \"dealLostAt\" IS NULL AND NOT deleted",
        "-- Won: \"dealWonAt\" IS NOT NULL AND NOT deleted | Lost: \"dealLostAt\" IS NOT NULL",
        "-- TEXT date cols: NULLIF(col,'')::timestamptz  e.g. NULLIF(\"closeDate\",'')::timestamptz",
        "-- FK join companies: LEFT JOIN \"companies\" c ON c._id = d.company  (column='company' NOT 'companyId')",
        "-- ⚠ deals has NO 'items' or 'product' JSONB column — only sales and invoices have items JSONB",
        "-- d.type is a deal category TEXT label (e.g. 'Cross-sell','Upsell','New Business') — NOT a product FK",
        "-- For product+deal-type analysis: SELECT d.type, COUNT(*), SUM(d.grand_total) FROM \"deals\" d GROUP BY d.type",
    ],
    "invoices": [
        "-- Revenue: SUM(grand_total) WHERE payment_status='paid' AND NOT deleted",
        "-- Overdue: NULLIF(due_date,'')::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')",
        "-- Year: EXTRACT(YEAR FROM NULLIF(payment_date,'')::timestamptz) = EXTRACT(YEAR FROM CURRENT_DATE)",
        "-- ⚠ FK join companies: LEFT JOIN \"companies\" c ON c._id = i.company  (column='company' NOT 'companyId'!)",
        "-- ⚠ TEXT date cols (invoice_date, due_date, payment_date): NULLIF(col,'')::timestamptz — NEVER raw cast",
        "-- ⚠ companies table has NO 'currency' column — currency is on invoices (i.currency)",
        "-- ⚠ NO productId column on invoices — products are not directly joinable via invoices",
    ],
    "sales": [
        "-- Revenue: SUM(grand_total) WHERE status='Confirm' AND NOT deleted",
        "-- JOIN users: LEFT JOIN \"users\" u ON u._id = s.\"salesOwner\"",
        "-- ⚠ CORRECT date EXTRACT: EXTRACT(YEAR FROM NULLIF(s.sales_date,'')::timestamptz) = t.year",
        "-- ✗ WRONG:                EXTRACT(YEAR FROM NULLIF(s.sales_date,''))::numeric  (cast INSIDE NULLIF!)",
        "-- ⚠ NO 'closeDate' column on sales — date col is sales_date (TEXT), cast: NULLIF(sales_date,'')::timestamptz",
        "-- items column is JSONB — line items: jsonb_array_elements(s.items)->>'name' AS product_name",
        "-- Target join: LEFT JOIN targets t ON t.\"userId\"=s.\"salesOwner\" AND EXTRACT(YEAR FROM NULLIF(s.sales_date,'')::timestamptz)=t.year AND EXTRACT(MONTH FROM NULLIF(s.sales_date,'')::timestamptz)=t.month",
    ],
    "targets": [
        "-- ⚠ year and month are NUMERIC INTEGERS — NEVER cast to timestamptz!",
        "-- This year: WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE)",
        "-- month is 1-indexed (Jan=1). Achieved = JOIN sales ON salesOwner=userId AND sales year+month = target year+month",
    ],
    "outreaches": [
        "-- ⚠ soft delete is \"isDeleted\" NOT deleted! → WHERE NOT \"isDeleted\"",
        "-- ⚠ NO 'leadId' column — outreaches link to contacts via email or assignedTo field",
        "-- Leads not contacted: contacts WHERE lifecycleStage='Lead' AND _id NOT IN (outreach assignedTo subquery)",
    ],
    "companies": [
        "-- ⚠ FK source: c.source (NOT c.sourceId, NOT c.source_id) → LEFT JOIN \"sources\" s ON s._id = c.source",
        "-- ⚠ NO 'currency' column on companies — currency lives on invoices/deals/sales tables",
    ],
    "vendors": [
        "-- ⚠ NO deleted column on vendors — NEVER add WHERE NOT deleted for vendors!",
        "-- Count all: SELECT COUNT(*) FROM \"vendors\"  (no filter needed)",
    ],
    "createtasks": [
        "-- ⚠ table name is 'createtasks' NOT 'tasks'",
        "-- Title column is \"Task\" (capital T)",
        "-- Overdue: NULLIF(due_date,'')::timestamptz < NOW() AND status!='Completed' AND NOT deleted",
        "-- FK companyId IS correct on createtasks (unlike invoices which uses 'company')",
    ],
    "products": [
        "-- Standalone product catalog — no direct FK from deals or sales",
        "-- ✗ NO sales.product or deals.product or invoices.productId column — those do NOT exist",
        "-- Line items embedded as JSONB in sales.items and invoices.items",
        "-- Top products from sales: SELECT jsonb_array_elements(s.items)->>'name' AS product_name, COUNT(*) FROM \"sales\" s WHERE s.status='Confirm' AND NOT s.deleted GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
        "-- Direct product list: SELECT name, unit_cost, currency FROM \"products\" WHERE \"isActive\"=true ORDER BY unit_cost DESC",
    ],
    "activitylogs": [
        "-- ⚠ createdAt is TEXT — ALWAYS cast: NULLIF(al.\"createdAt\",'')::timestamptz",
        "-- ✗ WRONG: al.\"createdAt\" > (CURRENT_DATE - INTERVAL '7 day')  — TEXT vs timestamp fails!",
        "-- ✓ CORRECT: NULLIF(al.\"createdAt\",'')::timestamptz > NOW() - INTERVAL '7 days'",
        "-- Filter recent activity: WHERE NULLIF(al.\"createdAt\",'')::timestamptz >= NOW() - INTERVAL '7 days'",
        "-- Deals that changed stage: JOIN activitylogs al ON al.\"recordId\" = d._id AND al.module='deals'",
        "-- NO deleted column on activitylogs — omit soft-delete filter",
    ],
}

# Soft delete info per table (from real DB introspection)
_SOFT_DELETE = {
    "deals":            "NOT deleted",
    "invoices":         "NOT deleted",
    "sales":            "NOT deleted",
    "companies":        "NOT deleted",
    "contacts":         "NOT deleted",
    "createtasks":      "NOT deleted",
    "dealstagesettings":"NOT deleted",
    "outreaches":       'NOT "isDeleted"',
    "vendors":          None,  # no soft delete
    "users":            None,
    "departments":      None,
    "regions":          None,
    "products":         None,
    "targets":          None,
    "meetings":         None,
    "campaigns":        None,
    "bills":            None,
}

# Which cols to show for each table (top-priority ones, others trimmed for token budget)
_PRIORITY_COLS = {
    "deals":    ["_id","name","stage","owner","company","deleted","grand_total_in_usd",
                 "currency","type","closeDate","dealWonAt","dealLostAt","createdAt"],
    "invoices": ["_id","invoice_number","payment_status","grandtotal_in_usd","grand_total",
                 "currency","company","invoice_date","due_date","payment_date","deleted",
                 "companyName","createdBy"],
    "sales":    ["_id","sales_number","status","salesOwner","company","grand_total_in_usd",
                 "currency","sales_date","deleted","createdAt"],
    "companies":["_id","companyName","deleted","companyOwner","industry","country","region",
                 "lifecycleStage","leadStatus","createdAt","leadWonAt","clientHealth","userType"],
    "contacts": ["_id","firstName","lastName","email","jobTitle","phoneNumber",
                 "lifecycleStage","leadStatus","contactOwner","company","deleted","createdAt"],
    "users":    ["_id","name","email","department","isActive","isAdmin","createdAt"],
    "createtasks":["_id","Task","status","priority","createdBy","due_date","companyId",
                   "dealsId","invoiceId","salesId","deleted","createdAt"],
    "targets":  ["_id","userId","month","year","targetInUSD","teamName","createdAt"],
    "outreaches":["_id","name","email","status","leadStatus","campaign","region",
                  "assignedTo","isDeleted","createdAt"],
    "vendors":  ["_id","companyName","email","phone","currency","stage","country","createdAt"],
    "bills":    ["_id","vendor","systemBillNo","billDate","dueDate","status",
                 "netPayableAmount","subtotal","gstPercent","billType","createdAt"],
    "departments":["_id","name","createdAt"],
}


def _build_table_schema_line(table: str) -> str:
    """Build one compact schema line for a table from live DB data."""
    cols   = _get_columns(table)
    col_map = {c[0]: (c[1], c[2]) for c in cols}

    # Choose which columns to show
    priority = _PRIORITY_COLS.get(table)
    if priority:
        show_cols = [c for c in priority if c in col_map]
    else:
        # Show first 12 non-jsonb, non-internal columns
        show_cols = [
            c[0] for c in cols
            if c[0] not in _EXCLUDE_COLS
            and _short_type(c[1]) not in ("JSONB",)
        ][:12]

    # Build column list
    parts = []
    for col in show_cols:
        if col not in col_map:
            continue
        pg_type, nullable = col_map[col]
        short   = _short_type(pg_type)
        quoted  = f'"{col}"' if _needs_quote(col) else col

        # FK reference
        fk = _FK_MAP.get(col.lower())
        if fk and fk in _CRM_TABLES:
            parts.append(f"{quoted}→{fk}")
            continue

        # Type annotation (only if not TEXT, since TEXT is default)
        if short == "TEXT":
            parts.append(quoted)
        else:
            parts.append(f"{quoted} {short}")

    # Enum values for important columns
    enum_hints = []
    for ecol in _ENUM_COLUMNS.get(table, []):
        if ecol not in col_map:
            continue
        vals = _get_enum_values(table, ecol)
        if vals:
            quoted = f'"{ecol}"' if _needs_quote(ecol) else ecol
            vals_str = "|".join(repr(v) for v in vals[:8])
            enum_hints.append(f"  {quoted}: {vals_str}")

    # Soft delete
    sd = _SOFT_DELETE.get(table, "NOT deleted")  # default assume deleted col exists

    # Check actual columns to override
    actual_col_names = {c[0] for c in cols}
    if "deleted" not in actual_col_names and "isDeleted" not in actual_col_names:
        sd = None   # no soft delete column
    elif "isDeleted" in actual_col_names:
        sd = 'NOT "isDeleted"'
    elif "deleted" in actual_col_names:
        sd = "NOT deleted"

    # Header
    sd_hint = f" | filter: {sd}" if sd else " | NO soft-delete column"
    line = f'{table}({", ".join(parts)}){sd_hint}'

    # Add enum hints and notes
    lines = [line] + enum_hints
    for note in _TABLE_NOTES.get(table, []):
        lines.append(note)

    return "\n".join(lines)


def build_schema_for_query(
    query:       str,
    hint_tables: List[str] = [],
    max_tables:  int = 8,
) -> str:
    """Build compact live schema for the query — only relevant tables.

    Returns a compact multi-line schema string built from real DB metadata.
    Token estimate: ~80-120 tokens per table × max 8 tables = ~700-960 tokens max.
    """
    q_lower = query.lower()
    scores  = score_tables(query)

    # Start with hint tables (forced-include)
    selected = list(dict.fromkeys(hint_tables))  # preserve order, dedup

    # Join-requiring keywords → ensure anchor tables
    needs_join = any(w in q_lower for w in [
        "with", "by", "owner", "rep", "user", "per", "department",
        "team", "leaderboard", "360", "health", "each", "all"
    ])
    if needs_join and "users" not in selected:
        selected.append("users")

    # Add top-scoring tables
    for tbl, score in scores:
        if score > 0 and tbl not in selected:
            selected.append(tbl)
        if len(selected) >= max_tables:
            break

    # Fallback: at least one table
    if not selected:
        selected = ["deals"]

    # Build lines
    header = (
        "-- SCHEMA (live from DB) — CRITICAL RULES:\n"
        "-- 1. Always alias tables in JOINs. Prefix ALL cols: d.name NOT name\n"
        "-- 2. Mixed-case columns NEED double quotes: s.\"salesOwner\" NOT s.salesOwner\n"
        "-- 3. Date TEXT cols: NULLIF(col,'')::timestamptz  Year: EXTRACT(YEAR FROM NULLIF(col,'')::timestamptz)=2026\n"
        "-- 4. targets.year / targets.month are NUMERIC — never cast to timestamp\n"
    )
    table_lines = []
    for tbl in selected:
        try:
            table_lines.append(_build_table_schema_line(tbl))
        except Exception as exc:
            LOGGER.debug("Schema build error for %s: %s", tbl, exc)

    return header + "\n\n".join(table_lines)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN CORRECTION  (for self-healing SQL repair)
# ══════════════════════════════════════════════════════════════════════════════

def find_correct_column(bad_col: str, tables: List[str]) -> Optional[str]:
    """
    Given a bad column reference (e.g. 'salesowner'), find the correct
    column name with proper casing across all given tables.
    Returns the correctly quoted form, or None if not found.
    """
    bad_lower = bad_col.lower().strip('"')
    for tbl in tables:
        col_map = get_column_map(tbl)
        if bad_lower in col_map:
            actual = col_map[bad_lower]
            return f'"{actual}"' if _needs_quote(actual) else actual
    return None
