"""schema_compact.py — Dynamic schema compression for SQL generation.

Reduces token cost from ~3500 (full schema) to ~300-700 (relevant tables only).

All schema content is now read LIVE from the DB via pipeline/db_schema.py —
zero hardcoded column names, table names, or enum values.

Strategy:
  1. Score each table by keyword overlap with the query (keyword map preserved from before)
  2. Include anchor tables (users, companies) when JOINs are likely
  3. Build compact schema lines from live DB column metadata
  4. Cache for 10 minutes (db_schema handles caching internally)

Public API (unchanged):
  get_compact_schema(query: str, extra_tables: list = []) -> str
  COMPACT_SYSTEM_PROMPT  (use instead of full _SYSTEM_SQL_EXPERT)
"""
from __future__ import annotations

from typing import Dict, List, Set

from pipeline.db_schema import (
    build_schema_for_query,
    score_tables,
    get_all_crm_tables,
)


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD → TABLE RELEVANCE MAP
# (preserved for fast scoring without hitting the DB)
# ══════════════════════════════════════════════════════════════════════════════

_TABLE_KEYWORDS: Dict[str, List[str]] = {
    "deals":            ["deal","deals","pipeline","stage","won","lost","close","open","negotiation",
                         "contract","opportunity","prospect","bid","quote","proposal","opportunity"],
    "invoices":         ["invoice","invoices","payment","revenue","paid","unpaid","overdue","billing",
                         "amount","inr","usd","gbp","currency","due","receipt","collect","aging"],
    "sales":            ["sales","order","confirm","so","sale","salesowner","rep","confirmed"],
    "companies":        ["company","companies","customer","client","account","business","firm",
                         "organization","industry","lifecycle","health","active","lead","partner"],
    "contacts":         ["contact","contacts","person","people","firstname","lastname","job",
                         "title","mobile","phone"],
    "users":            ["user","users","rep","owner","employee","staff","assigned","manager",
                         "team","active","admin","sales rep","people"],
    "createtasks":      ["task","tasks","pending","overdue","priority","to-do","todo","due",
                         "assign","high priority","medium","low","productivity"],
    "targets":          ["target","targets","achievement","quota","kpi","goal","achieved",
                         "performance","gap","percentage","incentive","attain"],
    "meetings":         ["meeting","meetings","scheduled","calendar","call","appointment","event"],
    "outreaches":       ["outreach","outreaches","prospect","contacted","campaign","cold",
                         "not contacted","funnel","conversion","converted"],
    "departments":      ["department","departments","team","division","group","dept"],
    "regions":          ["region","regions","geography","location","area","zone"],
    "vendors":          ["vendor","vendors","supplier","suppliers"],
    "bills":            ["bill","bills","payable","expense","vendor bill","invoice to vendor"],
    "products":         ["product","products","item","service","sku","price","unit cost","active product"],
    "sources":          ["source","sources","lead source","channel","origin","where from"],
    "technologies":     ["technology","technologies","tech","stack","platform","framework"],
    "taxes":            ["tax","taxes","gst","vat","rate","percent","18%"],
    "categories":       ["category","categories","type","classification"],
    "campaigns":        ["campaign","campaigns","marketing","campaign name"],
    "lead_statuses":    ["lead status","leadstatus","qualification","qualified","unqualified"],
    "lifecycle_stages": ["lifecycle","lifecycle stage","life cycle","stage name"],
    "dealstagesettings":["deal stage","pipeline stage","stage setting"],
    "payments":         ["payment method","payment mode","bank","wire","transfer"],
    "emails":           ["email","emails","message","subject","sent","inbox","mail"],
    "commonnotes":      ["note","notes","comment","pin","pinned","remark"],
    "activitylogs":     ["activity","log","history","audit","track","change","who did"],
    "countryregions":   ["country","countries","country region"],
    "projecttypes":     ["project type","engagement","dedicated","fixed price","t&m"],
    "notes":            ["outreach note","outreach activity","contact method"],
    "publicleads":      ["public lead","web lead","form lead","inbound","website lead"],
    "meetings":         ["meeting","meetings","calendar","scheduled","call"],
}

# Anchor tables: always included when JOINs are likely
_ANCHOR_TABLES: Set[str] = {"users", "companies"}

# All anchor tables for complex/leaderboard/360 queries
_COMPLEX_ANCHORS: Set[str] = {"deals", "invoices", "sales", "users", "companies", "createtasks"}


def _score_table(query_lower: str, table: str) -> float:
    """Return keyword hit score for a table given the lowercased query."""
    kws = _TABLE_KEYWORDS.get(table, [])
    return sum(1.0 for kw in kws if kw in query_lower)


def get_compact_schema(
    query:        str,
    extra_tables: List[str] = [],
    max_tables:   int = 9,
) -> str:
    """Return compact LIVE schema string for tables relevant to the query.

    All content comes from the real DB (via db_schema.build_schema_for_query).
    Keyword scoring selects which tables to include; db_schema provides
    the actual column names, types, enum values, and soft-delete rules.

    Args:
        query:        Natural language query from the user.
        extra_tables: Additional table names to force-include.
        max_tables:   Hard cap on table count (limits token usage).

    Returns:
        Multi-line compact schema string, ~300-700 tokens.
    """
    q = query.lower()

    # Score every known table
    all_crm = get_all_crm_tables()
    scores  = {t: _score_table(q, t) for t in all_crm}

    # Include anchors when JOINs are likely
    needs_join = any(w in q for w in [
        "with", "by", "owner", "user", "rep", "company", "per",
        "department", "team", "leaderboard", "360", "health", "all",
    ])
    if needs_join:
        for t in _ANCHOR_TABLES:
            if scores.get(t, 0) == 0:
                scores[t] = 0.5

    # Complex/report queries get all core anchors
    is_complex = any(w in q for w in [
        "leaderboard", "360", "health", "summary", "report", "dashboard",
        "all reps", "each", "per user", "per company", "vendor list", "executive",
    ])
    if is_complex:
        for t in _COMPLEX_ANCHORS:
            scores[t] = max(scores.get(t, 0), 1.0)

    # Force-include explicitly requested extra tables
    for t in extra_tables:
        if t in scores:
            scores[t] = max(scores.get(t, 0), 2.0)

    # Select top-N tables
    ranked   = sorted(
        [(t, s) for t, s in scores.items() if s > 0],
        key=lambda x: (-x[1], x[0]),
    )[:max_tables]
    selected = [t for t, _ in ranked]

    if not selected:
        selected = ["deals", "users"]  # safe fallback

    # ── Delegate to db_schema for LIVE content ────────────────────────────────
    # build_schema_for_query builds each table line from real DB metadata
    return build_schema_for_query(query, hint_tables=selected, max_tables=max_tables)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSED SYSTEM PROMPT FOR SQL GENERATION  (~180 tokens)
# ══════════════════════════════════════════════════════════════════════════════

COMPACT_SYSTEM_PROMPT = """\
You are an expert PostgreSQL SQL generator. Output ONLY a SELECT statement — no markdown, no explanations.

RULES:
1. Start directly with SELECT. Double-quote table names: FROM "deals"
2. NEVER SELECT * — always list explicit columns
3. JOIN key is _id: LEFT JOIN "users" u ON u._id = d.owner
4. JOINs: ALWAYS prefix all columns with alias — d.name NOT name, d.deleted NOT deleted
5. Soft delete: WHERE NOT t.deleted  — EXCEPT vendors (NO deleted column — omit filter)
6. outreaches: WHERE NOT "isDeleted"  (isDeleted, not deleted — different column!)
7. Date TEXT cast: NULLIF(col,'')::timestamptz  NEVER: col::timestamptz
8. Year from TEXT date: EXTRACT(YEAR FROM NULLIF(col,'')::timestamptz) = EXTRACT(YEAR FROM CURRENT_DATE)
9. targets.year and targets.month are NUMERIC INTEGERS — never cast to timestamptz
10. NULL-safe numeric: COALESCE(SUM(col),0) | NULLIF(col,'')::numeric
11. MIXED-CASE COLUMNS need double quotes: s."salesOwner" NOT s.salesOwner (will fail!)
    Examples: s."salesOwner", t."userId", t."targetInUSD", c."companyName", u."isActive"
12. Only SELECT — never UPDATE/DELETE/INSERT/DROP.
13. MULTI-CURRENCY RULE — applies to ALL revenue/total/amount queries (invoices, sales, deals):
    ALWAYS select: currency column  +  native amount (NOT *_in_usd columns)
    ALWAYS GROUP BY currency so the system can convert each currency to USD automatically.
    invoices  → SELECT UPPER(COALESCE(i.currency,'USD')) AS currency, COALESCE(SUM(i.grand_total),0) AS amount FROM "invoices" i ... GROUP BY 1 ORDER BY 2 DESC
    sales     → SELECT UPPER(COALESCE(s.currency,'USD')) AS currency, COALESCE(SUM(s.grand_total),0) AS amount FROM "sales" s ... GROUP BY 1 ORDER BY 2 DESC
    deals     → SELECT UPPER(COALESCE(d.currency,'USD')) AS currency, COALESCE(SUM(d.grand_total),0) AS amount FROM "deals" d ... GROUP BY 1 ORDER BY 2 DESC
    bills     → SELECT 'USD' AS currency, COALESCE(SUM(b."netPayableAmount"),0) AS amount FROM "bills" b ... GROUP BY 1
    invoices native amount column: grand_total  (not grandtotal, not grandtotal_in_usd)
    ✗ NEVER use grandtotal_in_usd / grand_total_in_usd for SUM — use native grand_total + currency."""
