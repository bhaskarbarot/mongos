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
5. Soft delete: WHERE NOT t.deleted  — tables with NO deleted column (never add filter): vendors, users, departments, regions, products, targets, meetings, campaigns, bills, emails, commonnotes, activitylogs, sources, technologies, taxes, categories, lead_statuses, lifecycle_stages, payments, countryregions, projecttypes, notes, publicleads
6. outreaches: WHERE NOT "isDeleted"  (isDeleted, not deleted — different column!)
   notifications: WHERE NOT "isDeleted"
7. GLOBAL TEXT DATE RULE — ALL date-looking columns in EVERY table are stored as TEXT (ISO-8601 strings from MongoDB). This covers: createdAt, updatedAt, start, end, date, billDate, dueDate, due_date, invoice_date, payment_date, sales_date, closeDate, dealWonAt, dealLostAt, interestedDate, reminderDate, ReminderDate, leadWonAt, deletedAt, lastLogin, lastEmailSync, birthday — ALL of them.
   ALWAYS use: NULLIF(col,'')::timestamptz before ANY comparison or EXTRACT.
   ✗ NEVER write: col < CURRENT_DATE  or  col > NOW()  without the cast — causes "operator does not exist: text < date" error.
8. Date TEXT cast — EXACT PATTERN (memorize character by character):
    NULLIF(col,'')::timestamptz
    ✗ NEVER: col::timestamptz
    ✗ NEVER: NULLIF(col,''::timestamptz)   ← casting '' not the result — WRONG
    ✗ NEVER: NULLIF(col,'')               ← missing cast — stays TEXT, EXTRACT will fail
8. EXTRACT from TEXT date — copy these exactly:
    EXTRACT(YEAR  FROM NULLIF(i.due_date,'')::timestamptz) = EXTRACT(YEAR  FROM CURRENT_DATE)
    EXTRACT(MONTH FROM NULLIF(i.due_date,'')::timestamptz) = EXTRACT(MONTH FROM CURRENT_DATE)
    EXTRACT(DAY   FROM NOW() - NULLIF(i.due_date,'')::timestamptz)::INT AS days_overdue
    ✗ NEVER: EXTRACT(MONTH FROM NULLIF(i.due_date,''))::timestamptz   ← cast outside EXTRACT
    ✗ NEVER: EXTRACT(MONTH FROM NULLIF(i.due_date,''::timestamptz))   ← cast on wrong token
9. targets.year and targets.month are NUMERIC INTEGERS — never cast to timestamptz
10. NULL-safe numeric: COALESCE(SUM(col),0) | NULLIF(col,'')::numeric
11. MIXED-CASE COLUMNS need double quotes: s."salesOwner" NOT s.salesOwner (will fail!)
    Examples: s."salesOwner", t."userId", t."targetInUSD", c."companyName", u."isActive"
12. Only SELECT — never UPDATE/DELETE/INSERT/DROP.
13. NEVER use :param or $1 placeholders. Write concrete SQL only (use EXTRACT, CURRENT_DATE, literals).
14. FK COLUMN NAMES — use EXACT names, never guess with Id/id suffix:
    invoices  → company      (LEFT JOIN "companies" c ON c._id = i.company)
    deals     → company      (LEFT JOIN "companies" c ON c._id = d.company)
    sales     → company      (LEFT JOIN "companies" c ON c._id = s.company)
    companies → source       (LEFT JOIN "sources"   s ON s._id = c.source)
    companies → region       (LEFT JOIN "regions"   r ON r._id = c.region)
    createtasks→ companyId   (createtasks is the ONLY table with companyId column)
    ✗ NEVER write: i.companyId, d.companyId, s.companyId, c.sourceId, c.regionId
15. COLUMN EXISTENCE RULES (these columns do NOT exist — never generate them):
    ✗ companies.currency  — currency is on invoices/deals/sales, NOT on companies
    ✗ sales.closeDate     — sales uses sales_date (TEXT). closeDate is on deals only.
    ✗ invoices.productId  — invoices have no direct product FK column
    ✗ outreaches.leadId   — outreaches use assignedTo or email to link contacts
16. SINGLE RECORD LOOKUP — when the query mentions a specific ID, number, or name to fetch details:
    Signals: "details of", "show me X", "get invoice ELSN/...", "deal named X", "info on SO-..."
    → Use WHERE + ILIKE, NO GROUP BY, NO SUM, NO aggregate functions.
    → Select all useful columns for that record.
    Invoice detail example:
      SELECT i.invoice_number, i."companyName", i.grand_total, i.currency, i.payment_status,
             i.invoice_date, i.due_date, i.payment_date, i.subtotal, i.approval_status
      FROM "invoices" i
      WHERE i.invoice_number ILIKE '%ELSN/2026/005%' AND NOT i.deleted
    Deal detail example:
      SELECT d.name, d.stage, d.grand_total, d.currency, d."closeDate", d."dealWonAt", d.type
      FROM "deals" d WHERE d.name ILIKE '%deal name%' AND NOT d.deleted
    ✗ NEVER use SUM/GROUP BY/aggregate when fetching details of a specific record.

17. MULTI-CURRENCY RULE — applies ONLY to aggregate revenue/total/amount queries (NOT single-record lookups):
    ALWAYS select: currency column  +  native amount (NOT *_in_usd columns)
    ALWAYS GROUP BY currency so the system can convert each currency to USD automatically.
    invoices  → SELECT UPPER(COALESCE(i.currency,'USD')) AS currency, COALESCE(SUM(i.grand_total),0) AS amount FROM "invoices" i ... GROUP BY 1 ORDER BY 2 DESC
    sales     → SELECT UPPER(COALESCE(s.currency,'USD')) AS currency, COALESCE(SUM(s.grand_total),0) AS amount FROM "sales" s ... GROUP BY 1 ORDER BY 2 DESC
    deals     → SELECT UPPER(COALESCE(d.currency,'USD')) AS currency, COALESCE(SUM(d.grand_total),0) AS amount FROM "deals" d ... GROUP BY 1 ORDER BY 2 DESC
    bills     → SELECT 'USD' AS currency, COALESCE(SUM(b."netPayableAmount"),0) AS amount FROM "bills" b ... GROUP BY 1
    invoices native amount column: grand_total  (not grandtotal, not grandtotal_in_usd)
    ✗ NEVER use grandtotal_in_usd / grand_total_in_usd for SUM — use native grand_total + currency.

18. ORDINAL SEARCHES ("first", "1st", "last N", "2nd", "second", "last 5"):
    "first" / "1st" / "oldest"   → ORDER BY "createdAt" ASC  LIMIT 1
    "last"  / "latest" / "recent" → ORDER BY "createdAt" DESC LIMIT 1
    "last N" / "recent N"         → ORDER BY "createdAt" DESC LIMIT N
    "2nd" / "second"              → ORDER BY "createdAt" ASC  OFFSET 1 LIMIT 1
    ✗ NEVER write WHERE _id = 1 or WHERE _id = '1' — _id is a MongoDB ObjectId TEXT string, never an integer.
    Example: "give me 1st invoice" → SELECT ... FROM "invoices" i WHERE NOT i.deleted ORDER BY i."createdAt" ASC LIMIT 1

19. ENTITY NAME SEARCH — ALWAYS use ILIKE for user-provided names (case-insensitive):
    Company name : WHERE c."companyName" ILIKE '%Wiegand LLC%'
    Contact name : WHERE (c."firstName" ILIKE '%ketul%' OR c."lastName" ILIKE '%ketul%')
    User name    : WHERE u.name ILIKE '%kartik%'
    Invoice no.  : WHERE i.invoice_number ILIKE '%ELSN%'
    Sales order  : WHERE s.sales_number ILIKE '%SO00080%'
    ✗ NEVER exact-match user-provided names: WHERE u.name = 'ketul' → use ILIKE '%ketul%' instead

20. PRODUCTS — embedded in JSONB on sales/invoices only; deals has NO items column:
    ✗ NEVER write: d.product, s.product, i.productId, d.items — these DO NOT exist
    ✓ Only sales.items and invoices.items are JSONB line-item arrays
    Top products from sales JSONB items:
      SELECT jsonb_array_elements(s.items)->>'name' AS product_name, COUNT(*) AS order_count
      FROM "sales" s WHERE s.status='Confirm' AND NOT s.deleted GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    For "product + project type" analysis using deals:
      deals.type is a deal-category TEXT ('Cross-sell','Upsell','New Business') — NOT a product FK
      SELECT d.type AS deal_type, COUNT(*) AS deal_count, COALESCE(SUM(d.grand_total),0) AS total_value,
             ROUND(100.0*SUM(CASE WHEN d."dealWonAt" IS NOT NULL THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),1) AS win_rate_pct
      FROM "deals" d WHERE NOT d.deleted GROUP BY d.type ORDER BY total_value DESC
    Products standalone list: SELECT name, unit_cost, currency FROM "products" WHERE "isActive"=true"""
