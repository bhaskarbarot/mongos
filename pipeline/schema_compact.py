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
    # ── New tables ─────────────────────────────────────────────────────────────
    "technologycategories": ["technology category","tech category","technologycategories"],
    "status":            ["status list","status names","status types"],
    "mails":             ["mail","mails","sent mail","inbox mail"],
    "activityevents":    ["activity event","activityevent","event log","crm event"],
    "activities":        ["activity type","activity name","activities list"],
    "notifications":     ["notification","notifications","alert","reminder notification"],
    "conversations":     ["conversation","conversations","chat log","message thread"],
    "companynotes":      ["company note","companynotes","note for company"],
    "contactsnotes":     ["contact note","contactsnotes","note for contact"],
    "dealsnotes":        ["deal note","dealsnotes","note for deal"],
    "salesnotes":        ["sales note","salesnotes","note for sale","note for order"],
    "remotejobnotes":    ["remote job note","remotejobnotes","job note"],
    "ai_notes":          ["ai note","ai notes","smart note","automated note"],
    "tasks":             ["task list","other tasks","tasks collection"],
    "remotejobs":        ["remote job","remotejobs","job","job listing","remote work"],
    "vendormagiclinks":  ["vendor link","vendor magic","vendor invite","vendormagiclinks"],
    "outreachactivities":["outreach activity","outreachactivities","outreach event","outreach count"],
    "deletedcompanies":  ["deleted company","deletedcompanies","archived company","removed company"],
    "prompts":           ["prompt","prompts","ai prompt","query prompt","llm prompt"],
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
   JOIN soft delete: ALWAYS add deleted filter ON the JOIN clause too:
     ✓ LEFT JOIN "deals" d ON d.company = c._id AND NOT d.deleted
     ✗ NEVER: LEFT JOIN "deals" d ON d.company = c._id  (missing deleted filter = counts deleted rows!)
     ✓ LEFT JOIN "invoices" i ON i.company = c._id AND NOT i.deleted
     ✓ LEFT JOIN "sales" s ON s."salesOwner" = u._id AND NOT s.deleted AND s.status='Confirm'
     ✓ LEFT JOIN "contacts" ct ON ct.company = c._id AND NOT ct.deleted
6. outreaches: WHERE NOT "isDeleted"  (isDeleted, not deleted — different column!)
7. Date cols are TIMESTAMPTZ — use directly, NO cast needed: d."createdAt" >= NOW() - INTERVAL '6 months'
8. Year from date: EXTRACT(YEAR FROM d."createdAt") = EXTRACT(YEAR FROM CURRENT_DATE)
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
16. REVENUE / AMOUNT QUERIES — MANDATORY RULES (always apply):

    REVENUE = paid invoices only, dated by when payment was received.
    TWO mandatory filters for EVERY revenue query on invoices:
      ✓ payment_status = 'paid'          — unpaid/overdue invoices are NOT revenue
      ✓ date filter on payment_date      — NOT invoice_date, NOT createdAt
    ✗ NEVER filter revenue by invoice_date or createdAt — use payment_date always.

    MODE A — "total revenue", "how much revenue", single USD number:
    ✓ ALWAYS use grandtotal_in_usd — already USD-converted, gives a clean single number.
    ✗ NEVER SUM(grand_total) for a total — it mixes INR+USD+GBP into a meaningless number.

    Canonical revenue query (copy this pattern exactly):
    SELECT COALESCE(SUM(i.grandtotal_in_usd), 0) AS total_revenue_usd
    FROM "invoices" i
    WHERE NOT i.deleted
      AND i.payment_status = 'paid'
      AND EXTRACT(YEAR FROM i.payment_date) = EXTRACT(YEAR FROM CURRENT_DATE)

    For a date range: AND i.payment_date >= '2026-01-01' AND i.payment_date < '2026-04-01'
    For last N months: AND i.payment_date >= NOW() - INTERVAL 'N months'

    MODE B — "revenue by currency", user explicitly asks per-currency breakdown:
    ✓ THEN use: SELECT UPPER(COALESCE(i.currency,'USD')) AS currency, COALESCE(SUM(i.grand_total),0) AS amount
    ✓ Still keep: AND i.payment_status = 'paid'  AND date filter on payment_date
    ✓ GROUP BY currency — to separate each currency bucket.

    Column names (exact spelling — double-check before writing):
    invoices  → grandtotal_in_usd  (no underscore: grandtotal_in_usd, NOT grand_total_in_usd)
    sales     → grand_total_in_usd (with underscore: grand_total_in_usd)
    deals     → grand_total_in_usd (with underscore: grand_total_in_usd)
    invoices native (MODE B only) → grand_total

17. ORDINAL SEARCHES ("first", "1st", "last N", "2nd", "second", "last 5"):
    "first" / "1st" / "oldest"   → ORDER BY "createdAt" ASC  LIMIT 1
    "last"  / "latest" / "recent" → ORDER BY "createdAt" DESC LIMIT 1
    "last N" / "recent N"         → ORDER BY "createdAt" DESC LIMIT N
    "2nd" / "second"              → ORDER BY "createdAt" ASC  OFFSET 1 LIMIT 1
    ✗ NEVER write WHERE _id = 1 or WHERE _id = '1' — _id is a MongoDB ObjectId TEXT string, never an integer.
    Example: "give me 1st invoice" → SELECT ... FROM "invoices" i WHERE NOT i.deleted ORDER BY i."createdAt" ASC LIMIT 1

18. ENTITY NAME SEARCH — ALWAYS use ILIKE for user-provided names (case-insensitive):
    Company name : WHERE c."companyName" ILIKE '%Wiegand LLC%'
    Contact name : WHERE (c."firstName" ILIKE '%ketul%' OR c."lastName" ILIKE '%ketul%')
    User name    : WHERE u.name ILIKE '%kartik%'
    Invoice no.  : WHERE i.invoice_number ILIKE '%ELSN%'
    Sales order  : WHERE s.sales_number ILIKE '%SO00080%'
    ✗ NEVER exact-match user-provided names: WHERE u.name = 'ketul' → use ILIKE '%ketul%' instead

19. PRODUCTS — embedded in JSONB on sales/invoices only; deals has NO items column:
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
    Products standalone list: SELECT name, unit_cost, currency FROM "products" WHERE "isActive"=true

20. LEAST / LOWEST / LESS / LOW / MINIMUM and HIGHEST / MOST / HIGH / HIGHER / MAXIMUM — GROUP QUERIES:
    These words mean: find ALL entities that share the minimum OR maximum value in a group.
    NEVER return just 1 row with LIMIT 1 — that gives wrong results.
    NEVER use ORDER BY + LIMIT — that only returns 1 entity, not ALL tied entities.

    ✓ CORRECT pattern — use RANK() window function to get ALL entities with min/max:

    "companies with LEAST/LOWEST/LESS/LOW/MINIMUM deals":
    SELECT company_name, deal_count
    FROM (
      SELECT c."companyName" AS company_name, COUNT(d._id) AS deal_count,
        RANK() OVER (ORDER BY COUNT(d._id) ASC) AS rnk
      FROM "companies" c
      LEFT JOIN "deals" d ON d.company = c._id AND NOT d.deleted
      WHERE NOT c.deleted
      GROUP BY c._id, c."companyName"
    ) ranked
    WHERE rnk = 1
    ORDER BY company_name;
    -- CRITICAL: outer SELECT uses plain alias 'company_name', NOT 'c."companyName"'
    -- Table aliases (c., u., d., s.) are ONLY valid inside the subquery — NEVER in outer SELECT!

    "companies with HIGHEST/MOST/HIGH/HIGHER/MAXIMUM deals":
    SELECT company_name, deal_count
    FROM (
      SELECT c."companyName" AS company_name, COUNT(d._id) AS deal_count,
        RANK() OVER (ORDER BY COUNT(d._id) DESC) AS rnk
      FROM "companies" c
      LEFT JOIN "deals" d ON d.company = c._id AND NOT d.deleted
      WHERE NOT c.deleted
      GROUP BY c._id, c."companyName"
    ) ranked
    WHERE rnk = 1
    ORDER BY company_name;
    -- CRITICAL: outer SELECT uses plain alias 'company_name', NOT 'c."companyName"'

    -- For USERS with least/highest invoices/sales/deals — same RANK pattern:
    SELECT user_name, item_count
    FROM (
      SELECT u.name AS user_name, COUNT(i._id) AS item_count,
        RANK() OVER (ORDER BY COUNT(i._id) ASC) AS rnk
      FROM "users" u
      LEFT JOIN "invoices" i ON i."createdBy" = u._id AND NOT i.deleted
      WHERE u."isActive" = true
      GROUP BY u._id, u.name
    ) ranked
    WHERE rnk = 1 ORDER BY user_name;

    Same pattern for: users with least/highest sales, reps with lowest/highest revenue, etc.
    Key rule: ORDER BY ASC + RANK WHERE rnk=1 for LEAST/LOWEST/LESS/LOW/MINIMUM
              ORDER BY DESC + RANK WHERE rnk=1 for HIGHEST/MOST/HIGH/HIGHER/MAXIMUM

    ✗ WRONG: SELECT "companyName" ... ORDER BY deal_count ASC LIMIT 1  (only 1 row, misses ties)
    ✓ RIGHT:  Use RANK() pattern above to get ALL companies with the minimum count

21. AGGREGATE FILTER RULE — CRITICAL:
    ✗ NEVER put SUM()/COUNT()/AVG() inside a WHERE clause — PostgreSQL will error.
    ✓ Use HAVING after GROUP BY for aggregate comparisons.
    ✓ OR use a correlated subquery in WHERE.

    TARGET vs ACHIEVED pattern (who achieved / who did not achieve targets):
    -- Who ACHIEVED (HAVING pattern):
    SELECT u.name
    FROM "users" u
    JOIN "targets" t ON t."userId" = u._id
    LEFT JOIN "sales" s ON s."salesOwner" = u._id
      AND EXTRACT(MONTH FROM s.sales_date) = t.month
      AND EXTRACT(YEAR  FROM s.sales_date) = t.year
      AND s.status = 'Confirm' AND NOT s.deleted
    WHERE t.month = 4 AND t.year = 2026
    GROUP BY u.name, t."targetInUSD"
    HAVING COALESCE(SUM(s.grand_total_in_usd), 0) >= t."targetInUSD"

    -- Who DID NOT ACHIEVE (HAVING pattern):
    SELECT u.name
    FROM "users" u
    JOIN "targets" t ON t."userId" = u._id
    LEFT JOIN "sales" s ON s."salesOwner" = u._id
      AND EXTRACT(MONTH FROM s.sales_date) = t.month
      AND EXTRACT(YEAR  FROM s.sales_date) = t.year
      AND s.status = 'Confirm' AND NOT s.deleted
    WHERE t.month = 4 AND t.year = 2026
    GROUP BY u.name, t."targetInUSD"
    HAVING COALESCE(SUM(s.grand_total_in_usd), 0) < t."targetInUSD"

    -- For LAST N MONTHS range — use IN list (never chained <=):
    ✗ WRONG: WHERE EXTRACT(MONTH FROM CURRENT_DATE) - 2 <= t.month <= EXTRACT(MONTH FROM CURRENT_DATE)
    ✓ CORRECT: WHERE t.month IN (2, 3, 4) AND t.year = 2026
    ✓ CORRECT: WHERE t.month BETWEEN 2 AND 4 AND t.year = 2026"""
