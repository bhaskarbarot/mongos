"""schema_compact.py — Dynamic schema compression for SQL generation.

Reduces token cost from ~3500 (full schema) to ~300-700 (relevant tables only).
Strategy:
  1. Score each table by keyword overlap with the query
  2. Always include anchor tables (deals, users, companies) for JOIN availability
  3. Render only relevant tables in compact single-line format

Compact format per table (~15-25 tokens vs ~300 for CREATE TABLE):
  deals(name,stage,owner→u,company→c,deleted,value_usd,closeDate,wonAt,lostAt)
  -- open=wonAt IS NULL AND lostAt IS NULL; won=wonAt IS NOT NULL

Token budget:
  System prompt:  ~180 tokens  (compressed SQL rules)
  Compact schema: ~400 tokens  (avg 5-7 tables × ~60 tokens each)
  Query:          ~20 tokens
  Total:          ~600 tokens per SQL call   ← vs 3500 before  (83% reduction)

Public API:
  get_compact_schema(query: str, extra_tables: list = []) -> str
  COMPACT_SYSTEM_PROMPT  (use instead of full _SYSTEM_SQL_EXPERT)
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

# ══════════════════════════════════════════════════════════════════════════════
# COMPACT SCHEMA STRINGS  (one entry per table)
# ══════════════════════════════════════════════════════════════════════════════
# Format: TABLE_NAME → (compact_def, usage_notes)
# compact_def: table(col1 TYPE,col2,...) !soft_delete_note
# notes: short bullet hints for SQL generation

_TABLE_SCHEMA: Dict[str, str] = {

    "deals": """\
deals(_id,name,stage,owner→users,company→companies,deleted BOOL,grand_total_in_usd NUM,currency,type,"closeDate" TEXT,"dealWonAt" TEXT,"dealLostAt" TEXT)
-- stage: 'Analysis - To be Quoted'|'Quotation Sent'|'Negotiation'|'Contract Under Review'|'On Hold'|'Closed Won'|'Closed Lost'
-- Open: d."dealWonAt" IS NULL AND d."dealLostAt" IS NULL AND NOT d.deleted
-- Won:  d."dealWonAt" IS NOT NULL AND NOT d.deleted | Lost: d."dealLostAt" IS NOT NULL AND NOT d.deleted
-- camelCase cols need quotes: d."dealWonAt" NOT d.dealWonAt (will fail without quotes!)
-- closeDate/dealWonAt/dealLostAt: TEXT → NULLIF(col,'')::timestamptz""",

    "invoices": """\
invoices(_id,invoice_number,payment_status,grandtotal_in_usd NUM,grand_total NUM,currency,company→companies,invoice_date TEXT,due_date TEXT,payment_date TEXT,deleted BOOL,"companyName","createdBy"→users,"payment_mode")
-- payment_status: 'paid'|'unpaid'|'cancelled'|'draft'|'partial_payment'|'confirmed'
-- Revenue: SUM(grandtotal_in_usd) WHERE payment_status='paid' AND NOT deleted
-- Overdue: NULLIF(due_date,'')::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')
-- Year: EXTRACT(YEAR FROM NULLIF(payment_date,'')::timestamptz) = 2026""",

    "sales": """\
sales(_id,sales_number,status,"salesOwner"→users,company→companies,grand_total_in_usd NUM,currency,sales_date TEXT,deleted BOOL)
-- status: 'Confirm'|'Draft'|'Cancel'  Revenue: status='Confirm' AND NOT deleted
-- JOIN: LEFT JOIN "sales" s ON s."salesOwner" = u._id  (quotes required — mixed case!)
-- Year: EXTRACT(YEAR FROM NULLIF(s.sales_date,'')::timestamptz) = EXTRACT(YEAR FROM CURRENT_DATE)""",

    "companies": """\
companies(_id,"companyName",deleted BOOL,industry,country,region→regions,"lifecycleStage","leadStatus","companyOwner"→users,"clientHealth","createdAt")
-- lifecycleStage: 'Lead'|'Customer'|'Partner'|'Inactive Customer'|'Dead Customer'
-- Active: NOT IN ('Inactive Customer','Dead Customer') AND NOT deleted""",

    "contacts": """\
contacts(_id,"firstName","lastName",email,"jobTitle","phoneNumber","lifecycleStage","contactOwner"→users,company→companies,deleted BOOL,"createdAt")
-- Full name: TRIM(CONCAT(COALESCE("firstName",''),' ',COALESCE("lastName",'')))
-- Search: "firstName" ILIKE '%name%' OR "lastName" ILIKE '%name%'""",

    "users": """\
users(_id,name,email,department→departments,"isActive" BOOL,"isAdmin" BOOL,"createdAt")
-- Active users: WHERE "isActive" = true""",

    "createtasks": """\
createtasks(_id,"Task" TEXT,status,priority,"createdBy"→users,due_date TEXT,company→companies,"dealsId"→deals,deleted BOOL)
-- ⚠ table name is 'createtasks' NOT 'tasks'
-- status: 'Pending'|'Completed'|'Open'  priority: 'Low'|'Medium'|'High'
-- Assignee JOIN: LEFT JOIN "users" u ON u._id = t."createdBy"  (quotes on "createdBy"!)
-- Overdue: NULLIF(due_date,'')::timestamptz < NOW() AND status!='Completed' AND NOT deleted""",

    "targets": """\
targets(_id,"userId"→users,"targetInUSD" NUM,month NUM,year NUM,"teamName","createdAt")
-- ⚠ month and year are NUMERIC INTEGERS — NEVER cast to timestamptz!
-- This year: WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE)
-- Achieved JOIN: LEFT JOIN "sales" s ON s."salesOwner" = t."userId" AND s.status='Confirm' AND NOT s.deleted AND EXTRACT(YEAR FROM NULLIF(s.sales_date,'')::timestamptz) = t.year
-- All camelCase cols need double quotes: t."userId", t."targetInUSD", s."salesOwner" """,

    "meetings": """\
meetings(_id,title,start TEXT,"end" TEXT,location,"createdBy"→users,"createdAt")
-- start is TEXT → cast: start::timestamptz  (no empty strings expected)""",

    "outreaches": """\
outreaches(_id,name,email,status,"leadStatus",campaign→campaigns,region→regions,"assignedTo"→users,"isDeleted" BOOL,"createdAt")
-- ⚠ soft delete is "isDeleted" NOT deleted! → WHERE NOT "isDeleted"
-- status: 'Not Contacted'|'Contacted'|'Interested'|'Converted to Deal'""",

    "departments": """\
departments(_id,name)
-- name: 'accounts team'|'Business Analyst'|'Lead Generation'|'outreach team'
-- JOIN with users: users.department = departments._id""",

    "regions": """\
regions(_id,"regionName")
-- regionName: 'USA'|'Europe'|'APAC' etc.""",

    "vendors": """\
vendors(_id,"companyName",email,phone,currency,stage,country,"createdBy"→users,"createdAt")
-- CRITICAL: vendors has NO deleted/isDeleted column. NEVER use WHERE NOT deleted or WHERE NOT v.deleted
-- Count all: SELECT COUNT(*) FROM "vendors"  — no filter needed
-- stage: 'Pending Approval'|'Active'|'Inactive'""",

    "bills": """\
bills(_id,vendor→vendors,"systemBillNo","dueDate" TEXT,status,"netPayableAmount" NUM,subtotal NUM,"billType","createdBy"→users,"createdAt")
-- status: 'Payment Scheduled'|'Paid'|'Pending'|'Draft'
-- Unpaid: status NOT IN ('Paid')""",

    "products": """\
products(_id,name,unit_cost NUM,currency,"isActive" BOOL,sku,billing_frequency,"createdAt")
-- Active: WHERE "isActive" = true""",

    "taxes": """\
taxes(_id,name,amount NUM)
-- amount = tax rate e.g. 18 (for 18%)""",

    "campaigns": """\
campaigns(_id,"campaignName","categoryId"→categories,"createdBy","createdAt")""",

    "sources": """\
sources(_id,"sourceName")
-- sourceName: 'Old Client'|'LinkedIn'|'Referral' etc.""",

    "technologies": """\
technologies(_id,name,category)
-- name: 'Magento'|'React'|'PHP' etc.""",

    "categories": """\
categories(_id,"categoryName",name)""",

    "lead_statuses": """\
lead_statuses(_id,name)
-- name: 'New'|'Open'|'In Progress'|'Unqualified' etc.""",

    "lifecycle_stages": """\
lifecycle_stages(_id,name)
-- name: 'Lead'|'Customer'|'Partner' etc.""",

    "dealstagesettings": """\
dealstagesettings(_id,"dealStageName",deleted BOOL)
-- All active deal stage names""",

    "payments": """\
payments(_id,"payment_name","payment_fee" NUM,description)
-- Payment method names e.g. 'IDFC FIRST BANK International'""",

    "emails": """\
emails(_id,"user"→users,"from","to",subject,snippet,body,date TEXT,"createdAt")
-- date: TEXT → cast NULLIF(date,'')::timestamptz""",

    "commonnotes": """\
commonnotes(_id,note,type,"createdBy"→users,"companyId"→companies,"dealId"→deals,"contactId"→contacts,"isPinned" BOOL,"createdAt")
-- type: 'Company'|'Deal'|'Contact'|'Invoice'|'Sales'""",

    "activitylogs": """\
activitylogs(_id,action,module,"recordId","recordName","userId"→users,"createdAt")
-- action: 'create'|'update'|'delete'  module: 'Companies'|'Deals'|'Invoices' etc.""",

    "countryregions": """\
countryregions(_id,country,region)""",

    "projecttypes": """\
projecttypes(_id,name)
-- name: 'Dedicated'|'Fixed Price'|'T&M' etc.""",

    "notes": """\
notes(_id,"outreachId"→outreaches,"contactMethod",message,"reminderDate","createdBy"→users,"createdAt")""",

    "publicleads": """\
publicleads(_id,"firstName","lastName",email,"phoneNumber",source,"leadStatus","lifecycleStage","userType","createdAt")
-- Inbound/web form leads""",
}


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD → TABLE RELEVANCE MAP
# ══════════════════════════════════════════════════════════════════════════════

_TABLE_KEYWORDS: Dict[str, List[str]] = {
    "deals":            ["deal","deals","pipeline","stage","won","lost","close","open","negotiation",
                         "contract","opportunity","prospect","bid","quote","proposal","revenue","opportunity"],
    "invoices":         ["invoice","invoices","payment","revenue","paid","unpaid","overdue","billing",
                         "amount","INR","USD","GBP","currency","due","receipt","collect","aging"],
    "sales":            ["sales","order","confirm","SO","sale","salesOwner","rep","confirmed"],
    "companies":        ["company","companies","customer","client","account","business","firm",
                         "organization","industry","lifecycle","health","active","lead","partner"],
    "contacts":         ["contact","contacts","person","people","firstName","lastName","job",
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
    "projecttypes":     ["project type","engagement","dedicated","fixed price","T&M"],
    "notes":            ["outreach note","outreach activity","contact method"],
    "publicleads":      ["public lead","web lead","form lead","inbound","website lead"],
}

# Anchor tables always included when any JOIN or multi-entity query detected
_ANCHOR_TABLES: Set[str] = {"users", "companies"}

# Always include for complex/leaderboard/360 queries
_COMPLEX_ANCHORS: Set[str] = {"deals", "invoices", "sales", "users", "companies", "createtasks"}


def _score_table(query_lower: str, table: str) -> int:
    """Return keyword hit count for a table given the lowercased query."""
    kws = _TABLE_KEYWORDS.get(table, [])
    score = 0
    for kw in kws:
        if kw in query_lower:
            score += 1
    return score


def get_compact_schema(
    query: str,
    extra_tables: List[str] = [],
    max_tables:   int = 9,
) -> str:
    """Return compact schema string containing only tables relevant to the query.

    Args:
        query:        Natural language query from the user.
        extra_tables: Additional table names to force-include.
        max_tables:   Hard cap on table count (limits token usage).

    Returns:
        Multi-line compact schema string, ~300-700 tokens.
    """
    q = query.lower()
    all_tables = list(_TABLE_SCHEMA.keys())

    # Score every table
    scores = {t: _score_table(q, t) for t in all_tables}

    # Always include anchors if JOINs are likely
    needs_join = any(w in q for w in [
        "with", "by", "owner", "user", "rep", "company", "per",
        "department", "team", "leaderboard", "360", "health", "all"
    ])
    if needs_join:
        for t in _ANCHOR_TABLES:
            if scores.get(t, 0) == 0:
                scores[t] = 0.5  # small boost to include anchor tables

    # Complex multi-entity queries get all anchors
    is_complex = any(w in q for w in [
        "leaderboard", "360", "health", "summary", "report", "dashboard",
        "all reps", "each", "per user", "per company", "vendor list"
    ])
    if is_complex:
        for t in _COMPLEX_ANCHORS:
            scores[t] = max(scores.get(t, 0), 1)

    # Force-include explicitly requested tables
    for t in extra_tables:
        if t in scores:
            scores[t] = max(scores.get(t, 0), 2)

    # Select top-N tables by score (only tables with score > 0)
    ranked = sorted(
        [(t, s) for t, s in scores.items() if s > 0],
        key=lambda x: (-x[1], x[0]),
    )[:max_tables]

    selected = [t for t, _ in ranked]

    # Fallback: if nothing scored, include the 3 most universal tables
    if not selected:
        selected = ["deals", "invoices", "users"]

    # Build schema string
    lines = [
        "-- CRITICAL JOIN RULE: In JOINs always prefix columns with alias (d.name NOT name)",
        "-- Date TEXT cast: NULLIF(col,'')::timestamptz  |  Year: EXTRACT(YEAR FROM NULLIF(col,'')::ts) = 2026",
    ]
    for t in selected:
        schema = _TABLE_SCHEMA.get(t)
        if schema:
            lines.append(schema)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSED SYSTEM PROMPT FOR SQL GENERATION  (~180 tokens vs ~600 before)
# ══════════════════════════════════════════════════════════════════════════════

COMPACT_SYSTEM_PROMPT = """\
You are an expert PostgreSQL SQL generator. Output ONLY a SELECT statement — no markdown, no explanations.

RULES:
1. Start directly with SELECT. Double-quote table names: FROM "deals"
2. NEVER SELECT * — always list explicit columns
3. JOIN key is _id: LEFT JOIN "users" u ON u._id = d.owner
4. JOINs: ALWAYS prefix all columns with alias — d.name NOT name, d.deleted NOT deleted
5. Soft delete: WHERE NOT t.deleted  — EXCEPT vendors (NO deleted column at all, omit filter)
6. outreaches: WHERE NOT "isDeleted"  (not "deleted" — different column name!)
7. Date TEXT cast: NULLIF(col,'')::timestamptz  NEVER: col::timestamptz
8. Year from TEXT date: EXTRACT(YEAR FROM NULLIF(col,'')::timestamptz) = 2026
9. targets.year and targets.month are NUMERIC INTEGERS — never cast to timestamptz
10. NULL-safe numeric: COALESCE(SUM(col),0) | NULLIF(doc->>'n','')::numeric
11. MIXED-CASE COLUMNS need double quotes: s."salesOwner" NOT s.salesOwner (will fail!)
    Examples: s."salesOwner", t."userId", t."targetInUSD", c."companyName", u."isActive"
12. Only SELECT, never UPDATE/DELETE/INSERT/DROP."""
