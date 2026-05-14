"""text2sql.py — Dual-model Text2SQL engine with parallel execution.

Model priority:
  1. PRIMARY:  debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned  (fast, 8s)
  2. FALLBACK: a-kore/Arctic-Text2SQL-R1-7B                      (accurate, 30s)

Both fire in parallel — first valid SQL wins.
If parallel fails → retry with fallback model alone (more thorough).

Key design:
  • CREATE TABLE schema format (what fine-tuned Text2SQL models expect)
  • Ollama /api/chat endpoint so each model uses its correct template
  • Accepts BOTH direct column access AND document->>'field' JSONB access
    (both work: flat columns AND document JSONB column are populated)
  • JOIN key: always use _id (the actual PK text column)

Public API:
    run(query, agent)          -> Optional[Dict]
    generate_sql(query, agent) -> Optional[str]
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from pipeline.schema import (
    build_text2sql_schema,
    get_table_names,
    run_sql,
)
from pipeline.utils import (
    Timer,
    coerce_number,
    fmt_number,
    format_rows_as_markdown_table,
)

LOGGER = logging.getLogger("sql_chatbot")

# ── Constants ──────────────────────────────────────────────────────────────────
_MAX_SQL_RETRIES    = 1     # retry once with fallback model on failure
_PARALLEL_TIMEOUT_S = 20    # max wait for the parallel pool (was 35)
_MAX_RESULT_ROWS    = 500   # return all rows; user can add "top N" to limit
_SQL_EXEC_TIMEOUT_S = 20    # SQL execution timeout (was 30)
_GROQ_SQL_TIMEOUT   = 15    # Groq SQL generation timeout


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA — CREATE TABLE format (what fine-tuned Text2SQL models understand)
# ══════════════════════════════════════════════════════════════════════════════

# Static CREATE TABLE schema built from the actual DB columns.
# Includes only the fields most relevant to CRM queries.
# Both direct column access (deals.name) AND document->>'name' work — the DB
# has flat columns AND a populated JSONB document column.

_CREATE_TABLE_SCHEMA = """-- ══════════════════════════════════════════════════════════════════════
-- PostgreSQL CRM Database — Full Schema
-- ══════════════════════════════════════════════════════════════════════
-- PRIMARY KEY  : _id TEXT (every table)
-- SOFT DELETE  : WHERE NOT deleted   (most tables, BOOLEAN column)
--                outreaches: WHERE NOT "isDeleted"
-- JOIN KEY     : always use _id flat column
-- ⚠️  JSONB RULE : mixed-case fields MUST use d.document->>'field'
--                 In JOINs ALWAYS prefix with alias: d.document, i.document, etc.
--                 NEVER write document->>'field' without table alias in a JOIN!
-- NUMERIC COLS : grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal,
--                grand_total, "targetInUSD", month, year — no cast needed
-- DATE COLS    : stored as TEXT → cast: NULLIF(col,'')::timestamptz
--                For month filter: DATE_TRUNC('month', NULLIF(col,'')::timestamptz) = DATE '2025-09-01'

-- ─── CORE CRM TABLES ────────────────────────────────────────────────────────

CREATE TABLE "deals" (
    _id TEXT PRIMARY KEY,
    name TEXT,              -- deal/project name
    stage TEXT,             -- 'Analysis - To be Quoted' | 'Quotation Sent' | 'Negotiation'
                            -- 'Contract Under Review' | 'On Hold' | 'Closed Won' | 'Closed Lost'
    owner TEXT,             -- → users._id (sales rep who owns the deal)
    company TEXT,           -- → companies._id
    contact TEXT,           -- → contacts._id
    deleted BOOLEAN,        -- WHERE NOT deleted
    grand_total_in_usd NUMERIC, -- deal value in USD
    grand_total NUMERIC,        -- deal value in base currency
    currency TEXT,
    type TEXT,              -- deal category/type e.g. 'Support','Dedicated','Project'
    "closeDate" TEXT,       -- expected close date (TEXT → cast ::timestamptz)
    "dealWonAt" TEXT,       -- NULL=not yet won; NOT NULL=won date
    "dealLostAt" TEXT,      -- NULL=not yet lost; NOT NULL=lost date
    "createdAt" TEXT,
    document JSONB          -- ALL fields also accessible here
);
-- ✓ Open deals:  WHERE d."dealWonAt" IS NULL AND d."dealLostAt" IS NULL AND NOT d.deleted
-- ✓ Won deals:   WHERE d."dealWonAt" IS NOT NULL AND NOT d.deleted
-- ✓ Lost deals:  WHERE d."dealLostAt" IS NOT NULL AND NOT d.deleted
-- ✓ With owner:  FROM "deals" d LEFT JOIN "users" u ON u._id = d.owner
--   In JOIN use: d."dealWonAt", d.stage, d.deleted  (NOT just dealWonAt — AMBIGUOUS!)
-- ✓ By category: WHERE d.type = 'Support'

CREATE TABLE "invoices" (
    _id TEXT PRIMARY KEY,
    invoice_number TEXT,    -- e.g. 'ELSN/2025/101' — search with ILIKE
    payment_status TEXT,    -- 'paid' | 'unpaid' | 'confirmed' | 'draft' | 'cancelled'
                            -- 'partial_payment' | 'approved' | 'rejected' | 'submitted'
    approval_status TEXT,   -- 'approved' | 'rejected' | 'pending' | 'submitted'
    grandtotal_in_usd NUMERIC, -- USD amount (use for revenue/comparison)
    grand_total NUMERIC,    -- base currency amount
    currency TEXT,          -- 'USD','INR','AUD','GBP', etc.
    company TEXT,           -- → companies._id
    invoice_date TEXT,      -- invoice creation date (TEXT → cast ::timestamptz)
    due_date TEXT,          -- payment due date (TEXT → cast ::timestamptz)
    payment_date TEXT,      -- actual payment received date (TEXT → cast ::timestamptz)
    deleted BOOLEAN,
    "createdBy" TEXT,       -- → users._id
    "companyName" TEXT,     -- company name snapshot (denormalised)
    "invoiceFor" TEXT,      -- 'Elsner Technologies Pvt. Ltd.' etc.
    "payment_mode" TEXT,    -- bank/payment method name
    document JSONB
);
-- ✓ Revenue by period: SUM(grandtotal_in_usd) WHERE payment_status='paid'
--   AND DATE_TRUNC('month', NULLIF(payment_date,'')::timestamptz) = DATE '2025-09-01'
-- ✓ Overdue: WHERE NULLIF(due_date,'')::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')
-- ✓ By currency: WHERE UPPER(currency) = 'INR'
-- ✓ By invoice#: WHERE invoice_number ILIKE '%ELSN/2025/1%'
-- ✓ With company name: LEFT JOIN "companies" c ON c._id = i.company

CREATE TABLE "sales" (
    _id TEXT PRIMARY KEY,
    sales_number TEXT,      -- e.g. 'S000080'
    status TEXT,            -- 'Confirm' | 'Draft' | 'Cancel'
    "salesOwner" TEXT,      -- → users._id
    company TEXT,           -- → companies._id
    grand_total_in_usd NUMERIC,
    grand_total NUMERIC,
    currency TEXT,
    sales_date TEXT,        -- TEXT → cast ::timestamptz
    deleted BOOLEAN,
    document JSONB
);
-- ✓ Revenue: SUM(grand_total_in_usd) WHERE status='Confirm' AND NOT deleted
-- ✓ By rep: LEFT JOIN "users" u ON u._id = s."salesOwner"  → use s.grand_total_in_usd

CREATE TABLE "companies" (
    _id TEXT PRIMARY KEY,
    "companyName" TEXT,         -- company display name
    deleted BOOLEAN,
    industry TEXT,
    email TEXT,
    country TEXT,
    region TEXT,                -- → regions._id
    "lifecycleStage" TEXT,      -- 'Lead' | 'Customer' | 'Partner' | 'Inactive Customer' | 'Dead Customer'
    "leadStatus" TEXT,          -- 'New' | 'Open' | 'In Progress' | 'Unqualified' | 'Bad Timing' etc.
    "companyOwner" TEXT,        -- → users._id (account manager)
    "annualRevenue" TEXT,
    "websiteUrl" TEXT,
    "phoneNumber" TEXT,
    source TEXT,                -- → sources._id
    "clientHealth" TEXT,
    "createdAt" TEXT,
    "leadWonAt" TEXT,           -- date company became customer
    "inActiveSince" TEXT,
    document JSONB
);
-- ✓ Active customers: WHERE c."lifecycleStage" NOT IN ('Inactive Customer','Dead Customer') AND NOT c.deleted
-- ✓ Company name: c."companyName"  (flat column, no JSONB needed)
-- ✓ With region: LEFT JOIN "regions" r ON r._id = c.region
-- ✓ Vendors: this table stores all account types. Filter: WHERE c."userType"='vendor' if needed

CREATE TABLE "contacts" (
    _id TEXT PRIMARY KEY,
    "firstName" TEXT,
    "lastName" TEXT,
    email TEXT,
    "jobTitle" TEXT,
    "phoneNumber" TEXT,
    "lifecycleStage" TEXT,  -- 'Lead' | 'Customer' | 'Partner' etc.
    "leadStatus" TEXT,
    "contactOwner" TEXT,    -- → users._id
    company TEXT,           -- → companies._id
    deleted BOOLEAN,
    source TEXT,
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Full name: TRIM(CONCAT(COALESCE(c."firstName",''),' ',COALESCE(c."lastName",'')))
-- ✓ Search by name: WHERE c."firstName" ILIKE '%kartik%' OR c."lastName" ILIKE '%kartik%'
-- ✓ With company: LEFT JOIN "companies" co ON co._id = c.company

CREATE TABLE "users" (
    _id TEXT PRIMARY KEY,
    name TEXT,          -- full name e.g. 'Ketul Trivedi'
    email TEXT,
    department TEXT,    -- → departments._id
    "isActive" BOOLEAN, -- WHERE "isActive" = true  for active users
    "isAdmin" BOOLEAN,
    "isSuperAdmin" BOOLEAN,
    "createdAt" TEXT,
    document JSONB
);

CREATE TABLE "createtasks" (   -- ⚠️ table is 'createtasks' NOT 'tasks'
    _id TEXT PRIMARY KEY,
    "Task" TEXT,        -- task title (capital T — use flat column or document->>'Task')
    status TEXT,        -- 'Pending' | 'Completed' | 'Open'
    priority TEXT,      -- 'Low' | 'Medium' | 'High'
    "createdBy" TEXT,   -- → users._id (task assignee/owner)
    due_date TEXT,      -- TEXT → cast ::timestamptz
    company TEXT,       -- → companies._id
    "companyId" TEXT,   -- same as company
    "dealsId" TEXT,     -- → deals._id (if task linked to a deal)
    "invoiceId" TEXT,   -- → invoices._id
    deleted BOOLEAN,
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Pending: WHERE t.status='Pending' AND NOT t.deleted
-- ✓ Overdue: WHERE NULLIF(t.due_date,'')::timestamptz < NOW() AND t.status!='Completed' AND NOT t.deleted
-- ✓ By user: LEFT JOIN "users" u ON u._id = t."createdBy"  → use t.status, t."Task"

CREATE TABLE "targets" (
    _id TEXT PRIMARY KEY,
    "userId" TEXT,          -- → users._id
    "targetInUSD" NUMERIC,  -- monthly sales target in USD
    month NUMERIC,          -- 1–12
    year NUMERIC,           -- e.g. 2025, 2026
    "teamName" TEXT,        -- e.g. 'Accounts Team'
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Current month: WHERE t.month = EXTRACT(MONTH FROM CURRENT_DATE) AND t.year = EXTRACT(YEAR FROM CURRENT_DATE)
-- ✓ With user: LEFT JOIN "users" u ON u._id = t."userId"

CREATE TABLE "meetings" (
    _id TEXT PRIMARY KEY,
    title TEXT,             -- meeting title e.g. 'Client Call'
    description TEXT,
    start TEXT,             -- start datetime (TEXT → cast ::timestamptz)
    "end" TEXT,             -- end datetime
    location TEXT,
    attendees JSONB,        -- list of attendee objects
    "EventId" TEXT,         -- Google Calendar event ID
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Today: WHERE m.start::timestamptz::date = CURRENT_DATE
-- ✓ Previous: WHERE m.start::timestamptz < NOW() ORDER BY m.start DESC
-- ✓ By date: WHERE m.start::timestamptz::date = '2026-05-14'::date

-- ─── OUTREACH & MARKETING ───────────────────────────────────────────────────

CREATE TABLE "outreaches" (
    _id TEXT PRIMARY KEY,
    name TEXT,              -- prospect name
    email TEXT,
    phone TEXT,
    country TEXT,
    status TEXT,            -- 'Not Contacted' | 'Contacted' | 'Interested' | 'Converted to Deal'
    "leadStatus" TEXT,
    priority TEXT,
    campaign TEXT,          -- → campaigns._id
    region TEXT,            -- → regions._id
    "assignedTo" TEXT,      -- → users._id
    "isDeleted" BOOLEAN,    -- ⚠️ isDeleted (not deleted!) WHERE NOT "isDeleted"
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Filter: WHERE NOT o."isDeleted"
-- ✓ Conversion rate: COUNT(*) FILTER (WHERE status='Converted to Deal') / COUNT(*) * 100

CREATE TABLE "campaigns" (
    _id TEXT PRIMARY KEY,
    "campaignName" TEXT,
    "categoryId" TEXT,      -- → categories._id
    "createdBy" TEXT,
    "createdAt" TEXT,
    document JSONB
);

-- ─── FINANCE ────────────────────────────────────────────────────────────────

CREATE TABLE "vendors" (
    _id TEXT PRIMARY KEY,
    "companyName" TEXT,     -- vendor company name
    email TEXT,
    phone TEXT,
    currency TEXT,          -- vendor's billing currency
    stage TEXT,             -- 'Pending Approval' | 'Active' | 'Inactive'
    country TEXT,
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Count: SELECT COUNT(*) FROM "vendors"
-- ✓ Active vendors: WHERE v.stage = 'Active'

CREATE TABLE "bills" (
    _id TEXT PRIMARY KEY,
    vendor TEXT,            -- → vendors._id
    "systemBillNo" TEXT,    -- e.g. 'BILL-14'
    "vendorInvoiceNo" TEXT,
    "billDate" TEXT,        -- TEXT → cast ::timestamptz
    "dueDate" TEXT,         -- payment due date
    status TEXT,            -- 'Payment Scheduled' | 'Paid' | 'Pending' | 'Draft'
    "netPayableAmount" NUMERIC,
    subtotal NUMERIC,
    "gstPercent" NUMERIC,
    "billType" TEXT,        -- 'Service' | 'Product'
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Unpaid bills: WHERE b.status NOT IN ('Paid') AND b."dueDate" is set
-- ✓ By vendor: LEFT JOIN "vendors" v ON v._id = b.vendor

-- ─── LOOKUP / REFERENCE TABLES ──────────────────────────────────────────────

CREATE TABLE "departments" (
    _id TEXT PRIMARY KEY,
    name TEXT               -- 'accounts team' | 'Business Analyst' | 'Lead Generation' | 'outreach team'
);
-- ✓ Users in dept: JOIN "users" u ON u.department = d._id

CREATE TABLE "regions" (
    _id TEXT PRIMARY KEY,
    "regionName" TEXT       -- 'USA' | 'Europe' | 'APAC' etc.
);

CREATE TABLE "products" (
    _id TEXT PRIMARY KEY,
    name TEXT,
    product_type TEXT,      -- → projecttypes._id
    unit_cost NUMERIC,
    currency TEXT,
    "isActive" BOOLEAN,     -- WHERE "isActive" = true
    sku TEXT,
    description_short TEXT,
    description_long TEXT,
    billing_frequency TEXT,
    "createdAt" TEXT,
    document JSONB
);

CREATE TABLE "sources" (
    _id TEXT PRIMARY KEY,
    "sourceName" TEXT   -- 'Old Client' | 'LinkedIn' | 'Referral' etc.
);

CREATE TABLE "technologies" (
    _id TEXT PRIMARY KEY,
    name TEXT,          -- 'Magento' | 'React' | 'PHP' etc.
    category TEXT       -- → technologycategories._id
);

CREATE TABLE "taxes" (
    _id TEXT PRIMARY KEY,
    name TEXT,          -- 'Tax 18%' | 'GST' etc.
    amount NUMERIC      -- tax rate e.g. 18
);

CREATE TABLE "categories" (
    _id TEXT PRIMARY KEY,
    "categoryName" TEXT,
    name TEXT
);

CREATE TABLE "dealstagesettings" (
    _id TEXT PRIMARY KEY,
    "dealStageName" TEXT,   -- all active deal stages
    deleted BOOLEAN
);

CREATE TABLE "lead_statuses" (
    _id TEXT PRIMARY KEY,
    name TEXT   -- 'New' | 'Open' | 'In Progress' | 'Unqualified' etc.
);

CREATE TABLE "lifecycle_stages" (
    _id TEXT PRIMARY KEY,
    name TEXT   -- 'Lead' | 'Customer' | 'Partner' etc.
);

CREATE TABLE "payments" (
    _id TEXT PRIMARY KEY,
    "payment_name" TEXT,    -- payment method name e.g. 'IDFC FIRST BANK International'
    "payment_fee" NUMERIC,
    description TEXT
);

-- ─── ACTIVITY / NOTES ───────────────────────────────────────────────────────

CREATE TABLE "emails" (
    _id TEXT PRIMARY KEY,
    "user" TEXT,            -- → users._id (who sent/received)
    "from" TEXT,
    "to" TEXT,
    subject TEXT,
    snippet TEXT,
    body TEXT,
    date TEXT,              -- TEXT → cast ::timestamptz
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Search emails: WHERE e.subject ILIKE '%keyword%' OR e.snippet ILIKE '%keyword%'

CREATE TABLE "commonnotes" (
    _id TEXT PRIMARY KEY,
    note TEXT,              -- note content (HTML)
    type TEXT,              -- 'Company' | 'Deal' | 'Contact' | 'Invoice' | 'Sales'
    "createdBy" TEXT,       -- → users._id
    "companyId" TEXT,       -- → companies._id
    "dealId" TEXT,          -- → deals._id
    "contactId" TEXT,       -- → contacts._id
    "invoiceId" TEXT,       -- → invoices._id
    "salesId" TEXT,         -- → sales._id
    "isPinned" BOOLEAN,
    "isLog" BOOLEAN,
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Notes for company: WHERE cn."companyId" = 'company_id_here'
-- ✓ Notes for deal: WHERE cn."dealId" = 'deal_id_here'

CREATE TABLE "activitylogs" (
    _id TEXT PRIMARY KEY,
    action TEXT,            -- 'create' | 'update' | 'delete'
    module TEXT,            -- 'Companies' | 'Deals' | 'Invoices' | 'Sales' etc.
    "recordId" TEXT,
    "recordName" TEXT,
    "userId" TEXT,          -- → users._id (who performed the action)
    "ipAddress" TEXT,
    "createdAt" TEXT,
    document JSONB
);
-- ✓ Recent activity: ORDER BY "createdAt" DESC LIMIT 20
-- ✓ By module: WHERE module = 'Deals'

CREATE TABLE "countryregions" (
    _id TEXT PRIMARY KEY,
    country TEXT,
    region TEXT     -- region name string (not FK)
);

CREATE TABLE "projecttypes" (
    _id TEXT PRIMARY KEY,
    name TEXT   -- 'Dedicated' | 'Fixed Price' | 'T&M' etc.
);

-- ─── KEY JOIN PATTERNS (use table alias prefix on ALL columns in JOINs) ─────
-- Deals + owner:     FROM "deals" d LEFT JOIN "users" u ON u._id = d.owner
--                    SELECT d.name, d.stage, u.name AS owner_name WHERE NOT d.deleted
-- Invoices + co:     FROM "invoices" i LEFT JOIN "companies" c ON c._id = i.company
--                    SELECT i.invoice_number, c."companyName", i.grandtotal_in_usd
-- Tasks + user:      FROM "createtasks" t LEFT JOIN "users" u ON u._id = t."createdBy"
--                    WHERE t.status='Pending' AND NOT t.deleted
-- Targets + user:    FROM "targets" t LEFT JOIN "users" u ON u._id = t."userId"
-- Sales + owner:     FROM "sales" s LEFT JOIN "users" u ON u._id = s."salesOwner"
-- Companies + dept:  FROM "users" u LEFT JOIN "departments" d ON d._id = u.department
-- Outreaches + camp: FROM "outreaches" o LEFT JOIN "campaigns" c ON c._id = o.campaign
-- Bills + vendor:    FROM "bills" b LEFT JOIN "vendors" v ON v._id = b.vendor
-- Contacts + co:     FROM "contacts" ct LEFT JOIN "companies" c ON c._id = ct.company
--
-- ⚠️ JOIN RULE: In any JOIN query, ALWAYS write d.document, i.document, c.document etc.
--    NEVER write bare 'document' — it is AMBIGUOUS when multiple tables are joined.
"""


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CALLER  (uses /api/chat so each model applies its own template)
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama_chat(
    model: str,
    system_msg: str,
    user_msg: str,
    timeout: int,
    max_tokens: int = 600,
) -> Optional[str]:
    """Call Ollama /api/chat endpoint with system + user messages.

    Using the chat endpoint ensures each model applies its correct prompt
    template (Qwen ChatML, Arctic instruct format, etc.) rather than raw text.
    """
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream":  False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "stop": ["Question:", "\n\n\n", "```\n\n"],
        },
    }
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data   = json.loads(resp.read())
            result = data.get("message", {}).get("content", "").strip()
            if result:
                LOGGER.debug("Ollama [%s] responded (%d chars)", model, len(result))
            return result or None
    except Exception as exc:
        LOGGER.debug("Ollama [%s] chat error: %s", model, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING  (model-specific)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_SQL_EXPERT = """You are an expert PostgreSQL SQL generator for a CRM database.
Given a schema and a question, generate ONE correct SELECT SQL query.

STRICT OUTPUT RULES:
1. Output ONLY the SQL — no explanation, no markdown, no text before or after.
2. Start directly with SELECT.
3. Table names in double-quotes: FROM "deals", FROM "createtasks"
4. JOIN key is always _id: LEFT JOIN "users" u ON u._id = d.owner
5. NEVER use SELECT * — always list explicit columns. SELECT * includes the
   large JSONB `document` column which breaks result parsing.
   ✗  SELECT * FROM "invoices"
   ✓  SELECT invoice_number, payment_status, grand_total, currency FROM "invoices"

INVOICE NUMBER LOOKUP — CRITICAL:
  invoice_number may be in the flat column OR in the document JSONB.
  Always search BOTH:
    WHERE invoice_number ILIKE '%ELSN/2025/1%'
       OR document->>'invoice_number' ILIKE '%ELSN/2025/1%'
  Use ILIKE (case-insensitive) not = (exact match).

CRITICAL — MIXED-CASE COLUMNS:
  Many columns have mixed case (dealWonAt, companyName, firstName, etc.).
  ALWAYS use document->>'fieldName' (JSONB) for these — it is case-safe.
  NEVER use bare column names for mixed-case fields without double-quoting.
  Examples:
    ✓ document->>'dealWonAt'     (JSONB — always works)
    ✓ "dealWonAt"                (quoted flat — also works)
    ✗ dealWonAt                  (unquoted — PostgreSQL lowercases it, FAILS)

CRITICAL — JOIN QUERIES: ALWAYS prefix `document` with table alias:
  Every table has a `document` JSONB column. In JOINs the column is AMBIGUOUS.
  ALWAYS write: d.document->>'field'   NOT   document->>'field'
  Examples:
    ✓ FROM "deals" d LEFT JOIN "users" u ON u._id = d.owner
      WHERE d.document->>'dealWonAt' IS NOT NULL    ← d. prefix required
      AND NOT d.deleted                              ← d. prefix required
    ✗ WHERE document->>'dealWonAt' IS NOT NULL       ← AMBIGUOUS — SQL ERROR
  Rule: ANY time you use FROM ... JOIN ..., prefix ALL column references with alias.

SOFT DELETE:
  Most tables: WHERE NOT deleted          (deleted is BOOLEAN)
  outreaches:  WHERE NOT "isDeleted"      (isDeleted is BOOLEAN)

DATE FILTERING — CRITICAL RULES:
  Date columns are TEXT — ALWAYS use NULLIF to avoid empty-string cast errors:
    NULLIF(invoice_date, '')::timestamptz       ← correct
    invoice_date::timestamptz                   ← FAILS on empty strings

  For TODAY:
    WHERE NULLIF(due_date,'')::date = CURRENT_DATE

  For a SPECIFIC MONTH (e.g. "December 2025"):
    WHERE DATE_TRUNC('month', NULLIF(due_date,'')::timestamptz) = DATE '2025-12-01'
    ✗ NEVER: WHERE due_date::date = '2025-12-01'  ← checks one day only, WRONG

  For THIS MONTH:
    WHERE DATE_TRUNC('month', NULLIF(col,'')::timestamptz) = DATE_TRUNC('month', CURRENT_DATE)

  For LAST MONTH:
    WHERE DATE_TRUNC('month', NULLIF(col,'')::timestamptz) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')

  For a YEAR (e.g. "2025"):
    WHERE DATE_TRUNC('year', NULLIF(col,'')::timestamptz) = DATE '2025-01-01'

  For BETWEEN dates:
    WHERE NULLIF(col,'')::timestamptz BETWEEN '2025-01-01' AND '2025-12-31'

NULL HANDLING — always wrap TEXT-to-number casts:
  NULLIF(document->>'grand_total', '')::numeric   ← safe
  (document->>'grand_total')::numeric             ← FAILS on NULL/empty rows

NUMERIC COLUMNS (no casting needed — already numeric):
  grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal, "targetInUSD"

OPEN/WON/LOST DEALS — use JSONB (avoids mixed-case quoting):
  Open:  document->>'dealWonAt' IS NULL AND document->>'dealLostAt' IS NULL AND NOT deleted
  Won:   document->>'dealWonAt' IS NOT NULL AND NOT deleted
  Lost:  document->>'dealLostAt' IS NOT NULL AND NOT deleted

SALES TABLE STATUS VALUES — EXACT strings (case-sensitive):
  status = 'Confirm'   ← confirmed/active sales orders
  status = 'Draft'     ← drafts
  status = 'Cancel'    ← cancelled
  Revenue query: WHERE status = 'Confirm' AND NOT deleted

INVOICE STATUS VALUES:
  payment_status: 'paid','unpaid','cancelled','draft','partial_payment','approved','rejected'
  approval_status: 'approved','rejected','pending','submitted'

TASKS TABLE:
  Table name:    "createtasks"   (NOT "tasks")
  Title column:  document->>'Task'   (capital T — use JSONB)
  Assignee:      "createdBy" = users._id

SELECT COLUMNS — always include a human-readable name first:
  deals:       d.name, d.stage, d.grand_total_in_usd
  invoices:    i.invoice_number, i.payment_status, i.grandtotal_in_usd
  companies:   document->>'companyName' (JSONB — mixed case)
  contacts:    document->>'firstName', document->>'lastName' (JSONB)
  users:       u.name, u.email
  createtasks: document->>'Task', t.status, t.priority

AGGREGATION — always handle NULLs:
  COALESCE(SUM(grand_total_in_usd), 0) AS total
  COUNT(*) AS count   (never returns NULL)

ONLY generate SELECT. Never UPDATE, DELETE, INSERT, DROP, CREATE."""


def _build_primary_prompt(query: str) -> Tuple[str, str]:
    """Prompt for Qwen2.5-Coder 3B fine-tuned (fast, direct SQL generation)."""
    user_msg = (
        f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
        f"Question: {query}\n\n"
        "Write ONLY the SQL query:"
    )
    return _SYSTEM_SQL_EXPERT, user_msg


def _build_fallback_prompt(query: str, prev_sql: str = "", error: str = "") -> Tuple[str, str]:
    """Prompt for Arctic-Text2SQL-R1-7B (accurate, with reasoning context)."""
    system = (
        _SYSTEM_SQL_EXPERT
        + "\n\nThink carefully about the correct table names, JOIN keys, and filters. "
        "The primary key is always _id. Use it for all JOINs."
    )
    if prev_sql and error:
        user_msg = (
            f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
            f"Question: {query}\n\n"
            f"Previous SQL attempt failed:\n{prev_sql}\n"
            f"Error: {error}\n\n"
            "Fix the SQL and write ONLY the corrected query:"
        )
    else:
        user_msg = (
            f"Database Schema:\n{_CREATE_TABLE_SCHEMA}\n\n"
            f"Question: {query}\n\n"
            "Write ONLY the SQL query:"
        )
    return system, user_msg


# ══════════════════════════════════════════════════════════════════════════════
# SQL EXTRACTION AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sql(raw: str) -> Optional[str]:
    """Extract a clean SELECT statement from model output."""
    if not raw:
        return None

    cleaned = raw.strip()
    cleaned = re.sub(r"^(?:Here is|The SQL|SQL query|Answer|Result)\s*:?\s*", "", cleaned, flags=re.I)

    # ```sql ... ``` code block
    code_blocks = re.findall(r"```(?:sql|SQL|postgresql)?\s*(SELECT.+?)```", cleaned, re.DOTALL)
    if code_blocks:
        return _clean_sql(code_blocks[-1])

    # <execute>...</execute>
    exec_m = re.search(r"<execute>\s*(SELECT.+?)(?:</execute>|```|$)", cleaned, re.DOTALL | re.I)
    if exec_m:
        return _clean_sql(exec_m.group(1))

    # Bare SELECT statement
    m = re.search(
        r"(SELECT\b.+?)(?:;|\n\n\n|Explanation:|Note:|Question:|$)",
        cleaned, re.DOTALL | re.I,
    )
    if m:
        return _clean_sql(m.group(1))

    return None


def _clean_sql(sql: str) -> str:
    """Normalise extracted SQL: strip semicolons, trailing comments, whitespace."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s*--[^\n]*$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def _validate_sql(sql: str, table_names: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate SQL for safety and correctness.

    Accepts BOTH:
      • Direct column access:  SELECT name FROM "deals"
      • JSONB document access: SELECT document->>'name' FROM "deals"
    Both work because the DB has flat columns AND a populated document JSONB.
    """
    if not sql or len(sql) < 10:
        return False, "empty or too short SQL"

    if not sql.upper().strip().startswith("SELECT"):
        return False, "not a SELECT statement"

    # Block destructive operations
    if re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|GRANT|REVOKE|CREATE)\b", sql, re.I):
        return False, "destructive operation detected"

    # Wrong dialect functions
    if re.search(r"\bIFNULL\b|\bNVL\b|\bDATEDIFF\b|\bSTRFTIME\b", sql, re.I):
        return False, "wrong SQL dialect (MySQL/SQLite function)"

    # Unknown table reference check
    if table_names:
        used_tables = re.findall(r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', sql, re.I)
        unknown = [t for t in used_tables if t.lower() not in [x.lower() for x in table_names]]
        if unknown:
            return False, f"unknown tables: {unknown}"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL SQL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_model(
    model:       str,
    system_msg:  str,
    user_msg:    str,
    timeout:     int,
    table_names: List[str],
    label:       str,
) -> Optional[str]:
    """Generate + validate SQL with one model. Returns valid SQL or None."""
    raw = _call_ollama_chat(model, system_msg, user_msg, timeout)
    if not raw:
        LOGGER.debug("[%s] no response from Ollama", label)
        return None

    sql = _extract_sql(raw)
    if not sql:
        LOGGER.debug("[%s] could not extract SQL from: %.100s", label, raw)
        return None

    ok, err = _validate_sql(sql, table_names)
    if not ok:
        LOGGER.debug("[%s] SQL validation failed: %s | SQL: %.80s", label, err, sql)
        return None

    LOGGER.debug("[%s] valid SQL (%d chars): %.80s", label, len(sql), sql)
    return sql


def _generate_sql_groq(query: str, table_names: List[str]) -> Optional[str]:
    """Generate SQL using Groq (fast cloud model, ~2s).

    This is the PRIMARY path — tried before Ollama because Groq is ~10x faster.
    Falls back gracefully when Groq is unavailable or rate-limited.
    """
    try:
        from config import settings
        groq_keys = settings.groq_api_keys
        if not groq_keys:
            return None

        import json as _json
        import urllib.request as _ur
        import urllib.error as _ue

        # Compact schema (enough for SQL generation, avoids token waste)
        compact_schema = _CREATE_TABLE_SCHEMA

        sys_prompt = _SYSTEM_SQL_EXPERT
        user_prompt = (
            f"Database Schema:\n{compact_schema}\n\n"
            f"Question: {query}\n\n"
            "Write ONLY the SQL query (no explanation, no markdown):"
        )

        payload = {
            "model":       settings.groq_sql_model,
            "messages":    [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens":  600,
        }

        _GROQ_HEADERS = {
            "Content-Type": "application/json",
            "User-Agent":   "groq-python/0.9.0",
            "Authorization": f"Bearer {groq_keys[0]}",
        }

        import time as _time
        import re as _re

        def _parse_retry_after(exc: "_ue.HTTPError") -> float:
            """Extract wait seconds from Groq 429 response headers/body."""
            try:
                # Try retry-after header first
                ra = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
                if ra:
                    return max(1.0, float(ra) + 0.5)
                # Try parsing body: "Please try again in 5.1s"
                body = exc.read().decode(errors="ignore")
                m = _re.search(r"try again in ([\d.]+)s", body)
                if m:
                    return max(1.0, float(m.group(1)) + 0.5)
            except Exception:
                pass
            return 10.0  # safe default

        def _try_key(key: str):
            """Returns (sql_or_sentinel, wait_seconds_if_rate_limited)."""
            _GROQ_HEADERS["Authorization"] = f"Bearer {key}"
            try:
                req = _ur.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=_json.dumps(payload).encode(),
                    headers=_GROQ_HEADERS,
                    method="POST",
                )
                with _ur.urlopen(req, timeout=_GROQ_SQL_TIMEOUT) as resp:
                    data = _json.loads(resp.read())
                    raw = data["choices"][0]["message"]["content"].strip()
                    sql = _extract_sql(raw)
                    if not sql:
                        LOGGER.debug("Groq SQL: could not extract SQL from response")
                        return "INVALID", 0
                    ok, err = _validate_sql(sql, table_names)
                    if ok:
                        LOGGER.info("Groq SQL generated (%d chars): %.80s", len(sql), sql)
                        return sql, 0
                    LOGGER.debug("Groq SQL invalid: %s | sql=%.80s", err, sql)
                    return "INVALID", 0
            except _ue.HTTPError as e:
                if e.code == 429:
                    wait = _parse_retry_after(e)
                    LOGGER.warning("Groq SQL: 429 on key ...%s (retry in %.1fs)", key[-6:], wait)
                    return "RATE_LIMIT", wait
                LOGGER.warning("Groq SQL HTTP %d", e.code)
                return "ERROR", 0
            except Exception as exc:
                LOGGER.debug("Groq SQL error: %s", exc)
                return "ERROR", 0

        # Try each key; if rate-limited, track the max wait needed
        max_wait = 0.0
        all_rate_limited = True
        for key in groq_keys:
            result, wait = _try_key(key)
            if result not in ("RATE_LIMIT", "INVALID", "ERROR"):
                return result
            if result != "RATE_LIMIT":
                all_rate_limited = False
            else:
                max_wait = max(max_wait, wait)
                _time.sleep(0.2)  # brief pause before next key

        # All keys rate-limited → wait for the longest reset, then retry once
        if all_rate_limited and max_wait > 0:
            LOGGER.warning("Groq SQL: all keys rate-limited — waiting %.1fs", max_wait)
            _time.sleep(max_wait)
            for key in groq_keys:
                result, wait = _try_key(key)
                if result not in ("RATE_LIMIT", "INVALID", "ERROR"):
                    return result
                if result == "RATE_LIMIT":
                    _time.sleep(0.2)

        return None
    except Exception as exc:
        LOGGER.debug("Groq SQL path error: %s", exc)
        return None


def generate_sql(query: str, agent) -> Optional[str]:
    """Generate SQL for a query.

    Strategy (fastest first):
      0. Groq cloud (llama-3.3-70b) — ~2s. PRIMARY path when API key available.
      1. Ollama PRIMARY (Qwen 3B) + FALLBACK (Arctic 7B) in parallel — ~8-20s
      2. Retry with FALLBACK model alone — up to 15s

    Returns validated SQL string, or None if all attempts fail.
    """
    table_names = get_table_names(agent)

    # ── Step 0: Groq fast path (~2s) ─────────────────────────────────────────
    with Timer("groq_sql") as groq_timer:
        groq_sql = _generate_sql_groq(query, table_names)
    if groq_sql:
        LOGGER.info("SQL via Groq in %.0fms: %.80s", groq_timer.elapsed_ms, groq_sql)
        return groq_sql

    LOGGER.info("Groq SQL unavailable (%.0fms) → trying Ollama", groq_timer.elapsed_ms)

    primary_sys,  primary_user  = _build_primary_prompt(query)
    fallback_sys, fallback_user = _build_fallback_prompt(query)

    # ── Parallel execution ────────────────────────────────────────────────────
    with Timer("text2sql_parallel") as timer:
        primary_sql:  Optional[str] = None
        fallback_sql: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2s") as pool:
            f_primary = pool.submit(
                _run_model,
                settings.ollama_primary_model,
                primary_sys,
                primary_user,
                settings.ollama_primary_timeout,
                table_names,
                "primary(Qwen3B)",
            )
            f_fallback = pool.submit(
                _run_model,
                settings.ollama_fallback_model,
                fallback_sys,
                fallback_user,
                settings.ollama_fallback_timeout,
                table_names,
                "fallback(Arctic7B)",
            )

            futures = {f_primary: "primary", f_fallback: "fallback"}
            for future in as_completed(futures, timeout=_PARALLEL_TIMEOUT_S):
                label = futures[future]
                try:
                    sql = future.result(timeout=2)
                    if sql:
                        if label == "primary":
                            primary_sql = sql
                        else:
                            fallback_sql = sql
                        LOGGER.info(
                            "Text2SQL: %s model produced SQL (%.0fms): %.80s",
                            label, timer.elapsed_ms, sql,
                        )
                except Exception as exc:
                    LOGGER.debug("Text2SQL %s future error: %s", label, exc)

    # Prefer primary if both succeeded; otherwise use whichever worked
    best_sql = primary_sql or fallback_sql
    if best_sql:
        LOGGER.info("Text2SQL parallel: selected %s SQL", "primary" if primary_sql else "fallback")
        return best_sql

    # ── Retry with fallback model (more powerful, more time) ─────────────────
    LOGGER.info("Text2SQL parallel both failed → retry with Arctic-7B fallback")
    retry_sys, retry_user = _build_fallback_prompt(query)
    retry_raw = _call_ollama_chat(
        settings.ollama_fallback_model,
        retry_sys,
        retry_user,
        timeout=settings.ollama_fallback_timeout,
        max_tokens=800,
    )
    if retry_raw:
        retry_sql = _extract_sql(retry_raw)
        if retry_sql:
            ok, err = _validate_sql(retry_sql, table_names)
            if ok:
                LOGGER.info("Text2SQL retry success: %.80s", retry_sql)
                return retry_sql
            # One more attempt with error feedback
            LOGGER.debug("Text2SQL retry SQL invalid (%s) — trying with error hint", err)
            fix_sys, fix_user = _build_fallback_prompt(query, retry_sql, err)
            fix_raw = _call_ollama_chat(
                settings.ollama_fallback_model,
                fix_sys,
                fix_user,
                timeout=settings.ollama_fallback_timeout,
                max_tokens=800,
            )
            if fix_raw:
                fix_sql = _extract_sql(fix_raw)
                if fix_sql:
                    ok2, _ = _validate_sql(fix_sql, table_names)
                    if ok2:
                        LOGGER.info("Text2SQL fix-retry success: %.80s", fix_sql)
                        return fix_sql

    LOGGER.warning("Text2SQL: all attempts failed for query: %.60s", query)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names referenced in a SQL query."""
    return sorted({
        m for m in re.findall(
            r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            sql, re.IGNORECASE,
        ) if m
    })


def _extract_column_headers(sql: str) -> List[str]:
    """Extract column aliases or names from the SELECT clause."""
    select_m = re.search(r"SELECT\s+(.+?)\s+FROM\b", sql, re.DOTALL | re.I)
    if not select_m:
        return []

    select_clause = select_m.group(1)
    headers: List[str] = []

    # Split by top-level commas
    depth, current = 0, ""
    for ch in select_clause:
        if ch == "(":   depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            headers.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        headers.append(current.strip())

    result = []
    for expr in headers:
        as_m = re.search(r"\bAS\s+\"?(\w+)\"?\s*$", expr, re.I)
        if as_m:
            result.append(as_m.group(1))
            continue
        doc_m = re.search(r"document->>'(\w+)'\s*$", expr)
        if doc_m:
            result.append(doc_m.group(1))
            continue
        agg_m = re.search(r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(", expr, re.I)
        if agg_m:
            result.append(agg_m.group(1).lower())
            continue
        # bare column or table.column
        bare = re.search(r'(?:\w+\.)?"?(\w+)"?\s*$', expr)
        if bare:
            result.append(bare.group(1))
            continue
        result.append(expr[:20].strip())

    return result


def _format_result(rows: List[Any], sql: str, query: str) -> str:
    """Format SQL result rows into user-friendly output.

    CRITICAL: NEVER return 'No data found' when rows actually exist.
    Only return empty-result message when row count is genuinely 0.
    """
    # ── Truly empty result ────────────────────────────────────────────────────
    if not rows:
        return "No records found for this query."

    first = rows[0]

    # Single-row all-NULL result (e.g. COUNT(*) with no matches = 0, not NULL)
    if isinstance(first, (list, tuple)) and len(rows) == 1 and all(v is None for v in first):
        return "No records found for this query."

    if first is None:
        return "No records found for this query."

    # ── Single scalar value (e.g. COUNT(*) = 176) ────────────────────────────
    if len(rows) == 1 and not isinstance(first, (list, tuple)):
        n = coerce_number(first)
        if isinstance(n, (int, float)):
            return f"**{fmt_number(n)}**"
        return f"**{first}**"

    if len(rows) == 1 and isinstance(first, (list, tuple)) and len(first) == 1:
        val = first[0]
        # Treat 0 as a valid count — not "no data"
        if val is None:
            return "No records found for this query."
        n = coerce_number(val)
        if isinstance(n, (int, float)):
            return f"**{fmt_number(n)}**"
        return f"**{val}**"

    # ── Multi-column single row (detail view) ─────────────────────────────────
    if len(rows) == 1 and isinstance(first, (list, tuple)):
        headers = _extract_column_headers(sql)
        if headers and len(headers) == len(first):
            lines = []
            for h, v in zip(headers, first):
                if v is not None and str(v).strip():
                    lines.append(f"**{h.replace('_',' ').title()}:** {v}")
            if lines:
                return "\n\n".join(lines)

    # ── Multi-row result ──────────────────────────────────────────────────────
    headers      = _extract_column_headers(sql)
    display_rows = rows[:_MAX_RESULT_ROWS]

    table = format_rows_as_markdown_table(
        display_rows,
        headers=headers or None,
        max_rows=_MAX_RESULT_ROWS,
        show_overflow=len(rows) > _MAX_RESULT_ROWS,
    )

    total     = len(rows)
    showing   = min(total, _MAX_RESULT_ROWS)
    count_str = (
        f"**{total}**" if total == showing
        else f"**{showing}** of **{total}** total"
    )
    return f"Found {count_str} record(s):\n\n{table}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """Generate SQL → execute → return formatted result dict.

    Includes auto-repair: if the first SQL fails at execution, we retry
    once with the error message fed back to the fallback model.

    Returns result dict {answer, tables_used, confidence, sql_queries}
    or None if SQL generation AND repair both fail entirely.
    """
    with Timer("text2sql_total") as timer:
        LOGGER.info("Text2SQL: processing: %.60s", query)

        sql = generate_sql(query, agent)
        if not sql:
            LOGGER.warning("Text2SQL: no valid SQL generated for: %.60s", query)
            return None

        _res = run_sql(agent, sql)

        # ── Auto-repair: retry with error hint if execution failed ────────────
        if _res.error:
            LOGGER.warning(
                "Text2SQL exec failed (attempt 1/2): %s | SQL: %.120s",
                _res.error, sql,
            )
            # Feed the error back to the fallback model and retry once
            try:
                table_names_r = get_table_names(agent)
                fix_sys, fix_user = _build_fallback_prompt(query, sql, _res.error)
                fix_raw = _call_ollama_chat(
                    settings.ollama_fallback_model,
                    fix_sys, fix_user,
                    timeout=settings.ollama_fallback_timeout,
                    max_tokens=800,
                )
                if fix_raw:
                    fix_sql = _extract_sql(fix_raw)
                    if fix_sql:
                        ok2, _ = _validate_sql(fix_sql, table_names_r)
                        if ok2:
                            _res2 = run_sql(agent, fix_sql)
                            if not _res2.error:
                                LOGGER.info(
                                    "Text2SQL auto-repair succeeded: %.80s", fix_sql
                                )
                                sql   = fix_sql
                                _res  = _res2
                            else:
                                LOGGER.warning(
                                    "Text2SQL repair also failed: %s", _res2.error
                                )
                                return None
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            except Exception as exc:
                LOGGER.warning("Text2SQL auto-repair error: %s", exc)
                return None

        rows        = _res.rows
        tables_used = _extract_tables_from_sql(sql)
        body        = _format_result(rows, sql, query)

        # Confidence reflects execution quality, not row count.
        # An empty valid result is still a confident answer (0.88).
        confidence = 0.90 if rows else 0.88

    LOGGER.info(
        "Text2SQL: done %.0fms | rows=%d | tables=%s | sql=%.80s",
        timer.elapsed_ms, len(rows) if rows else 0, tables_used, sql,
    )

    return {
        "answer":      body,
        "tables_used": tables_used,
        "confidence":  confidence,
        "sql_queries": [sql],
    }
