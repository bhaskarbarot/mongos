"""text2sql.py — SQL generation engine.

SQL generation priority (via llm_router):
  1. Groq Scout-17b    (compact schema, ~1-2s, high TPM budget)
  2. Gemini flash-lite (fallback when Groq blocked)
  3. OpenRouter DeepSeek (second fallback)
  4. Ollama local models (always-available last resort)

Key design:
  • Compact dynamic schema (schema_compact.py) — only relevant tables sent
    Reduces tokens from ~3500 → ~400-700 per SQL call (83% reduction)
  • llm_router.py — per-key backoff, rotation, SQL cache, no global sleeps
  • temperature=0, max_tokens=250 for deterministic SQL
  • Auto-repair: if SQL fails at execution, retry with error hint via router

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
_MAX_SQL_RETRIES    = 3  # 3-attempt self-heal loop
_PARALLEL_TIMEOUT_S = 20
_MAX_RESULT_ROWS    = 500
_SQL_EXEC_TIMEOUT_S = 20


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
-- SOFT DELETE  : WHERE NOT deleted   (deals, invoices, sales, companies, contacts, createtasks, bills)
--                outreaches ONLY: WHERE NOT "isDeleted"
--                vendors: NO deleted column — no soft delete filter needed
-- JOIN KEY     : always use _id column directly
-- DIRECT COLS  : ALL fields are individual columns — use them directly (NO document JSONB column)
--                mixed-case columns need double-quotes: d."createdAt", d."dealWonAt", u."isActive"
-- NUMERIC COLS : grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal,
--                "targetInUSD", "netPayableAmount" — already NUMERIC, no cast needed
-- TARGETS COLS : targets.year and targets.month are NUMERIC integers — NOT date columns
--                NEVER cast them to timestamptz. Use: WHERE t.year = 2026
-- DATE COLS    : stored as TIMESTAMPTZ — use directly, NO NULLIF cast needed:
--                WHERE d."createdAt" >= NOW() - INTERVAL '6 months'       ← correct
--                WHERE NULLIF(d."createdAt",'')::timestamptz >= ...        ← WRONG, causes error!
--                Month filter : DATE_TRUNC('month', d."createdAt") = DATE '2025-09-01'
--                Year filter  : EXTRACT(YEAR FROM d."createdAt") = 2026
--                This year    : EXTRACT(YEAR FROM d."createdAt") = EXTRACT(YEAR FROM CURRENT_DATE)
--                Last N months: d."createdAt" >= NOW() - INTERVAL '6 months'

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
    "closeDate" TIMESTAMPTZ,       -- expected close date (TEXT → cast ::timestamptz)
    "dealWonAt" TIMESTAMPTZ,       -- NULL=not yet won; NOT NULL=won date
    "dealLostAt" TIMESTAMPTZ,      -- NULL=not yet lost; NOT NULL=lost date
    "createdAt" TEXT,
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
    invoice_date TIMESTAMPTZ,      -- invoice creation date (TEXT → cast ::timestamptz)
    due_date TIMESTAMPTZ,          -- payment due date (TEXT → cast ::timestamptz)
    payment_date TIMESTAMPTZ,      -- actual payment received date (TEXT → cast ::timestamptz)
    deleted BOOLEAN,
    "createdBy" TEXT,       -- → users._id
    "companyName" TEXT,     -- company name snapshot (denormalised)
    "invoiceFor" TEXT,      -- 'Elsner Technologies Pvt. Ltd.' etc.
    "payment_mode" TEXT,    -- bank/payment method name
);
-- ✓ Revenue by period: SUM(grandtotal_in_usd) WHERE payment_status='paid'
--   AND DATE_TRUNC('month', payment_date) = DATE '2025-09-01'
-- ✓ Overdue: WHERE due_date < NOW() AND payment_status NOT IN ('paid','cancelled')
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
    sales_date TIMESTAMPTZ,        -- TEXT → cast ::timestamptz
    deleted BOOLEAN,
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
);

CREATE TABLE "createtasks" (   -- ⚠️ table is 'createtasks' NOT 'tasks'
    _id TEXT PRIMARY KEY,
    "Task" TEXT,        -- task title (capital T — double-quote: t."Task")
    status TEXT,        -- 'Pending' | 'Completed' | 'Open'
    priority TEXT,      -- 'Low' | 'Medium' | 'High'
    "createdBy" TEXT,   -- → users._id (task assignee/owner)
    due_date TIMESTAMPTZ,      -- TEXT → cast ::timestamptz
    company TEXT,       -- → companies._id
    "companyId" TEXT,   -- same as company
    "dealsId" TEXT,     -- → deals._id (if task linked to a deal)
    "invoiceId" TEXT,   -- → invoices._id
    deleted BOOLEAN,
    "createdAt" TEXT,
);
-- ✓ Pending: WHERE t.status='Pending' AND NOT t.deleted
-- ✓ Overdue: WHERE t.due_date < NOW() AND t.status!='Completed' AND NOT t.deleted
-- ✓ By user: LEFT JOIN "users" u ON u._id = t."createdBy"  → use t.status, t."Task"

CREATE TABLE "targets" (
    _id TEXT PRIMARY KEY,
    "userId" TEXT,          -- → users._id
    "targetInUSD" NUMERIC,  -- monthly sales target in USD
    month NUMERIC,          -- INTEGER 1–12  ← NOT a date column, NO timestamptz cast!
    year NUMERIC,           -- INTEGER e.g. 2026 ← NOT a date column, NO timestamptz cast!
    "teamName" TEXT,        -- e.g. 'Accounts Team'
    "createdAt" TEXT,
);
-- ✓ This year targets:  WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE)
-- ✓ This month targets: WHERE t.month = EXTRACT(MONTH FROM CURRENT_DATE) AND t.year = EXTRACT(YEAR FROM CURRENT_DATE)
-- ✓ With user: LEFT JOIN "users" u ON u._id = t."userId"
-- ✓ Target vs achieved (sales rep this year):
--   FROM "targets" t
--   LEFT JOIN "users" u ON u._id = t."userId"
--   LEFT JOIN "sales" s ON s."salesOwner" = t."userId"
--     AND EXTRACT(YEAR FROM s.sales_date) = t.year
--     AND s.status = 'Confirm' AND NOT s.deleted
--   WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE)
--   GROUP BY u.name, t."targetInUSD"
--   → achieved = COALESCE(SUM(s.grand_total_in_usd), 0), gap = target - achieved

CREATE TABLE "meetings" (
    _id TEXT PRIMARY KEY,
    title TEXT,             -- meeting title e.g. 'Client Call'
    description TEXT,
    start TIMESTAMPTZ,             -- start datetime (TEXT → cast ::timestamptz)
    "end" TIMESTAMPTZ,             -- end datetime
    location TEXT,
    attendees JSONB,        -- list of attendee objects
    "EventId" TEXT,         -- Google Calendar event ID
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
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
);
-- ✓ Filter: WHERE NOT o."isDeleted"
-- ✓ Conversion rate: COUNT(*) FILTER (WHERE status='Converted to Deal') / COUNT(*) * 100

CREATE TABLE "campaigns" (
    _id TEXT PRIMARY KEY,
    "campaignName" TEXT,
    "categoryId" TEXT,      -- → categories._id
    "createdBy" TEXT,
    "createdAt" TEXT,
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
    -- ⚠️ NO deleted column — vendors has no soft-delete. Omit WHERE NOT deleted for vendors.
);
-- ✓ Count all vendors: SELECT COUNT(*) FROM "vendors"
-- ✓ Active vendors: WHERE v.stage = 'Active'
-- ✓ With bill count: LEFT JOIN "bills" b ON b.vendor = v._id  → COUNT(b._id), SUM(b."netPayableAmount")

CREATE TABLE "bills" (
    _id TEXT PRIMARY KEY,
    vendor TEXT,            -- → vendors._id
    "systemBillNo" TEXT,    -- e.g. 'BILL-14'
    "vendorInvoiceNo" TEXT,
    "billDate" TIMESTAMPTZ,        -- TEXT → cast ::timestamptz
    "dueDate" TIMESTAMPTZ,         -- payment due date
    status TEXT,            -- 'Payment Scheduled' | 'Paid' | 'Pending' | 'Draft'
    "netPayableAmount" NUMERIC,
    subtotal NUMERIC,
    "gstPercent" NUMERIC,
    "billType" TEXT,        -- 'Service' | 'Product'
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
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
    date TIMESTAMPTZ,              -- TEXT → cast ::timestamptz
    "createdAt" TEXT,
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
    "createdAt" TIMESTAMPTZ  -- use directly, NO cast
);
-- ⚠ "notes" table is for OUTREACH notes only — it has NO "type" column
-- ⚠ commonnotes is for CRM entity notes — it HAS "type" TEXT column
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

CREATE TABLE "notes" (
    _id TEXT PRIMARY KEY,
    "outreachId" TEXT,      -- → outreaches._id
    "contactMethod" TEXT,   -- e.g. 'Email' | 'Call' | 'Meeting'
    message TEXT,
    "reminderDate" TEXT,
    "createdBy" TEXT,       -- → users._id
    "createdAt" TEXT,
);
-- ✓ Outreach notes: WHERE n."outreachId" = outreach_id

CREATE TABLE "publicleads" (
    _id TEXT PRIMARY KEY,
    "firstName" TEXT,
    "lastName" TEXT,
    email TEXT,
    "phoneNumber" TEXT,
    source TEXT,
    "leadStatus" TEXT,
    "lifecycleStage" TEXT,
    "userType" TEXT,
    description TEXT,
    "createdAt" TEXT,
);
-- ✓ Public leads: list of inbound/web form leads

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
-- Emails + user:     FROM "emails" e LEFT JOIN "users" u ON u._id = e."user"
--
-- ─── COMPLEX MULTI-TABLE PATTERNS ───────────────────────────────────────────
--
-- SALES LEADERBOARD (rank, rep, dept, revenue, deals won, pending tasks):
--   SELECT RANK() OVER (ORDER BY COALESCE(SUM(s.grand_total_in_usd),0) DESC) AS rank,
--     u.name AS rep, d.name AS department,
--     COALESCE(SUM(s.grand_total_in_usd),0) AS revenue,
--     COUNT(DISTINCT CASE WHEN dl."dealWonAt" IS NOT NULL THEN dl._id END) AS deals_won,
--     COUNT(DISTINCT CASE WHEN t.status='Pending' THEN t._id END) AS pending_tasks
--   FROM "users" u
--   LEFT JOIN "departments" d ON d._id = u.department
--   LEFT JOIN "sales" s ON s."salesOwner" = u._id AND s.status='Confirm' AND NOT s.deleted
--   LEFT JOIN "deals" dl ON dl.owner = u._id AND NOT dl.deleted
--   LEFT JOIN "createtasks" t ON t."createdBy" = u._id AND NOT t.deleted
--   WHERE u."isActive" = true
--   GROUP BY u._id, u.name, d.name ORDER BY revenue DESC
--
-- COMPANY HEALTH (deals + invoices + tasks per company):
--   SELECT c."companyName",
--     COUNT(DISTINCT d._id) AS open_deals,
--     COUNT(DISTINCT CASE WHEN i.payment_status NOT IN ('paid','cancelled') THEN i._id END) AS pending_invoices,
--     COUNT(DISTINCT CASE WHEN t.status='Pending' THEN t._id END) AS open_tasks
--   FROM "companies" c
--   LEFT JOIN "deals" d ON d.company = c._id AND NOT d.deleted
--   LEFT JOIN "invoices" i ON i.company = c._id AND NOT i.deleted
--   LEFT JOIN "createtasks" t ON t.company = c._id AND NOT t.deleted
--   WHERE NOT c.deleted GROUP BY c._id, c."companyName"
--
-- CUSTOMER 360 (company + deals + revenue + unpaid + tasks):
--   SELECT c."companyName",
--     COUNT(DISTINCT d._id) AS deal_count,
--     COALESCE(SUM(CASE WHEN i.payment_status='paid' THEN i.grandtotal_in_usd END),0) AS paid_revenue,
--     COALESCE(SUM(CASE WHEN i.payment_status NOT IN ('paid','cancelled') THEN i.grandtotal_in_usd END),0) AS unpaid,
--     COUNT(DISTINCT CASE WHEN t.status='Pending' THEN t._id END) AS open_tasks
--   FROM "companies" c
--   LEFT JOIN "deals" d ON d.company = c._id AND NOT d.deleted
--   LEFT JOIN "invoices" i ON i.company = c._id AND NOT i.deleted
--   LEFT JOIN "createtasks" t ON t.company = c._id AND NOT t.deleted
--   WHERE NOT c.deleted GROUP BY c._id, c."companyName"
--
-- DEPT USER COUNT + PENDING TASKS:
--   SELECT d.name AS department, COUNT(DISTINCT u._id) AS user_count,
--     COUNT(CASE WHEN t.status='Pending' AND NOT t.deleted THEN 1 END) AS pending_tasks
--   FROM "departments" d
--   LEFT JOIN "users" u ON u.department = d._id AND u."isActive" = true
--   LEFT JOIN "createtasks" t ON t."createdBy" = u._id
--   GROUP BY d._id, d.name ORDER BY pending_tasks DESC
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
  invoice_number is a direct column. Search with ILIKE:
    WHERE invoice_number ILIKE '%ELSN/2025/1%'
  Use ILIKE (case-insensitive) not = (exact match).

CRITICAL — MIXED-CASE COLUMNS:
  Many columns have mixed case (dealWonAt, companyName, firstName, etc.).
  ALWAYS double-quote mixed-case column names:
    ✓ d."dealWonAt"              (double-quoted direct column — correct)
    ✓ u."isActive"               (double-quoted direct column — correct)
    ✗ dealWonAt                  (unquoted — PostgreSQL lowercases it, FAILS)

CRITICAL — JOIN QUERIES: ALWAYS prefix columns with table alias:
  In JOINs column references are AMBIGUOUS without a table alias prefix.
  ALWAYS write: d."dealWonAt"  NOT  "dealWonAt"
  Examples:
    ✓ FROM "deals" d LEFT JOIN "users" u ON u._id = d.owner
      WHERE d."dealWonAt" IS NOT NULL    ← d. prefix required
      AND NOT d.deleted                   ← d. prefix required
    ✗ WHERE "dealWonAt" IS NOT NULL       ← AMBIGUOUS — SQL ERROR
  Rule: ANY time you use FROM ... JOIN ..., prefix ALL column references with alias.

SOFT DELETE:
  Most tables: WHERE NOT deleted          (deleted is BOOLEAN)
  outreaches:  WHERE NOT "isDeleted"      (isDeleted is BOOLEAN)

DATE FILTERING — CRITICAL RULES:
  Date columns are TIMESTAMPTZ — use them DIRECTLY, NO cast needed:
    WHERE "createdAt" >= NOW() - INTERVAL '6 months'  ← correct
    NULLIF("createdAt",'')::timestamptz               ← WRONG, causes error!

  For TODAY:
    WHERE due_date::date = CURRENT_DATE

  For a SPECIFIC MONTH (e.g. "December 2025"):
    WHERE DATE_TRUNC('month', due_date) = DATE '2025-12-01'

  For THIS MONTH:
    WHERE DATE_TRUNC('month', col) = DATE_TRUNC('month', CURRENT_DATE)

  For LAST MONTH:
    WHERE DATE_TRUNC('month', col) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')

  For a YEAR filter:
    ✓ EXTRACT(YEAR FROM "createdAt") = 2026
    ✓ EXTRACT(YEAR FROM "createdAt") = EXTRACT(YEAR FROM CURRENT_DATE)

  For LAST N MONTHS:
    WHERE "createdAt" >= NOW() - INTERVAL '6 months'

  For BETWEEN dates:
    WHERE "createdAt" BETWEEN '2025-01-01' AND '2025-12-31'

TARGETS TABLE — year/month are INTEGER columns, NOT date columns:
  ✓ WHERE t.year = 2026                            ← integer comparison
  ✓ WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE) ← correct
  ✓ WHERE t.month = 5 AND t.year = 2026
  ✗ NEVER: NULLIF(t.year,'')::timestamptz          ← year is NUMERIC, not TEXT!
  ✗ NEVER: DATE_TRUNC('year', t.year)              ← year is already an integer!

  Target vs Achieved pattern:
  SELECT u.name, t."targetInUSD",
    COALESCE(SUM(s.grand_total_in_usd),0) AS achieved,
    t."targetInUSD" - COALESCE(SUM(s.grand_total_in_usd),0) AS gap
  FROM "targets" t
  LEFT JOIN "users" u ON u._id = t."userId"
  LEFT JOIN "sales" s ON s."salesOwner" = t."userId"
    AND EXTRACT(YEAR FROM s.sales_date) = t.year
    AND s.status = 'Confirm' AND NOT s.deleted
  WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE)
  GROUP BY u.name, t."targetInUSD"

VENDORS TABLE — NO deleted column:
  ✗ NEVER add WHERE NOT v.deleted  ← vendors has no deleted column, will CRASH!
  ✓ SELECT COUNT(*) FROM "vendors"            (no filter needed)
  ✓ WHERE v.stage = 'Active'                  (use stage for status filter)

NULL HANDLING — always use COALESCE for aggregations:
  COALESCE(SUM(grand_total_in_usd), 0)   ← safe
  COALESCE(grand_total_in_usd, 0)         ← safe for single values

NUMERIC COLUMNS (already NUMERIC — never cast to timestamptz):
  grand_total_in_usd, grandtotal_in_usd, grand_total, subtotal, "targetInUSD",
  "netPayableAmount", targets.year, targets.month

OPEN/WON/LOST DEALS — use direct quoted columns:
  Open:  d."dealWonAt" IS NULL AND d."dealLostAt" IS NULL AND NOT d.deleted
  Won:   d."dealWonAt" IS NOT NULL AND NOT d.deleted
  Lost:  d."dealLostAt" IS NOT NULL AND NOT d.deleted

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
  Title column:  t."Task"   (capital T, double-quoted)
  Assignee:      "createdBy" = users._id

SELECT COLUMNS — always include a human-readable name first:
  deals:       d.name, d.stage, d.grand_total_in_usd
  invoices:    i.invoice_number, i.payment_status, i.grandtotal_in_usd
  companies:   c."companyName"  (double-quoted — mixed case)
  contacts:    c."firstName", c."lastName"  (double-quoted)
  users:       u.name, u.email
  createtasks: t."Task", t.status, t.priority

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
    """Extract a clean SELECT or WITH (CTE) statement from model output."""
    if not raw:
        return None

    cleaned = raw.strip()
    # Strip ALL markdown fences anywhere in the string
    cleaned = re.sub(r"```(?:sql|SQL|postgresql)?\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    # Convert backtick-quoted identifiers to double-quoted (LLM MySQL habit)
    cleaned = re.sub(r"`([^`]+)`", r'"\1"', cleaned)
    # Strip common LLM preamble
    cleaned = re.sub(r"^(?:Here is|The SQL|SQL query|Answer|Result)\s*:?\s*", "", cleaned, flags=re.I)

    def _safe(candidate: str) -> Optional[str]:
        """Return cleaned SQL if safe, else None."""
        candidate = _clean_sql(candidate)
        upper = candidate.upper().lstrip()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return None
        if len(candidate) < 10:
            return None
        first_kw = candidate.strip().split()[0].upper()
        if first_kw in ("DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"):
            return None
        return candidate

    # Priority 1: XML-style tags the model uses: <sql>, <execute>, <query>
    for tag in ("sql", "execute", "query"):
        tag_m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", cleaned, re.DOTALL | re.IGNORECASE)
        if tag_m:
            candidate = tag_m.group(1).strip()
            candidate = re.split(r"\n{3,}|Explanation:|Note:|Question:", candidate, flags=re.IGNORECASE)[0].strip()
            result = _safe(candidate)
            if result:
                return result

    # Priority 2: find first SELECT (strict — no WITH-as-prose false positives)
    m = re.search(r"\bSELECT\b.+", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(0)
        sql = re.split(r";[\r\n]", sql)[0]   # Ollama fine-tuned: SQL ends at ; then explanation
        sql = re.split(r"\n{3,}|Explanation:|Note:|Question:|This query|The query|The SQL", sql, flags=re.IGNORECASE)[0]
        result = _safe(sql)
        if result:
            return result

    # Priority 3: WITH CTE — only if followed by identifier+AS+( (not English prose)
    m = re.search(r"\bWITH\s+\w+\s+AS\s*\(.+", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(0)
        sql = re.split(r";[\r\n]", sql)[0]
        sql = re.split(r"\n{3,}|Explanation:|Note:|Question:|This query|The query|The SQL", sql, flags=re.IGNORECASE)[0]
        result = _safe(sql)
        if result:
            return result

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

    # Unknown table reference check — skip SQL keywords/functions that appear after FROM
    _SQL_PSEUDO_TABLES = {
        "nullif","coalesce","current_date","current_timestamp","now","extract",
        "date_trunc","date_part","interval","rank","row_number","dense_rank",
        "lateral","unnest","generate_series","values","dual","information_schema",
        "pg_tables","json_array_elements","jsonb_array_elements",
    }
    if table_names:
        used_tables = re.findall(r'\b(?:FROM|JOIN)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', sql, re.I)
        known_lower = {x.lower() for x in table_names}
        unknown = [
            t for t in used_tables
            if t.lower() not in known_lower and t.lower() not in _SQL_PSEUDO_TABLES
        ]
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




# ══════════════════════════════════════════════════════════════════════════════
# SQL GENERATION  (via llm_router — compact schema, multi-provider fallback)
# ══════════════════════════════════════════════════════════════════════════════

from pipeline.llm_router import call as _router_call, sql_cache_get, sql_cache_set
from pipeline.schema_compact import get_compact_schema, COMPACT_SYSTEM_PROMPT


def _generate_sql_via_router(query: str, table_names: List[str]) -> Optional[str]:
    """Generate SQL using the centralized llm_router.

    Chain: Groq Scout-17b → Gemini flash-lite → OpenRouter DeepSeek → Ollama
    Uses compact dynamic schema (~400-700 tokens vs 3500 before).
    Results are cached for 5 minutes to avoid redundant API calls.
    """
    # Cache check (avoids re-calling LLM for repeated identical queries)
    cached = sql_cache_get(query)
    if cached:
        ok, _ = _validate_sql(cached, table_names)
        if ok:
            return cached

    # Build compact schema — only relevant tables
    compact_schema = get_compact_schema(query, extra_tables=table_names or [])

    user_prompt = (
        f"Schema:\n{compact_schema}\n\n"
        f"Question: {query}\n\n"
        "SQL:"
    )

    raw = _router_call("sql", COMPACT_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    sql = _extract_sql(raw)
    if not sql:
        LOGGER.debug("Router SQL: could not extract SQL from: %.80s", raw)
        return None

    ok, err = _validate_sql(sql, table_names)
    if ok:
        LOGGER.info("Router SQL generated (%d chars): %.80s", len(sql), sql)
        sql_cache_set(query, sql)
        return sql

    LOGGER.debug("Router SQL invalid: %s | sql=%.80s", err, sql)
    return None


# Keep these names so main.py SIMPLE-retry still imports them without change
def _generate_sql_groq(query: str, table_names: List[str]) -> Optional[str]:
    """Alias → routes through llm_router (Groq Scout first)."""
    return _generate_sql_via_router(query, table_names)


def _generate_sql_gemini(query: str, table_names: List[str]) -> Optional[str]:
    """Legacy alias — router already includes Gemini in fallback chain."""
    return _generate_sql_via_router(query, table_names)


def generate_sql(query: str, agent) -> Optional[str]:
    """Generate SQL for a query.

    Strategy:
      1. llm_router: Groq Scout → Gemini → OpenRouter → Ollama  (~1-5s)
      2. Ollama parallel (PRIMARY Qwen + FALLBACK Arctic) — if router fails
      3. Ollama Arctic alone with extra time

    Returns validated SQL string, or None if all attempts fail.
    """
    table_names = get_table_names(agent)

    # ── Step 1: Router (cloud LLMs with compact schema) ──────────────────────
    with Timer("router_sql") as router_timer:
        router_sql = _generate_sql_via_router(query, table_names)
    if router_sql:
        LOGGER.info("SQL via router in %.0fms: %.80s", router_timer.elapsed_ms, router_sql)
        return router_sql

    LOGGER.info("Router SQL unavailable (%.0fms) → Ollama parallel", router_timer.elapsed_ms)

    primary_sys,  primary_user  = _build_primary_prompt(query)
    fallback_sys, fallback_user = _build_fallback_prompt(query)

    # ── Step 2: Ollama parallel ───────────────────────────────────────────────
    with Timer("text2sql_parallel") as timer:
        primary_sql:  Optional[str] = None
        fallback_sql: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2s") as pool:
            f_primary = pool.submit(
                _run_model,
                settings.ollama_primary_model,
                primary_sys, primary_user,
                settings.ollama_primary_timeout,
                table_names, "primary(Qwen3B)",
            )
            f_fallback = pool.submit(
                _run_model,
                settings.ollama_fallback_model,
                fallback_sys, fallback_user,
                settings.ollama_fallback_timeout,
                table_names, "fallback(Arctic7B)",
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
                        LOGGER.info("Text2SQL %s SQL (%.0fms): %.80s", label, timer.elapsed_ms, sql)
                except Exception as exc:
                    LOGGER.debug("Text2SQL %s future error: %s", label, exc)

    best_sql = primary_sql or fallback_sql
    if best_sql:
        return best_sql

    # ── Step 3: Ollama Arctic alone (last resort) ─────────────────────────────
    LOGGER.info("Text2SQL parallel failed → Arctic fallback")
    retry_raw = _call_ollama_chat(
        settings.ollama_fallback_model, fallback_sys, fallback_user,
        timeout=settings.ollama_fallback_timeout, max_tokens=800,
    )
    if retry_raw:
        retry_sql = _extract_sql(retry_raw)
        if retry_sql:
            ok, err = _validate_sql(retry_sql, table_names)
            if ok:
                return retry_sql

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

    3-attempt self-heal loop:
      Attempt 1: Normal SQL generation via router (compact schema)
      Attempt 2: If column error → db_schema.find_correct_column() + rebuild prompt
      Attempt 3: Wider schema (max_tables=10) + explicit column correction hint

    Returns result dict {answer, tables_used, confidence, sql_queries}
    or None if all 3 attempts fail.
    """
    from pipeline.db_schema import find_correct_column, get_all_crm_tables

    with Timer("text2sql_total") as timer:
        LOGGER.info("Text2SQL: processing: %.60s", query)

        table_names  = get_table_names(agent)
        all_crm      = get_all_crm_tables()
        heal_hint:   str = ""
        last_sql:    Optional[str] = None
        current_query = query

        for attempt in range(1, _MAX_SQL_RETRIES + 1):
            # ── Generate SQL ──────────────────────────────────────────────────
            sql = _generate_sql_via_router(current_query, table_names)

            if not sql:
                # Router failed — try Ollama parallel on first attempt only
                if attempt == 1:
                    primary_sys,  primary_user  = _build_primary_prompt(query)
                    fallback_sys, fallback_user = _build_fallback_prompt(query)
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2s") as pool:
                        f_primary  = pool.submit(
                            _run_model, settings.ollama_primary_model,
                            primary_sys, primary_user,
                            settings.ollama_primary_timeout, table_names, "primary",
                        )
                        f_fallback = pool.submit(
                            _run_model, settings.ollama_fallback_model,
                            fallback_sys, fallback_user,
                            settings.ollama_fallback_timeout, table_names, "fallback",
                        )
                        for future in as_completed({f_primary, f_fallback}, timeout=_PARALLEL_TIMEOUT_S):
                            try:
                                candidate = future.result(timeout=2)
                                if candidate:
                                    sql = candidate
                                    break
                            except Exception:
                                pass

            if not sql:
                LOGGER.warning("Text2SQL attempt %d: no SQL generated", attempt)
                if attempt < _MAX_SQL_RETRIES:
                    continue
                return None

            last_sql = sql

            # ── Execute ───────────────────────────────────────────────────────
            _res = run_sql(agent, sql)

            if not _res.error:
                # Success
                rows        = _res.rows
                tables_used = _extract_tables_from_sql(sql)
                body        = _format_result(rows, sql, query)
                confidence  = 0.90 if rows else 0.88

                LOGGER.info(
                    "Text2SQL: done %.0fms (attempt %d) | rows=%d | tables=%s | sql=%.80s",
                    timer.elapsed_ms, attempt, len(rows) if rows else 0, tables_used, sql,
                )
                return {
                    "answer":      body,
                    "tables_used": tables_used,
                    "confidence":  confidence,
                    "sql_queries": [sql],
                }

            # ── Self-heal on error ────────────────────────────────────────────
            err_lower = _res.error.lower()
            LOGGER.warning(
                "Text2SQL attempt %d exec failed: %s | sql=%.120s",
                attempt, _res.error, sql,
            )

            if attempt >= _MAX_SQL_RETRIES:
                break  # exhausted all attempts

            # ── Error-type diagnosis → targeted repair prompt ─────────────────
            if "column" in err_lower and "exist" in err_lower:
                # Find the bad column name from the error message
                import re as _re
                bad_col_match = _re.search(
                    r'column\s+"?([^"\s]+)"?\s+(?:does not exist|of relation)',
                    _res.error, _re.IGNORECASE,
                )
                bad_col = bad_col_match.group(1).strip().strip('"') if bad_col_match else None

                if bad_col:
                    correct = find_correct_column(bad_col, all_crm)
                    if correct:
                        heal_hint = (
                            f"Column '{bad_col}' does not exist. "
                            f"Use '{correct}' instead. "
                        )
                        LOGGER.info("Text2SQL self-heal: %s → %s", bad_col, correct)
                    else:
                        heal_hint = (
                            f"Column '{bad_col}' does not exist in this DB. "
                            "Check the schema for the correct column name."
                        )
                else:
                    heal_hint = f"Column error: {_res.error[:200]}"

            elif "ambiguous" in err_lower:
                heal_hint = (
                    "Column reference is ambiguous. Prefix ALL columns with the "
                    "table alias (e.g. d.name, u.name — never bare 'name' in JOINs)."
                )

            elif "syntax" in err_lower or "parse" in err_lower:
                heal_hint = (
                    f"SQL syntax error: {_res.error[:150]}. "
                    "Check parentheses, quoting of mixed-case column names, "
                    "and NULLIF cast syntax."
                )

            else:
                heal_hint = f"Fix this error: {_res.error[:200]}"

            # Rebuild query with heal hint for next attempt
            current_query = (
                f"{query}\n\n"
                f"[SELF-HEAL ATTEMPT {attempt+1}: {heal_hint} "
                f"Previous SQL was: {sql[:300]}. Write corrected SQL.]"
            )

        LOGGER.warning("Text2SQL: all %d attempts failed for: %.60s", _MAX_SQL_RETRIES, query)
        return None
