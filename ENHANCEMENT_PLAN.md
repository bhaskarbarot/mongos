# CRM Chatbot — Complete Enhancement Plan
**Goal: 90%+ query accuracy | All query types working**

---

## ROOT CAUSE ANALYSIS

### Why queries fail today

| Problem | Impact | Root Cause |
|---|---|---|
| Fast path DISABLED | ~70% queries go to slow LLM pipeline | Was disabled during JSONB→column migration |
| SO01241 not found | Sales order lookups fail | No handler for SO/ELSN/ELS number patterns |
| Company shows as hex ID | Bad UX in answers | No JOIN to companies table in results |
| Line items not shown | Items JSONB not expanded | No JSONB array handler |
| Intent router misses | Many queries escalate to slow path | Too few pattern examples |
| Fast path uses document->> | Old JSONB syntax | Not updated after schema migration |

---

## PRIORITY 1 — RE-ENABLE FAST PATH (Biggest Impact)

Fast path is your highest-accuracy layer. It's currently **disabled** because handlers use old
`document->>'field'` syntax. Fix each handler → re-enable → 70% of queries answered in <500ms.

### Step 1: Fix fast_path.py handlers

**File:** `/home/elsner/Documents/mongos/pipeline/fast_path.py`

Every handler that uses `document->>` needs to be updated. The pattern is always the same:

```python
# OLD (broken - uses JSONB syntax)
WHERE COALESCE(document->>'deleted','false') != 'true'
AND document->>'stage' = 'Closed Won'
AND NULLIF(document->>'grand_total_in_usd','')::numeric > 0

# NEW (correct - direct column access)
WHERE (deleted = false OR deleted IS NULL)
AND stage = 'Closed Won'
AND grand_total_in_usd > 0
```

**Key replacements to make across ALL handlers:**

```python
# Soft delete
"COALESCE(document->>'deleted','false') != 'true'"
→ "(deleted = false OR deleted IS NULL)"

# For outreaches only (uses isDeleted not deleted)
"COALESCE(document->>'isDeleted','false') != 'true'"
→ '("isDeleted" = false OR "isDeleted" IS NULL)'

# Field access
"document->>'fieldName'"
→ '"fieldName"'   (quote mixed-case names)

# Number fields
"NULLIF(document->>'grand_total_in_usd','')::numeric"
→ "grand_total_in_usd"   (already NUMERIC column)

# Date fields
"NULLIF(document->>'closeDate','')::timestamptz"
→ 'NULLIF("closeDate",\'\')::timestamptz'   (TEXT column, needs cast)

# Text search
"document->>'name' ILIKE '%x%'"
→ '"name"::text ILIKE \'%x%\''

# Amount fields - NOTE: invoices uses grandtotal_in_usd (no underscore before 'in')
"document->>'grandtotal_in_usd'"  → '"grandtotal_in_usd"'
"document->>'grand_total_in_usd'" → '"grand_total_in_usd"'   (deals/sales)
```

**Handlers to fix (in order of importance):**

1. `_fp_revenue` — fix revenue coalesce expressions
2. `_fp_count` — fix deleted filter
3. `_fp_deals_filter` — fix stage, won/lost, deleted
4. `_fp_tasks` — fix status, deleted
5. `_fp_list_records` — fix deleted, all field access
6. `_fp_group_by` — fix group by field expressions
7. `_fp_targets` — fix userId field access
8. `_fp_pipeline_summary` — fix stage, deleted, amount
9. `_fp_top_customers` — fix amount, company join
10. `_fp_overdue_aging` — fix due_date, payment_status
11. `_fp_pending_invoices` — fix payment_status, due_date
12. `_fp_active_customers` — fix lifecycleStage
13. `_fp_no_activity` — fix date fields
14. `_fp_search` — fix name/email field access
15. `_fp_quarterly` — fix date and amount fields

### Step 2: Re-enable fast path in .env

```bash
# Change in /home/elsner/Documents/mongos/.env
FAST_PATH_ENABLED=true
```

---

## PRIORITY 2 — RECORD LOOKUP BY NUMBER (SO/Invoice/Deal)

### Create new file: `/home/elsner/Documents/mongos/pipeline/record_lookup.py`

```python
"""record_lookup.py — Fast record lookup by business number (SO, ELSN, ELS).

Handles: SO01241, ELSN/2026/018, ELS042
Expands JSONB items arrays as markdown tables.
Resolves company/user IDs to names via JOINs.
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List

from pipeline.schema import run_sql

LOGGER = logging.getLogger("sql_chatbot")


def _expand_items(items_raw, currency: str = "USD") -> str:
    """Convert JSONB items array to readable markdown table."""
    if not items_raw:
        return ""
    try:
        items = json.loads(items_raw) if isinstance(items_raw, str) else items_raw
        if not items:
            return ""
        lines = [
            "\n**Line Items:**\n",
            "| Product | Qty | Unit Price | Total ({}) |".format(currency),
            "| --- | --- | --- | --- |",
        ]
        for item in items:
            name     = item.get("product_name", "—")
            qty      = item.get("quantity", 0)
            unit     = item.get("unit_price", 0)
            total    = item.get("total_price", 0)
            total_usd = item.get("total_price_in_usd", 0)
            lines.append(f"| {name} | {qty} | {unit:,.2f} | {total:,.2f} (USD {total_usd:,.2f}) |")
        return "\n".join(lines)
    except Exception:
        return ""


def _so_lookup(agent, so_number: str, want_items: bool) -> Optional[Dict[str, Any]]:
    """Look up a sales order by SO number with company and owner."""
    # Normalize: SO01241 or SO1241 both work
    digits = re.search(r"\d+", so_number)
    if not digits:
        return None
    padded = "SO" + digits.group(0).zfill(5)
    bare   = "SO" + digits.group(0)

    sql = f"""
        SELECT s.sales_number, s.status, s.grand_total, s.grand_total_in_usd,
               s.currency, s.sales_date, s.items,
               c."companyName", u.name AS sales_rep,
               s.tax_amount, s.discount_amount, s.subtotal
        FROM sales s
        LEFT JOIN companies c ON c._id = s.company::text
        LEFT JOIN users     u ON u._id = s."salesOwner"::text
        WHERE s.sales_number IN ('{padded}', '{bare}')
          AND (s.deleted = false OR s.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row = res.rows[0]
    cols = ["sales_number","status","grand_total","grand_total_in_usd",
            "currency","sales_date","items","companyName","sales_rep",
            "tax_amount","discount_amount","subtotal"]
    data = dict(zip(cols, row))

    lines = [
        f"**Sales Order: {data['sales_number']}**\n",
        f"- **Company:** {data.get('companyName') or '—'}",
        f"- **Status:** {data.get('status') or '—'}",
        f"- **Sales Rep:** {data.get('sales_rep') or '—'}",
        f"- **Date:** {str(data.get('sales_date',''))[:10]}",
        f"- **Currency:** {data.get('currency') or 'USD'}",
        f"- **Subtotal:** {data.get('subtotal') or 0:,.2f}",
        f"- **Grand Total:** {data.get('grand_total') or 0:,.2f} "
        f"(USD {data.get('grand_total_in_usd') or 0:,.2f})",
    ]

    # Always show items for sales orders
    items_text = _expand_items(data.get("items"), data.get("currency","USD"))
    if items_text:
        lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["sales", "companies", "users"],
        "confidence":  0.98,
        "sql_queries": [sql],
    }


def _invoice_lookup(agent, inv_number: str) -> Optional[Dict[str, Any]]:
    """Look up an invoice by invoice_number."""
    # Normalize ELSN2026018 → ELSN/2026/018 or keep as-is
    safe = inv_number.replace("'", "''")
    sql = f"""
        SELECT i.invoice_number, i.payment_status, i.approval_status,
               i.grandtotal_in_usd, i.currency, i.invoice_date, i.due_date,
               i.payment_date, i.items, i."companyName",
               u.name AS owner_name, i.notes
        FROM invoices i
        LEFT JOIN users u ON u._id = i."invoiceOwner"::text
        WHERE i.invoice_number ILIKE '%{safe}%'
          AND (i.deleted = false OR i.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row = res.rows[0]
    cols = ["invoice_number","payment_status","approval_status","grandtotal_in_usd",
            "currency","invoice_date","due_date","payment_date","items",
            "companyName","owner_name","notes"]
    data = dict(zip(cols, row))

    lines = [
        f"**Invoice: {data['invoice_number']}**\n",
        f"- **Company:** {data.get('companyName') or '—'}",
        f"- **Payment Status:** {data.get('payment_status') or '—'}",
        f"- **Approval Status:** {data.get('approval_status') or '—'}",
        f"- **Amount:** USD {data.get('grandtotal_in_usd') or 0:,.2f}",
        f"- **Currency:** {data.get('currency') or 'USD'}",
        f"- **Invoice Date:** {str(data.get('invoice_date',''))[:10]}",
        f"- **Due Date:** {str(data.get('due_date',''))[:10]}",
        f"- **Payment Date:** {str(data.get('payment_date','') or 'Not paid')[:10]}",
        f"- **Owner:** {data.get('owner_name') or '—'}",
    ]
    if data.get("notes"):
        lines.append(f"- **Notes:** {data['notes'][:100]}")

    items_text = _expand_items(data.get("items"), data.get("currency","USD"))
    if items_text:
        lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["invoices", "users"],
        "confidence":  0.98,
        "sql_queries": [sql],
    }


def _deal_lookup(agent, deal_ref: str) -> Optional[Dict[str, Any]]:
    """Look up a deal by ELS number (e.g. ELS042 → sequence_number=42)."""
    digits = re.search(r"\d+", deal_ref)
    if not digits:
        return None
    seq_num = int(digits.group(0))

    sql = f"""
        SELECT d.name, d.stage, d.grand_total_in_usd, d.currency,
               d."closeDate", d."dealWonAt", d."dealLostAt", d.sequence_number,
               c."companyName", u.name AS owner_name, d."lineItems"
        FROM deals d
        LEFT JOIN companies c ON c._id = d.company::text
        LEFT JOIN users     u ON u._id = d.owner::text
        WHERE d.sequence_number = {seq_num}
          AND (d.deleted = false OR d.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row = res.rows[0]
    cols = ["name","stage","grand_total_in_usd","currency","closeDate",
            "dealWonAt","dealLostAt","sequence_number","companyName","owner_name","lineItems"]
    data = dict(zip(cols, row))

    deal_num = "ELS" + str(seq_num).zfill(3)
    status = ("Won" if data.get("dealWonAt") else
              "Lost" if data.get("dealLostAt") else "Open")

    lines = [
        f"**Deal {deal_num}: {data.get('name','—')}**\n",
        f"- **Company:** {data.get('companyName') or '—'}",
        f"- **Stage:** {data.get('stage') or '—'}",
        f"- **Status:** {status}",
        f"- **Owner:** {data.get('owner_name') or '—'}",
        f"- **Value:** USD {data.get('grand_total_in_usd') or 0:,.2f}",
        f"- **Close Date:** {str(data.get('closeDate',''))[:10]}",
    ]
    if data.get("lineItems"):
        items_text = _expand_items(data["lineItems"], data.get("currency","USD"))
        if items_text:
            lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["deals", "companies", "users"],
        "confidence":  0.98,
        "sql_queries": [sql],
    }


def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """
    Try to handle the query as a record-by-number lookup.
    Returns None if no pattern matched — caller should continue to next layer.
    """
    text = query.strip()

    # SO number: SO01241, SO 01241, so1241
    so_m = re.search(r'\bSO\s*(\d{4,6})\b', text, re.IGNORECASE)
    if so_m:
        want_items = bool(re.search(
            r'\b(item|line|product|detail|breakdown|content)\b', text, re.IGNORECASE
        ))
        result = _so_lookup(agent, "SO" + so_m.group(1), want_items)
        if result:
            LOGGER.info("RecordLookup HIT: SO%s", so_m.group(1))
            return result

    # Invoice number: ELSN/2026/018, ELSN2026018, ELSN/2025/24
    inv_m = re.search(r'\b(ELSN[/\s]?\d{4}[/\s]?\d+)\b', text, re.IGNORECASE)
    if inv_m:
        result = _invoice_lookup(agent, inv_m.group(1))
        if result:
            LOGGER.info("RecordLookup HIT: %s", inv_m.group(1))
            return result

    # Deal number: ELS042, ELS42
    deal_m = re.search(r'\bELS(\d{2,4})\b', text, re.IGNORECASE)
    if deal_m:
        result = _deal_lookup(agent, "ELS" + deal_m.group(1))
        if result:
            LOGGER.info("RecordLookup HIT: ELS%s", deal_m.group(1))
            return result

    return None
```

### Add record_lookup to pipeline/main.py

In `main.py`, add this BEFORE the intent router section (around line 255):

```python
# Add import at top of file:
from pipeline import record_lookup

# In the run() function, add this block AFTER fast_path and BEFORE intent router:

# ── Layer 0.5: Record Lookup — handles SO/ELSN/ELS number queries ──────────
with Timer("record_lookup") as rl_timer:
    try:
        rl_result = record_lookup.run(user_query, agent)
    except Exception as exc:
        LOGGER.warning("[RID:%s] RecordLookup exception: %s", request_id, exc)
        rl_result = None
metrics["record_lookup_ms"] = round(rl_timer.elapsed_ms)

if rl_result is not None:
    LOGGER.info("[RID:%s] RecordLookup HIT | %.0fms", request_id, rl_timer.elapsed_ms)
    metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
    return _build_response(
        rl_result["answer"],
        rl_result.get("tables_used", []),
        rl_result.get("sql_queries", []),
        rl_result.get("confidence", 0.98),
        started, layer="record_lookup", metrics=metrics,
    )
```

---

## PRIORITY 3 — FIX COMPANY NAME RESOLUTION

Currently, company fields store hex ObjectIds like `688c551ccd8bbb5c117fc83a`.
When the LLM shows these to users it looks wrong.

### Fix in intent_router.py `_build_sql()` for list/detail actions

For `deals`, `invoices`, `sales` — always JOIN companies:

```python
# In _build_sql(), for list/detail of deals:
if table == "deals":
    sql = f"""
        SELECT d.name, d.stage, d.grand_total_in_usd, d.currency, d."closeDate",
               c."companyName", u.name AS owner
        FROM "deals" d
        LEFT JOIN "companies" c ON c._id = d.company::text
        LEFT JOIN "users" u ON u._id = d.owner::text
        WHERE {where}
        ORDER BY d."createdAt" DESC NULLS LAST
        LIMIT {limit}
    """
    return sql

# For invoices:
if table == "invoices":
    sql = f"""
        SELECT i.invoice_number, i.payment_status, i.grandtotal_in_usd,
               i.currency, i.invoice_date, i.due_date,
               COALESCE(c."companyName", i."companyName", '—') AS company
        FROM "invoices" i
        LEFT JOIN "companies" c ON c._id = i.company::text
        WHERE {where}
        ORDER BY i."invoice_date" DESC NULLS LAST
        LIMIT {limit}
    """
    return sql
```

---

## PRIORITY 4 — IMPROVE INTENT ROUTER SYSTEM PROMPT

Add more examples and patterns to `_SYSTEM` in `intent_router.py`:

```python
# Add to _SYSTEM examples:
"tasks for deal ELS042"→{"action":"list","entity":"createtasks","filters":{"deal_ref":"ELS042"},"confidence":0.92}
"contacts of company ABC"→{"action":"list","entity":"contacts","filters":{"search":"ABC","related_to":"company"},"confidence":0.90}
"deals of ketul"→{"action":"list","entity":"deals","filters":{"owner":"ketul"},"confidence":0.95}
"revenue by company"→{"action":"top_n","entity":"invoices","filters":{"limit":20},"confidence":0.90}
"monthly revenue trend"→{"action":"compare","entity":"invoices","filters":{},"confidence":0.10}
"show all contacts of company [X]"→{"action":"list","entity":"contacts","filters":{"company_search":"X"},"confidence":0.90}
"which deals are overdue"→{"action":"list","entity":"deals","filters":{"stage":"open","overdue":true},"confidence":0.90}
```

### Add `company_search` filter in `_build_where()`:

```python
company_search = filters.get("company_search")
if company_search:
    cs = str(company_search).replace("'","''")
    # Find company _id first, then filter
    clauses.append(
        f'company IN (SELECT _id FROM "companies" '
        f'WHERE "companyName" ILIKE \'%{cs}%\' LIMIT 5)'
    )

# Add overdue deals filter
if filters.get("overdue") and entity == "deals":
    clauses.append('NULLIF("closeDate", \'\')::timestamptz < NOW()')

# Add owner name filter (search by name, not ID)
owner = filters.get("owner")
if owner and entity in ("deals", "sales", "contacts"):
    o = str(owner).replace("'","''")
    owner_col = '"salesOwner"' if entity == "sales" else "owner"
    clauses.append(
        f'{owner_col}::text IN (SELECT _id FROM "users" '
        f'WHERE name ILIKE \'%{o}%\' LIMIT 3)'
    )
```

---

## PRIORITY 5 — FIX TEXT2SQL PROMPT WITH FULL SCHEMA CONTEXT

**File:** `pipeline/text2sql.py` `_build_prompt()`

Replace the rules section with this comprehensive version:

```python
return (
    f"{schema}\n\n"
    f"{hint_block}"
    f"Question: {query}\n\n"
    "Write ONLY the SQL query. No explanation, no markdown.\n"
    "RULES:\n"
    "- Direct column names (NOT document->>'field')\n"
    "- Soft delete: WHERE deleted=false OR deleted IS NULL\n"
    "- outreaches: WHERE \"isDeleted\"=false OR \"isDeleted\" IS NULL\n"
    "- invoices amount: grandtotal_in_usd (no underscore before 'in')\n"
    "- deals/sales amount: grand_total_in_usd (with underscore)\n"
    "- contacts name: CONCAT(\"firstName\", ' ', \"lastName\")\n"
    "- companies name: \"companyName\"\n"
    "- tasks: table=createtasks, description=\"Task\" (capital T)\n"
    "- Revenue = invoices WHERE payment_status='paid'\n"
    "- targets month is 0-indexed (Jan=0, Dec=11)\n"
    "- For owner/company lookups: JOIN users/companies ON _id = field::text\n"
    "- First/oldest: ORDER BY \"createdAt\" ASC LIMIT 1\n"
    "- Last/latest: ORDER BY \"createdAt\" DESC LIMIT 1\n"
    "- Output ONLY a SELECT statement\n\n"
    "SQL:"
)
```

---

## PRIORITY 6 — IMPROVE DECOMPOSER WITH FULL DOMAIN CONTEXT

**File:** `prompt.py` `build_decompose_prompt()`

Add these critical patterns:

```python
# Add to build_decompose_prompt():
EXTRA_PATTERNS = """
RECORD LOOKUP PATTERNS:
- "SO01241" → SELECT from sales WHERE sales_number='SO01241', JOIN companies, users
- "ELSN/2026/018" → SELECT from invoices WHERE invoice_number='ELSN/2026/018'
- "ELS042" → SELECT from deals WHERE sequence_number=42

JOIN PATTERNS (always resolve IDs to names):
- deals.owner → JOIN users ON users._id = deals.owner::text → users.name
- deals.company → JOIN companies ON companies._id = deals.company::text → companyName
- invoices.company → same as above
- sales.salesOwner → JOIN users → users.name

LINE ITEMS:
- sales.items → JSONB array: [{product_name, quantity, unit_price, total_price, total_price_in_usd}]
- invoices.items → same structure
- deals.lineItems → same structure

ANALYTICAL QUERIES:
- "revenue by company" → GROUP BY company, SUM(grandtotal_in_usd), JOIN companies
- "deals by owner" → GROUP BY owner, COUNT(*), JOIN users for name
- "monthly trend" → GROUP BY DATE_TRUNC('month', invoice_date::timestamptz)
- "win rate" → COUNT won / COUNT total = won/(won+lost)*100
"""
```

---

## PRIORITY 7 — ADD CUSTOM QUERY HANDLERS FOR HIGH-FREQUENCY FAILURES

Create these specific handlers in `intent_router.py` `_build_sql()`:

### Handler: Tasks for a deal/company

```python
# After existing handlers, add:
if action == "list" and entity == "createtasks":
    deal_ref = filters.get("deal_ref")
    if deal_ref:
        digits = re.search(r'\d+', deal_ref)
        if digits:
            sql = f"""
                SELECT t."Task", t.status, t.priority, t."due_date", u.name AS assigned_to
                FROM "createtasks" t
                LEFT JOIN "users" u ON u._id = t."createdBy"::text
                WHERE t."dealsId"::text IN (
                    SELECT _id FROM "deals" WHERE sequence_number = {int(digits.group())}
                )
                AND (t.deleted = false OR t.deleted IS NULL)
                ORDER BY t."due_date" ASC NULLS LAST
            """
            return sql
```

### Handler: Revenue by company (top N)

```python
if action == "top_n" and entity == "invoices":
    amount_f = "grandtotal_in_usd"
    sql = f"""
        SELECT COALESCE(c."companyName", i."companyName", 'Unknown') AS company,
               COUNT(i._id)::int AS invoices,
               ROUND(SUM(i.{amount_f})::numeric, 2) AS total_revenue
        FROM invoices i
        LEFT JOIN companies c ON c._id = i.company::text
        WHERE (i.deleted = false OR i.deleted IS NULL)
          AND i.payment_status = 'paid'
          {('AND ' + where_extra) if where_extra else ''}
        GROUP BY 1
        ORDER BY total_revenue DESC NULLS LAST
        LIMIT {limit}
    """
    return sql
```

---

## QUICK WIN CHECKLIST (implement in order)

```
[ ] 1. Create pipeline/record_lookup.py  (fix SO01241 immediately)
[ ] 2. Add record_lookup to pipeline/main.py  (2 lines of code)
[ ] 3. Set FAST_PATH_ENABLED=true in .env
[ ] 4. Fix fast_path.py handlers (replace document->> with column names)
[ ] 5. Add company JOIN to intent_router deals/invoices queries
[ ] 6. Add owner name resolution in intent_router
[ ] 7. Update text2sql.py _build_prompt with full rules
[ ] 8. Update prompt.py decomposer with join patterns and line items
[ ] 9. Add overdue deals filter to intent_router
[ ] 10. Add company_search filter to intent_router
```

---

## TESTING QUERIES (run after each fix)

```
# Should work after Priority 2 (record_lookup):
SO01241 line items of this share me
give me this sales id details SO01241
ELSN/2026/018 invoice details
ELS042 deal details

# Should work after Priority 3 (company name fix):
give me all deals   ← should show company name not hex ID
overdue invoices   ← should show company name

# Should work after Priority 1 (fast path):
how many open deals
total revenue this month
pending tasks
top 10 customers by revenue
pipeline by stage

# Should work after Priority 4-6 (intent + decomposer):
deals of ketul
tasks for deal ELS042
contacts of company ABC
revenue by company this year
monthly revenue trend
win rate this quarter
```

---

## MODEL CONFIGURATION (already done)

Current best free models in `.env`:
- **Classify/Decompose:** `meta-llama/llama-4-scout-17b-16e-instruct` (Groq, newest)
- **Synthesize:** `llama-3.3-70b-versatile` (Groq, best quality)
- **Gemini fallback:** `gemini-2.5-flash` (latest free)
- **OpenRouter synthesize fallback:** `nousresearch/hermes-3-llama-3.1-405b:free` (405B!)
- **OpenRouter decompose fallback:** `meta-llama/llama-3.3-70b-instruct:free`

---

## DATABASE FACTS (key for SQL writing)

```
sales_number format:     SO00001  (5-digit padded)
invoice_number format:   ELSN/2026/018
deal number format:      ELS + sequence_number (ELS042 = sequence_number 42)

Revenue source:          invoices WHERE payment_status = 'paid'
Revenue field invoices:  grandtotal_in_usd  (NO underscore before 'in')
Revenue field deals:     grand_total_in_usd (WITH underscore)

Soft delete:             deleted = false OR deleted IS NULL
Outreach soft delete:    "isDeleted" = false OR "isDeleted" IS NULL

JSONB columns (still JSONB, need special handling):
  sales.items, invoices.items, deals.lineItems
  lastActivity, BillingAddress, lineupEmailComments

Date columns (TEXT, cast for comparison):
  "closeDate", "createdAt", "updatedAt", "dealWonAt", "dealLostAt"
  → NULLIF("closeDate",'')::timestamptz

Foreign keys (TEXT storing ObjectId hex):
  deals.owner      → users._id
  deals.company    → companies._id
  invoices.company → companies._id
  sales.salesOwner → users._id
  contacts.company → companies._id
  createtasks.createdBy → users._id
```

---

## EXPECTED RESULTS AFTER ALL FIXES

| Query Type | Before | After |
|---|---|---|
| SO/ELSN/ELS number lookup | ❌ No data | ✅ Full detail + line items |
| Simple counts | ✅ Works | ✅ Works (faster via fast path) |
| Revenue queries | ✅ Works | ✅ Works (faster) |
| Deals with company names | ⚠️ Shows hex IDs | ✅ Shows company name |
| "Deals of [person]" | ❌ Fails | ✅ Joins users by name |
| "Contacts of [company]" | ❌ Fails | ✅ Subquery by company name |
| Pipeline by stage | ✅ Works | ✅ Works (faster via fast path) |
| Monthly trend analysis | ⚠️ Sometimes | ✅ Consistent |
| Tasks for a deal | ❌ Fails | ✅ Via dealsId JOIN |
| Line items display | ❌ Shows JSON blob | ✅ Markdown table |

**Expected accuracy: 90-95%** after all priorities implemented.
