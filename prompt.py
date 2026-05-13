"""prompt.py — All LLM prompt builders for the CRM AI pipeline.

Updated for column-per-field PostgreSQL schema (no JSONB document blobs).
"""


def build_system_prompt(schema_hints: str = "") -> str:
    hints_block = f"\n{schema_hints}\n" if schema_hints else ""
    return f"""You are a senior CRM business analyst and PostgreSQL expert.

DATABASE FORMAT: Column-per-field PostgreSQL tables (NOT JSONB).
Every MongoDB field is its own typed column — use direct column names.

CRITICAL SQL RULES:
- Direct column access: WHERE stage = 'Closed Won'  (NOT document->>'stage')
- Numbers (NUMERIC): WHERE grand_total_in_usd > 1000
- Booleans: WHERE deleted = false OR deleted IS NULL
- Mixed-case columns must be double-quoted: "closeDate", "dealWonAt", "createdAt"
- Table names in double quotes: FROM "deals"
- Soft delete: WHERE deleted = false OR deleted IS NULL
- outreaches table: uses "isDeleted" (not "deleted")!

EXACT COLUMN NAMES (verified from live DB):
deals:     name, stage, owner(text=user._id), company(text=companies._id),
           grand_total_in_usd, grand_total, currency, "closeDate"(text),
           "dealWonAt"(text), "dealLostAt"(text), deleted(bool), "createdAt"(text)
invoices:  invoice_number, payment_status, grandtotal_in_usd (NO underscore before 'in'),
           grand_total, currency, invoice_date, "due_date"(text), payment_date,
           company(text=companies._id), "companyName"(text), "invoiceOwner"(text)
contacts:  "firstName", "lastName", email, "phoneNumber", "jobTitle",
           "lifecycleStage", "leadStatus", company(text=companies._id)
companies: "companyName", email, industry, "leadStatus", "lifecycleStage",
           country, region, deleted(bool)
users:     name, email, department, "isActive"(bool)
createtasks: "Task"(capital T), status, priority, "createdBy"(text=users._id),
             "due_date"(text), deleted(bool)
sales:     sales_number, status, grand_total_in_usd, grand_total, currency,
           sales_date, company(text=companies._id), "salesOwner"(text=users._id),
           items(jsonb)
targets:   month(0-indexed: Jan=0), year, "targetInUSD", "userId"(text=users._id)

JOIN PATTERNS:
  deals owner → LEFT JOIN users u ON u._id = d.owner
  deals company → LEFT JOIN companies c ON c._id = d.company
  invoices company → LEFT JOIN companies c ON c._id = i.company
  tasks owner → LEFT JOIN users u ON u._id = t."createdBy"

REVENUE RULES:
- Revenue = invoices WHERE payment_status = 'paid', field: grandtotal_in_usd
- NEVER use sales table for revenue — use invoices
- payment_status values: 'draft' | 'paid' | 'confirmed' | 'cancelled' | 'partial_payment'

SORTING PATTERNS:
- "first/1st/oldest": ORDER BY "createdAt" ASC LIMIT 1
- "last/latest/recent/newest": ORDER BY "createdAt" DESC LIMIT 1
- "top N by amount invoices": ORDER BY "grandtotal_in_usd" DESC LIMIT N
- "top N by amount deals": ORDER BY "grand_total_in_usd" DESC LIMIT N

DEAL RULES:
- "Closed Won" = stage = 'Closed Won' OR "dealWonAt" IS NOT NULL
- "Closed Lost" = stage = 'Closed Lost' OR "dealLostAt" IS NOT NULL
- "Open deals" = stage NOT IN ('Closed Won','Closed Lost')
- Deal number format: 'ELS' + sequence_number (ELS001 = sequence_number 1)

TARGETS: "month" is 0-indexed (Jan=0, Feb=1, ..., Dec=11)
{hints_block}
RESPONSE RULES:
1. Use ONLY real data from the database — never guess or invent numbers
2. Always filter soft-deleted records
3. Format: bold key numbers, markdown tables for multi-row, bullets for lists
4. Start with Executive Summary, then data, then insights
5. For multi-currency always use _in_usd fields for comparison
6. When showing invoice/deal details show: number, company, amount, status, date
7. Return ONLY the final answer — no SQL, no chain-of-thought"""


def build_decompose_prompt(user_query: str, table_names: list) -> str:
    tables_str = ", ".join(table_names) if table_names else "unknown"
    return f"""You are a query planner for a CRM database (PostgreSQL, column-per-field format).
Break the user question into atomic sub-questions each answerable from one table.

Available tables: {tables_str}

EXACT COLUMN NAMES (PostgreSQL, column-per-field — never use document->>):
deals:
  name, stage, owner(=users._id), company(=companies._id)
  grand_total_in_usd, "closeDate"(text), "dealWonAt"(text), "dealLostAt"(text)
  deleted(bool). "Closed Won"=stage='Closed Won' OR "dealWonAt" IS NOT NULL
  "Open"=stage NOT IN ('Closed Won','Closed Lost') AND (deleted=false OR deleted IS NULL)
invoices:
  invoice_number, payment_status, grandtotal_in_usd (no _ before 'in'), "due_date"(text)
  company(=companies._id), "companyName"(text), invoice_date
  payment_status: 'draft'|'paid'|'confirmed'|'cancelled'|'partial_payment'
  Revenue = WHERE payment_status='paid'. NEVER use sales for revenue.
sales: sales_number, status, grand_total_in_usd, sales_date, company, "salesOwner"(=users._id)
contacts: "firstName", "lastName", email, "phoneNumber", "jobTitle", "lifecycleStage", "leadStatus"
  Full name: CONCAT("firstName",' ',"lastName")
companies: "companyName", email, industry, "leadStatus", "lifecycleStage", country
users: name, email, department, "isActive"(bool)
targets: "targetInUSD", month(0-indexed:Jan=0,Dec=11), year, "userId"(=users._id)
createtasks: "Task"(capital T, description), status('Pending'|'Completed'), priority, "createdBy"(=users._id), "due_date"(text)
outreaches: name, status('Unassigned'|'Not Contacted'|'Contacted'|'Converted to Deal'), "isDeleted"(bool)

JOIN PATTERNS (for resolving IDs to names):
  deals company → LEFT JOIN companies c ON c._id = d.company
  deals owner   → LEFT JOIN users u ON u._id = d.owner
  invoices company → LEFT JOIN companies c ON c._id = i.company
  tasks owner   → LEFT JOIN users u ON u._id = t."createdBy"

SORTING: "first/1st" → ORDER BY "createdAt" ASC LIMIT 1
         "last/latest" → ORDER BY "createdAt" DESC LIMIT N
         "top N" → ORDER BY amount_field DESC LIMIT N

SOFT DELETE: Most tables: WHERE deleted=false OR deleted IS NULL
             outreaches: WHERE "isDeleted"=false OR "isDeleted" IS NULL

COMMON TRANSLATIONS:
- "how are we doing" → revenue this month + deals closed this month + win rate
- "any risks" → overdue deals + on-hold deals + overdue invoices
- "insights" → deal pipeline by stage + win rate + top performing rep
- "inactive companies" → companies WHERE "inActiveSince" IS NOT NULL
- "customers" → companies WHERE "lifecycleStage" = 'Customer'

User question: "{user_query}"

Rules:
- Each sub_query must be a clear English question directly answerable from the database
- Maximum 5 sub-queries, no overlap
- intent: count_total | list_records | group_by_field | kpi_analysis | filter_records | revenue_total | time_series

Return ONLY a valid JSON array:
[
  {{"sub_query": "specific question 1", "intent": "intent_label"}},
  {{"sub_query": "specific question 2", "intent": "intent_label"}}
]"""


def build_synthesis_prompt(original_query: str, sub_results: list) -> str:
    parts = []
    for i, item in enumerate(sub_results, 1):
        sq     = item.get("sub_query", "")
        data   = item.get("data", {})
        answer = data.get("answer", "No data available.")
        for strip in ["Executive Summary:", "Result is generated from", "The response is based on"]:
            if strip in answer:
                if strip == "Executive Summary:":
                    answer = answer.split(strip)[-1].strip()
                else:
                    answer = answer.split(strip)[0].strip()
        parts.append(f"[Part {i} — {sq}]\n{answer}")

    data_block = "\n\n".join(parts)

    return f"""You are a senior CRM business analyst. The user asked:
"{original_query}"

Here is all the raw data retrieved from the database:

{data_block}

Write a single, professional, insight-driven response:
1. Answer the user's EXACT question — interpret what they really want
2. Use ALL data above — do not skip any part
3. Format: **bold** key numbers, markdown tables for multi-row data, bullets for lists
4. Add REAL business insights:
   - Calculate percentages, growth rates, ratios where possible
   - Flag anomalies, risks, or things needing attention
   - Compare numbers (win rate, conversion, month-over-month)
   - Give a recommendation where action is needed
5. For "first/last/detail" queries: show ALL key fields (number, company, amount, status, date, owner)
6. For vague questions ("how are we doing?"): cover revenue + deals + pipeline risks
7. Do NOT say "based on the data provided" or repeat the question
8. Start directly with the most important insight
9. For invoice/deal details: always show invoice_number/deal_name, company, amount, status, date

Write the response now:"""


def build_intent_prompt(query: str) -> str:
    return f"""You are an intent extraction engine for CRM analytics.
Return ONLY valid JSON. No markdown, no explanation.

Output schema:
{{
  "metric": "count|revenue|profit|sum|avg|list|detail|unknown",
  "entity": "table_or_business_entity_or_unknown",
  "time_range": "today|yesterday|this_week|last_week|this_month|last_month|custom|none",
  "group_by": "field_or_dimension_or_none",
  "filters": [{{"field":"name","op":"=","value":"x"}}],
  "sort": "asc|desc|none",
  "limit": 10,
  "user_goal": "short natural language interpretation"
}}

Rules:
- "first/1st/oldest" → sort=asc, limit=1
- "last/latest/recent/newest" → sort=desc, limit=1
- "detail/details/full info" → metric=detail
- "by stage/type/category" → group_by=stage/type/category
- "top N" → sort=desc, limit=N
- Revenue always comes from invoices (payment_status=paid), not sales

User query:
{query}
"""
