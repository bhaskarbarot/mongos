Now I have the full picture. Let me compose the complete workflow:

CRM AI Assistant — Complete Pipeline Workflow
System Architecture Overview

MongoDB (ecrm DB)
      │
      ▼ real-time sync
PostgreSQL :5433 (mongos_sync)
      │
      ▼ SQL queries
FastAPI Backend :8000
      │
      ▼ JSON API
React Frontend :5173
Layer 0 — Data Sync (automation/)
This runs continuously in the background, keeping PostgreSQL in sync with MongoDB.

File	What it does
sync.js	Entry point — starts both watcher + polling fallback, handles log rotation
mongoWatcher.js	Primary: Listens to MongoDB Change Streams (real-time inserts/updates/deletes)
mongoPoller.js	Fallback: Polls MongoDB every N seconds if Change Streams fail
sqlHandler.js	Writes to PostgreSQL — ensureTable, insertRow, deleteRow, deleteMissingRows
mapper.js	Transforms MongoDB BSON documents → flat SQL rows (JSONB format: {id, document, updated_at})
How it works: Every MongoDB collection (deals, companies, contacts, etc.) maps to a PostgreSQL table with 3 columns: id TEXT, document JSONB, updated_at TIMESTAMPTZ. All business data lives inside the document column as raw JSON.

Layer 1 — Startup & Agent Initialization
File	What it does
config.py	All settings — Postgres credentials, Ollama URL, all API keys, all model names
db.py	Connects to PostgreSQL via SQLAlchemy, filters out empty tables
agent.py	Builds a LangChain SQL Agent (AgentExecutor) with sql_db_query + sql_db_list_tables tools. Pre-warms schema cache
prompt.py	Builds system prompts for: agent, decomposition, synthesis, intent routing
app.py	CLI entry point (non-API mode) — load_database → load_agent → initialize_session → main loop
Layer 2 — HTTP API (api.py)
FastAPI app, entry point for all chat requests.


POST /chat   →  chat()
               ├─ rate check (_check_rate)
               ├─ get/init agent (_get_agent)
               ├─ rebuild memory (_rebuild_memory)
               ├─ call pipeline (pipeline/main.py → run())
               ├─ build query plan (_build_query_plan)
               ├─ extract structured data (_extract_structured_data)
               └─ return JSON response
POST /transcribe  →  Whisper voice-to-text
GET  /health      →  DB + agent status
GET  /sources     →  list available tables
POST /feedback    →  save user feedback
Layer 3 — Pipeline Orchestrator (pipeline/main.py)
This is the brain — routes every query through 4 layers in order:


User Query
    │
    ▼
[GUARD] is_greeting / is_blocked / is_vague_query
    │ pass
    ▼
[LAYER 1] fast_path.run()          ← tries FIRST, ~70-80% of queries handled here
    │ miss (None returned)
    ▼
[LAYER 2] classify(query)          ← SIMPLE or COMPLEX?
    │
    ├─ SIMPLE → text2sql.run()     ← fine-tuned SQL model, no decomposition
    │               │ miss
    │               ▼
    └─ COMPLEX → decompose()       ← break into sub-queries
                    │
                    ▼
                 run_parallel()    ← execute sub-queries concurrently
                    │
                    ▼
                 synthesize()      ← combine results into final answer
Layer 4 — Fast Path (pipeline/fast_path.py)
No LLM needed. Pure regex + SQL pattern matching. Returns in <300ms.

_classify_route(query) → dispatches to one of 24 specialized handlers:

Handler	Triggers on
_fp_kpi_report	"give me kpi report", "kpi dashboard", "manager dashboard"
_fp_revenue	"revenue", "income", "billing", "total sales"
_fp_count	"how many", "count", "total number"
_fp_deals_filter	"closed won deals", "lost deals", "open deals"
_fp_targets	"target", "achievement", "performance", "kpi"
_fp_pipeline_summary	"pipeline by stage", "pipeline distribution"
_fp_top_customers	"top N customers by revenue"
_fp_quarterly	"quarterly revenue", "Q1", "Q2"
_fp_overdue_aging	"overdue invoice aging", "outstanding by company"
_fp_pending_invoices	"pending invoices", "unpaid invoices"
_fp_tasks	"pending tasks", "overdue tasks"
_fp_search	"who is X", "find contact X"
_fp_active_customers	"active customers", "inactive customers"
_fp_no_activity	"customers with no business", "no invoice"
_fp_dept_users	"users in department", "department members"
_fp_lookup_list	"all technologies", "list sources", "all regions"
_fp_group_by	"deals by stage", "contacts by source"
_fp_list_records	"give me all deals", "show invoices"
_fp_sales	"sales orders", "list sales"
_fp_targets / _fp_target_unachieved	"who missed targets", "below target"
_fp_user_task_map	"tasks by user", "tasks per person"
_fp_invoice_status	"approved invoices", "rejected invoices"
_fp_system_config	"SMTP settings", "upload limit"
Key helpers inside fast_path:

_parse_time_condition() — understands "last month", "this year", "Q3 2025", "FY2025", "last 3 months" etc.
_natural_delay() — adds 6–10s human-feeling delay so responses don't look hardcoded
Layer 5 — Intent Classifier (pipeline/classifier.py)
Only runs if fast_path misses. Uses LLM to label query as SIMPLE or COMPLEX.


_rule_prescreen(query)     ← regex fast-screen first (no LLM needed for obvious cases)
     │ unclear
     ▼
_call_llm_classify(query)  ← llm.call("classify", ...)
Layer 6 — Text2SQL (pipeline/text2sql.py)
For SIMPLE queries only. Runs two fine-tuned SQL models in parallel, returns first valid result.


generate_sql(query)
    │
    ├─ Model 1: debopam/Text-to-SQL__Qwen2.5-Coder-3B  (fast, 8s timeout)
    ├─ Model 2: a-kore/Arctic-Text2SQL-R1-7B           (slower, 30s timeout)
    │   ← first valid SQL wins
    ▼
_validate_sql()    ← checks for SELECT only, no DROP/DELETE etc.
_format_result()   ← runs SQL → formats as markdown table
Layer 7 — Decomposer (pipeline/decomposer.py)
For COMPLEX queries. Breaks one question into multiple focused sub-queries.


decompose("top customers by revenue with their overdue invoices")
    │
    ▼  llm.call("decompose", ...)
    │
    └─ returns: [
         {sub_query: "top customers by revenue", intent: "ranking"},
         {sub_query: "overdue invoices by company", intent: "filter"},
       ]
Layer 8 — Executor (pipeline/executor.py)
Runs all sub-queries in parallel (ThreadPoolExecutor).


run_parallel([sub_query_1, sub_query_2, ...])
    │
    ├─ Each sub-query → fast_path.run() first
    │                 → text2sql.run() if fast_path misses
    │                 → stub answer if both fail (timeout)
    │
    └─ Returns: [SubQueryResult, SubQueryResult, ...]
Layer 9 — Synthesizer (pipeline/synthesizer.py)
Combines all sub-query results into one final answer.


synthesize(original_query, [sub_result_1, sub_result_2, ...])
    │
    ▼  llm.call("synthesize", ...)
    │
    └─ Formats as: table / list / report / kpi_card / paragraph
       (detected by _get_format_instructions based on query type)
Layer 10 — Schema & Memory Support
File	What it does
pipeline/schema.py	Discovers PostgreSQL tables/fields, caches schema, builds Text2SQL schema string, run_sql() wrapper
pipeline/utils.py	normalize_text, fmt_number, format_rows_as_markdown_table, is_greeting, is_blocked, is_vague_query
pipeline/chat_memory.py	Per-session memory — resolves pronouns ("those deals" → "deals"), tracks last entity, finds similar past queries
pipeline/llm.py	Central LLM router — call(task, system, user)
All LLM Models Used
Text2SQL (SQL Generation) — Ollama Local
Model	Size	Role	Timeout
debopam/Text-to-SQL__Qwen2.5-Coder-3B	3B	Primary SQL generator	8s
a-kore/Arctic-Text2SQL-R1-7B	7B	Fallback SQL generator	30s
Reasoning (Classify / Decompose / Synthesize) — Cloud with local fallback
Task	Provider 1	Provider 2	Provider 3	Provider 4 (always available)
Classify	Groq llama-3.1-8b-instant	Gemini gemini-2.5-flash-lite	—	Ollama qwen2.5:1.5b
Decompose	Groq llama-3.1-8b-instant	Gemini gemini-2.5-flash-lite	OpenRouter llama-3.1-8b	Ollama qwen2.5:7b
Synthesize	Groq llama-3.3-70b-versatile	Gemini gemini-2.5-flash-lite	OpenRouter llama-3.1-8b	Ollama qwen2.5:7b
Fallback order is automatic — if Groq hits rate limit → Gemini → OpenRouter → Ollama (offline fallback, always works).

Frontend (chat-ui/src/)
File	What it does
DataAnalysisChat.jsx	Full production chat UI — sessions, voice input (MediaRecorder + Whisper), markdown rendering, query plan display, feedback widget
DataAnalysisChatPreview.jsx	Demo/preview mode with static mock responses
App.jsx	Root component
What Works Well ✓
Fast path handles ~70-80% of queries with no LLM — sub-300ms response
Full offline capable — if all cloud APIs fail, Ollama handles everything locally
Auto-fallback chain — Groq → Gemini → OpenRouter → Ollama, no manual intervention
Real-time sync — MongoDB → PostgreSQL stays current via Change Streams
KPI report — 27 SQL queries across 12 tables, LLM-narrated enterprise report
Time parsing — handles "last month", "FY2025", "Q3", "last 3 months", "between Jan and Mar" etc.
Voice input — Whisper transcription via /transcribe endpoint
Session memory — resolves "those deals" / "them" / "their" to last queried entity





Correct — fast path uses zero LLM to understand the query. It's 100% regex + pattern matching.

Here's exactly how it works:

Step 1 — normalize_text(query)
Just lowercases and strips punctuation. Example:


"Give me KPI Report!" → "give me kpi report"
Step 2 — _classify_route(text)
Pure re.search() patterns, checked top to bottom:


if re.search(r"\bkpi\s+report\b|\bgive\s+(me\s+)?kpi\b|...", t):
    return "kpi_report"

if re.search(r"\btargets?\b|\bperformance\b|\bachievement\b", t):
    return "targets"

if re.search(r"\bpending\s+invoice|invoice.*unpaid\b", t):
    return "pending_invoices"
# ... 20+ more patterns
Step 3 — Handler runs direct SQL
Each handler builds SQL from regex-extracted pieces:


# _parse_time_condition() extracts: "last month" → WHERE clause
# _fp_deals_filter() regex-detects: "closed won" → stage = 'Closed Won'
# No LLM involved — just string matching → hardcoded SQL templates
The ONE exception — KPI Report
The _fp_kpi_report handler does use LLM — but only at the very end, after all 27 SQL queries have already finished collecting data. The LLM just narrates the numbers, it doesn't understand or route the query:


User: "give me kpi report"
         │
         ▼ regex matched → _fp_kpi_report
         │
         ▼ 27 SQL queries run (no LLM)
         │
         ▼ llm.call("synthesize", data_summary)  ← LLM only here, to write prose
         │
         ▼ answer returned
Why no LLM for understanding?
Fast Path (regex)	LLM understanding
Speed	<5ms	500ms – 3s
Cost	Free	API tokens
Reliability	100% deterministic	Can hallucinate
Works offline	Yes	Only if Ollama
The tradeoff is that fast path only works for known query patterns. Anything it doesn't recognize returns None and the query escalates to the classifier → LLM pipeline.



Now I have the full picture. Let me explain the exact decision flow clearly:

How Fast Path Decides — Step by Step

Every query → fast_path.run(query)
                    │
                    ▼
         Step 1: _classify_route(query)
                    │
          ┌─────────┴──────────┐
     returns a               returns
     specific route          "general"
          │                      │
          ▼                      ▼
    Step 2A:                Step 2B:
    Try that ONE            Try ALL 11 general
    specialized handler     handlers in order
Step 1 — _classify_route() in Detail
It checks 22 regex patterns in order, first match wins, returns a route string:


def _classify_route(text):
    t = normalize_text(text)   # just lowercases the text

    # Each check is: does query contain THIS pattern?
    if "who is" / "find contact"        → "search"
    if "INV/2025/001" pattern            → "invoice_lookup"
    if "quarterly" + no target words     → "quarterly"
    if "no business" / "not active"      → "no_activity"
    if "department" + "users/members"    → "dept_users"
    if "overdue tasks" / "missed tasks"  → "overdue_tasks"
    if "who created" / "created by whom" → "chain_lookup"
    if "company named X" + "invoice"     → "company_invoice"
    if "target" + "not achieved/missed"  → "target_unachieved"
    if "kpi report" / "kpi dashboard"    → "kpi_report"
    if "target" / "performance" / "kpi"  → "targets"
    if "outreach" / "interested leads"   → "outreach"
    if "all technologies" / "list sources" → "lookup_list"
    if "SO123" / "sales order detail"    → "sales_order_detail"
    if "smtp" / "upload limit"           → "system_config"
    if "overdue invoice aging"           → "overdue_aging"
    if "pipeline by stage"               → "pipeline_summary"
    if "pending invoice" / "unpaid"      → "pending_invoices"
    if "active" + "customers"            → "active_customers"
    if "users" + "tasks" + "by/per"      → "user_task_map"
    if "invoices" + "approved/rejected"  → "invoice_status_filter"
    if "get ... for company X"           → "entity_lookup"

    # NOTHING matched above?
    return "general"    ← default fallback
Key rule: If NO pattern matches → route is always "general". There is no "reject" — everything enters fast_path, it just goes to different handlers.

Step 2A — Specialized Route (specific handler called)

_SPECIALIZED = {
    "kpi_report":        _fp_kpi_report,
    "search":            _fp_search,
    "targets":           _fp_targets,
    "quarterly":         _fp_quarterly,
    "pending_invoices":  _fp_pending_invoices,
    # ... all 20+ routes
}

if route in _SPECIALIZED:
    result = _SPECIALIZED[route](agent, query)

    if result is not None:   ← handler succeeded
        return result        ← DONE. Goes back to user.

    # result is None → handler recognized the route but
    # couldn't build a valid answer (e.g. table not found)
    # FALLS THROUGH to general handlers below ↓
So a specialized handler can ALSO return None if something is wrong internally (missing table, no data found in schema). When that happens it falls through.

Step 2B — General Route (chain of 11 handlers tried in order)
When route is "general" OR a specialized handler returned None:


_GENERAL = [
    _fp_sales,            # 1st tried
    _fp_revenue,          # 2nd
    _fp_top_customers,    # 3rd
    _fp_deals_filter,     # 4th
    _fp_active_customers, # 5th
    _fp_user_task_map,    # 6th
    _fp_invoice_status,   # 7th
    _fp_tasks,            # 8th
    _fp_count,            # 9th
    _fp_group_by,         # 10th
    _fp_list_records,     # 11th ← most generic, catches almost everything
]

for handler in _GENERAL:
    result = handler(agent, query)

    if result is not None:  ← this handler matched and answered
        return result        ← DONE. Goes back to user.

    # None = this handler said "not my job", try next one
Each general handler has its OWN internal check at the start:


def _fp_revenue(agent, query):
    text = normalize_text(query)

    # Handler checks itself if it applies
    if not any(kw in text for kw in ["revenue", "income", "billing"]):
        return None   ← "not my query, skip me"

    # ... otherwise build SQL and return result

def _fp_count(agent, query):
    if not any(kw in text for kw in ["how many", "count", "total number"]):
        return None   ← skip

    # ... otherwise answer
Step 3 — Fast Path MISS → Escalate

# All _GENERAL handlers returned None:
return None   ← fast_path gives up
Back in main.py:


fp_result = fast_path.run(query, agent)

if fp_result is not None:
    return fp_result   ← SUCCESS, send to user, STOP HERE

# fp_result is None → fast_path missed
# Escalate to classifier:
classification = classify(query)   ← LLM decides SIMPLE vs COMPLEX
Answer to Your Key Question: "Does fast_path re-check after processing?"
No. Once fast_path returns a non-None result, it is FINAL. No re-checking, no validation pass. The answer goes directly back to the user.


fast_path returns result
         │
         ▼
    main.py checks: is result None?
         │
    NO → return to user immediately ✓
    YES → escalate to classifier
The only "checking" is inside each handler before it answers — it checks if the query is relevant to it. After it answers, that's it.

Full Decision Tree (visual)

User Query: "show closed won deals last quarter"
                        │
                        ▼
              _classify_route()
              no specific pattern → "general"
                        │
                        ▼
              _fp_sales()        → None  (no "sales order" keyword)
              _fp_revenue()      → None  (no "revenue" keyword)
              _fp_top_customers()→ None  (no "top N" keyword)
              _fp_deals_filter() → ✓ MATCH! ("closed won" detected)
                        │
                        ▼
              Builds SQL: WHERE stage='Closed Won'
                          AND closeDate in last quarter
                        │
                        ▼
              Returns result → sent to user
              ══════ DONE ══════



User Query: "what do you think about our sales strategy?"
                        │
                        ▼
              _classify_route() → "general"
                        │
                        ▼
              _fp_sales()        → None
              _fp_revenue()      → None
              _fp_top_customers()→ None
              _fp_deals_filter() → None
              _fp_active_customers()→ None
              _fp_user_task_map()→ None
              _fp_invoice_status()→ None
              _fp_tasks()        → None
              _fp_count()        → None
              _fp_group_by()     → None
              _fp_list_records() → None (no entity found)
                        │
                        ▼
              fast_path returns None
                        │
                        ▼
              classifier() → COMPLEX
                        │
                        ▼
              decompose → execute → synthesize → LLM answer
Summary Table
Query type	Route	Handled by
"who is Ketul"	search	_fp_search (specialized)
"kpi report"	kpi_report	_fp_kpi_report (specialized)
"total revenue this month"	general	_fp_revenue (general chain #2)
"how many deals"	general	_fp_count (general chain #9)
"give me all contacts"	general	_fp_list_records (general chain #11)
"what is our strategy?"	general → all miss	Escalates to LLM
"complex multi-part question"	general → all miss	Decomposer + Synthesizer
