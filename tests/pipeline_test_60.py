"""
pipeline_test_60.py — 65 complex query test (fast path DISABLED)
Uses llm_router: Groq Scout-17b → Gemini → OpenRouter DeepSeek → Ollama
Compact schema: only relevant tables sent (~400-700 tok vs 3500 before)
Delay: 20s between queries = 3 queries/min = zero rate-limit risk
Output: tests/results_60.csv
"""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from pipeline.text2sql import (
    _generate_sql_via_router, _format_result,
    _extract_tables_from_sql, _validate_sql,
)
from pipeline.schema import _run_sql_direct
from pipeline.synthesizer import narrate_response
from pipeline.schema_compact import get_compact_schema
from pipeline.llm_router import estimate_tokens

TABLE_NAMES = [
    'deals','invoices','sales','companies','contacts','users','createtasks',
    'targets','meetings','outreaches','departments','regions','vendors','bills',
    'products','sources','technologies','taxes','categories','campaigns',
    'lead_statuses','lifecycle_stages','dealstagesettings','payments',
    'emails','commonnotes','activitylogs','countryregions','projecttypes',
    'notes','publicleads',
]

# ── 65 Queries ───────────────────────────────────────────────────────────────
QUERIES = [
    # COUNTS
    "How many deals are there?",
    "How many departments do we have?",
    "How many vendors are registered?",
    "How many contacts are in the system?",
    "How many outreaches have been done?",
    "How many products are active?",
    "How many invoices are unpaid?",
    "How many users are active?",
    "How many meetings have been scheduled?",
    "How many bills are pending payment?",

    # LISTS
    "Give me names of all departments",
    "List all deal stages",
    "Show me all lead statuses",
    "Give me list of all technologies",
    "List all active products with price",
    "Show me all tax types and their rates",
    "List all payment methods",
    "Give me all regions",
    "Show me lifecycle stages",

    # DEALS
    "Give me all open deals with their owner and value",
    "Show me closed won deals with total amount and owner",
    "Which deals are overdue past their close date with how many days",
    "Give me deals grouped by stage with count and total pipeline value",
    "Show me deals by type/category with count",
    "Give me top 5 deals by value with company name",
    "Show me all negotiation stage deals with owner contact",
    "Which deals were lost last 3 months?",

    # REVENUE
    "What is total revenue this year from paid invoices?",
    "Compare revenue between 2025 and 2026",
    "Give me monthly revenue trend for last 12 months",
    "Show revenue by currency with invoice count",
    "Who are the top 10 customers by revenue?",
    "Give me revenue by sales rep this year",
    "What is total value of pending/unpaid invoices?",
    "Show me invoice aging: overdue by 0-30, 31-60, 61-90, 90+ days",

    # INVOICES
    "Give me list of paid invoices in INR currency",
    "Show me top 10 invoices by amount with company name",
    "List overdue invoices with company name and days overdue",
    "Give me all draft invoices",
    "Show invoices by payment status with count and total",
    "Give me details of first invoice in the system",

    # TASKS & PRODUCTIVITY
    "Who has the most pending tasks and how many?",
    "Show all overdue tasks with user name and due date",
    "Give me task summary per user: pending, completed, total",
    "Which department has most overdue tasks?",
    "Show me pending high priority tasks with assigned user",
    "Give me all tasks for the outreach team",

    # TARGETS & PERFORMANCE
    "Show target vs achieved for each sales rep this year with gap and percentage",
    "Which sales reps have not achieved their targets?",
    "Give me top performing sales rep by achievement percentage",

    # COMPANIES & CONTACTS
    "Give me active customers with industry and region",
    "Which companies have not given any business in last 3 months?",
    "Show me company health: open deals, pending invoices, open tasks per company",
    "Give me contacts with their company and job title",
    "Show me lead companies by lifecycle stage with count",

    # OUTREACH & CAMPAIGNS
    "Show outreach funnel: status breakdown with conversion rate",
    "Give me campaign performance: outreaches, contacted, converted per campaign",
    "Which outreaches are not contacted yet?",

    # COMPLEX MULTI-TABLE
    "Give me full sales leaderboard: rank, rep, department, revenue, deals won, pending tasks",
    "Show customer 360 view: company, deals, paid revenue, unpaid invoices, open tasks",
    "Give me vendor list with their bill count and total bill amount",
    "Show me email activity per user: email count and latest email date",
    "Which companies have deals but no paid invoices?",
    "Give me department-wise user count and their total pending tasks",
    "Show me stalled deals with company name, owner, value, and days past close date",
]


def run_query(query: str) -> dict:
    result = {
        "query":            query,
        "status":           "",
        "sql_generated":    "",
        "tables_used":      "",
        "schema_tokens":    0,
        "rows_returned":    0,
        "sql_time_ms":      0,
        "db_time_ms":       0,
        "narrate_time_ms":  0,
        "total_time_ms":    0,
        "response_preview": "",
        "full_response":    "",
        "error":            "",
    }

    total_start = time.perf_counter()

    # Step 1 — SQL generation via router (compact schema)
    compact = get_compact_schema(query)
    result["schema_tokens"] = estimate_tokens(compact)

    t1 = time.perf_counter()
    sql = _generate_sql_via_router(query, TABLE_NAMES)
    result["sql_time_ms"] = round((time.perf_counter() - t1) * 1000)

    if not sql:
        result["status"]        = "FAIL_NO_SQL"
        result["error"]         = "Router failed (all providers exhausted)"
        result["total_time_ms"] = round((time.perf_counter() - total_start) * 1000)
        return result

    result["sql_generated"] = sql
    result["tables_used"]   = ", ".join(_extract_tables_from_sql(sql))

    # Step 2 — DB execution
    t2 = time.perf_counter()
    db_result = _run_sql_direct(sql)
    result["db_time_ms"] = round((time.perf_counter() - t2) * 1000)

    if db_result is None:
        result["status"]        = "FAIL_DB_CONNECT"
        result["error"]         = "DB connection failed"
        result["total_time_ms"] = round((time.perf_counter() - total_start) * 1000)
        return result

    if db_result.error:
        result["status"]        = "FAIL_SQL_ERROR"
        result["error"]         = db_result.error[:200]
        result["total_time_ms"] = round((time.perf_counter() - total_start) * 1000)
        return result

    rows = db_result.rows
    result["rows_returned"] = len(rows)

    if not rows:
        result["status"]        = "WARN_EMPTY"
        result["error"]         = "Query OK but 0 rows returned"
        result["total_time_ms"] = round((time.perf_counter() - total_start) * 1000)
        return result

    # Step 3 — Format + narrate
    body     = _format_result(rows, sql, query)
    tables   = _extract_tables_from_sql(sql)

    t3 = time.perf_counter()
    narrated = narrate_response(query, body, tables)
    result["narrate_time_ms"] = round((time.perf_counter() - t3) * 1000)

    result["total_time_ms"]    = round((time.perf_counter() - total_start) * 1000)
    result["status"]           = "PASS"
    result["full_response"]    = narrated.replace('\n', ' | ').strip()
    result["response_preview"] = narrated.replace('\n', ' ').strip()[:200]

    return result


def main():
    os.makedirs("tests", exist_ok=True)
    out_path = "tests/results_60.csv"

    fieldnames = [
        "query","status","tables_used","schema_tokens","rows_returned",
        "sql_time_ms","db_time_ms","narrate_time_ms","total_time_ms",
        "response_preview","sql_generated","full_response","error",
    ]

    passed = failed = warned = 0
    all_results = []

    # 20s gap = 3 queries/minute = safe under Scout-17b free-tier limits
    # Scout: 30,000 TPM/key × 3 keys = 90,000 TPM/min
    # Compact schema: ~500 tok/query → 180 queries/min budget vs 3 used = 1.7%
    QUERY_DELAY_S = 20.0

    print("=" * 72)
    print(f"  Running {len(QUERIES)} queries | delay={QUERY_DELAY_S}s | model=Groq Scout-17b")
    print(f"  Token budget: Scout 90K TPM/min, using ~{round(60/QUERY_DELAY_S*500):,} TPM/min")
    print("=" * 72)

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i:02d}/{len(QUERIES)}] {query[:68]}")
        r = run_query(query)
        all_results.append(r)

        icon = "✅" if r["status"] == "PASS" else ("⚠️ " if r["status"] == "WARN_EMPTY" else "❌")
        print(
            f"       {icon} {r['status']} | "
            f"schema={r['schema_tokens']}tok | "
            f"SQL:{r['sql_time_ms']}ms DB:{r['db_time_ms']}ms "
            f"Narr:{r['narrate_time_ms']}ms | "
            f"TOTAL:{r['total_time_ms']}ms | rows={r['rows_returned']}"
        )
        if r["response_preview"]:
            print(f"       → {r['response_preview'][:110]}")
        if r["error"]:
            print(f"       ✗ {r['error'][:90]}")

        if r["status"] == "PASS":
            passed += 1
        elif r["status"] == "WARN_EMPTY":
            warned += 1
        else:
            failed += 1

        if i < len(QUERIES):
            time.sleep(QUERY_DELAY_S)

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed} PASS | {warned} WARN(empty) | {failed} FAIL")
    print(f"  CSV → {out_path}")
    if all_results:
        avg = sum(r["total_time_ms"] for r in all_results) / len(all_results)
        avg_tok = sum(r["schema_tokens"] for r in all_results) / len(all_results)
        print(f"  Avg total time : {avg:.0f}ms/query")
        print(f"  Avg schema size: {avg_tok:.0f} tokens (vs 3500 full schema = {100-round(avg_tok/3500*100)}% reduction)")
    print("=" * 72)


if __name__ == "__main__":
    main()
