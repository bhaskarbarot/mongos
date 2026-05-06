"""
QA test script — 10 complex production CRM queries.
Runs each query through the real agent pipeline and prints results + logs.
"""
import logging, os, sys, time
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for noisy in ["transformers","sentence_transformers","httpx","urllib3","sqlalchemy.engine","langchain"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

from db import get_database
from agent import get_sql_agent, run_agent_query, ConversationMemory

QUERIES = [
    ("Q1 — Deals KPI Dashboard",
     "give me complete deals KPI report: total deals count, closed won count with win rate percentage, "
     "closed lost count, average deal value, total pipeline value by stage analysis and how many deals "
     "are currently in negotiation stage"),

    ("Q2 — Invoice Revenue Report",
     "how many invoices are paid and what is total paid revenue, also show me all draft invoices "
     "count with total outstanding amount and give me overall invoice count by payment status"),

    ("Q3 — Sales vs Target Performance",
     "give me sales performance report for all users showing their targets versus achieved sales amounts "
     "with achievement percentage, highlight who has overachieved and who is below target"),

    ("Q4 — Pipeline Stage Analysis",
     "give me all deals grouped by stage with count and total deal value per stage, "
     "also show me deals currently in negotiation stage with company names and deal amounts"),

    ("Q5 — Task Status Analysis",
     "give me all overdue tasks with task name, priority, due date and company they belong to, "
     "also show me total count of pending versus completed tasks and high priority task breakdown"),

    ("Q6 — Company Lifecycle Funnel",
     "give me all companies with their lifecycle stage count category wise, "
     "and also show me lead status distribution and which companies have the most deals"),

    ("Q7 — Monthly Revenue Trend 2025",
     "give me monthly sales revenue trend for year 2025 month by month, "
     "which sales owner has the highest total sales revenue, "
     "and also give me total closed won deals count for 2025"),

    ("Q8 — Overdue Invoice Aging Report",
     "show me all overdue invoices that are not paid or cancelled, with company name, "
     "outstanding grand total amount, due date, and group by company to show total outstanding per company"),

    ("Q9 — Business Overview Executive Report",
     "give me a complete business overview report: total number of companies, total contacts, "
     "total deals with closed won count and win rate, total invoices with paid count, "
     "total revenue from paid invoices, and top 3 companies by invoice value"),

    ("Q10 — Contact and Deal Funnel Analysis",
     "give me total contacts grouped by lead status category wise with count, "
     "and deals category wise by stage with count and pipeline value, "
     "also give me win rate and average deal size as KPI summary for the business"),
]


def run_test():
    print("\n" + "="*80)
    print("   CRM CHATBOT QA — 10 Complex Query Tests")
    print("="*80 + "\n")

    print("Loading database and agent (schema discovery runs once)...")
    db = get_database()
    agent = get_sql_agent(db)
    memory = ConversationMemory()
    print("Agent ready.\n")

    passed = 0
    failed = 0

    for i, (label, query) in enumerate(QUERIES, 1):
        print("\n" + "─"*80)
        print(f"[{label}]")
        print(f"QUERY: {query}")
        print("─"*80)

        t0 = time.perf_counter()
        try:
            result = run_agent_query(agent, query, memory=memory)
            elapsed = round((time.perf_counter() - t0) * 1000)

            answer  = result.get("answer", "")
            conf    = result.get("confidence", 0)
            tables  = result.get("tables_used", [])
            sqls    = result.get("sql_queries", [])
            sub_res = result.get("_sub_results")

            is_empty = (
                not answer.strip()
                or "could not generate" in answer.lower()
                or "unable to complete" in answer.lower()
                or (len(answer) < 80 and "no data" in answer.lower())
            )

            status = "FAIL" if is_empty else "PASS"
            if is_empty: failed += 1
            else: passed += 1

            flag = "PASS" if status == "PASS" else "FAIL"
            print(f"STATUS    : {flag}")
            print(f"LATENCY   : {elapsed} ms")
            print(f"CONFIDENCE: {conf}")
            print(f"TABLES    : {tables}")
            print(f"SQL COUNT : {len(sqls)}")
            if sub_res:
                print(f"SUB-PARTS : {len(sub_res)} sub-queries (parallel pipeline)")
                for j, s in enumerate(sub_res, 1):
                    print(f"  Part {j}: [{s['intent']}] {s['sub_query'][:90]}")
            print(f"\nANSWER (first 700 chars):\n{answer[:700]}")
            if len(answer) > 700:
                print("  ...[truncated]")

        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000)
            print(f"STATUS    : EXCEPTION  ({elapsed} ms)")
            print(f"ERROR     : {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"   RESULTS: {passed}/10 PASSED | {failed}/10 FAILED")
    print("="*80)
    if failed == 0:
        print("\nGREEN FLAG -- All 10 complex queries handled correctly.\n")
    else:
        print(f"\nRED FLAG -- {failed} query/queries need attention.\n")

if __name__ == "__main__":
    run_test()
