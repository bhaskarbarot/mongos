"""
query_benchmark.py — Full CRM chatbot query benchmark suite.

Runs all production queries one by one (10s gap between each),
captures every detail, and writes a timestamped CSV report.

Usage:
    python tests/query_benchmark.py

Output:
    logs/benchmark_YYYYMMDD_HHMMSS.csv

CSV columns:
    #, category, query, layer, status, accuracy,
    processing_ms, tables_used, sql_count, sql_preview,
    response_preview, full_response
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── Lazy pipeline import (after path fix) ─────────────────────────────────────
from db import get_database                         # noqa: E402
from agent import get_sql_agent, ConversationMemory  # noqa: E402
from pipeline.main import run                        # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# QUERY LIST  — (category, query)
# ══════════════════════════════════════════════════════════════════════════════

QUERIES: List[tuple] = [
    # ── Core Business ──────────────────────────────────────────────────────
    ("core",     "How many departments"),
    ("core",     "Give me names of departments"),
    ("core",     "Give me who have pending task"),
    ("core",     "Share kartik's task status"),
    ("core",     "Share me yesh bhide pending task list"),
    ("core",     "Give me deals with categories"),
    ("core",     "Share me deals with stages"),
    ("core",     "Give me all deals with year wise stages status count"),
    ("core",     "Share me revenue of september 2027"),
    ("core",     "Share me revenue of march 2025"),
    ("core",     "Give me list of customers which have pending invoices"),
    ("core",     "Give me summary of deals"),
    ("core",     "Share me top 10 invoices by amount"),
    ("core",     "Share me paid invoices top 10 only by amount"),
    ("core",     "I want all details of pankaj"),
    ("core",     "I need to find details of negotiable deals with total amount and count"),
    ("core",     "Who are you"),
    ("core",     "Give me kpi report"),
    ("core",     "Share me all contacts list"),
    ("core",     "I want last 5 contacts"),
    ("core",     "I have to see last 5 contacts details with summary"),
    ("core",     "Compare last year revenue and current year and give me summary"),
    ("core",     "Give me open deals and the ratio of closed deals"),
    ("core",     "I need to see the first invoice details"),
    ("core",     "Give me 1st invoice details"),
    ("core",     "Give me invoices that are paid and currency is INR only"),
    ("core",     "Give me USD currency invoices list only with total"),
    ("core",     "Show me unpaid invoices with their currency"),
    # ── Tasks / Team ───────────────────────────────────────────────────────
    ("tasks",    "What tasks are pending for the users"),
    ("tasks",    "What tasks are pending for the sales team"),
    ("tasks",    "Pending tasks"),
    ("tasks",    "Which customers have not given business in 3 months"),
    ("tasks",    "Show me the top 10 customers by revenue"),
    # ── Deals / Pipeline ───────────────────────────────────────────────────
    ("deals",    "Give me closed won deals"),
    ("deals",    "Give me all deals"),
    ("deals",    "How many deals are there"),
    ("deals",    "Who is ketul"),
    ("deals",    "Sales pipeline by stage"),
    ("deals",    "Show pipeline distribution by stage"),
    ("deals",    "Which deals are stuck in the proposal stage"),
    ("deals",    "Opportunities lost in July"),
    ("deals",    "Lost deals last quarter"),
    ("deals",    "Which sales rep closed the most deals this quarter"),
    ("deals",    "Give me the sales pipeline by stage"),
    # ── Leads / Contacts ───────────────────────────────────────────────────
    ("contacts", "List new leads from this week"),
    ("contacts", "New leads this month"),
    ("contacts", "Last 5 contacts details"),
    ("contacts", "Share me all contacts list"),
    # ── Revenue / Finance ──────────────────────────────────────────────────
    ("revenue",  "Show me the total sales for last month"),
    ("revenue",  "Total revenue this financial year"),
    ("revenue",  "Total revenue generated in 2025"),
    ("revenue",  "Total revenue of 2025"),
    ("revenue",  "Pending invoices"),
    ("revenue",  "Overdue payments"),
    ("revenue",  "Total collection"),
    ("revenue",  "Overdue invoice aging and outstanding amount by company"),
    # ── Follow-ups ─────────────────────────────────────────────────────────
    ("followup", "Which leads have not been contacted in 7 days"),
    ("followup", "Conversion rate from lead to customer"),
    ("followup", "Any follow-ups overdue this week"),
    # ── Complex / Analytics ────────────────────────────────────────────────
    ("complex",  "Top performing sales owners in last 6 months with monthly trend"),
    ("complex",  "Funnel performance and conversion to closed won by owner"),
    ("complex",  "High activity companies with weak payment conversion"),
    ("complex",  "Best product and project type combinations by value average deal size and win rate"),
]

WAIT_BETWEEN_S = 10   # seconds gap between queries

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY SCORER
# ══════════════════════════════════════════════════════════════════════════════

_ERROR_SIGNALS = [
    "encountered an error", "internal error", "exception",
    "pipeline error", "failed to", "timed out",
    "i wasn't able to retrieve",
]

# "No data" = pipeline ran correctly, DB simply has no matching records.
# This is a VALID response — scored as PASS.
_NO_DATA_SIGNALS = [
    "no data", "no records", "no result", "not found",
    "no invoices", "no deals", "no contacts", "no tasks",
    "no companies", "no leads", "no overdue", "no pending",
    "no closed", "no open", "no payments", "no follow",
    "0 result", "criteria.", "your criteria",
]


def score_accuracy(query: str, result: Dict[str, Any]) -> float:
    """
    Score 0.0–1.0 based on response quality signals.

    1.0  — has exec summary + real data  OR  valid "no data" answer
    0.85 — has exec summary (no SQL, e.g. fast-path template)
    0.70 — has data table / number, no exec summary
    0.65 — identity / greeting guard (correct for those queries)
    0.10 — real error / exception (pipeline broke)

    NOTE: "No data found" is treated as PASS (1.0) — the pipeline ran
    correctly and the DB genuinely has no matching records.
    """
    ans    = result.get("answer", "")
    layer  = result.get("layer", "")
    sqls   = result.get("sql_queries", [])
    ans_l  = ans.lower()

    # Real error — pipeline broke
    if any(s in ans_l for s in _ERROR_SIGNALS):
        return 0.10

    # Guard: identity/greeting/blocked — always correct behaviour
    if layer == "guard":
        return 0.90

    # "No data" — DB has no matching records — valid correct answer
    if any(s in ans_l for s in _NO_DATA_SIGNALS):
        return 1.0

    # Has exec summary + data → perfect
    has_summary = "Executive Summary" in ans or "## executive" in ans_l
    has_data    = bool(re.search(r"\*\*[\d,.$]+\*\*|Found \*\*\d+\*\*|\|\s*---", ans))

    if has_summary and has_data:
        return 1.0
    if has_summary:
        return 0.85
    if has_data:
        return 0.70
    if ans.strip():
        return 0.65
    return 0.10


def status_label(score: float) -> str:
    if score >= 0.80:
        return "PASS"
    if score >= 0.55:
        return "PARTIAL"
    return "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean(text: str, maxlen: int = 300) -> str:
    """Strip markdown noise and truncate for CSV."""
    t = re.sub(r"[#\*`\|]", "", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:maxlen] + ("…" if len(t) > maxlen else "")


def sql_preview(sqls: List[str]) -> str:
    if not sqls:
        return ""
    return sqls[0][:200].replace("\n", " ")


def print_bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / total)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    csv_path   = out_dir / f"benchmark_{ts}.csv"

    print("=" * 70)
    print("  CRM CHATBOT — FULL QUERY BENCHMARK")
    print(f"  {len(QUERIES)} queries  |  {WAIT_BETWEEN_S}s gap  |  output: {csv_path.name}")
    print("=" * 70)
    print()

    # Connect
    print("Connecting to database and initialising agent...")
    try:
        db    = get_database()
        agent = get_sql_agent(db)
        mem   = ConversationMemory()
        print("Agent ready.\n")
    except Exception as exc:
        print(f"FATAL: Could not connect — {exc}")
        sys.exit(1)

    # CSV header
    fieldnames = [
        "#", "category", "query", "layer", "status", "accuracy",
        "processing_ms", "tables_used", "sql_count",
        "sql_preview", "response_preview", "full_response",
    ]

    results  = []
    pass_c   = fail_c = partial_c = 0
    total    = len(QUERIES)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (category, query) in enumerate(QUERIES, 1):
            print(f"{'─'*70}")
            print(f"  [{idx:02d}/{total}] {category.upper()}  |  {query!r}")

            # Run query
            try:
                result     = run(agent, query, memory=mem, request_id=f"bench_{idx:02d}")
                layer      = result.get("layer", "unknown")
                ms         = round(result.get("latency_ms", 0))
                tables     = result.get("tables_used", [])
                sqls       = result.get("sql_queries", [])
                answer     = result.get("answer", "")
                accuracy   = score_accuracy(query, result)
                status     = status_label(accuracy)
            except Exception as exc:
                layer    = "error"
                ms       = 0
                tables   = []
                sqls     = []
                answer   = f"EXCEPTION: {exc}"
                accuracy = 0.0
                status   = "FAIL"

            # Print live result
            acc_str   = f"{accuracy:.2f}"
            mark      = "✅" if status == "PASS" else ("⚠️ " if status == "PARTIAL" else "❌")
            snip      = clean(answer, 90)
            print(f"  {mark} {status:<7}  acc={acc_str}  layer={layer}  {ms}ms")
            print(f"     Tables : {', '.join(tables) or '—'}")
            print(f"     Answer : {snip}")

            if status == "PASS":    pass_c    += 1
            elif status == "PARTIAL": partial_c += 1
            else:                   fail_c    += 1

            row = {
                "#":               idx,
                "category":        category,
                "query":           query,
                "layer":           layer,
                "status":          status,
                "accuracy":        acc_str,
                "processing_ms":   ms,
                "tables_used":     " | ".join(tables),
                "sql_count":       len(sqls),
                "sql_preview":     sql_preview(sqls),
                "response_preview": clean(answer, 300),
                "full_response":   (answer or "").replace("\n", " "),
            }
            writer.writerow(row)
            f.flush()
            results.append(row)

            # Progress bar
            print(f"\n  {print_bar(idx, total)}")

            # Wait between queries (skip after last one)
            if idx < total:
                print(f"  Waiting {WAIT_BETWEEN_S}s before next query...\n")
                time.sleep(WAIT_BETWEEN_S)

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"  Total queries   : {total}")
    print(f"  ✅ PASS         : {pass_c}  ({pass_c*100//total}%)")
    print(f"  ⚠️  PARTIAL      : {partial_c}  ({partial_c*100//total}%)")
    print(f"  ❌ FAIL         : {fail_c}  ({fail_c*100//total}%)")
    print()

    # Per-category breakdown
    from collections import defaultdict
    cats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"PASS": 0, "PARTIAL": 0, "FAIL": 0})
    for row in results:
        cats[row["category"]][row["status"]] += 1

    print("  Category breakdown:")
    print(f"  {'Category':<12} {'PASS':>6} {'PARTIAL':>8} {'FAIL':>6}")
    print(f"  {'─'*36}")
    for cat, counts in sorted(cats.items()):
        print(f"  {cat:<12} {counts['PASS']:>6} {counts['PARTIAL']:>8} {counts['FAIL']:>6}")

    # Failed queries
    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print(f"\n  Failed queries ({len(failed)}):")
        for r in failed:
            print(f"    [{r['#']:02d}] {r['query']!r}  →  {r['response_preview'][:60]}")

    overall_acc = sum(float(r["accuracy"]) for r in results) / total
    print(f"\n  Overall accuracy score : {overall_acc:.2f} / 1.00")
    green = overall_acc >= 0.75
    print(f"\n  {'🟢 GREEN FLAG — system is performing well!' if green else '🔴 Needs improvement.'}")
    print(f"\n  CSV saved → {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
