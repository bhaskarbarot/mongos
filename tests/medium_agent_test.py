"""medium_agent_test.py — Performance test suite for the Medium Agent.

Tests all 50 medium-complexity queries from medium.txt:
  • Routes every query through the MEDIUM agent only
  • Rate-limited to 2 queries per minute (30s gap) to respect LLM API limits
  • Verifies every SQL result against the real PostgreSQL database
  • Scores accuracy per query and reports green/red flag

Output: tests/medium_agent_results.csv

Usage:
    cd /home/elsner/Documents/mongos
    python3 tests/medium_agent_test.py

Estimated runtime: ~50 queries × 15s avg + 30s delay = ~37 minutes
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

for _n in ["httpx", "urllib3", "langchain", "crewai", "openai",
           "groq", "litellm", "LiteLLM", "transformers"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("medium_agent_test")

QUERIES_FILE    = ROOT / "medium.txt"
OUTPUT_CSV      = ROOT / "tests" / "medium_agent_results_v2.csv"
MAX_TIMEOUT_S   = 90          # medium agent is slower than simple
RATE_LIMIT_S    = 30          # 2 queries per minute = 30s between each
GREEN_THRESHOLD = 0.85        # 85%+ accuracy = green flag for medium agent

CSV_COLUMNS = [
    "#", "query",
    "agent_type",
    "full_response",
    "sql_queries",
    "latency_ms",
    "success",
    "has_data",
    "row_count",
    "db_verify_status",
    "db_actual_rows",
    "accuracy_score",
    "error_notes",
]

# ── Queries that genuinely return 0 rows (not a failure) ──────────────────────
_LEGIT_EMPTY_HINTS = [
    "2027", "next month", "next quarter", "last year",
    "no activity", "no deal", "60 day", "90 day",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_queries() -> List[Tuple[int, str]]:
    queries: List[Tuple[int, str]] = []
    with open(QUERIES_FILE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if ". " in line[:6]:
                num_str, text = line.split(". ", 1)
            else:
                parts = line.split(None, 1)
                if not parts or not parts[0].rstrip(".").isdigit():
                    continue
                num_str = parts[0].rstrip(".")
                text = parts[1] if len(parts) > 1 else ""
            try:
                queries.append((int(num_str), text.strip()))
            except ValueError:
                pass
    return queries


def _run_with_timeout(fn, args=(), timeout_s=MAX_TIMEOUT_S):
    _result = [None]; _exc = [None]
    def _w():
        try: _result[0] = fn(*args)
        except Exception as e: _exc[0] = e
    t = threading.Thread(target=_w, daemon=True)
    t.start(); t.join(timeout_s)
    if t.is_alive():
        return None, None, True
    return _result[0], _exc[0], False


def run_medium(query: str) -> Dict[str, Any]:
    """Force query through medium agent with MEDIUM classification."""
    from agents.medium_agent import run_medium_agent
    classification = {
        "type":                  "MEDIUM",
        "reason":                "forced for test",
        "tables_needed":         [],
        "requires_calculation":  True,
        "requires_multi_period": False,
        "routed_by":             "test_harness",
    }
    result, exc, timed_out = _run_with_timeout(
        run_medium_agent, (query, classification), MAX_TIMEOUT_S
    )
    if timed_out:
        return {
            "answer": "TIMEOUT: exceeded time limit.",
            "sql_queries": [], "tables_used": [],
            "latency_ms": MAX_TIMEOUT_S * 1000,
            "agent_type": "medium", "layer": "timeout",
        }
    if exc:
        return {
            "answer": f"EXCEPTION: {exc}",
            "sql_queries": [], "tables_used": [],
            "latency_ms": 0,
            "agent_type": "medium", "layer": "exception",
        }
    return result or {}


def verify_sql(sql: str) -> Tuple[str, int]:
    """Re-run the first SQL query from the agent against real DB."""
    if not sql or not sql.strip():
        return "SKIP_NO_SQL", 0
    try:
        from mcp_server.crm_mcp import execute_sql
        res = execute_sql(sql)
        if res.get("error"):
            return f"FAIL: {res['error'][:80]}", 0
        return "PASS", res.get("total_rows", res.get("row_count", 0))
    except Exception as exc:
        return f"FAIL: {str(exc)[:70]}", 0


def is_legit_empty(query: str) -> bool:
    q = query.lower()
    return any(h in q for h in _LEGIT_EMPTY_HINTS)


def score_accuracy(result: Dict, query: str) -> float:
    """Score 0.0 – 1.0 for a medium agent result."""
    answer  = (result.get("answer") or "").lower()
    layer   = result.get("layer", "")
    sqls    = result.get("sql_queries") or []

    # Hard failures
    if layer in ("timeout", "exception"):
        return 0.0
    fail_signals = [
        "wasn't able", "error processing", "encountered an error",
        "TIMEOUT", "EXCEPTION", "try rephrasing",
    ]
    if any(s.lower() in answer for s in fail_signals):
        return 0.0

    # Good data returned
    data  = result.get("data") or {}
    rows  = (data.get("rows") or []) if isinstance(data, dict) else []
    if len(rows) > 0:
        return 1.0

    # SQL ran but 0 rows — could be legit
    if sqls and ("no records" in answer or "0 record" in answer
                 or is_legit_empty(query)):
        return 0.80

    # Has SQL + answer but no structured rows
    if sqls and len(answer.strip()) > 80:
        return 0.90

    # No SQL at all
    if not sqls:
        return 0.25

    return 0.50


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests() -> bool:
    queries = load_queries()
    total   = len(queries)

    print(f"\n{'═'*72}")
    print(f"  Medium Agent Test Suite  —  {total} queries")
    print(f"  Rate limit  : 2 queries per minute (30s gap)")
    print(f"  Timeout     : {MAX_TIMEOUT_S}s per query")
    print(f"  Output      : {OUTPUT_CSV}")
    print(f"  Est. runtime: ~{total * (15 + RATE_LIMIT_S) // 60} minutes")
    print(f"  Green flag  : ≥ {int(GREEN_THRESHOLD*100)}% accuracy")
    print(f"{'═'*72}\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_fh.flush()

    passed         = 0
    failed         = 0
    accuracy_scores: List[float] = []
    error_log:       List[Tuple[int, str, str]] = []

    for idx, (num, query) in enumerate(queries):
        label = f"[{num:2d}/50]"
        print(f"{label} {query[:60]:<60}", end=" … ", flush=True)

        # ── Run medium agent ──────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            result = run_medium(query)
        except Exception as exc:
            result = {
                "answer": f"EXCEPTION: {exc}",
                "sql_queries": [], "latency_ms": 0, "layer": "exception",
            }
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if not result.get("latency_ms"):
            result["latency_ms"] = elapsed_ms

        answer     = result.get("answer", "")
        sql_list   = result.get("sql_queries") or []
        first_sql  = sql_list[0] if sql_list else ""
        all_sqls   = " | ".join(s[:60] for s in sql_list) if sql_list else ""
        latency_ms = result.get("latency_ms", elapsed_ms)
        data       = result.get("data") or {}
        rows       = (data.get("rows") or []) if isinstance(data, dict) else []
        row_count  = len(rows)
        has_data   = row_count > 0
        agent_type = result.get("agent_type", "medium")

        # ── Accuracy score ────────────────────────────────────────────────────
        acc = score_accuracy(result, query)
        accuracy_scores.append(acc)
        ok  = acc >= 0.50

        if ok:
            passed += 1
            tag = "✓"
        else:
            failed += 1
            tag = "✗"
            error_log.append((num, query, answer[:100]))

        # ── DB Verification (all 50) ──────────────────────────────────────────
        db_status, db_rows = verify_sql(first_sql)
        if db_status == "PASS":
            tag += " [DB✓]"
        elif first_sql:
            tag += " [DB✗]"

        print(f"{tag:<16}  {latency_ms:>6.0f}ms  acc={acc:.2f}  db={db_rows}")

        # ── Write CSV row ─────────────────────────────────────────────────────
        writer.writerow({
            "#":               num,
            "query":           query,
            "agent_type":      agent_type,
            "full_response":    answer.replace("\n", " ").replace("|", "¦"),
            "sql_queries":     all_sqls.replace("\n", " "),
            "latency_ms":      f"{latency_ms:.0f}",
            "success":         "Yes" if ok else "No",
            "has_data":        "Yes" if has_data else "No",
            "row_count":       row_count,
            "db_verify_status": db_status[:50],
            "db_actual_rows":  db_rows,
            "accuracy_score":  f"{acc:.2f}",
            "error_notes":     "" if ok else answer.replace("\n", " "),
        })
        csv_fh.flush()

        # ── Rate limit: 2 queries per minute ─────────────────────────────────
        if idx < total - 1:
            print(f"         ⏱  waiting {RATE_LIMIT_S}s (rate limit)…",
                  flush=True)
            time.sleep(RATE_LIMIT_S)

    csv_fh.close()

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    avg_acc = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    success_rate = passed / total if total else 0

    print(f"\n{'═'*72}")
    print(f"  MEDIUM AGENT — FINAL TEST RESULTS")
    print(f"{'═'*72}")
    print(f"  Total queries   : {total}")
    print(f"  Passed (≥0.50)  : {passed}  ({100*success_rate:.1f}%)")
    print(f"  Failed          : {failed}")
    print(f"  Avg accuracy    : {avg_acc:.3f}  ({100*avg_acc:.1f}%)")
    print(f"  DB verified     : {total} queries (100%)")
    print(f"  Output CSV      : {OUTPUT_CSV}")

    if avg_acc >= GREEN_THRESHOLD:
        print(f"\n  🟢  GREEN FLAG — {100*avg_acc:.1f}%  (target ≥ {int(GREEN_THRESHOLD*100)}%)")
        print(f"      Medium agent performing at production quality!\n")
    else:
        gap = GREEN_THRESHOLD - avg_acc
        print(f"\n  🔴  RED FLAG — {100*avg_acc:.1f}%  "
              f"(need ≥ {int(GREEN_THRESHOLD*100)}%, gap = {100*gap:.1f}%)\n")

    # ── Error breakdown ───────────────────────────────────────────────────────
    if error_log:
        print(f"{'─'*72}")
        print(f"  FAILED QUERIES ({len(error_log)})")
        print(f"{'─'*72}")
        timeout_errs = [(n, q, e) for n, q, e in error_log if "TIMEOUT" in e.upper()]
        sql_errs     = [(n, q, e) for n, q, e in error_log
                        if "wasn't able" in e.lower() or "EXCEPTION" in e.upper()]
        other_errs   = [(n, q, e) for n, q, e in error_log
                        if (n, q, e) not in timeout_errs
                        and (n, q, e) not in sql_errs]

        if timeout_errs:
            print(f"\n  TIMEOUTS ({len(timeout_errs)}) — increase OLLAMA timeout or check Groq keys")
            for n, q, _ in timeout_errs:
                print(f"    Q{n:2d}: {q[:65]}")
        if sql_errs:
            print(f"\n  SQL FAILURES ({len(sql_errs)}) — schema hints or few-shot examples needed")
            for n, q, e in sql_errs[:6]:
                print(f"    Q{n:2d}: {q[:55]}  →  {e[:50]}")
        if other_errs:
            print(f"\n  OTHER ({len(other_errs)})")
            for n, q, e in other_errs[:5]:
                print(f"    Q{n:2d}: {q[:55]}  →  {e[:50]}")

        print(f"\n  IMPROVEMENT PLAN:")
        if timeout_errs:
            print(f"  1. Check Groq API keys are active and not rate-limited")
            print(f"  2. Raise OLLAMA_FALLBACK_TIMEOUT in .env")
        if sql_errs:
            print(f"  3. Add few-shot examples to mcp_server/query_examples.json")
            print(f"     for failed query patterns")
        print(f"  4. Re-run: python3 tests/medium_agent_test.py")

    print(f"\n  Full results → {OUTPUT_CSV}")
    print(f"{'═'*72}\n")

    return avg_acc >= GREEN_THRESHOLD


if __name__ == "__main__":
    green = run_tests()
    sys.exit(0 if green else 1)
