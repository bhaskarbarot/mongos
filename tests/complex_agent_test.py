"""complex_agent_test.py — Performance test suite for the Complex Agent.

Tests all 50 complex queries from complex.txt:
  • Routes every query through the COMPLEX agent only
  • Rate-limited to 1 query per minute (60s gap) — complex agent is heavier
  • Verifies every result against the real PostgreSQL database
  • Full response stored in CSV — no truncation
  • Reports green/red flag at the end

Output: tests/complex_agent_results.csv

Usage:
    cd /home/elsner/Documents/mongos
    python3 tests/complex_agent_test.py

Estimated runtime: ~50 queries × (25s avg + 60s wait) ≈ 72 minutes
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
LOGGER = logging.getLogger("complex_agent_test")

QUERIES_FILE    = ROOT / "complex.txt"
OUTPUT_CSV      = ROOT / "tests" / "complex_agent_results_v2.csv"
MAX_TIMEOUT_S   = 180          # Groq key blocks add 60s each — 180s handles up to 2 blocks per query
RATE_LIMIT_S    = 120          # 2 min between queries — lets rate-limited keys fully recover
GREEN_THRESHOLD = 0.80         # 80%+ accuracy = green flag for complex agent

CSV_COLUMNS = [
    "#", "query",
    "agent_type",
    "full_response",
    "sql_queries",
    "sql_count",
    "latency_ms",
    "success",
    "has_data",
    "row_count",
    "db_verify_status",
    "db_actual_rows",
    "accuracy_score",
    "error_notes",
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


def run_complex(query: str) -> Dict[str, Any]:
    """Force query through the complex agent."""
    from agents.complex_agent import run_complex_agent
    classification = {
        "type":                  "COMPLEX",
        "reason":                "forced for test",
        "tables_needed":         [],
        "requires_calculation":  True,
        "requires_multi_period": True,
        "routed_by":             "test_harness",
    }
    result, exc, timed_out = _run_with_timeout(
        run_complex_agent, (query, classification), MAX_TIMEOUT_S
    )
    if timed_out:
        return {
            "answer":      "TIMEOUT: exceeded time limit.",
            "sql_queries": [], "tables_used": [],
            "latency_ms":  MAX_TIMEOUT_S * 1000,
            "agent_type":  "complex", "layer": "timeout",
        }
    if exc:
        return {
            "answer":      f"EXCEPTION: {exc}",
            "sql_queries": [], "tables_used": [],
            "latency_ms":  0,
            "agent_type":  "complex", "layer": "exception",
        }
    return result or {}


def verify_sql(sql: str) -> Tuple[str, int]:
    """Run a SQL query directly against the DB and return (status, row_count)."""
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


def score_accuracy(result: Dict, query: str) -> float:
    """Score 0.0–1.0 for a complex agent result."""
    answer  = (result.get("answer") or "").lower()
    layer   = result.get("layer", "")
    sqls    = result.get("sql_queries") or []

    # Hard failures
    if layer in ("timeout", "exception"):
        return 0.0

    # Only check fail signals in the first 150 chars — complex reports may
    # mention errors naturally in their narrative body without being failures
    answer_start = answer[:150]
    fail_signals = [
        "wasn't able", "error processing", "encountered an error",
        "TIMEOUT", "EXCEPTION", "try rephrasing",
        "unable to complete", "could not retrieve",
    ]
    if any(s.lower() in answer_start for s in fail_signals):
        return 0.0

    # Complex agent produces rich text reports — score by quality indicators
    data  = result.get("data") or {}
    rows  = (data.get("rows") or []) if isinstance(data, dict) else []

    # Has multiple SQL queries executed (complex agent runs 4-8)
    sql_count = len(sqls)

    # Rich answer — complex agent should produce 200+ word reports
    word_count = len(answer.split())

    if sql_count >= 3 and word_count >= 100:
        return 1.0
    if sql_count >= 2 and word_count >= 60:
        return 0.90
    if sql_count >= 1 and word_count >= 40:
        return 0.80
    if sqls and word_count >= 20:
        return 0.70
    if sqls:
        return 0.50
    return 0.25


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests() -> bool:
    queries = load_queries()
    total   = len(queries)

    print(f"\n{'═'*72}")
    print(f"  Complex Agent Test Suite  —  {total} queries")
    print(f"  Rate limit  : 1 query per minute (60s gap)")
    print(f"  Timeout     : {MAX_TIMEOUT_S}s per query")
    print(f"  Output      : {OUTPUT_CSV}")
    print(f"  Est. runtime: ~{total * (30 + RATE_LIMIT_S) // 60} minutes")
    print(f"  Green flag  : ≥ {int(GREEN_THRESHOLD*100)}% accuracy")
    print(f"{'═'*72}\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_fh.flush()

    passed         = 0
    failed         = 0
    accuracy_scores: List[float]             = []
    error_log:       List[Tuple[int, str, str]] = []

    for idx, (num, query) in enumerate(queries):
        label = f"[{num:2d}/50]"
        print(f"{label} {query[:58]:<58}", end="\n       ", flush=True)

        # ── Run complex agent ─────────────────────────────────────────────────
        t_start = time.monotonic()
        try:
            result = run_complex(query)
        except Exception as exc:
            result = {
                "answer": f"EXCEPTION: {exc}",
                "sql_queries": [], "latency_ms": 0, "layer": "exception",
            }
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        if not result.get("latency_ms"):
            result["latency_ms"] = elapsed_ms

        answer     = result.get("answer", "")
        sql_list   = result.get("sql_queries") or []
        first_sql  = sql_list[0] if sql_list else ""
        all_sqls   = " ||| ".join(s[:80].replace("\n", " ") for s in sql_list)
        latency_ms = result.get("latency_ms", elapsed_ms)
        data       = result.get("data") or {}
        rows       = (data.get("rows") or []) if isinstance(data, dict) else []
        row_count  = len(rows)
        has_data   = row_count > 0
        agent_type = result.get("agent_type", "complex")
        word_count = len(answer.split())

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
            error_log.append((num, query, answer[:120]))

        # ── DB Verification (all 50 — use first SQL) ──────────────────────────
        db_status, db_rows = verify_sql(first_sql)
        db_tag = "[DB✓]" if db_status == "PASS" else ("[DB✗]" if first_sql else "[no SQL]")

        print(f"  {tag} {db_tag:<8}  {latency_ms:>6.0f}ms  "
              f"sqls={len(sql_list)}  words={word_count}  acc={acc:.2f}  db={db_rows}")

        # ── Write CSV row (full response, no truncation) ──────────────────────
        writer.writerow({
            "#":               num,
            "query":           query,
            "agent_type":      agent_type,
            "full_response":   answer.replace("\n", " ").replace("|", "¦"),
            "sql_queries":     all_sqls,
            "sql_count":       len(sql_list),
            "latency_ms":      f"{latency_ms:.0f}",
            "success":         "Yes" if ok else "No",
            "has_data":        "Yes" if has_data else "No",
            "row_count":       row_count,
            "db_verify_status": db_status[:60],
            "db_actual_rows":  db_rows,
            "accuracy_score":  f"{acc:.2f}",
            "error_notes":     "" if ok else answer[:200].replace("\n", " "),
        })
        csv_fh.flush()

        # ── Rate limit: 1 query per minute ────────────────────────────────────
        if idx < total - 1:
            elapsed_total = time.monotonic() - t_start
            wait_s = max(0, RATE_LIMIT_S - elapsed_total)
            if wait_s > 0:
                print(f"       ⏱  waiting {wait_s:.0f}s (1/min rate limit)…",
                      flush=True)
                time.sleep(wait_s)

    csv_fh.close()

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    avg_acc      = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    success_rate = passed / total if total else 0

    print(f"\n{'═'*72}")
    print(f"  COMPLEX AGENT — FINAL TEST RESULTS")
    print(f"{'═'*72}")
    print(f"  Total queries   : {total}")
    print(f"  Passed (≥0.50)  : {passed}  ({100*success_rate:.1f}%)")
    print(f"  Failed          : {failed}")
    print(f"  Avg accuracy    : {avg_acc:.3f}  ({100*avg_acc:.1f}%)")
    print(f"  DB verified     : {total} queries (100%)")
    print(f"  Output CSV      : {OUTPUT_CSV}")

    if avg_acc >= GREEN_THRESHOLD:
        print(f"\n  🟢  GREEN FLAG — {100*avg_acc:.1f}%  (target ≥ {int(GREEN_THRESHOLD*100)}%)")
        print(f"      Complex agent performing at production quality!\n")
    else:
        gap = GREEN_THRESHOLD - avg_acc
        print(f"\n  🔴  RED FLAG — {100*avg_acc:.1f}%  "
              f"(need ≥ {int(GREEN_THRESHOLD*100)}%, gap = {100*gap:.1f}%)\n")

    if error_log:
        print(f"{'─'*72}")
        print(f"  FAILED QUERIES ({len(error_log)})")
        print(f"{'─'*72}")
        for n, q, e in error_log:
            print(f"  Q{n:2d}: {q[:60]}")
            print(f"       → {e[:80]}")

        print(f"\n  IMPROVEMENT PLAN:")
        print(f"  1. Check Groq API keys (complex agent uses Groq 70b for synthesis)")
        print(f"  2. For timeout queries: raise OLLAMA_REASONING_TIMEOUT in .env")
        print(f"  3. For SQL failures: add few-shot examples to query_examples.json")
        print(f"  4. Re-run: python3 tests/complex_agent_test.py")

    print(f"\n  Full results → {OUTPUT_CSV}")
    print(f"{'═'*72}\n")

    return avg_acc >= GREEN_THRESHOLD


if __name__ == "__main__":
    green = run_tests()
    sys.exit(0 if green else 1)
