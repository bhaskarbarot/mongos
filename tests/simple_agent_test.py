"""simple_agent_test.py — Comprehensive test suite for the Simple Agent.

Tests all 125 simple queries from simple_queries.txt through the simple agent.
• 40% of queries are re-verified against the real PostgreSQL database.
• Accuracy is scored 0.0–1.0 per query.
• Outputs: tests/simple_agent_results.csv

Usage:
    cd /home/elsner/Documents/mongos
    python tests/simple_agent_test.py
"""
from __future__ import annotations

import csv
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # .env is loaded relative to cwd

# ── Silence noisy loggers ─────────────────────────────────────────────────────
for _noisy in [
    "httpx", "urllib3", "langchain", "crewai", "openai",
    "transformers", "groq", "anthropic", "litellm", "LiteLLM",
]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("simple_agent_test")

# ── Constants ──────────────────────────────────────────────────────────────────
QUERIES_FILE      = ROOT / "simple_queries.txt"
OUTPUT_CSV        = ROOT / "tests" / "simple_agent_results.csv"
VERIFY_PCT        = 0.40          # fraction of queries verified in real DB
RANDOM_SEED       = 42
MAX_TIMEOUT_S     = 60            # per-query wall-clock timeout
INTER_QUERY_DELAY = 0.3           # seconds between queries (rate-limit buffer)
GREEN_THRESHOLD   = 0.90          # 90 %+ average accuracy = green flag

CSV_COLUMNS = [
    "#",
    "query",
    "response_summary",
    "sql_query",
    "success",
    "has_data",
    "row_count",
    "latency_ms",
    "accuracy_score",
    "db_verified",
    "db_verify_status",
    "db_actual_rows",
    "error_notes",
]


# ══════════════════════════════════════════════════════════════════════════════
# QUERY LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_queries() -> List[Tuple[int, str]]:
    """Parse simple_queries.txt → [(num, text), ...]."""
    queries: List[Tuple[int, str]] = []
    with open(QUERIES_FILE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # Supports "1. text" and "1 text"
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


# ══════════════════════════════════════════════════════════════════════════════
# THREAD-BASED TIMEOUT
# ══════════════════════════════════════════════════════════════════════════════

def _run_with_timeout(fn, args=(), kwargs=None, timeout_s: float = MAX_TIMEOUT_S):
    """Run fn in a daemon thread. Returns (result, exception, timed_out)."""
    kwargs = kwargs or {}
    _result: List[Any]        = [None]
    _exc:    List[Optional[Exception]] = [None]

    def _worker():
        try:
            _result[0] = fn(*args, **kwargs)
        except Exception as exc:            # noqa: BLE001
            _exc[0] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, None, True       # timed out
    return _result[0], _exc[0], False


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER (with safe fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _classify(query: str) -> Dict:
    try:
        from pipeline.classifier import classify
        return classify(query)
    except Exception as exc:
        LOGGER.debug("Classifier error for '%s': %s", query[:50], exc)
        return {
            "type": "SIMPLE",
            "reason": f"classifier unavailable: {exc}",
            "tables_needed": [],
            "requires_calculation": False,
            "requires_multi_period": False,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE AGENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str) -> Dict[str, Any]:
    """Run one query through the simple agent and return its result dict."""
    from agents.simple_agent import run_simple_agent

    classification = _classify(query)
    # Force SIMPLE regardless of what the classifier says — this test targets
    # the simple agent exclusively.
    classification["type"] = "SIMPLE"

    result, exc, timed_out = _run_with_timeout(
        run_simple_agent,
        args=(query, classification),
        timeout_s=MAX_TIMEOUT_S,
    )

    if timed_out:
        return {
            "answer":      "TIMEOUT: query exceeded time limit.",
            "sql_queries": [],
            "tables_used": [],
            "confidence":  0.0,
            "agent_type":  "simple",
            "latency_ms":  int(MAX_TIMEOUT_S * 1000),
            "attempts":    0,
            "layer":       "timeout",
        }

    if exc is not None:
        return {
            "answer":      f"EXCEPTION: {exc}",
            "sql_queries": [],
            "tables_used": [],
            "confidence":  0.0,
            "agent_type":  "simple",
            "latency_ms":  0,
            "attempts":    0,
            "layer":       "exception",
        }

    return result or {}


# ══════════════════════════════════════════════════════════════════════════════
# DB VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_sql(sql: str) -> Tuple[str, int]:
    """Re-execute SQL against real DB. Returns (status_label, row_count)."""
    if not sql or not sql.strip():
        return "SKIP_NO_SQL", 0
    try:
        from mcp_server.crm_mcp import execute_sql
        res = execute_sql(sql)
        if res.get("error"):
            short_err = res["error"][:100]
            return f"FAIL: {short_err}", 0
        return "PASS", res.get("total_rows", res.get("row_count", 0))
    except Exception as exc:                          # noqa: BLE001
        return f"FAIL: {str(exc)[:80]}", 0


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY SCORING  (0.0 / 0.5 / 0.75 / 1.0)
# ══════════════════════════════════════════════════════════════════════════════

_FAIL_SIGNALS = [
    "wasn't able to retrieve",
    "error processing",
    "failed after",
    "TIMEOUT",
    "EXCEPTION",
    "encountered an error",
    "i encountered an error",
    "try rephrasing",
]

_GUARD_SIGNALS = [
    "i am mongos",
    "i'm mongos",
    "hello! i can help",
    "crm ai assistant",
    "i can help you query",
    "sql chatbot",
]


def score(result: Dict) -> float:
    """Return accuracy score 0.0–1.0 for a single query result."""
    answer  = (result.get("answer") or "").lower()
    layer   = result.get("layer", "")
    sql_lst = result.get("sql_queries") or []
    sql     = sql_lst[0] if sql_lst else ""

    # Hard failures
    if layer in ("timeout", "exception"):
        return 0.0
    if any(sig.lower() in answer for sig in _FAIL_SIGNALS):
        return 0.0

    # Guard-handled (identity / greeting) — valid but no SQL
    if any(sig.lower() in answer for sig in _GUARD_SIGNALS):
        return 0.75

    # No SQL generated at all but answer present
    if not sql:
        if len(answer.strip()) > 40:
            return 0.50    # partial — answered in natural language without SQL
        return 0.25

    # SQL present — check if answer has meaningful content
    data     = result.get("data") or {}
    rows     = data.get("rows") or []
    has_data = len(rows) > 0

    if has_data:
        return 1.0

    # SQL ran but 0 rows — could be correct (e.g., future date queries)
    if "no records found" in answer or "0 record" in answer or "none found" in answer:
        return 0.80   # legitimate empty result

    # SQL present, answer is non-trivial
    if len(answer.strip()) > 60:
        return 0.90

    return 0.50


def is_success(result: Dict, acc: float) -> bool:
    return acc >= 0.50


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    queries = load_queries()
    total   = len(queries)
    n_verify = int(total * VERIFY_PCT)

    print(f"\n{'═'*70}")
    print(f"  Simple Agent Test Suite  —  {total} queries")
    print(f"  DB verification on {n_verify} queries ({int(VERIFY_PCT*100)}%)")
    print(f"  Output  : {OUTPUT_CSV}")
    print(f"  Timeout : {MAX_TIMEOUT_S}s per query")
    print(f"{'═'*70}\n")

    # Fixed random sample for DB verification
    rng        = random.Random(RANDOM_SEED)
    verify_set = set(rng.sample(range(total), k=n_verify))

    # Prepare CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_fh.flush()

    passed         = 0
    failed         = 0
    accuracy_scores: List[float]             = []
    error_log:       List[Tuple[int, str, str]] = []

    try:
        for idx, (num, query) in enumerate(queries):
            label = f"[{num:3d}/125]"
            print(f"{label} {query[:58]:<58}", end=" … ", flush=True)

            # ── Run simple agent ──────────────────────────────────────────────
            try:
                result = run_agent(query)
            except Exception as exc:                   # noqa: BLE001
                result = {
                    "answer":      f"EXCEPTION: {exc}",
                    "sql_queries": [],
                    "tables_used": [],
                    "latency_ms":  0,
                    "layer":       "exception",
                }

            answer     = result.get("answer", "")
            sql_lst    = result.get("sql_queries") or []
            sql_query  = sql_lst[0] if sql_lst else ""
            latency_ms = result.get("latency_ms", 0)
            data       = result.get("data") or {}
            rows       = data.get("rows") or []
            row_count  = len(rows)
            has_data   = row_count > 0

            acc     = score(result)
            ok      = is_success(result, acc)
            accuracy_scores.append(acc)

            if ok:
                passed += 1
                tag = "✓"
            else:
                failed += 1
                tag = "✗"
                error_log.append((num, query, answer[:120]))

            # ── DB Verification (40 % sample) ─────────────────────────────────
            db_verified      = "No"
            db_verify_status = "SKIP"
            db_actual_rows   = ""

            if idx in verify_set:
                db_verified = "Yes"
                if sql_query:
                    db_verify_status, db_actual = verify_sql(sql_query)
                    db_actual_rows = str(db_actual)
                    if db_verify_status == "PASS":
                        tag += " [DB✓]"
                    else:
                        tag += " [DB✗]"
                else:
                    db_verify_status = "SKIP_NO_SQL"

            print(f"{tag:<14}  {latency_ms:>6.0f}ms  acc={acc:.2f}")

            # ── Write CSV row (incremental) ───────────────────────────────────
            resp_safe = answer[:200].replace("\n", " ").replace("|", "¦")
            sql_safe  = sql_query.replace("\n", " ") if sql_query else ""
            err_note  = "" if ok else answer[:150].replace("\n", " ")

            writer.writerow({
                "#":                num,
                "query":            query,
                "response_summary": resp_safe,
                "sql_query":        sql_safe,
                "success":          "Yes" if ok else "No",
                "has_data":         "Yes" if has_data else "No",
                "row_count":        row_count,
                "latency_ms":       f"{latency_ms:.0f}",
                "accuracy_score":   f"{acc:.2f}",
                "db_verified":      db_verified,
                "db_verify_status": db_verify_status,
                "db_actual_rows":   db_actual_rows,
                "error_notes":      err_note,
            })
            csv_fh.flush()

            time.sleep(INTER_QUERY_DELAY)

    finally:
        csv_fh.close()

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    avg_acc      = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    success_rate = passed / total if total else 0.0

    print(f"\n{'═'*70}")
    print(f"  SIMPLE AGENT — TEST RESULTS")
    print(f"{'═'*70}")
    print(f"  Total queries   : {total}")
    print(f"  Passed (≥0.50)  : {passed}  ({100*success_rate:.1f}%)")
    print(f"  Failed (<0.50)  : {failed}")
    print(f"  Avg accuracy    : {avg_acc:.3f}  ({100*avg_acc:.1f}%)")
    print(f"  DB verified     : {n_verify} queries ({int(VERIFY_PCT*100)}%)")
    print(f"  Output CSV      : {OUTPUT_CSV}")

    # ── Green / Red flag ──────────────────────────────────────────────────────
    if avg_acc >= GREEN_THRESHOLD:
        print(f"\n  🟢  GREEN FLAG — {100*avg_acc:.1f}% accuracy  (target ≥ 90%)")
        print(f"      Simple agent is performing at production quality!\n")
    else:
        gap = GREEN_THRESHOLD - avg_acc
        print(f"\n  🔴  RED FLAG — {100*avg_acc:.1f}% accuracy  (need ≥ 90%)")
        print(f"      Gap to target: {100*gap:.1f}% points\n")

    # ── Error analysis + improvement plan ─────────────────────────────────────
    if error_log:
        print(f"{'─'*70}")
        print(f"  FAILED QUERIES  ({len(error_log)} total)")
        print(f"{'─'*70}")

        timeout_errs  = [(n, q, e) for n, q, e in error_log if "TIMEOUT" in e.upper()]
        sql_gen_errs  = [(n, q, e) for n, q, e in error_log
                         if "wasn't able to retrieve" in e.lower() or "EXCEPTION" in e.upper()]
        empty_errs    = [(n, q, e) for n, q, e in error_log
                         if "no records" in e.lower()]
        other_errs    = [(n, q, e) for n, q, e in error_log
                         if (n, q, e) not in timeout_errs
                         and (n, q, e) not in sql_gen_errs
                         and (n, q, e) not in empty_errs]

        _print_group("TIMEOUT (LLM too slow)", timeout_errs,
                     "→ Increase OLLAMA_FALLBACK_TIMEOUT or ensure Groq keys are active")
        _print_group("SQL GENERATION FAILED", sql_gen_errs,
                     "→ Add more few-shot examples for these query patterns")
        _print_group("EMPTY RESULTS (possible data gap)", empty_errs,
                     "→ Verify real DB has the expected records; may be data quality issue")
        _print_group("OTHER", other_errs,
                     "→ Review individual answers — may need schema hints or reclassification")

        print(f"\n{'─'*70}")
        print("  IMPROVEMENT PLAN")
        print(f"{'─'*70}")
        _improvement_plan(timeout_errs, sql_gen_errs, empty_errs, other_errs, avg_acc)

    print(f"\n  Full results → {OUTPUT_CSV}")
    print(f"{'═'*70}\n")

    return avg_acc >= GREEN_THRESHOLD


# ── Reporting helpers ──────────────────────────────────────────────────────────

def _print_group(title: str, items: list, advice: str) -> None:
    if not items:
        return
    print(f"\n  [{len(items)}] {title}")
    print(f"       {advice}")
    for n, q, _ in items[:8]:
        print(f"       Q{n:3d}: {q[:65]}")
    if len(items) > 8:
        print(f"       … and {len(items)-8} more (see CSV for full list)")


def _improvement_plan(timeout_errs, sql_gen_errs, empty_errs, other_errs,
                      avg_acc: float) -> None:
    step = 1

    if timeout_errs:
        print(f"\n  {step}. FIX TIMEOUTS ({len(timeout_errs)} queries)")
        print("     • Ensure GROQ_API_KEY is valid and not rate-limited.")
        print("     • Set FAST_PATH_ENABLED=false (already done).")
        print("     • As fallback: raise OLLAMA_FALLBACK_TIMEOUT in .env.")
        step += 1

    if sql_gen_errs:
        print(f"\n  {step}. ADD FEW-SHOT EXAMPLES ({len(sql_gen_errs)} failed patterns)")
        print("     • Open mcp_server/query_examples.json")
        print("     • Add 2-3 new natural-language → SQL pairs for each failed pattern.")
        print("     • Focus on: date-range queries, named-entity lookups, SO/invoice IDs.")
        step += 1

    if empty_errs:
        print(f"\n  {step}. CHECK DATA FRESHNESS ({len(empty_errs)} empty-result queries)")
        print("     • Run the SQL from the CSV in psql to confirm empty vs. bad SQL.")
        print("     • If DB has no data for those entities, results are correctly empty.")
        step += 1

    if other_errs:
        print(f"\n  {step}. REVIEW BORDERLINE QUERIES ({len(other_errs)} others)")
        print("     • Check CSV column 'error_notes' for each failed query.")
        print("     • Consider reclassifying as MEDIUM if they require multi-table joins.")
        step += 1

    if avg_acc < GREEN_THRESHOLD:
        print(f"\n  {step}. SCHEMA KEYWORD TUNING")
        print("     • Review pipeline/db_schema.py _KEYWORDS dict.")
        print("     • Add missing trigger words for tables that failed queries reference.")
        step += 1
        print(f"\n  {step}. RE-RUN THIS TEST after fixes:")
        print("     python tests/simple_agent_test.py")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    green = run_tests()
    sys.exit(0 if green else 1)
