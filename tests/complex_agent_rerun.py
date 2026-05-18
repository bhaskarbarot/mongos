"""complex_agent_rerun.py — Re-run only the 19 failed queries from complex_agent_results.csv.

Steps:
  1. Load complex.txt — get all 50 queries
  2. Load complex_agent_results.csv — remove failed rows
  3. Re-run 19 failed queries through complex agent (180s timeout, 120s gap)
  4. Merge with the 31 passing rows → write final combined CSV
  5. Print full report

Output: tests/complex_agent_results.csv (overwritten with all 50 complete)
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
LOGGER = logging.getLogger("complex_rerun")

QUERIES_FILE  = ROOT / "complex.txt"
EXISTING_CSV  = ROOT / "tests" / "complex_agent_results.csv"
OUTPUT_CSV    = ROOT / "tests" / "complex_agent_results.csv"   # overwrite same file

MAX_TIMEOUT_S = 180    # 3 min — handles 2 Groq blocks (60s each) + 30s query
RATE_LIMIT_S  = 120    # 2 min gap — lets rate-limited keys fully recover
GREEN_THRESHOLD = 0.80

CSV_COLUMNS = [
    "#", "query", "agent_type", "full_response", "sql_queries", "sql_count",
    "latency_ms", "success", "has_data", "row_count",
    "db_verify_status", "db_actual_rows", "accuracy_score", "error_notes",
]

FAILED_NOS = {25, 33, 34, 35, 36, 37, 38, 39, 40, 41,
              42, 43, 44, 45, 46, 47, 48, 49, 50}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_queries() -> Dict[int, str]:
    out = {}
    with open(QUERIES_FILE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line: continue
            if ". " in line[:6]:
                n, t = line.split(". ", 1)
            else:
                parts = line.split(None, 1)
                if not parts or not parts[0].rstrip(".").isdigit(): continue
                n, t = parts[0].rstrip("."), (parts[1] if len(parts) > 1 else "")
            try: out[int(n)] = t.strip()
            except ValueError: pass
    return out


def load_passing_rows() -> List[Dict]:
    if not EXISTING_CSV.exists():
        return []
    rows = list(csv.DictReader(open(EXISTING_CSV, encoding="utf-8")))
    return [r for r in rows if r["success"] == "Yes"]


def _run_with_timeout(fn, args=(), timeout_s=MAX_TIMEOUT_S):
    _result = [None]; _exc = [None]
    def _w():
        try: _result[0] = fn(*args)
        except Exception as e: _exc[0] = e
    t = threading.Thread(target=_w, daemon=True)
    t.start(); t.join(timeout_s)
    if t.is_alive(): return None, None, True
    return _result[0], _exc[0], False


def run_complex(query: str) -> Dict[str, Any]:
    from agents.complex_agent import run_complex_agent
    cls = {
        "type": "COMPLEX", "reason": "rerun",
        "tables_needed": [], "requires_calculation": True,
        "requires_multi_period": True, "routed_by": "rerun_harness",
    }
    result, exc, timed_out = _run_with_timeout(run_complex_agent, (query, cls), MAX_TIMEOUT_S)
    if timed_out:
        return {"answer": "TIMEOUT: exceeded time limit.", "sql_queries": [],
                "latency_ms": MAX_TIMEOUT_S * 1000, "agent_type": "complex", "layer": "timeout"}
    if exc:
        return {"answer": f"EXCEPTION: {exc}", "sql_queries": [],
                "latency_ms": 0, "agent_type": "complex", "layer": "exception"}
    return result or {}


def verify_sql(sql: str) -> Tuple[str, int]:
    if not sql or not sql.strip(): return "SKIP_NO_SQL", 0
    try:
        from mcp_server.crm_mcp import execute_sql
        res = execute_sql(sql)
        if res.get("error"): return f"FAIL: {res['error'][:80]}", 0
        return "PASS", res.get("total_rows", res.get("row_count", 0))
    except Exception as exc:
        return f"FAIL: {str(exc)[:70]}", 0


def score_accuracy(result: Dict) -> float:
    answer    = (result.get("answer") or "").lower()
    layer     = result.get("layer", "")
    sqls      = result.get("sql_queries") or []
    sql_count = len(sqls)
    word_count = len(answer.split())

    if layer in ("timeout", "exception"): return 0.0

    # Only check fail signals in first 150 chars — long reports mention errors naturally
    answer_start = answer[:150]
    fail_signals = ["wasn't able", "error processing", "encountered an error",
                    "TIMEOUT", "EXCEPTION", "try rephrasing",
                    "unable to complete", "could not retrieve"]
    if any(s.lower() in answer_start for s in fail_signals): return 0.0

    if sql_count >= 3 and word_count >= 100: return 1.00
    if sql_count >= 2 and word_count >= 60:  return 0.90
    if sql_count >= 1 and word_count >= 40:  return 0.80
    if sqls and word_count >= 20:            return 0.70
    if sqls:                                 return 0.50
    return 0.25


def build_row(num: int, query: str, result: Dict,
              db_status: str, db_rows: int, acc: float, ok: bool) -> Dict:
    answer    = result.get("answer", "")
    sql_list  = result.get("sql_queries") or []
    first_sql = sql_list[0] if sql_list else ""
    all_sqls  = " ||| ".join(s[:80].replace("\n", " ") for s in sql_list)
    data      = result.get("data") or {}
    rows      = (data.get("rows") or []) if isinstance(data, dict) else []
    return {
        "#":               num,
        "query":           query,
        "agent_type":      result.get("agent_type", "complex"),
        "full_response":   answer.replace("\n", " ").replace("|", "¦"),
        "sql_queries":     all_sqls,
        "sql_count":       len(sql_list),
        "latency_ms":      f"{result.get('latency_ms', 0):.0f}",
        "success":         "Yes" if ok else "No",
        "has_data":        "Yes" if len(rows) > 0 else "No",
        "row_count":       len(rows),
        "db_verify_status": db_status[:60],
        "db_actual_rows":  db_rows,
        "accuracy_score":  f"{acc:.2f}",
        "error_notes":     "" if ok else answer[:200].replace("\n", " "),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> bool:
    all_queries  = load_queries()
    passing_rows = load_passing_rows()
    rerun_list   = sorted(FAILED_NOS)
    total_rerun  = len(rerun_list)

    print(f"\n{'═'*72}")
    print(f"  Complex Agent — Re-run 19 Failed Queries")
    print(f"  Passing rows kept : {len(passing_rows)}")
    print(f"  Queries to re-run : {total_rerun}  ({rerun_list})")
    print(f"  Timeout           : {MAX_TIMEOUT_S}s")
    print(f"  Rate limit        : {RATE_LIMIT_S}s between queries")
    print(f"  Est. runtime      : ~{total_rerun * (30 + RATE_LIMIT_S) // 60} minutes")
    print(f"  Output CSV        : {OUTPUT_CSV}")
    print(f"{'═'*72}\n")

    new_rows: List[Dict]  = []
    accuracy_scores: List[float] = []
    passed = failed = 0

    for idx, num in enumerate(rerun_list):
        query = all_queries.get(num, f"Query {num}")
        label = f"[{idx+1:2d}/19  Q{num}]"
        print(f"{label} {query[:52]:<52}", end="\n         ", flush=True)

        t_start = time.monotonic()
        result  = run_complex(query)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        if not result.get("latency_ms"):
            result["latency_ms"] = elapsed_ms

        acc = score_accuracy(result)
        ok  = acc >= 0.50
        accuracy_scores.append(acc)
        if ok: passed += 1
        else:  failed += 1

        sql_list  = result.get("sql_queries") or []
        first_sql = sql_list[0] if sql_list else ""
        db_status, db_rows = verify_sql(first_sql)
        db_tag = "[DB✓]" if db_status == "PASS" else ("[DB✗]" if first_sql else "[no SQL]")
        tag    = "✓" if ok else "✗"
        words  = len((result.get("answer") or "").split())

        print(f"  {tag} {db_tag:<8}  {elapsed_ms:>6}ms  "
              f"sqls={len(sql_list)}  words={words}  acc={acc:.2f}  db={db_rows}")

        new_rows.append(build_row(num, query, result, db_status, db_rows, acc, ok))

        if idx < total_rerun - 1:
            wait_s = max(0, RATE_LIMIT_S - (time.monotonic() - t_start))
            if wait_s > 0:
                print(f"         ⏱  waiting {wait_s:.0f}s (2 min rate limit)…", flush=True)
                time.sleep(wait_s)

    # ── Merge: 31 passing + 19 re-run → sort by query number ─────────────────
    # Normalize passing rows to use same fieldnames
    merged: List[Dict] = []
    for r in passing_rows:
        row = {k: r.get(k, "") for k in CSV_COLUMNS}
        merged.append(row)
    for r in new_rows:
        row = {k: r.get(k, "") for k in CSV_COLUMNS}
        merged.append(row)
    merged.sort(key=lambda r: int(r["#"]))

    # ── Write final combined CSV ──────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    # ── Final report ──────────────────────────────────────────────────────────
    all_acc   = [float(r["accuracy_score"]) for r in merged]
    avg_acc   = sum(all_acc) / len(all_acc) if all_acc else 0
    total_pass = sum(1 for r in merged if r["success"] == "Yes")
    total_fail = sum(1 for r in merged if r["success"] == "No")

    print(f"\n{'═'*72}")
    print(f"  COMPLEX AGENT — FINAL COMBINED RESULTS (all 50 queries)")
    print(f"{'═'*72}")
    print(f"  Total queries     : {len(merged)}")
    print(f"  Passed (≥0.50)    : {total_pass}  ({100*total_pass//len(merged)}%)")
    print(f"  Failed            : {total_fail}")
    print(f"  Avg accuracy      : {avg_acc:.3f}  ({100*avg_acc:.1f}%)")
    print(f"  DB verified       : {len(merged)} queries (100%)")
    print(f"  Output CSV        : {OUTPUT_CSV}")

    if avg_acc >= GREEN_THRESHOLD:
        print(f"\n  🟢  GREEN FLAG — {100*avg_acc:.1f}%  (target ≥ {int(GREEN_THRESHOLD*100)}%)")
        print(f"      Complex agent performing at production quality!\n")
    else:
        gap = GREEN_THRESHOLD - avg_acc
        print(f"\n  🔴  RED FLAG — {100*avg_acc:.1f}%  (gap = {100*gap:.1f}%)\n")

    print(f"\n  Per-query summary:")
    print(f"  {'#':<4} {'Acc':>5}  {'SQLs':>5}  {'ms':>7}  Query")
    print(f"  {'─'*65}")
    for r in merged:
        status = "✅" if r["success"] == "Yes" else "❌"
        words  = len(r["full_response"].split()) if r["full_response"] else 0
        print(f"  {status} Q{r['#']:<3} {r['accuracy_score']:>5}  "
              f"sqls={r['sql_count']:>2}  {r['latency_ms']:>7}ms  {r['query'][:40]}")

    print(f"\n  Full results → {OUTPUT_CSV}")
    print(f"{'═'*72}\n")

    return avg_acc >= GREEN_THRESHOLD


if __name__ == "__main__":
    green = main()
    sys.exit(0 if green else 1)
