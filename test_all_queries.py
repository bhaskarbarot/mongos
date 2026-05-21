"""
test_all_queries.py — Automated CRM AI Assistant Query Tester
=============================================================
Reads all_queries.txt, tests each query against the running API,
verifies the SQL result against live PostgreSQL, and APPENDS results
to test_results.csv — safe to stop and resume across multiple days.

Columns in CSV:
  no | section | type | table | query | agent_response | sql_used |
  response_ms | db_result | accuracy_score | match_ratio | issue

RESUME BEHAVIOR:
  - Already-tested queries (matched by query text in CSV) are SKIPPED.
  - New results are APPENDED — never overwrites previous runs.
  - Run with --batch N each day to test N more queries.

Usage:
  python3 test_all_queries.py               # resume + test ALL remaining
  python3 test_all_queries.py --batch 20    # test next 20 untested queries
  python3 test_all_queries.py --batch 10 --section A   # next 10 from Section A
  python3 test_all_queries.py --status      # just show progress, test nothing
  python3 test_all_queries.py --summary     # show summary of completed results
  python3 test_all_queries.py --reset       # clear CSV and start fresh
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_URL         = "http://localhost:8000/chat"
QUERIES_FILE    = os.path.join(os.path.dirname(__file__), "all_queries.txt")
OUTPUT_CSV      = os.path.join(os.path.dirname(__file__), "test_results.csv")
REQUEST_TIMEOUT = 120          # seconds per query
DELAY_BETWEEN   = 1.5          # seconds between queries (rate limit)
GREEN_THRESHOLD = 0.90         # 90% of queries must score >= 7/10

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_DB   = os.getenv("POSTGRES_DB", "mongos_sync")

# Queries containing these keywords are "no-data expected" (Section A)
NO_DATA_YEARS   = {"2027","2028","2029","2030","2031","2032","2033","2015",
                   "2016","2017","2018","2019","2020","2021","2022"}
NO_DATA_PHRASES = ["september 2027","december 2028","march 2030","january 2027",
                   "june 2029","october 2026","august 2027","year 2030",
                   "february 2028","2029","july 2027","november 2028","q3 2027",
                   "january 2020","2019","2015","2018","2017","2022","2021",
                   "march 2018","2016"]

# ── DB connection ─────────────────────────────────────────────────────────────
def _get_db():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASS, dbname=PG_DB, connect_timeout=10
    )


def run_sql_on_db(sql: str) -> Tuple[List, List, Optional[str]]:
    """Execute SQL on PostgreSQL. Returns (rows, columns, error)."""
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return [], [], "Not a SELECT statement"
    try:
        conn = _get_db()
        cur  = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows, cols, None
    except Exception as e:
        return [], [], str(e)


# ── Query parser ──────────────────────────────────────────────────────────────
def parse_queries(filepath: str) -> List[Dict[str, str]]:
    """
    Parse all_queries.txt and extract natural language queries with metadata.
    Returns list of dicts: {query, type, section, table}
    """
    queries: List[Dict[str, str]] = []
    current_table   = "general"
    current_section = "CORE"
    current_type    = "S"

    tag_re    = re.compile(r"^\[(S|M|C)\]\s+(.+)$")
    table_re  = re.compile(r"TABLE\s+\d+:\s+(\w+)", re.IGNORECASE)
    sect_re   = re.compile(r"SECTION\s+([A-C]):", re.IGNORECASE)
    pair_re   = re.compile(r"Query\s+[A-D]:\s+(.+)$", re.IGNORECASE)
    inline_re = re.compile(
        r"^(give me|show me|how many|what is|what are|which|list all|"
        r"who |compare |identify |build |find |show all|"
        r"revenue of|show deals|give me revenue|show invoices|"
        r"show contacts|total revenue|which reps|which company|"
        r"how much|show all|show top|give me last|give me total|"
        r"give me activity|net payable|which vendor)",
        re.IGNORECASE,
    )

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("-"):
                continue

            # Section header
            sm = sect_re.search(line)
            if sm:
                current_section = f"SECTION_{sm.group(1)}"
                continue

            # Table header
            tm = table_re.search(line)
            if tm:
                current_table = tm.group(1).lower()
                continue

            # [S]/[M]/[C] tagged queries
            mm = tag_re.match(line)
            if mm:
                qtype, qtext = mm.group(1), mm.group(2).strip()
                # Strip trailing comments after [  — keep only query text
                qtext = re.split(r"\s{2,}\[", qtext)[0].strip()
                queries.append({
                    "query":   qtext,
                    "type":    qtype,
                    "section": current_section,
                    "table":   current_table,
                })
                continue

            # Pair queries (Query A: ...)
            pm = pair_re.match(line)
            if pm:
                qtext = pm.group(1).strip()
                qtext = re.split(r"\s{2,}\[", qtext)[0].strip()
                queries.append({
                    "query":   qtext,
                    "type":    "M",
                    "section": "SECTION_B",
                    "table":   current_table,
                })
                continue

            # Inline queries in Section A / C (no [S/M/C] prefix)
            if current_section in ("SECTION_A", "SECTION_C"):
                if inline_re.match(line):
                    qtext = re.split(r"\s{2,}#|\s+\[", line)[0].strip()
                    if len(qtext) > 10 and not qtext.startswith("✓") and not qtext.startswith("✗"):
                        queries.append({
                            "query":   qtext,
                            "type":    "S",
                            "section": current_section,
                            "table":   current_table,
                        })

    # Drop legend/header lines (start with = or are very short)
    queries = [q for q in queries if not q["query"].startswith("=") and len(q["query"]) > 8]

    # Deduplicate while preserving order
    seen: set = set()
    unique = []
    for q in queries:
        key = q["query"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ── API caller ────────────────────────────────────────────────────────────────
def call_api(query: str) -> Dict[str, Any]:
    """Call the CRM API and return structured result."""
    start = time.time()
    try:
        resp = requests.post(
            API_URL,
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        if resp.status_code != 200:
            return {
                "answer": f"HTTP {resp.status_code}",
                "sql_used": "",
                "elapsed_ms": elapsed_ms,
                "error": f"HTTP error {resp.status_code}",
            }
        data = resp.json()
        return {
            "answer":     data.get("answer", ""),
            "sql_used":   data.get("query_used", ""),
            "elapsed_ms": elapsed_ms,
            "error":      None,
            "agent_type": data.get("agent_type", ""),
        }
    except requests.exceptions.Timeout:
        return {"answer": "TIMEOUT", "sql_used": "", "elapsed_ms": REQUEST_TIMEOUT*1000, "error": "Request timed out"}
    except Exception as e:
        return {"answer": "ERROR", "sql_used": "", "elapsed_ms": 0, "error": str(e)}


# ── Number extractor ──────────────────────────────────────────────────────────
def extract_numbers(text: str) -> List[float]:
    """Pull all numeric values from a text string."""
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    return [float(m) for m in re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)]


def is_no_data_response(answer: str) -> bool:
    """Returns True if the agent correctly returned a no-data response."""
    low = answer.lower()
    return any(p in low for p in [
        "no records", "no data", "not available", "0 records",
        "weren't able", "wasn't able", "couldn't find", "no result",
        "none found", "no matching", "found 0", "no invoices",
        "no deals", "no contacts", "no revenue", "no information",
        "unable to find", "not found", "no entries",
    ])


def is_no_data_query(query: str) -> bool:
    """Returns True if this query is expected to return no data."""
    ql = query.lower()
    return any(p in ql for p in NO_DATA_PHRASES)


# ── Accuracy scorer ───────────────────────────────────────────────────────────
def score_response(
    query: str,
    answer: str,
    sql: str,
    db_rows: List,
    db_cols: List,
    db_error: Optional[str],
    section: str,
) -> Tuple[int, str, str]:
    """
    Score the agent response 0-10 and return (score, db_summary, issue).

    Scoring logic:
      10 — SQL ran, response matches DB truth exactly
       9 — SQL ran, minor formatting diff but numbers match
       8 — SQL ran, answer roughly correct (within 5%)
       7 — SQL ran successfully, hard to auto-verify but no obvious error
       5 — SQL ran but response numbers differ from DB
       3 — SQL errored or agent said "no data" but DB has data
       1 — No SQL generated, or hard error
       0 — Future date query but agent returned fake numbers (hallucination)
    """
    issues = []

    # No SQL generated
    if not sql:
        if "timeout" in answer.lower():
            return 1, "N/A", "Request timed out"
        return 1, "N/A", "No SQL generated by agent"

    # API error
    if answer in ("TIMEOUT", "ERROR", "HTTP 422"):
        return 1, "N/A", answer

    # ── Section A: out-of-range / future date tests ───────────────────────────
    if section == "SECTION_A" or is_no_data_query(query):
        if is_no_data_response(answer):
            return 10, "0 rows (expected)", "None — correctly returned no data"
        # Agent returned numbers for a future/missing period
        nums = extract_numbers(answer)
        meaningful = [n for n in nums if n > 0]
        if meaningful:
            return 0, "N/A", f"HALLUCINATION: returned {meaningful} for out-of-range period"
        # Returned 0 or empty without explicit "no data" phrasing
        return 7, "0 rows", "Returned result but phrasing unclear — check manually"

    # ── DB execution error ────────────────────────────────────────────────────
    if db_error:
        issues.append(f"SQL error: {db_error[:100]}")
        return 3, f"SQL ERROR: {db_error[:80]}", "; ".join(issues)

    # ── DB returned 0 rows ────────────────────────────────────────────────────
    db_row_count = len(db_rows)
    if db_row_count == 0:
        if is_no_data_response(answer):
            return 10, "0 rows", "None — agent correctly said no data"
        # Agent gave numbers but DB has nothing
        nums = [n for n in extract_numbers(answer) if n > 0]
        if nums:
            return 2, "0 rows", f"Agent returned {nums} but DB has 0 rows — possible hallucination"
        return 8, "0 rows", "DB has 0 rows; agent response looks consistent"

    # ── DB has data — build summary ───────────────────────────────────────────
    db_summary = f"{db_row_count} rows"
    if db_rows and db_cols:
        # Show first row values
        first = dict(zip(db_cols, db_rows[0]))
        db_summary += f" | first row: {first}"

    # ── Single-value result (COUNT / SUM) ────────────────────────────────────
    if db_row_count == 1 and len(db_cols) == 1:
        db_val = db_rows[0][0]
        try:
            db_num = float(str(db_val).replace(",", ""))
            agent_nums = extract_numbers(answer)
            if not agent_nums:
                issues.append("Agent gave no numeric value")
                return 5, f"DB={db_num}", "Agent numeric value missing"
            # Check if any agent number is close to DB value
            for n in agent_nums:
                if db_num == 0 and n == 0:
                    return 10, f"DB=0", "None"
                if db_num > 0:
                    ratio = abs(n - db_num) / db_num
                    if ratio < 0.001:
                        return 10, f"DB={db_num}", "None — exact match"
                    if ratio < 0.05:
                        return 9, f"DB={db_num}", f"Minor diff: agent={n} db={db_num}"
                    if ratio < 0.20:
                        return 7, f"DB={db_num}", f"Approx match: agent={n} db={db_num} (diff {ratio*100:.1f}%)"
            # None matched
            best = min(agent_nums, key=lambda n: abs(n - db_num))
            return 4, f"DB={db_num}", f"MISMATCH: agent={best} db={db_num}"
        except (ValueError, TypeError):
            pass

    # ── Multi-row result ─────────────────────────────────────────────────────
    # Check row count mentioned in response
    agent_nums = extract_numbers(answer)
    if db_row_count > 0 and agent_nums:
        # See if agent mentioned the row count
        for n in agent_nums:
            if int(n) == db_row_count:
                return 10, db_summary, "None — row count matches"
        # Check if agent count is close
        close = [n for n in agent_nums if abs(int(n) - db_row_count) <= 2]
        if close:
            return 9, db_summary, f"Row count near-match: agent≈{close[0]} db={db_row_count}"

    # ── General: SQL ran, DB has data, hard to auto-verify ───────────────────
    # At minimum SQL executed without error
    if "no records" in answer.lower() and db_row_count > 0:
        issues.append(f"Agent said no records but DB returned {db_row_count} rows")
        return 3, db_summary, "; ".join(issues)

    return 7, db_summary, "SQL ran OK — verify response manually"


# ── Progress printer ──────────────────────────────────────────────────────────
def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"


# ── Main test runner ──────────────────────────────────────────────────────────


# ── Already-done tracker ─────────────────────────────────────────────────────
def get_done_queries(csv_path: str) -> set:
    """Return set of query texts already in the CSV (case-insensitive)."""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row.get("query", "").strip().lower())
    except Exception:
        pass
    return done


def show_status(csv_path: str, all_queries: List[Dict]) -> None:
    """Print progress status without running any tests."""
    done  = get_done_queries(csv_path)
    total = len(all_queries)
    done_count = sum(1 for q in all_queries if q["query"].lower().strip() in done)
    remaining  = total - done_count

    print(f"\n{'='*60}")
    print(f"  PROGRESS STATUS")
    print(f"{'='*60}")
    print(f"  Total queries   : {total}")
    print(f"  Completed       : {done_count}  ({done_count/total*100:.1f}%)")
    print(f"  Remaining       : {remaining}  ({remaining/total*100:.1f}%)")
    bar_done = int(40 * done_count / total)
    print(f"  [{('█'*bar_done).ljust(40,'░')}]")

    if done_count > 0 and os.path.exists(csv_path):
        scores = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    scores.append(int(row["accuracy_score"]))
                except Exception:
                    pass
        if scores:
            passed = sum(1 for s in scores if s >= 7)
            print(f"\n  So far: avg={sum(scores)/len(scores):.2f}/10  "
                  f"passed={passed}/{len(scores)} ({passed/len(scores)*100:.1f}%)")
    print(f"{'='*60}\n")


def show_summary(csv_path: str) -> None:
    """Print full summary of completed results from CSV."""
    if not os.path.exists(csv_path):
        print("  No results file yet. Run some tests first.")
        return
    rows   = []
    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            try:
                scores.append(int(row["accuracy_score"]))
            except Exception:
                pass
    if not scores:
        print("  No scored results found.")
        return

    total   = len(scores)
    avg     = sum(scores) / total
    passed  = sum(1 for s in scores if s >= 7)
    perfect = sum(1 for s in scores if s == 10)
    failed  = sum(1 for s in scores if s < 5)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY  ({total} queries tested)")
    print(f"{'='*60}")
    print(f"  Perfect (10/10) : {perfect}  ({perfect/total*100:.1f}%)")
    print(f"  Passed  (≥7/10) : {passed}  ({passed/total*100:.1f}%)")
    print(f"  Failed  (<5/10) : {failed}  ({failed/total*100:.1f}%)")
    print(f"  Average score   : {avg:.2f}/10")

    # By section
    from collections import defaultdict
    by_section: dict = defaultdict(list)
    for r in rows:
        try:
            by_section[r["section"]].append(int(r["accuracy_score"]))
        except Exception:
            pass
    print(f"\n  By section:")
    for sec, scs in sorted(by_section.items()):
        p = sum(1 for s in scs if s >= 7)
        print(f"    {sec:<15} {p}/{len(scs)} passed  avg={sum(scs)/len(scs):.1f}")

    # Issues list
    issues = [(r["query"][:55], int(r["accuracy_score"]), r["issue"])
              for r in rows if r.get("issue","None") not in ("None","")
              and int(r.get("accuracy_score","10")) < 7]
    if issues:
        print(f"\n  ⚠  Issues ({len(issues)}):")
        for q, s, iss in sorted(issues, key=lambda x: x[1])[:20]:
            print(f"    [{s}/10] {q}")
            print(f"           → {iss[:75]}")

    # Green/Red flag
    rate = passed / total
    print(f"\n{'='*60}")
    if rate >= GREEN_THRESHOLD:
        print(f"  🟢 GREEN FLAG — {rate*100:.1f}% passed (≥{GREEN_THRESHOLD*100:.0f}%)")
    else:
        gap = int((GREEN_THRESHOLD - rate) * total)
        print(f"  🔴 RED FLAG — {rate*100:.1f}% passed, need {gap} more to go green")
    print(f"{'='*60}\n")


# ── Main test runner ─────────────────────────────────────────────── UPDATED ──
def run_tests(
    batch:          Optional[int]  = None,
    section_filter: Optional[str]  = None,
    skip_empty:     bool           = False,
) -> None:
    print("\n" + "="*60)
    print("  CRM AI Assistant — Query Test Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_queries = parse_queries(QUERIES_FILE)
    print(f"  Total in file   : {len(all_queries)} queries")

    if section_filter:
        all_queries = [q for q in all_queries
                       if section_filter.upper() in q["section"].upper()]
        print(f"  Section filter  : {section_filter} → {len(all_queries)} queries")

    if skip_empty:
        all_queries = [q for q in all_queries if not is_no_data_query(q["query"])]

    # ── Skip already-done queries ─────────────────────────────────────────────
    done_set   = get_done_queries(OUTPUT_CSV)
    remaining  = [q for q in all_queries
                  if q["query"].lower().strip() not in done_set]
    done_count = len(all_queries) - len(remaining)

    print(f"  Already done    : {done_count}")
    print(f"  Remaining       : {len(remaining)}")

    if not remaining:
        print("\n  ✅ All queries already tested! Run --summary to see results.")
        show_summary(OUTPUT_CSV)
        return

    # Apply batch limit
    if batch:
        to_test = remaining[:batch]
        print(f"  This batch      : {len(to_test)} (--batch {batch})")
        after   = len(remaining) - len(to_test)
        print(f"  Still after run : {after} remaining")
    else:
        to_test = remaining
        print(f"  Running ALL remaining {len(to_test)} queries")

    print(f"\n  Appending to → {OUTPUT_CSV}\n")

    # ── Check API ─────────────────────────────────────────────────────────────
    try:
        h = requests.get("http://localhost:8000/health", timeout=5)
        if h.status_code != 200:
            print("  ❌ API not responding. Start server first.")
            sys.exit(1)
        print("  ✓ API is up")
    except Exception:
        print("  ❌ Cannot reach API at localhost:8000")
        sys.exit(1)

    try:
        conn = _get_db(); conn.close()
        print("  ✓ PostgreSQL connected\n")
    except Exception as e:
        print(f"  ❌ PostgreSQL error: {e}"); sys.exit(1)

    # ── Determine starting row number ─────────────────────────────────────────
    start_no = done_count + 1
    csv_exists = os.path.exists(OUTPUT_CSV)

    CSV_HEADERS = [
        "no","section","type","table","query",
        "agent_response","sql_used",
        "response_ms","db_result",
        "accuracy_score","match_ratio","issue",
    ]

    scores_this_run: List[int] = []

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        # Write header only if file is new
        if not csv_exists or os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()

        for i, qinfo in enumerate(to_test, 1):
            query   = qinfo["query"]
            qtype   = qinfo["type"]
            section = qinfo["section"]
            table   = qinfo["table"]
            row_no  = start_no + i - 1

            print(f"\r  {_bar(i, len(to_test))} | "
                  f"avg: {sum(scores_this_run)/len(scores_this_run):.1f}" if scores_this_run else
                  f"\r  {_bar(i, len(to_test))}", end="", flush=True)

            api_result = call_api(query)
            answer     = api_result.get("answer", "") or ""
            sql_used   = api_result.get("sql_used", "") or ""
            elapsed_ms = api_result.get("elapsed_ms", 0)
            api_error  = api_result.get("error")

            db_rows, db_cols, db_error = [], [], None
            if sql_used and not api_error:
                db_rows, db_cols, db_error = run_sql_on_db(sql_used)

            score, db_summary, issue = score_response(
                query, answer, sql_used,
                db_rows, db_cols, db_error, section,
            )
            scores_this_run.append(score)

            writer.writerow({
                "no":             row_no,
                "section":        section,
                "type":           qtype,
                "table":          table,
                "query":          query,
                "agent_response": answer.replace("\n", " ").strip()[:1000],
                "sql_used":       sql_used.replace("\n", " ").strip(),
                "response_ms":    elapsed_ms,
                "db_result":      db_summary,
                "accuracy_score": score,
                "match_ratio":    f"{score}/10",
                "issue":          issue,
            })
            f.flush()
            time.sleep(DELAY_BETWEEN)

    # ── This-run summary ──────────────────────────────────────────────────────
    total_r  = len(scores_this_run)
    avg_r    = sum(scores_this_run) / total_r if total_r else 0
    passed_r = sum(1 for s in scores_this_run if s >= 7)
    left     = len(remaining) - total_r

    print(f"\n\n{'='*60}")
    print(f"  THIS RUN: {total_r} queries tested")
    print(f"  Passed (≥7): {passed_r}/{total_r}  avg={avg_r:.2f}/10")
    if left > 0:
        print(f"\n  ⏳ {left} queries still remaining.")
        print(f"     Run again with --batch N to continue.")
    else:
        print(f"\n  ✅ All queries tested! Showing full summary:")
        show_summary(OUTPUT_CSV)
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRM AI Assistant Query Tester — resumable batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_all_queries.py --status          # check progress
  python3 test_all_queries.py --batch 15        # test next 15 queries
  python3 test_all_queries.py --batch 10 --section A  # next 10 out-of-range tests
  python3 test_all_queries.py --summary         # full report of done queries
  python3 test_all_queries.py --reset           # clear CSV and start over
        """
    )
    parser.add_argument("--batch",      type=int,  default=None,
                        help="Test next N untested queries (default: all remaining)")
    parser.add_argument("--section",    type=str,  default=None,
                        help="Filter by section: A, B, C, or CORE")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Skip out-of-range/no-data queries")
    parser.add_argument("--status",     action="store_true",
                        help="Show progress status only — no testing")
    parser.add_argument("--summary",    action="store_true",
                        help="Show full summary of completed results")
    parser.add_argument("--reset",      action="store_true",
                        help="Delete CSV and start fresh")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)
            print(f"  ✓ Cleared {OUTPUT_CSV} — ready to start fresh.")
        else:
            print("  No CSV file to clear.")
        sys.exit(0)

    if args.status:
        all_q = parse_queries(QUERIES_FILE)
        show_status(OUTPUT_CSV, all_q)
        sys.exit(0)

    if args.summary:
        show_summary(OUTPUT_CSV)
        sys.exit(0)

    run_tests(
        batch          = args.batch,
        section_filter = args.section,
        skip_empty     = args.skip_empty,
    )
