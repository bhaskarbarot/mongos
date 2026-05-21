"""
retest_failed.py — Re-test previously failed queries after bug fixes.

Reads test_results.csv, finds all rows with accuracy_score < 7, re-runs
them against the live API in S → M → C tier order, and writes results to
retest_results.csv.

Usage:
  python3 retest_failed.py           # all 3 tiers
  python3 retest_failed.py --tier S  # Simple only
  python3 retest_failed.py --tier M  # Medium only
  python3 retest_failed.py --tier C  # Complex only
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Reuse helpers from test_all_queries
from test_all_queries import run_sql_on_db, score_response, call_api, _bar

SOURCE_CSV = os.path.join(os.path.dirname(__file__), "test_results.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "retest_results.csv")
DELAY      = 1.5   # seconds between queries

CSV_HEADERS = [
    "no", "type", "section", "query",
    "old_score", "new_score",
    "new_sql", "new_response", "response_ms",
    "db_result", "issue", "fixed",
]


def load_failed(tier_filter: Optional[str] = None) -> Dict[str, List[dict]]:
    """Load failed rows from test_results.csv grouped by type."""
    if not os.path.exists(SOURCE_CSV):
        print(f"  ERROR: {SOURCE_CSV} not found.")
        sys.exit(1)

    with open(SOURCE_CSV) as f:
        rows = list(csv.DictReader(f))

    tiers: Dict[str, List[dict]] = {"S": [], "M": [], "C": []}
    for r in rows:
        try:
            score = int(r.get("accuracy_score", "10"))
        except ValueError:
            continue
        if score < 7:
            t = r.get("type", "S").upper()
            if t in tiers:
                if tier_filter is None or t == tier_filter.upper():
                    tiers[t].append(r)
    return tiers


def load_passing() -> List[dict]:
    """Load previously passing rows (score >= 7) for regression check."""
    with open(SOURCE_CSV) as f:
        rows = list(csv.DictReader(f))
    passing = [r for r in rows if int(r.get("accuracy_score", "0")) >= 7]
    return random.sample(passing, min(10, len(passing)))


def run_tier(tier: str, rows: List[dict], writer, f_out) -> dict:
    """Run one tier of failed queries. Returns counts."""
    total   = len(rows)
    fixed   = 0
    still   = 0
    regress = []

    print(f"\n{'='*60}")
    print(f"  TIER {tier} — {total} queries")
    print(f"{'='*60}")

    for i, row in enumerate(rows, 1):
        query   = row["query"]
        old_sc  = int(row.get("accuracy_score", "0"))
        section = row.get("section", "CORE")

        print(f"\r  {_bar(i, total)} | fixed: {fixed}", end="", flush=True)

        result     = call_api(query)
        answer     = result.get("answer", "") or ""
        sql        = result.get("sql_used", "") or ""
        elapsed_ms = result.get("elapsed_ms", 0)

        db_rows, db_cols, db_error = [], [], None
        if sql:
            db_rows, db_cols, db_error = run_sql_on_db(sql)

        new_score, db_summary, issue = score_response(
            query, answer, sql, db_rows, db_cols, db_error, section
        )

        is_fixed = "YES" if new_score >= 7 and old_sc < 7 else "NO"
        if is_fixed == "YES":
            fixed += 1
        else:
            still += 1

        writer.writerow({
            "no":           row.get("no", i),
            "type":         tier,
            "section":      section,
            "query":        query,
            "old_score":    old_sc,
            "new_score":    new_score,
            "new_sql":      sql.replace("\n", " ").strip(),
            "new_response": answer.replace("\n", " ").strip()[:800],
            "response_ms":  elapsed_ms,
            "db_result":    db_summary,
            "issue":        issue,
            "fixed":        is_fixed,
        })
        f_out.flush()
        time.sleep(DELAY)

    print(f"\n\n  Tier {tier}: {fixed}/{total} fixed | {still} still failing")
    return {"fixed": fixed, "still": still, "total": total}


def regression_check(passing_sample: List[dict]) -> bool:
    """Re-test a sample of previously passing queries. Returns True if all still pass."""
    print(f"\n{'='*60}")
    print(f"  REGRESSION CHECK — {len(passing_sample)} previously passing queries")
    print(f"{'='*60}")

    regressions = []
    for row in passing_sample:
        query   = row["query"]
        old_sc  = int(row.get("accuracy_score", "10"))
        section = row.get("section", "CORE")

        result     = call_api(query)
        answer     = result.get("answer", "") or ""
        sql        = result.get("sql_used", "") or ""
        elapsed_ms = result.get("elapsed_ms", 0)

        db_rows, db_cols, db_error = [], [], None
        if sql:
            db_rows, db_cols, db_error = run_sql_on_db(sql)

        new_score, _, issue = score_response(
            query, answer, sql, db_rows, db_cols, db_error, section
        )

        if new_score < 7:
            regressions.append((query[:60], old_sc, new_score, issue[:60]))
            print(f"  ❌ REGRESSION [{old_sc}→{new_score}] {query[:55]}")
        else:
            print(f"  ✅ OK [{old_sc}→{new_score}] {query[:55]}")
        time.sleep(DELAY)

    if regressions:
        print(f"\n  ⚠  {len(regressions)} REGRESSIONS FOUND — review changes!")
        return False
    print(f"\n  ✅ No regressions — all {len(passing_sample)} previously passing queries still pass.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-test previously failed queries")
    parser.add_argument("--tier", type=str, default=None,
                        help="Run only one tier: S, M, or C")
    args = parser.parse_args()

    tier_filter = args.tier.upper() if args.tier else None
    if tier_filter and tier_filter not in ("S", "M", "C"):
        print("--tier must be S, M, or C")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Re-test Failed Queries")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source : {SOURCE_CSV}")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"{'='*60}")

    # Load failed rows
    tiers = load_failed(tier_filter)
    total_failed = sum(len(v) for v in tiers.values())
    print(f"\n  Failed queries: S={len(tiers['S'])} | M={len(tiers['M'])} | C={len(tiers['C'])}")
    print(f"  Running: {total_failed} queries\n")

    if total_failed == 0:
        print("  Nothing to retest — no failed queries found.")
        return

    # Check API
    import requests
    try:
        h = requests.get("http://localhost:8000/health", timeout=5)
        if h.status_code != 200:
            print("  ❌ API not responding. Start server first.")
            sys.exit(1)
        print("  ✓ API is up")
    except Exception:
        print("  ❌ Cannot reach API at localhost:8000")
        sys.exit(1)

    # Run tiers
    csv_exists = os.path.exists(OUTPUT_CSV)
    totals = {"S": {}, "M": {}, "C": {}}

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_HEADERS)
        if not csv_exists or os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()

        for tier in ("S", "M", "C"):
            if tiers[tier]:
                totals[tier] = run_tier(tier, tiers[tier], writer, f_out)

    # Regression check (only when running all tiers)
    regression_ok = True
    if tier_filter is None:
        passing_sample = load_passing()
        regression_ok  = regression_check(passing_sample)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for tier in ("S", "M", "C"):
        if totals[tier]:
            t = totals[tier]
            pct = t['fixed'] / t['total'] * 100 if t['total'] else 0
            print(f"  Tier {tier}: {t['fixed']}/{t['total']} fixed ({pct:.0f}%)")

    all_fixed = sum(t.get('fixed', 0) for t in totals.values())
    all_total = sum(t.get('total', 0) for t in totals.values())
    all_still = sum(t.get('still', 0) for t in totals.values())

    print(f"\n  Total fixed : {all_fixed} / {all_total}")
    print(f"  Still failing: {all_still}")
    print(f"  Regressions : {'NONE ✅' if regression_ok else 'FOUND ❌ — check retest_results.csv'}")
    print(f"\n  Full results → {OUTPUT_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
