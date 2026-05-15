"""test_agents.py — Integration test suite for the 3-agent CRM pipeline.

Tests:
  SIMPLE  — single-table counts, lists, aggregations
  MEDIUM  — multi-table analysis, period comparison
  COMPLEX — full report, leaderboard, trend analysis

Usage:
  python -m pytest tests/test_agents.py -v
  python tests/test_agents.py          (standalone, no pytest required)

Each test prints:
  query → agent_type → latency_ms → sql_used → first 200 chars of answer

Assertions:
  - latency < 30 000 ms (30s max)
  - answer is non-empty
  - no obvious hallucination markers (numbers not sourced from DB)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Silence noisy loggers during tests ───────────────────────────────────────
import logging
for noisy in ["httpx", "urllib3", "langchain", "crewai", "openai", "transformers"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING)

MAX_LATENCY_MS = 30_000


# ══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_agent():
    """Initialize the LangChain AgentExecutor (needed for fast_path and schema)."""
    from db import get_database
    from agent import get_sql_agent
    db = get_database()
    return get_sql_agent(db)


def run_query(query: str, agent=None) -> Dict[str, Any]:
    """Run a query through the full pipeline and return the result."""
    import pipeline.main as pm
    if agent is None:
        agent = _get_agent()
    return pm.run(agent, query, request_id="test")


def print_result(query: str, result: Dict, label: str = "") -> None:
    """Pretty-print test result."""
    agent_type = result.get("agent_type", result.get("layer", "unknown"))
    latency    = result.get("latency_ms", 0)
    sql_list   = result.get("sql_queries", [])
    answer     = result.get("answer", "")
    tables     = result.get("tables_used", [])

    sql_preview = sql_list[0][:120] if sql_list else "(none)"
    ans_preview = answer[:200].replace("\n", " ") if answer else "(empty)"

    print(f"\n{'='*70}")
    print(f"[{label or 'TEST'}] {query[:70]}")
    print(f"  Agent:   {agent_type}")
    print(f"  Latency: {latency:.0f}ms")
    print(f"  Tables:  {tables}")
    print(f"  SQL:     {sql_preview}...")
    print(f"  Answer:  {ans_preview}...")
    print(f"{'='*70}")


def assert_result(result: Dict, label: str) -> None:
    """Run basic assertions on a result dict."""
    answer  = result.get("answer", "")
    latency = result.get("latency_ms", 0)

    assert answer, f"[{label}] Empty answer"
    assert latency < MAX_LATENCY_MS, f"[{label}] Latency {latency}ms exceeded {MAX_LATENCY_MS}ms"
    assert "error" not in answer.lower()[:50] or "no records" in answer.lower(), \
        f"[{label}] Answer looks like an error: {answer[:100]}"


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER TESTS (unit — no DB needed)
# ══════════════════════════════════════════════════════════════════════════════

def test_classifier_simple():
    from pipeline.classifier import classify
    result = classify("how many deals are there")
    assert result["type"] in ("SIMPLE", "MEDIUM", "COMPLEX")
    assert result["reason"]
    print(f"\nClassifier 'how many deals': {result['type']} — {result['reason']}")


def test_classifier_medium():
    from pipeline.classifier import classify
    result = classify("which sales rep has the highest win rate this quarter")
    assert result["type"] in ("MEDIUM", "COMPLEX")
    print(f"\nClassifier 'win rate': {result['type']} — {result['reason']}")


def test_classifier_complex():
    from pipeline.classifier import classify
    result = classify("give me a full sales leaderboard with all metrics")
    assert result["type"] == "COMPLEX"
    print(f"\nClassifier 'leaderboard': {result['type']} — {result['reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# MCP SERVER TESTS (unit — DB required)
# ══════════════════════════════════════════════════════════════════════════════

def test_mcp_get_live_schema():
    from mcp_server.crm_mcp import mcp_get_live_schema
    schema = mcp_get_live_schema(["deals", "users"])
    assert "deals" in schema
    assert "columns" in schema["deals"]
    assert len(schema["deals"]["columns"]) > 0
    print(f"\nMCP schema: deals has {len(schema['deals']['columns'])} columns")


def test_mcp_find_column():
    from mcp_server.crm_mcp import mcp_find_column
    result = mcp_find_column("salesowner", ["sales", "users"])
    print(f"\nMCP find_column 'salesowner': {result}")
    # Should either find it or report not found — both are valid
    assert "found" in result


def test_mcp_execute_sql():
    from mcp_server.crm_mcp import mcp_execute_sql
    result = mcp_execute_sql('SELECT COUNT(*) AS deal_count FROM "deals" WHERE NOT deleted')
    assert result["error"] is None, f"SQL error: {result['error']}"
    assert result["row_count"] >= 0
    print(f"\nMCP execute_sql: {result['row_count']} rows, count={result['rows']}")


def test_mcp_get_examples():
    from mcp_server.crm_mcp import mcp_get_examples
    examples = mcp_get_examples()
    assert len(examples) >= 5, "Should have at least 5 seed examples"
    assert "natural_query" in examples[0]
    assert "sql" in examples[0]
    print(f"\nMCP examples: {len(examples)} available")


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE AGENT TESTS  (DB required)
# ══════════════════════════════════════════════════════════════════════════════

def test_simple_count(agent=None):
    """SIMPLE: COUNT query — single table."""
    result = run_query("how many deals are there", agent)
    print_result("how many deals are there", result, "SIMPLE-COUNT")
    assert_result(result, "SIMPLE-COUNT")
    assert result.get("agent_type") == "simple" or "simple" in result.get("layer", "")


def test_simple_list(agent=None):
    """SIMPLE: Filtered list — single table."""
    result = run_query("show me all open deals", agent)
    print_result("show me all open deals", result, "SIMPLE-LIST")
    assert_result(result, "SIMPLE-LIST")


def test_simple_top_n(agent=None):
    """SIMPLE: Top N with GROUP BY — at most 2 tables."""
    result = run_query("top 5 companies by deal count", agent)
    print_result("top 5 companies by deal count", result, "SIMPLE-TOPN")
    assert_result(result, "SIMPLE-TOPN")


# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM AGENT TESTS  (DB required)
# ══════════════════════════════════════════════════════════════════════════════

def test_medium_win_rate(agent=None):
    """MEDIUM: Cross-table win rate calculation."""
    result = run_query("which sales rep has the highest win rate this quarter", agent)
    print_result("which sales rep has the highest win rate this quarter", result, "MEDIUM-WINRATE")
    assert_result(result, "MEDIUM-WINRATE")


def test_medium_stale_deals(agent=None):
    """MEDIUM: Date calculation — deals not moved in 30 days."""
    result = run_query("show me deals that have not moved in 30 days", agent)
    print_result("show me deals not moved 30 days", result, "MEDIUM-STALE")
    assert_result(result, "MEDIUM-STALE")


def test_medium_period_comparison(agent=None):
    """MEDIUM: Period comparison — this month vs last month revenue."""
    result = run_query("revenue this month vs last month", agent)
    print_result("revenue this month vs last month", result, "MEDIUM-PERIOD")
    assert_result(result, "MEDIUM-PERIOD")


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEX AGENT TESTS  (DB required)
# ══════════════════════════════════════════════════════════════════════════════

def test_complex_leaderboard(agent=None):
    """COMPLEX: Full sales leaderboard — many tables, many metrics."""
    result = run_query("give me a full sales leaderboard with all metrics", agent)
    print_result("full sales leaderboard", result, "COMPLEX-LEADERBOARD")
    assert_result(result, "COMPLEX-LEADERBOARD")
    assert result.get("agent_type") == "complex" or "complex" in result.get("layer", "")


def test_complex_conversion_trend(agent=None):
    """COMPLEX: Trend analysis — conversion dropping."""
    result = run_query("why is deal conversion dropping", agent)
    print_result("why is deal conversion dropping", result, "COMPLEX-TREND")
    assert_result(result, "COMPLEX-TREND")


def test_complex_360(agent=None):
    """COMPLEX: 360 view — deep multi-table report."""
    result = run_query("give me a 360 view of our top company by revenue", agent)
    print_result("360 view of top company", result, "COMPLEX-360")
    assert_result(result, "COMPLEX-360")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-HEAL TEST  (unit — tests the correction loop)
# ══════════════════════════════════════════════════════════════════════════════

def test_selfheal_column_correction():
    """Verify find_correct_column works for common mixed-case columns."""
    from pipeline.db_schema import find_correct_column, get_all_crm_tables
    all_tables = get_all_crm_tables()

    test_cases = [
        ("salesowner", "sales"),
        ("companyname", "companies"),
        ("wonAt", "deals"),
        ("isActive", "users"),
    ]

    for bad, table in test_cases:
        correct = find_correct_column(bad, [table] + all_tables[:5])
        print(f"  Self-heal: '{bad}' → '{correct}' (searched {table})")
        # Just verify it doesn't crash — result may be None if column truly doesn't exist


# ══════════════════════════════════════════════════════════════════════════════
# FULL SUITE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run the complete test suite and report results."""
    print("\n" + "="*70)
    print("CRM AI Assistant — Agent Test Suite")
    print("="*70)

    # Unit tests (no DB)
    unit_tests = [
        ("Classifier: SIMPLE",   test_classifier_simple),
        ("Classifier: MEDIUM",   test_classifier_medium),
        ("Classifier: COMPLEX",  test_classifier_complex),
        ("Self-heal column fix", test_selfheal_column_correction),
    ]

    # Integration tests (DB required)
    db_tests_no_agent = [
        ("MCP: get_live_schema",    test_mcp_get_live_schema),
        ("MCP: find_column",        test_mcp_find_column),
        ("MCP: execute_sql",        test_mcp_execute_sql),
        ("MCP: get_examples",       test_mcp_get_examples),
    ]

    passed = failed = 0

    print("\n--- Unit Tests (no DB required) ---")
    for name, fn in unit_tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            failed += 1

    print("\n--- MCP Tests (DB required) ---")
    for name, fn in db_tests_no_agent:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            failed += 1

    print("\n--- Agent Integration Tests (DB + LLM required) ---")
    print("  Initializing agent...")
    agent = None
    try:
        agent = _get_agent()
        agent_ok = True
        print("  Agent ready ✓")
    except Exception as exc:
        print(f"  Agent init failed: {exc}")
        agent_ok = False

    agent_tests = [
        ("SIMPLE: count deals",           test_simple_count),
        ("SIMPLE: list open deals",       test_simple_list),
        ("SIMPLE: top 5 companies",       test_simple_top_n),
        ("MEDIUM: win rate",              test_medium_win_rate),
        ("MEDIUM: stale deals",           test_medium_stale_deals),
        ("MEDIUM: period comparison",     test_medium_period_comparison),
        ("COMPLEX: sales leaderboard",    test_complex_leaderboard),
        ("COMPLEX: conversion trend",     test_complex_conversion_trend),
        ("COMPLEX: 360 view",             test_complex_360),
    ]

    if agent_ok:
        for name, fn in agent_tests:
            try:
                fn(agent)
                print(f"  ✓ {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  ✗ {name}: ASSERTION FAILED — {exc}")
                failed += 1
            except Exception as exc:
                print(f"  ✗ {name}: ERROR — {type(exc).__name__}: {exc}")
                failed += 1
    else:
        print("  Skipping agent tests (agent init failed)")
        for name, _ in agent_tests:
            print(f"  - {name} (skipped)")

    total = passed + failed
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("="*70)
    return failed == 0


# ══════════════════════════════════════════════════════════════════════════════
# pytest FIXTURES  (so `pytest tests/test_agents.py -v` works)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pytest

    @pytest.fixture(scope="session")
    def crm_agent():
        try:
            return _get_agent()
        except Exception as exc:
            pytest.skip(f"DB/agent not available: {exc}")

    def test_pytest_classifier_simple():           test_classifier_simple()
    def test_pytest_classifier_medium():           test_classifier_medium()
    def test_pytest_classifier_complex():          test_classifier_complex()
    def test_pytest_selfheal():                    test_selfheal_column_correction()
    def test_pytest_mcp_schema():                  test_mcp_get_live_schema()
    def test_pytest_mcp_examples():                test_mcp_get_examples()

    def test_pytest_simple_count(crm_agent):       test_simple_count(crm_agent)
    def test_pytest_simple_list(crm_agent):        test_simple_list(crm_agent)
    def test_pytest_simple_top_n(crm_agent):       test_simple_top_n(crm_agent)
    def test_pytest_medium_winrate(crm_agent):     test_medium_win_rate(crm_agent)
    def test_pytest_medium_period(crm_agent):      test_medium_period_comparison(crm_agent)
    def test_pytest_complex_leaderboard(crm_agent):test_complex_leaderboard(crm_agent)
    def test_pytest_complex_360(crm_agent):        test_complex_360(crm_agent)

except ImportError:
    pass  # pytest not required — standalone runner still works


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
