"""executor.py — Parallel sub-query execution engine.

Executes each sub-query concurrently through the full pipeline:
  fast_path → text2sql → stub fallback

Features:
  • ThreadPoolExecutor with configurable concurrency
  • Per-sub-query independent timeout (not total pool timeout)
  • Results always returned in original input order
  • Structured result objects with metadata for synthesizer
  • Graceful degradation — never raises, always returns a result per sub-query

Public API:
    run_parallel(sub_queries, agent) -> List[SubQueryResult]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline import fast_path, text2sql

LOGGER = logging.getLogger("sql_chatbot")

# ── Concurrency config ─────────────────────────────────────────────────────────
_MAX_WORKERS          = 6     # max concurrent SQL threads
_PER_QUERY_TIMEOUT_S  = 90    # seconds allowed per individual sub-query
_POOL_DRAIN_TIMEOUT_S = 300   # hard ceiling for the whole pool to finish (all sub-queries)


@dataclass
class SubQueryResult:
    """Structured result for a single sub-query execution."""
    index:       int                  # position in the original sub_queries list
    sub_query:   str                  # the sub-query text
    intent:      str                  # intent label from decomposer
    answer:      str                  # human-readable answer string
    tables_used: List[str]            = field(default_factory=list)
    sql_queries: List[str]            = field(default_factory=list)
    confidence:  float                = 0.0
    elapsed_ms:  int                  = 0
    source:      str                  = "unknown"   # "fast_path" | "text2sql" | "stub"
    error:       Optional[str]        = None
    raw:         Dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for synthesizer consumption."""
        return {
            "sub_query":   self.sub_query,
            "intent":      self.intent,
            "answer":      self.answer,
            "tables_used": self.tables_used,
            "sql_queries": self.sql_queries,
            "confidence":  self.confidence,
            "elapsed_ms":  self.elapsed_ms,
            "source":      self.source,
            "error":       self.error,
        }


def _make_stub(
    sub_query: str,
    intent: str,
    index: int,
    elapsed_ms: int = 0,
    error: Optional[str] = None,
) -> SubQueryResult:
    """Create a no-data stub result when all pipeline stages fail."""
    return SubQueryResult(
        index       = index,
        sub_query   = sub_query,
        intent      = intent,
        answer      = "Data unavailable for this part of the query.",
        tables_used = [],
        sql_queries = [],
        confidence  = 0.0,
        elapsed_ms  = elapsed_ms,
        source      = "stub",
        error       = error,
    )


def _build_result(
    data: Dict[str, Any],
    sub_query: str,
    intent: str,
    index: int,
    elapsed_ms: int,
    source: str,
) -> SubQueryResult:
    """Build a SubQueryResult from a pipeline handler's result dict."""
    return SubQueryResult(
        index       = index,
        sub_query   = sub_query,
        intent      = intent,
        answer      = data.get("answer", ""),
        tables_used = data.get("tables_used", []),
        sql_queries = data.get("sql_queries", []),
        confidence  = float(data.get("confidence", 0.0)),
        elapsed_ms  = elapsed_ms,
        source      = source,
        error       = None,
        raw         = data,
    )


def _execute_one(
    index:     int,
    sub_query: str,
    intent:    str,
    agent,
) -> SubQueryResult:
    """
    Execute a single sub-query through the full pipeline.

    Pipeline order:
      1. fast_path  — pure SQL, no LLM, <300ms
      2. text2sql   — Ollama model, ~5-20s
      3. stub       — returns "unavailable" — never raises

    Args:
        index:     Position index in the original sub_queries list
        sub_query: Natural language sub-query text
        intent:    Intent label (count/list/sum/etc.)
        agent:     LangChain AgentExecutor or DB agent (thread-safe for reads)

    Returns:
        SubQueryResult — always, never raises
    """
    t_start = time.monotonic()

    # ── Stage 1: Fast path ────────────────────────────────────────────────────
    try:
        result = fast_path.run(sub_query, agent)
        if result is not None:
            elapsed = int((time.monotonic() - t_start) * 1000)
            LOGGER.debug(
                "[%d] fast_path HIT (%dms): %.60s", index, elapsed, sub_query
            )
            return _build_result(result, sub_query, intent, index, elapsed, "fast_path")
    except Exception as exc:
        LOGGER.debug("[%d] fast_path error: %s | query: %.60s", index, exc, sub_query)

    # ── Stage 2: Text2SQL ─────────────────────────────────────────────────────
    try:
        result = text2sql.run(sub_query, agent)
        if result is not None:
            elapsed = int((time.monotonic() - t_start) * 1000)
            LOGGER.debug(
                "[%d] text2sql HIT (%dms): %.60s", index, elapsed, sub_query
            )
            return _build_result(result, sub_query, intent, index, elapsed, "text2sql")
    except Exception as exc:
        LOGGER.debug("[%d] text2sql error: %s | query: %.60s", index, exc, sub_query)

    # ── Stage 3: Stub fallback ────────────────────────────────────────────────
    elapsed = int((time.monotonic() - t_start) * 1000)
    LOGGER.warning(
        "[%d] all stages failed (%dms) — returning stub: %.60s",
        index, elapsed, sub_query,
    )
    return _make_stub(sub_query, intent, index, elapsed, error="all pipeline stages failed")


def run_parallel(
    sub_queries: List[Dict[str, str]],
    agent,
) -> List[SubQueryResult]:
    """
    Execute all sub-queries concurrently using a thread pool.

    Each sub-query runs independently through:
      fast_path → text2sql → stub

    Results are ALWAYS returned in the same order as the input list,
    regardless of which sub-query finishes first.

    Args:
        sub_queries: List of {"sub_query": str, "intent": str}
                     (output from decomposer.decompose())
        agent:       DB agent — must be thread-safe for concurrent reads.
                     LangChain agents with a connection pool satisfy this.

    Returns:
        List[SubQueryResult] — same length as input, same order,
        each with answer/tables/sql/confidence/timing/source metadata.
    """
    n       = len(sub_queries)
    workers = min(n, _MAX_WORKERS)

    LOGGER.info(
        "Executor: starting %d sub-queries | workers=%d | per-query timeout=%ds",
        n, workers, _PER_QUERY_TIMEOUT_S,
    )

    # Map future → index for order-preserving collection
    results_map: Dict[int, SubQueryResult] = {}
    future_to_idx: Dict[Future, int]       = {}

    t_pool_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sq_exec") as pool:
        for i, sq in enumerate(sub_queries):
            fut = pool.submit(
                _execute_one,
                i,
                sq.get("sub_query", ""),
                sq.get("intent", "general"),
                agent,
            )
            future_to_idx[fut] = i

        # Drain futures — respect per-query timeout and hard pool ceiling
        remaining_pool = max(
            _POOL_DRAIN_TIMEOUT_S,
            n * _PER_QUERY_TIMEOUT_S,
        )

        for future in as_completed(future_to_idx, timeout=remaining_pool):
            idx = future_to_idx[future]
            sq  = sub_queries[idx]
            try:
                sqr = future.result(timeout=_PER_QUERY_TIMEOUT_S)
            except TimeoutError:
                LOGGER.warning(
                    "SubQuery [%d] timed out after %ds: %.60s",
                    idx, _PER_QUERY_TIMEOUT_S, sq.get("sub_query", ""),
                )
                sqr = _make_stub(
                    sq.get("sub_query", ""), sq.get("intent", "general"),
                    idx, elapsed_ms=_PER_QUERY_TIMEOUT_S * 1000,
                    error=f"timed out after {_PER_QUERY_TIMEOUT_S}s",
                )
            except Exception as exc:
                LOGGER.warning(
                    "SubQuery [%d] future raised: %s | query: %.60s",
                    idx, exc, sq.get("sub_query", ""),
                )
                sqr = _make_stub(
                    sq.get("sub_query", ""), sq.get("intent", "general"),
                    idx, error=str(exc),
                )
            results_map[idx] = sqr

    # Handle any sub-queries that didn't complete (pool drain timeout)
    for i, sq in enumerate(sub_queries):
        if i not in results_map:
            LOGGER.error(
                "SubQuery [%d] missing from results (pool drain timeout): %.60s",
                i, sq.get("sub_query", ""),
            )
            results_map[i] = _make_stub(
                sq.get("sub_query", ""), sq.get("intent", "general"),
                i, error="pool drain timeout — sub-query never completed",
            )

    # Build ordered output
    ordered = [results_map[i] for i in range(n)]

    total_ms = int((time.monotonic() - t_pool_start) * 1000)
    sources  = [r.source for r in ordered]
    hits     = sum(1 for r in ordered if r.source != "stub")

    LOGGER.info(
        "Executor: completed %d/%d sub-queries in %dms | sources=%s",
        hits, n, total_ms, sources,
    )

    return ordered