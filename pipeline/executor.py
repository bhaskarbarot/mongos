"""executor.py — Parallel sub-query execution engine.

Executes each sub-query concurrently through the full pipeline:
  fast_path → text2sql → stub fallback

Features:
  • ThreadPoolExecutor with configurable concurrency
  • E8: concurrent.futures.wait() — never raises, always returns ordered results
  • Results always returned in original input order
  • Structured result objects with metadata for synthesizer
  • Graceful degradation — never raises, always returns a result per sub-query

Public API:
    run_parallel(sub_queries, agent, request_id="") -> List[SubQueryResult]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import threading

from pipeline import fast_path, text2sql

LOGGER = logging.getLogger("sql_chatbot")

# ── Concurrency config ─────────────────────────────────────────────────────────
_MAX_WORKERS             = 6   # max concurrent SQL threads
PER_QUERY_TIMEOUT_SECONDS = 45  # E8: wall-clock seconds for the entire parallel batch

# Semaphore: limit concurrent Groq SQL calls so parallel sub-queries don't
# exhaust the Groq rate limit simultaneously. DB execution still runs in parallel.
_GROQ_SQL_SEMAPHORE = threading.Semaphore(2)  # max 2 concurrent Groq SQL calls


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


def _make_timeout_stub(
    sub_query: str,
    intent: str,
    index: int,
) -> SubQueryResult:
    """Create a timeout stub result when a sub-query exceeded PER_QUERY_TIMEOUT_SECONDS."""
    return SubQueryResult(
        index       = index,
        sub_query   = sub_query,
        intent      = intent,
        answer      = "Sub-query timed out.",
        tables_used = [],
        sql_queries = [],
        confidence  = 0.0,
        elapsed_ms  = PER_QUERY_TIMEOUT_SECONDS * 1000,
        source      = "stub",
        error       = "timeout",
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
    index:      int,
    sub_query:  str,
    intent:     str,
    agent,
    request_id: str = "",
) -> SubQueryResult:
    """
    Execute a single sub-query through the full pipeline.

    Pipeline order:
      1. fast_path  — pure SQL, no LLM, <300ms
      2. text2sql   — Ollama model, ~5-20s
      3. stub       — returns "unavailable" — never raises

    Args:
        index:      Position index in the original sub_queries list
        sub_query:  Natural language sub-query text
        intent:     Intent label (count/list/sum/etc.)
        agent:      LangChain AgentExecutor or DB agent (thread-safe for reads)
        request_id: Correlation ID for log tracing (E5)

    Returns:
        SubQueryResult — always, never raises
    """
    t_start = time.monotonic()
    rid_sub = f"[RID:{request_id}][SUB:{index}]"  # E5: log prefix

    # ── Stage 1: Fast path ────────────────────────────────────────────────────
    try:
        result = fast_path.run(sub_query, agent)
        if result is not None:
            elapsed = int((time.monotonic() - t_start) * 1000)
            LOGGER.debug(
                "%s fast_path HIT (%dms): %.60s", rid_sub, elapsed, sub_query
            )
            return _build_result(result, sub_query, intent, index, elapsed, "fast_path")
    except Exception as exc:
        LOGGER.debug("%s fast_path error: %s | query: %.60s", rid_sub, exc, sub_query)

    # ── Stage 2: Text2SQL ─────────────────────────────────────────────────────
    try:
        result = text2sql.run(sub_query, agent)
        if result is not None:
            elapsed = int((time.monotonic() - t_start) * 1000)
            LOGGER.debug(
                "%s text2sql HIT (%dms): %.60s", rid_sub, elapsed, sub_query
            )
            return _build_result(result, sub_query, intent, index, elapsed, "text2sql")
    except Exception as exc:
        LOGGER.debug("%s text2sql error: %s | query: %.60s", rid_sub, exc, sub_query)

    # ── Stage 3: Stub fallback ────────────────────────────────────────────────
    elapsed = int((time.monotonic() - t_start) * 1000)
    LOGGER.warning(
        "%s all stages failed (%dms) — returning stub: %.60s",
        rid_sub, elapsed, sub_query,
    )
    return _make_stub(sub_query, intent, index, elapsed, error="all pipeline stages failed")


def run_parallel(
    sub_queries: List[Dict[str, str]],
    agent,
    request_id: str = "",
) -> List[SubQueryResult]:
    """
    Execute all sub-queries concurrently using a thread pool.

    E8: Uses concurrent.futures.wait() which never raises — futures that do not
    complete within PER_QUERY_TIMEOUT_SECONDS are cancelled and replaced with a
    timeout stub. The function ALWAYS returns a list of the same length as the input.

    Results are ALWAYS returned in the same order as the input list.

    Args:
        sub_queries: List of {"sub_query": str, "intent": str}
        agent:       DB agent — must be thread-safe for concurrent reads.
        request_id:  Correlation ID for log tracing (E5)

    Returns:
        List[SubQueryResult] — same length as input, same order.
    """
    n       = len(sub_queries)
    workers = min(n, _MAX_WORKERS)

    LOGGER.info(
        "[RID:%s] Executor: starting %d sub-queries | workers=%d | timeout=%ds",
        request_id, n, workers, PER_QUERY_TIMEOUT_SECONDS,
    )

    # Map future → original index for order-preserving assembly
    future_to_idx: Dict[Future, int] = {}
    results_map:   Dict[int, SubQueryResult] = {}

    t_pool_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sq_exec") as pool:
        for i, sq in enumerate(sub_queries):
            fut = pool.submit(
                _execute_one,
                i,
                sq.get("sub_query", ""),
                sq.get("intent", "general"),
                agent,
                request_id,
            )
            future_to_idx[fut] = i

        # E8: wait() never raises — returns (done, not_done)
        done, not_done = futures_wait(
            future_to_idx,
            timeout=PER_QUERY_TIMEOUT_SECONDS,
        )

        # Collect completed futures
        for future in done:
            idx = future_to_idx[future]
            sq  = sub_queries[idx]
            try:
                results_map[idx] = future.result()
            except Exception as exc:
                LOGGER.warning(
                    "[RID:%s] Sub-query %d future raised: %s | query: %.60s",
                    request_id, idx, exc, sq.get("sub_query", ""),
                )
                results_map[idx] = _make_stub(
                    sq.get("sub_query", ""), sq.get("intent", "general"),
                    idx, error=str(exc),
                )

        # E8: handle timed-out futures — cancel and create stub.
        # Note: future.cancel() only stops futures that haven't started yet.
        # Futures already running in a thread cannot be interrupted — those threads
        # will continue in the background until their underlying HTTP timeout fires
        # (Ollama has its own timeout). Their results are simply never collected.
        for future in not_done:
            idx = future_to_idx[future]
            sq  = sub_queries[idx]
            future.cancel()
            LOGGER.warning(
                "[RID:%s] Sub-query %d timed out after %ds: %.60s",
                request_id, idx, PER_QUERY_TIMEOUT_SECONDS, sq.get("sub_query", ""),
            )
            results_map[idx] = _make_timeout_stub(
                sq.get("sub_query", ""), sq.get("intent", "general"), idx,
            )

    # Build ordered output (all indices guaranteed to be in results_map)
    ordered  = [results_map[i] for i in range(n)]
    total_ms = int((time.monotonic() - t_pool_start) * 1000)
    sources  = [r.source for r in ordered]
    hits     = sum(1 for r in ordered if r.source != "stub")

    LOGGER.info(
        "[RID:%s] Executor: completed %d/%d sub-queries in %dms | sources=%s",
        request_id, hits, n, total_ms, sources,
    )

    return ordered
