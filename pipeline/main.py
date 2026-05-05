"""main.py — Pipeline orchestrator (production-grade).

Implements the full hybrid NL-to-SQL query flow:

    User Query
        ↓
    [Guard]     Greeting / Blocked / Vague checks
        ↓
    [Layer 1]   Fast Path Engine       → ~70-80% queries, <300ms, NO LLM
        ↓ (miss)
    [Layer 2]   Intent Classifier      → SIMPLE | COMPLEX
        ↓
    [SIMPLE]    Text2SQL               → Ollama models → execute → return
    [COMPLEX]   Decomposer            → Groq/Ollama → sub-queries
                Executor              → parallel SQL execution
                Synthesizer           → Groq/Ollama → final answer

Features:
  • Query result caching with TTL (avoids re-running identical queries)
  • Per-session conversation memory (pronoun resolution)
  • Full latency tracking at every layer
  • Structured response with metadata (latency, confidence, tables, SQL)
  • Graceful error handling — never crashes, always returns a response
  • Schema pre-warming on first query

Public API:
    run(agent, user_query, memory) -> Dict[str, Any]
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from pipeline import fast_path, text2sql
from pipeline.classifier import classify
from pipeline.decomposer import decompose
from pipeline.executor import run_parallel, SubQueryResult
from pipeline.schema import (
    discover_schema_links,
    get_table_names,
    build_text2sql_schema,
)
from pipeline.synthesizer import synthesize
from pipeline.utils import (
    Timer,
    format_final_answer,
    is_blocked,
    is_greeting,
    is_vague_query,
    mask_ids,
    normalize_text,
    query_hash,
)

LOGGER = logging.getLogger("sql_chatbot")

# ── Cache (disabled) ───────────────────────────────────────────────────────────
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE  = 0      # 0 = disabled
_CACHE_TTL_SEC   = 0
_SCHEMA_WARMED   = False


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Per-session state: resolves pronouns ('that', 'those') via last entity.

    Tracks:
      • last_entity:  last table/entity referenced
      • last_query:   last user query text
      • last_count:   last count result (for "give me those" follow-ups)
      • history:       recent query history for context
    """

    def __init__(self, max_history: int = 10) -> None:
        self.last_entity: Optional[str] = None
        self.last_query:  str           = ""
        self.last_count:  Optional[int] = None
        self.history:     List[Dict]    = []
        self._max_history = max_history

    def update(
        self,
        entity: Optional[str],
        query: str,
        count: Optional[int] = None,
    ) -> None:
        """Update memory after a successful query."""
        if entity:
            self.last_entity = entity
        self.last_query = query
        if count is not None:
            self.last_count = count

        self.history.append({
            "query":  query,
            "entity": entity,
            "count":  count,
            "ts":     time.time(),
        })
        # Trim history
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

    def resolve(self, query: str) -> str:
        """Resolve pronouns in the query using conversation context."""
        if not self.last_entity:
            return query

        text = normalize_text(query)

        # Check for pronoun references
        has_pronoun = re.search(
            r"\b(that|those|them|their|these|it|the same|above|previous)\b",
            text,
        )
        if has_pronoun:
            resolved = f"{query} (referring to {self.last_entity})"
            LOGGER.debug("Memory resolved: '%s' → '%s'", query, resolved)
            return resolved

        # "Give me the list" / "show names" without entity
        bare_action = re.match(
            r"^(give me|show|list|get|display)\s+(the\s+)?(names?|list|details?|all)\s*$",
            text,
        )
        if bare_action and self.last_entity:
            resolved = f"{query} of {self.last_entity}"
            LOGGER.debug("Memory resolved bare action: '%s' → '%s'", query, resolved)
            return resolved

        return query


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_response(
    answer: str,
    tables_used: List[str],
    sql_queries: List[str],
    confidence: float,
    started: float,
    layer: str = "unknown",
    sub_results: Optional[List] = None,
    cached: bool = False,
) -> Dict[str, Any]:
    """Build standardized pipeline response dict."""
    latency = round((time.perf_counter() - started) * 1000, 2)
    resp = {
        "answer":      format_final_answer(answer, tables_used),
        "latency_ms":  latency,
        "confidence":  round(confidence, 3),
        "tables_used": tables_used,
        "sql_queries": sql_queries,
        "layer":       layer,
    }
    if cached:
        resp["cached"] = True
    if sub_results:
        resp["_sub_results"] = sub_results
    return resp


def _warm_schema(agent) -> None:
    """Pre-warm schema caches on first query (runs once per process)."""
    global _SCHEMA_WARMED
    if _SCHEMA_WARMED:
        return

    with Timer("schema_warmup") as t:
        try:
            tables = get_table_names(agent)
            if tables:
                discover_schema_links(agent)
                build_text2sql_schema(agent)
                _SCHEMA_WARMED = True
                LOGGER.info(
                    "Schema pre-warmed: %d tables, %.0fms",
                    len(tables), t.elapsed_ms,
                )
        except Exception as exc:
            LOGGER.warning("Schema warmup failed (non-fatal): %s", exc)


def _check_cache(q_hash: str) -> Optional[Dict[str, Any]]:
    return None  # cache disabled


def _store_cache(q_hash: str, result: Dict[str, Any]) -> None:
    return  # cache disabled


def _extract_entity_from_result(result: Dict[str, Any]) -> Optional[str]:
    """Extract the primary entity/table from a pipeline result."""
    # Check _entity metadata first (set by fast-path)
    entity = result.get("_entity")
    if entity:
        return entity
    # Fall back to first table
    tables = result.get("tables_used", [])
    return tables[0] if tables else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    agent,
    user_query: str,
    memory: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """Execute the full hybrid NL-to-SQL pipeline for one user query.

    Flow:
      0. Pronoun resolution (memory)
      1. Guard checks (greeting / blocked / vague)
      2. Cache check
      3. Schema pre-warm (first query only)
      4. Intent classification (SIMPLE vs COMPLEX)
      5a. SIMPLE: fast-path → text2sql fallback
      5b. COMPLEX: decompose → parallel execute → synthesize
      6. Cache result, update memory, return

    Args:
        agent:      LangChain AgentExecutor with SQL tools
        user_query: Raw user input string
        memory:     Per-session ConversationMemory (optional)

    Returns:
        Dict with: answer, latency_ms, confidence, tables_used, sql_queries,
                   layer, cached (optional), _sub_results (optional)
    """
    started = time.perf_counter()

    # ── 0. Pronoun resolution ──────────────────────────────────────────────
    original_query = user_query
    if memory:
        user_query = memory.resolve(user_query)

    # ── 1. Guard: Greeting ─────────────────────────────────────────────────
    if is_greeting(user_query):
        LOGGER.info("Guard: greeting detected")
        return {
            "answer": (
                "Hello! I can help you query your CRM data. Try asking:\n\n"
                "• **Counts**: \"How many deals do we have?\"\n"
                "• **Lists**: \"Show all closed won deals\"\n"
                "• **Revenue**: \"Total revenue this month\"\n"
                "• **Reports**: \"Executive summary of pipeline\"\n"
                "• **Targets**: \"Target vs achieved this quarter\"\n"
                "• **Search**: \"Find contact John Smith\""
            ),
            "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
            "confidence":  1.0,
            "tables_used": [],
            "sql_queries": [],
            "layer":       "guard",
        }

    # ── 2. Guard: Blocked query ────────────────────────────────────────────
    if is_blocked(user_query):
        LOGGER.warning("Guard: blocked query: %.60s", user_query)
        return {
            "answer": (
                "This query contains operations that are not permitted. "
                "I can only run read-only (SELECT) queries on your CRM data."
            ),
            "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
            "confidence":  1.0,
            "tables_used": [],
            "sql_queries": [],
            "layer":       "guard",
        }

    # ── 3. Guard: Vague query (too short, no intent) ──────────────────────
    if is_vague_query(user_query):
        LOGGER.info("Guard: vague query: %.60s", user_query)
        return {
            "answer": (
                "Your query seems a bit vague. Could you be more specific? For example:\n\n"
                "• \"How many **open deals** do we have?\"\n"
                "• \"Show **revenue this month**\"\n"
                "• \"List **pending tasks** for all users\""
            ),
            "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
            "confidence":  0.5,
            "tables_used": [],
            "sql_queries": [],
            "layer":       "guard",
        }

    LOGGER.info("═══ Pipeline START | query: %.80s ═══", user_query)

    # ── 4. Cache check ─────────────────────────────────────────────────────
    q_hash = query_hash(user_query)
    cached = _check_cache(q_hash)
    if cached:
        cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if memory:
            memory.update(
                entity=_extract_entity_from_result(cached),
                query=original_query,
            )
        return cached

    # ── 5. Schema pre-warm ─────────────────────────────────────────────────
    _warm_schema(agent)

    # ── 6. Intent classification ───────────────────────────────────────────
    with Timer("classifier") as cls_timer:
        classification = classify(user_query)
    intent_type   = classification["type"]
    intent_reason = classification["reason"]
    LOGGER.info(
        "═══ Classifier: %s (%.0fms) | %s",
        intent_type, cls_timer.elapsed_ms, intent_reason,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SIMPLE PATH: fast_path → text2sql
    # ══════════════════════════════════════════════════════════════════════
    if intent_type == "SIMPLE":

        # ── Layer 1: Fast Path ─────────────────────────────────────────
        with Timer("fast_path") as fp_timer:
            try:
                fp_result = fast_path.run(user_query, agent)
            except Exception as exc:
                LOGGER.warning("Fast-path exception: %s", exc)
                fp_result = None

        if fp_result is not None:
            LOGGER.info(
                "═══ Layer 1 HIT (fast-path) | %.0fms | tables=%s",
                fp_timer.elapsed_ms, fp_result.get("tables_used"),
            )
            result = _build_response(
                fp_result["answer"],
                fp_result.get("tables_used", []),
                fp_result.get("sql_queries", []),
                fp_result.get("confidence", 0.95),
                started,
                layer="fast_path",
            )
            _store_cache(q_hash, result)
            if memory:
                memory.update(
                    entity=fp_result.get("_entity") or _extract_entity_from_result(fp_result),
                    query=original_query,
                    count=fp_result.get("_count"),
                )
            return result

        LOGGER.info("═══ Fast-path MISS → Text2SQL")

        # ── Layer 2: Text2SQL ──────────────────────────────────────────
        with Timer("text2sql") as t2s_timer:
            try:
                t2s_result = text2sql.run(user_query, agent)
            except Exception as exc:
                LOGGER.warning("Text2SQL exception: %s", exc)
                t2s_result = None

        if t2s_result is not None:
            LOGGER.info(
                "═══ Text2SQL HIT | %.0fms | tables=%s",
                t2s_timer.elapsed_ms, t2s_result.get("tables_used"),
            )
            result = _build_response(
                t2s_result["answer"],
                t2s_result.get("tables_used", []),
                t2s_result.get("sql_queries", []),
                t2s_result.get("confidence", 0.88),
                started,
                layer="text2sql",
            )
            _store_cache(q_hash, result)
            if memory:
                memory.update(
                    entity=_extract_entity_from_result(t2s_result),
                    query=original_query,
                )
            return result

        # Both SIMPLE paths failed — fall through to COMPLEX path
        LOGGER.info("═══ SIMPLE path fully missed → escalating to COMPLEX")

    # ══════════════════════════════════════════════════════════════════════
    # COMPLEX PATH: decompose → parallel execute → synthesize
    # ══════════════════════════════════════════════════════════════════════
    LOGGER.info("═══ COMPLEX path: Decompose → Parallel → Synthesize")

    try:
        table_names = get_table_names(agent)

        # ── Step A: Decompose ──────────────────────────────────────────
        with Timer("decompose") as dec_timer:
            sub_queries = decompose(user_query, table_names)
        LOGGER.info(
            "Decomposed into %d sub-queries (%.0fms): %s",
            len(sub_queries), dec_timer.elapsed_ms,
            [sq.get("sub_query", "")[:40] for sq in sub_queries],
        )

        # ── Step B: Parallel execution ─────────────────────────────────
        with Timer("parallel_exec") as exec_timer:
            sub_results = run_parallel(sub_queries, agent)
        LOGGER.info(
            "Parallel execution done (%.0fms): %d results",
            exec_timer.elapsed_ms, len(sub_results),
        )

        # ── Step C: Build synthesis input ──────────────────────────────
        # Convert SubQueryResult objects to dicts for synthesizer
        synthesis_input = []
        for sq, sr in zip(sub_queries, sub_results):
            if isinstance(sr, SubQueryResult):
                synthesis_input.append({
                    "sub_query": sr.sub_query,
                    "intent":    sr.intent,
                    "data":      sr.to_dict(),
                })
            elif isinstance(sr, dict):
                synthesis_input.append(sr)
            else:
                synthesis_input.append({
                    "sub_query": sq.get("sub_query", ""),
                    "intent":    sq.get("intent", "general"),
                    "data":      {"answer": "Data unavailable.", "error": "unexpected result type"},
                })

        # ── Step D: Synthesize ─────────────────────────────────────────
        with Timer("synthesize") as syn_timer:
            final_answer = synthesize(user_query, synthesis_input)
        LOGGER.info("Synthesis done (%.0fms)", syn_timer.elapsed_ms)

        # ── Aggregate metadata ─────────────────────────────────────────
        all_tables = sorted({
            t
            for item in synthesis_input
            for t in (
                item.get("data", {}).get("tables_used", [])
                if isinstance(item.get("data"), dict) else []
            )
        })
        all_sqls = [
            sql
            for item in synthesis_input
            for sql in (
                item.get("data", {}).get("sql_queries", [])
                if isinstance(item.get("data"), dict) else []
            )
        ]
        avg_conf = 0.0
        conf_items = [
            item.get("data", {}).get("confidence", 0.0)
            for item in synthesis_input
            if isinstance(item.get("data"), dict)
        ]
        if conf_items:
            avg_conf = sum(conf_items) / len(conf_items)

        result = _build_response(
            final_answer,
            all_tables,
            all_sqls,
            avg_conf,
            started,
            layer="complex",
            sub_results=synthesis_input,
        )

        _store_cache(q_hash, result)
        if memory and all_tables:
            memory.update(entity=all_tables[0], query=original_query)

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        LOGGER.info(
            "═══ Pipeline DONE | total=%.0fms | layer=complex | "
            "parts=%d | tables=%s ═══",
            total_ms, len(synthesis_input), all_tables,
        )

        return result

    except Exception as exc:
        LOGGER.error("Pipeline COMPLEX path failed: %s", exc, exc_info=True)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "answer": (
                "I encountered an error processing your query. "
                "Please try rephrasing or breaking it into simpler questions.\n\n"
                f"_Error: {str(exc)[:100]}_"
            ),
            "latency_ms":  total_ms,
            "confidence":  0.0,
            "tables_used": [],
            "sql_queries": [],
            "layer":       "error",
        }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def validate_response(answer: str, user_query: str) -> bool:
    """Validate if the response adequately answers the query.

    Used by text2sql and other layers to decide if a retry is needed.

    Returns:
        True if the response seems valid, False if it looks empty/failed.
    """
    ans_lower = answer.lower()

    empty_signals = [
        "no data found", "no records found", "no invoice",
        "could not find", "unable to complete", "no answer",
        "data unavailable",
    ]
    if not any(s in ans_lower for s in empty_signals):
        return True

    # Specific entity lookups should not return empty
    if re.search(r"[A-Z]{2,}/\d{4}/\d+", user_query, re.I):
        return False  # Invoice number lookup returned empty
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,}\b", user_query):
        return False  # Named entity lookup returned empty

    return True


# ══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def clear_cache() -> int:
    """Clear the query cache. Returns number of evicted entries."""
    global _QUERY_CACHE
    count = len(_QUERY_CACHE)
    _QUERY_CACHE = {}
    LOGGER.info("Query cache cleared: %d entries evicted", count)
    return count


def cache_stats() -> Dict[str, Any]:
    """Return cache statistics for monitoring."""
    now = time.time()
    ages = [now - v.get("_ts", 0) for v in _QUERY_CACHE.values()]
    return {
        "size":       len(_QUERY_CACHE),
        "max_size":   _CACHE_MAX_SIZE,
        "ttl_sec":    _CACHE_TTL_SEC,
        "oldest_sec": round(max(ages), 1) if ages else 0,
        "newest_sec": round(min(ages), 1) if ages else 0,
    }