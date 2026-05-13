"""main.py — Pipeline orchestrator (production-grade).

Implements the full hybrid NL-to-SQL query flow:

    User Query
        ↓
    [Guard]        Greeting / Blocked / Vague checks
        ↓
    [Layer 1]      Fast Path Engine    → regex rules, <50ms, NO LLM
        ↓ (miss)
    [Layer 1.5]    Intent Router       → LLM brain (~400ms, local Ollama)
                   Understands ANY phrasing → SQL template → execute
        ↓ (miss or low confidence)
    [Layer 2]      Intent Classifier   → SIMPLE | COMPLEX
        ↓
    [SIMPLE]       Text2SQL            → Ollama fine-tuned → execute
    [COMPLEX]      Decomposer          → Groq/Ollama → sub-queries
                   Executor            → parallel SQL execution
                   Synthesizer         → Groq/Ollama → final answer

    All SIMPLE-path results (fast_path, intent_router, text2sql) are wrapped
    by narrate_response() which adds an executive summary paragraph.

Public API:
    run(agent, user_query, memory, request_id="") -> Dict[str, Any]
    ConversationMemory                             ← legacy alias for ChatMemory
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from config import settings
from pipeline import fast_path, record_lookup, text2sql
from pipeline.chat_memory import ChatMemory
from pipeline.classifier import classify
from pipeline.decomposer import decompose
from pipeline.executor import run_parallel, SubQueryResult
from pipeline.intent_router import run as intent_router_run
from pipeline.schema import (
    discover_schema_links,
    get_table_names,
    build_text2sql_schema,
)
from pipeline.synthesizer import narrate_response, synthesize
from pipeline.utils import (
    Timer,
    _IDENTITY_ANSWER,
    format_final_answer,
    is_blocked,
    is_greeting,
    is_identity_question,
    is_vague_query,
    mask_ids,
    normalize_text,
    sanitize_user_input,
)

LOGGER = logging.getLogger("sql_chatbot")

_SCHEMA_WARMED = False

# Legacy alias so any code that imported ConversationMemory from here still works
ConversationMemory = ChatMemory


# ConversationMemory is now ChatMemory (imported above).
# The class definition has moved to pipeline/chat_memory.py.


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
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build standardized pipeline response dict."""
    latency = round((time.perf_counter() - started) * 1000, 2)
    resp: Dict[str, Any] = {
        "answer":      format_final_answer(answer, tables_used),
        "latency_ms":  latency,
        "confidence":  round(confidence, 3),
        "tables_used": tables_used,
        "sql_queries": sql_queries,
        "layer":       layer,
    }
    if sub_results:
        resp["_sub_results"] = sub_results
    if metrics is not None:
        resp["metrics"] = metrics
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
                LOGGER.info("Schema pre-warmed: %d tables, %.0fms", len(tables), t.elapsed_ms)
        except Exception as exc:
            LOGGER.warning("Schema warmup failed (non-fatal): %s", exc)


def _extract_entity_from_result(result: Dict[str, Any]) -> Optional[str]:
    entity = result.get("_entity")
    if entity:
        return entity
    tables = result.get("tables_used", [])
    return tables[0] if tables else None


def _empty_metrics() -> Dict[str, Any]:
    return {
        "guard_ms": 0, "fastpath_ms": 0, "record_lookup_ms": 0,
        "intent_router_ms": 0, "classifier_ms": 0,
        "text2sql_ms": 0, "decomposer_ms": 0,
        "executor_ms": 0, "synthesizer_ms": 0, "narrate_ms": 0, "total_ms": 0,
    }


def _narrate(query: str, raw: Dict[str, Any]) -> str:
    """Wrap a SIMPLE-path result with an executive summary paragraph."""
    return narrate_response(
        query,
        raw.get("answer", ""),
        raw.get("tables_used", []),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    agent,
    user_query: str,
    memory: Optional[ChatMemory] = None,
    request_id: str = "",
) -> Dict[str, Any]:
    """Execute the full hybrid NL-to-SQL pipeline for one user query."""
    started = time.perf_counter()
    metrics = _empty_metrics()

    # ── 0. Input sanitization ─────────────────────────────────────────────────
    user_query, was_truncated = sanitize_user_input(user_query)
    if was_truncated:
        LOGGER.warning("[RID:%s] Input truncated to 500 chars", request_id)

    # ── 1. Memory: pronoun / bare-action resolution ───────────────────────────
    original_query = user_query
    if memory:
        user_query = memory.resolve(user_query)

    # ── 2. Guard: Identity question ("who are you") ───────────────────────────
    if is_identity_question(user_query):
        LOGGER.info("[RID:%s] Guard: identity question", request_id)
        metrics["guard_ms"] = metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        return {
            "answer":     _IDENTITY_ANSWER,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "confidence": 1.0, "tables_used": [], "sql_queries": [],
            "layer": "guard", "metrics": metrics,
        }

    # ── 3. Guard: Greeting ─────────────────────────────────────────────────────
    if is_greeting(user_query):
        LOGGER.info("[RID:%s] Guard: greeting", request_id)
        metrics["guard_ms"] = metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
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
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "confidence": 1.0, "tables_used": [], "sql_queries": [],
            "layer": "guard", "metrics": metrics,
        }

    # ── 3. Guard: Blocked query ────────────────────────────────────────────────
    if is_blocked(user_query):
        LOGGER.warning("[RID:%s] Guard: blocked: %.60s", request_id, user_query)
        metrics["guard_ms"] = metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        return {
            "answer": (
                "This query contains operations that are not permitted. "
                "I can only run read-only (SELECT) queries on your CRM data."
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "confidence": 1.0, "tables_used": [], "sql_queries": [],
            "layer": "guard", "metrics": metrics,
        }

    # ── 4. Guard: Vague query ──────────────────────────────────────────────────
    if is_vague_query(user_query):
        LOGGER.info("[RID:%s] Guard: vague: %.60s", request_id, user_query)
        metrics["guard_ms"] = metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        return {
            "answer": (
                "Your query seems a bit vague. Could you be more specific? For example:\n\n"
                "• \"How many **open deals** do we have?\"\n"
                "• \"Show **revenue this month**\"\n"
                "• \"List **pending tasks** for all users\""
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "confidence": 0.5, "tables_used": [], "sql_queries": [],
            "layer": "guard", "metrics": metrics,
        }

    metrics["guard_ms"] = round((time.perf_counter() - started) * 1000)
    LOGGER.info("[RID:%s] ═══ Pipeline START | query: %.80s ═══", request_id, user_query)

    # ── 5. Schema pre-warm ─────────────────────────────────────────────────────
    _warm_schema(agent)

    # Build memory context once — used by intent_router
    mem_context = memory.get_context(user_query) if memory else ""

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1: Fast Path — pure regex, <50ms, NO LLM
    # Disable via .env: FAST_PATH_ENABLED=false
    # ══════════════════════════════════════════════════════════════════════
    fp_result = None
    if settings.fast_path_enabled:
        with Timer("fast_path") as fp_timer:
            try:
                fp_result = fast_path.run(user_query, agent, apply_delay=True)
            except Exception as exc:
                LOGGER.warning("[RID:%s] Fast-path exception: %s", request_id, exc)
                fp_result = None
        metrics["fastpath_ms"] = round(fp_timer.elapsed_ms)
    else:
        LOGGER.info("[RID:%s] Fast-path DISABLED (FAST_PATH_ENABLED=false)", request_id)

    if fp_result is not None:
        LOGGER.info(
            "[RID:%s] ═══ Layer 1 HIT (fast-path) | %.0fms | tables=%s",
            request_id, fp_timer.elapsed_ms, fp_result.get("tables_used"),
        )
        with Timer("narrate") as nar_timer:
            narrated = _narrate(original_query, fp_result)
        metrics["narrate_ms"]  = round(nar_timer.elapsed_ms)
        metrics["total_ms"]    = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
        result = _build_response(
            narrated,
            fp_result.get("tables_used", []),
            fp_result.get("sql_queries", []),
            fp_result.get("confidence", 0.95),
            started, layer="fast_path", metrics=metrics,
        )
        if memory:
            memory.update(original_query, fp_result)
        return result

    LOGGER.info("[RID:%s] ═══ Layer 1 MISS → Record Lookup", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1.5a: Record Lookup — SO/ELSN/ELS number patterns, no LLM
    # Handles: "SO01241 line items", "ELSN/2026/018 details", "ELS042"
    # ══════════════════════════════════════════════════════════════════════
    with Timer("record_lookup") as rl_timer:
        try:
            rl_result = record_lookup.run(user_query, agent)
        except Exception as exc:
            LOGGER.warning("[RID:%s] RecordLookup exception: %s", request_id, exc)
            rl_result = None
    metrics["record_lookup_ms"] = round(rl_timer.elapsed_ms)

    if rl_result is not None:
        LOGGER.info(
            "[RID:%s] ═══ Layer 1.5a HIT (record-lookup) | %.0fms | tables=%s",
            request_id, rl_timer.elapsed_ms, rl_result.get("tables_used"),
        )
        with Timer("narrate") as nar_timer:
            narrated = _narrate(original_query, rl_result)
        metrics["narrate_ms"] = round(nar_timer.elapsed_ms)
        metrics["total_ms"]   = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
        result = _build_response(
            narrated,
            rl_result.get("tables_used", []),
            rl_result.get("sql_queries", []),
            rl_result.get("confidence", 0.98),
            started, layer="record_lookup", metrics=metrics,
        )
        if memory:
            memory.update(original_query, rl_result)
        return result

    LOGGER.info("[RID:%s] ═══ Record Lookup MISS → Intent Router", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1.5b: Intent Router — LLM brain, understands ANY phrasing
    # "tell me all deals" = "show deals" = "give me deals" → same SQL
    # ══════════════════════════════════════════════════════════════════════
    with Timer("intent_router") as ir_timer:
        try:
            ir_result = intent_router_run(user_query, agent, memory_context=mem_context)
        except Exception as exc:
            LOGGER.warning("[RID:%s] Intent router exception: %s", request_id, exc)
            ir_result = None
    metrics["intent_router_ms"] = round(ir_timer.elapsed_ms)

    if ir_result is not None:
        LOGGER.info(
            "[RID:%s] ═══ Layer 1.5 HIT (intent-router) | %.0fms | tables=%s",
            request_id, ir_timer.elapsed_ms, ir_result.get("tables_used"),
        )
        with Timer("narrate") as nar_timer:
            narrated = _narrate(original_query, ir_result)
        metrics["narrate_ms"]  = round(nar_timer.elapsed_ms)
        metrics["total_ms"]    = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
        result = _build_response(
            narrated,
            ir_result.get("tables_used", []),
            ir_result.get("sql_queries", []),
            ir_result.get("confidence", 0.88),
            started, layer="intent_router", metrics=metrics,
        )
        if memory:
            memory.update(original_query, ir_result)
        return result

    LOGGER.info("[RID:%s] ═══ Intent Router MISS → Classifier", request_id)

    # ── 6. Intent classification ───────────────────────────────────────────────
    with Timer("classifier") as cls_timer:
        classification = classify(user_query)
    intent_type   = classification["type"]
    intent_reason = classification["reason"]
    metrics["classifier_ms"] = round(cls_timer.elapsed_ms)
    LOGGER.info(
        "[RID:%s] ═══ Classifier: %s (%.0fms) | %s",
        request_id, intent_type, cls_timer.elapsed_ms, intent_reason,
    )

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2: Text2SQL — SIMPLE queries that both fast_path and intent_router missed
    # ══════════════════════════════════════════════════════════════════════
    if intent_type == "SIMPLE":

        with Timer("text2sql") as t2s_timer:
            try:
                t2s_result = text2sql.run(user_query, agent)
            except Exception as exc:
                LOGGER.warning("[RID:%s] Text2SQL exception: %s", request_id, exc)
                t2s_result = None
        metrics["text2sql_ms"] = round(t2s_timer.elapsed_ms)

        if t2s_result is not None:
            LOGGER.info(
                "[RID:%s] ═══ Text2SQL HIT | %.0fms | tables=%s",
                request_id, t2s_timer.elapsed_ms, t2s_result.get("tables_used"),
            )
            with Timer("narrate") as nar_timer:
                narrated = _narrate(original_query, t2s_result)
            metrics["narrate_ms"]  = round(nar_timer.elapsed_ms)
            metrics["total_ms"]    = round((time.perf_counter() - started) * 1000)
            LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
            result = _build_response(
                narrated,
                t2s_result.get("tables_used", []),
                t2s_result.get("sql_queries", []),
                t2s_result.get("confidence", 0.88),
                started, layer="text2sql", metrics=metrics,
            )
            if memory:
                memory.update(original_query, t2s_result)
            return result

        LOGGER.info("[RID:%s] ═══ SIMPLE path missed → escalating to COMPLEX", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # COMPLEX PATH: decompose → parallel execute → synthesize
    # ══════════════════════════════════════════════════════════════════════
    LOGGER.info("[RID:%s] ═══ COMPLEX path: Decompose → Parallel → Synthesize", request_id)

    try:
        table_names = get_table_names(agent)

        # ── Step A: Decompose ──────────────────────────────────────────────
        # Pass the full schema so the LLM knows actual field names and table
        # structures (e.g. approval_status vs payment_status split).
        schema_ctx = build_text2sql_schema(agent)
        with Timer("decompose") as dec_timer:
            sub_queries = decompose(user_query, table_names, schema_context=schema_ctx)
        metrics["decomposer_ms"] = round(dec_timer.elapsed_ms)
        LOGGER.info(
            "[RID:%s] Decomposed into %d sub-queries (%.0fms): %s",
            request_id, len(sub_queries), dec_timer.elapsed_ms,
            [sq.get("sub_query", "")[:40] for sq in sub_queries],
        )

        # ── Step B: Parallel execution ─────────────────────────────────────
        with Timer("parallel_exec") as exec_timer:
            sub_results = run_parallel(sub_queries, agent, request_id=request_id)
        metrics["executor_ms"] = round(exec_timer.elapsed_ms)
        LOGGER.info(
            "[RID:%s] Parallel execution done (%.0fms): %d results",
            request_id, exec_timer.elapsed_ms, len(sub_results),
        )

        # ── Step C: Build synthesis input ──────────────────────────────────
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

        # ── Step D: Synthesize ─────────────────────────────────────────────
        with Timer("synthesize") as syn_timer:
            final_answer = synthesize(user_query, synthesis_input)
        metrics["synthesizer_ms"] = round(syn_timer.elapsed_ms)
        LOGGER.info("[RID:%s] Synthesis done (%.0fms)", request_id, syn_timer.elapsed_ms)

        # ── Aggregate metadata ─────────────────────────────────────────────
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
        conf_items = [
            item.get("data", {}).get("confidence", 0.0)
            for item in synthesis_input
            if isinstance(item.get("data"), dict)
        ]
        avg_conf = sum(conf_items) / len(conf_items) if conf_items else 0.0

        metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))

        result = _build_response(
            final_answer, all_tables, all_sqls, avg_conf, started,
            layer="complex", sub_results=synthesis_input, metrics=metrics,
        )

        if memory and all_tables:
            memory.update(original_query, result)

        LOGGER.info(
            "[RID:%s] ═══ Pipeline DONE | total=%.0fms | parts=%d | tables=%s ═══",
            request_id, metrics["total_ms"], len(synthesis_input), all_tables,
        )
        return result

    except Exception as exc:
        LOGGER.error("[RID:%s] Pipeline COMPLEX path failed: %s", request_id, exc, exc_info=True)
        metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
        return {
            "answer": (
                "I encountered an error processing your query. "
                "Please try rephrasing or breaking it into simpler questions."
            ),
            "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
            "confidence":  0.0,
            "tables_used": [],
            "sql_queries": [],
            "layer":       "error",
            "metrics":     metrics,
        }

