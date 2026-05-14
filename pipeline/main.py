"""main.py — Pipeline orchestrator.

Exact query flow:

    User Query
        ↓
    [Guard]       Greeting / Blocked / Vague / Identity checks  (0ms, no LLM)
        ↓ pass
    [Layer 1]     Fast Path Engine                              (~300ms, NO LLM)
                  Rule-based regex + SQL templates.
                  Handles ~70% of all CRM queries instantly.
        ↓ miss (None)
    [Layer 2]     Text2SQL — Ollama fine-tuned models           (5–30s, local)
                  PRIMARY : Qwen2.5-Coder-3B  (fast, 15s timeout)
                  FALLBACK: Arctic-Text2SQL-7B (accurate, 30s timeout)
                  Both fire in parallel — first valid SQL wins.
        ↓ hit → narrate → return to user
        ↓ miss (both models fail or SQL error)
    [Layer 3]     Intent Classifier                             (~300ms, Groq)
                  Decides: SIMPLE | COMPLEX
        ↓
    [SIMPLE]      Text2SQL retry with fallback model only       (30s, local)
        ↓ hit → narrate → return
        ↓ miss
    [COMPLEX]     Decomposer  → sub-queries                     (Groq/Ollama)
                  Executor    → parallel SQL execution          (threads)
                  Synthesizer → merge into final answer         (Groq/Ollama)

Public API:
    run(agent, user_query, memory, request_id="") -> Dict[str, Any]
    ConversationMemory                             ← alias for ChatMemory
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from config import settings
from pipeline import fast_path, text2sql
from pipeline.chat_memory import ChatMemory
from pipeline.classifier import classify
from pipeline.decomposer import decompose
from pipeline.executor import run_parallel, SubQueryResult
from pipeline.schema import (
    build_text2sql_schema,
    discover_schema_links,
    get_table_names,
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
    sanitize_user_input,
)

LOGGER = logging.getLogger("sql_chatbot")

_SCHEMA_WARMED = False

# Legacy alias — any code that imported ConversationMemory from here still works
ConversationMemory = ChatMemory


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_response(
    answer:      str,
    tables_used: List[str],
    sql_queries: List[str],
    confidence:  float,
    started:     float,
    layer:       str = "unknown",
    sub_results: Optional[List] = None,
    metrics:     Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
                LOGGER.info("Schema pre-warmed: %d tables in %.0fms", len(tables), t.elapsed_ms)
        except Exception as exc:
            LOGGER.warning("Schema warmup failed (non-fatal): %s", exc)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "guard_ms":      0,
        "fastpath_ms":   0,
        "text2sql_ms":   0,
        "classifier_ms": 0,
        "decomposer_ms": 0,
        "executor_ms":   0,
        "synthesizer_ms":0,
        "narrate_ms":    0,
        "total_ms":      0,
    }


def _narrate(query: str, raw: Dict[str, Any]) -> str:
    """Add executive summary paragraph to a SIMPLE-path answer."""
    return narrate_response(
        query,
        raw.get("answer", ""),
        raw.get("tables_used", []),
    )


def _guard_response(answer: str, started: float, metrics: Dict) -> Dict[str, Any]:
    metrics["guard_ms"] = metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
    return {
        "answer":      answer,
        "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
        "confidence":  1.0,
        "tables_used": [],
        "sql_queries": [],
        "layer":       "guard",
        "metrics":     metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    agent,
    user_query:  str,
    memory:      Optional[ChatMemory] = None,
    request_id:  str = "",
) -> Dict[str, Any]:
    """Execute the full pipeline for one user query and return a result dict."""
    started = time.perf_counter()
    metrics = _empty_metrics()

    # ── Input sanitization ────────────────────────────────────────────────────
    user_query, was_truncated = sanitize_user_input(user_query)
    if was_truncated:
        LOGGER.warning("[RID:%s] Input truncated to 500 chars", request_id)

    # ── Memory: resolve pronouns / bare-action references ─────────────────────
    original_query = user_query
    if memory:
        user_query = memory.resolve(user_query)

    # ══════════════════════════════════════════════════════════════════════
    # GUARD — fast checks, no DB, no LLM (0ms)
    # ══════════════════════════════════════════════════════════════════════
    if is_identity_question(user_query):
        LOGGER.info("[RID:%s] Guard: identity", request_id)
        return _guard_response(_IDENTITY_ANSWER, started, metrics)

    if is_greeting(user_query):
        LOGGER.info("[RID:%s] Guard: greeting", request_id)
        return _guard_response(
            "Hello! I can help you query your CRM data. Try asking:\n\n"
            "• **Counts**: \"How many deals do we have?\"\n"
            "• **Lists**: \"Show all closed won deals\"\n"
            "• **Revenue**: \"Total revenue this month\"\n"
            "• **Reports**: \"KPI report\" / \"Executive summary\"\n"
            "• **Targets**: \"Target vs achieved this quarter\"\n"
            "• **Search**: \"Find contact John Smith\"",
            started, metrics,
        )

    if is_blocked(user_query):
        LOGGER.warning("[RID:%s] Guard: blocked: %.60s", request_id, user_query)
        return _guard_response(
            "This query contains operations that are not permitted. "
            "I can only run read-only (SELECT) queries on your CRM data.",
            started, metrics,
        )

    if is_vague_query(user_query):
        LOGGER.info("[RID:%s] Guard: vague: %.60s", request_id, user_query)
        return {
            "answer":      (
                "Your query seems a bit vague. Could you be more specific? For example:\n\n"
                "• \"How many **open deals** do we have?\"\n"
                "• \"Show **revenue this month**\"\n"
                "• \"List **pending tasks** for all users\""
            ),
            "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
            "confidence":  0.5, "tables_used": [], "sql_queries": [],
            "layer":       "guard", "metrics": {**metrics, "guard_ms": round((time.perf_counter()-started)*1000), "total_ms": round((time.perf_counter()-started)*1000)},
        }

    metrics["guard_ms"] = round((time.perf_counter() - started) * 1000)
    LOGGER.info("[RID:%s] ══ Pipeline START | query: %.80s ══", request_id, user_query)

    # ── Schema pre-warm (runs once per process) ───────────────────────────────
    _warm_schema(agent)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 — FAST PATH (regex + SQL templates, NO LLM, ~300ms)
    # Controlled by FAST_PATH_ENABLED in .env (default: true)
    # ══════════════════════════════════════════════════════════════════════
    if settings.fast_path_enabled:
        with Timer("fast_path") as fp_timer:
            try:
                fp_result = fast_path.run(user_query, agent, apply_delay=True)
            except Exception as exc:
                LOGGER.warning("[RID:%s] Fast-path error: %s", request_id, exc)
                fp_result = None
        metrics["fastpath_ms"] = round(fp_timer.elapsed_ms)

        if fp_result is not None:
            LOGGER.info(
                "[RID:%s] ══ Layer 1 HIT (fast-path) %.0fms | tables=%s",
                request_id, fp_timer.elapsed_ms, fp_result.get("tables_used"),
            )
            with Timer("narrate") as nar_timer:
                narrated = _narrate(original_query, fp_result)
            metrics["narrate_ms"] = round(nar_timer.elapsed_ms)
            metrics["total_ms"]   = round((time.perf_counter() - started) * 1000)
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

        LOGGER.info("[RID:%s] ══ Layer 1 MISS → Text2SQL", request_id)
    else:
        LOGGER.info("[RID:%s] ══ Fast path DISABLED → Text2SQL", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2 — TEXT2SQL (Ollama fine-tuned models, local, no API cost)
    #
    # Both models fire in parallel — first valid SQL wins:
    #   PRIMARY : Qwen2.5-Coder-3B  (fast, OLLAMA_PRIMARY_TIMEOUT)
    #   FALLBACK: Arctic-7B         (accurate, OLLAMA_FALLBACK_TIMEOUT)
    #
    # This runs DIRECTLY after fast path — no Intent Router in between.
    # The fine-tuned models understand natural language CRM queries and
    # generate SQL from the CREATE TABLE schema provided in the prompt.
    # ══════════════════════════════════════════════════════════════════════
    with Timer("text2sql") as t2s_timer:
        try:
            t2s_result = text2sql.run(user_query, agent)
        except Exception as exc:
            LOGGER.warning("[RID:%s] Text2SQL error: %s", request_id, exc)
            t2s_result = None
    metrics["text2sql_ms"] = round(t2s_timer.elapsed_ms)

    if t2s_result is not None:
        LOGGER.info(
            "[RID:%s] ══ Layer 2 HIT (text2sql) %.0fms | tables=%s",
            request_id, t2s_timer.elapsed_ms, t2s_result.get("tables_used"),
        )
        with Timer("narrate") as nar_timer:
            narrated = _narrate(original_query, t2s_result)
        metrics["narrate_ms"] = round(nar_timer.elapsed_ms)
        metrics["total_ms"]   = round((time.perf_counter() - started) * 1000)
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

    LOGGER.info("[RID:%s] ══ Text2SQL MISS → Classifier", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3 — CLASSIFIER (Groq → Gemini → Ollama fallback)
    # Decides: SIMPLE | COMPLEX
    # SIMPLE  → Text2SQL retry with fallback model
    # COMPLEX → Decompose → Parallel Execute → Synthesize
    # ══════════════════════════════════════════════════════════════════════
    with Timer("classifier") as cls_timer:
        classification = classify(user_query)
    intent_type   = classification["type"]
    intent_reason = classification["reason"]
    metrics["classifier_ms"] = round(cls_timer.elapsed_ms)
    LOGGER.info(
        "[RID:%s] ══ Classifier: %s (%.0fms) | %s",
        request_id, intent_type, cls_timer.elapsed_ms, intent_reason,
    )

    # ── SIMPLE retry — try Groq first, then Ollama fallback ─────────────────────
    if intent_type == "SIMPLE":
        LOGGER.info("[RID:%s] ══ SIMPLE: retry SQL generation", request_id)
        from pipeline.text2sql import (
            _build_fallback_prompt, _extract_sql, _validate_sql,
            _call_ollama_chat, _extract_tables_from_sql, _format_result,
            _generate_sql_groq,
        )
        from pipeline.schema import run_sql, get_table_names as _gtn
        try:
            table_names_retry = _gtn(agent)

            # Groq retry first (fast)
            fsql = _generate_sql_groq(user_query, table_names_retry)

            # Fall through to Ollama if Groq failed
            if not fsql:
                fsys, fuser = _build_fallback_prompt(user_query)
                fraw = _call_ollama_chat(
                    settings.ollama_fallback_model, fsys, fuser,
                    timeout=min(settings.ollama_fallback_timeout, 15),  # cap at 15s
                    max_tokens=800,
                )
                fsql = _extract_sql(fraw) if fraw else None
            fok, ferr = _validate_sql(fsql, table_names_retry) if fsql else (False, "no SQL")
            if fok and fsql:
                _res = run_sql(agent, fsql)
                if not _res.error:
                    body = _format_result(_res.rows or [], fsql, user_query)
                    retry_result = {
                        "answer":      body,
                        "tables_used": _extract_tables_from_sql(fsql),
                        "confidence":  0.82,
                        "sql_queries": [fsql],
                    }
                    with Timer("narrate") as nar_timer:
                        narrated = _narrate(original_query, retry_result)
                    metrics["narrate_ms"]  = round(nar_timer.elapsed_ms)
                    metrics["total_ms"]    = round((time.perf_counter() - started) * 1000)
                    LOGGER.info("[RID:%s] ══ SIMPLE retry HIT %.0fms", request_id, metrics["total_ms"])
                    LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
                    result = _build_response(
                        narrated,
                        retry_result.get("tables_used", []),
                        retry_result.get("sql_queries", []),
                        retry_result.get("confidence", 0.82),
                        started, layer="text2sql_retry", metrics=metrics,
                    )
                    if memory:
                        memory.update(original_query, retry_result)
                    return result
        except Exception as exc:
            LOGGER.warning("[RID:%s] SIMPLE retry error: %s", request_id, exc)

        LOGGER.info("[RID:%s] ══ SIMPLE retry failed → COMPLEX path", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # COMPLEX PATH: Decompose → Parallel Execute → Synthesize
    # ══════════════════════════════════════════════════════════════════════
    LOGGER.info("[RID:%s] ══ COMPLEX: Decompose → Execute → Synthesize", request_id)

    try:
        table_names = get_table_names(agent)
        schema_ctx  = build_text2sql_schema(agent)

        # Step A — Decompose into sub-queries
        with Timer("decompose") as dec_timer:
            sub_queries = decompose(user_query, table_names, schema_context=schema_ctx)
        metrics["decomposer_ms"] = round(dec_timer.elapsed_ms)
        LOGGER.info(
            "[RID:%s] Decomposed → %d sub-queries (%.0fms): %s",
            request_id, len(sub_queries), dec_timer.elapsed_ms,
            [sq.get("sub_query", "")[:40] for sq in sub_queries],
        )

        # Step B — Parallel execution (fast_path + text2sql per sub-query)
        with Timer("parallel_exec") as exec_timer:
            sub_results = run_parallel(sub_queries, agent, request_id=request_id)
        metrics["executor_ms"] = round(exec_timer.elapsed_ms)
        LOGGER.info(
            "[RID:%s] Parallel done %.0fms | %d results",
            request_id, exec_timer.elapsed_ms, len(sub_results),
        )

        # Step C — Build synthesis input
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

        # Step D — Synthesize (Groq 70b → Gemini → OpenRouter → Ollama)
        with Timer("synthesize") as syn_timer:
            final_answer = synthesize(user_query, synthesis_input)
        metrics["synthesizer_ms"] = round(syn_timer.elapsed_ms)
        LOGGER.info("[RID:%s] Synthesis done %.0fms", request_id, syn_timer.elapsed_ms)

        # Aggregate metadata from all sub-results
        all_tables = sorted({
            t
            for item in synthesis_input
            for t in (item.get("data", {}).get("tables_used", [])
                      if isinstance(item.get("data"), dict) else [])
        })
        all_sqls = [
            sql
            for item in synthesis_input
            for sql in (item.get("data", {}).get("sql_queries", [])
                        if isinstance(item.get("data"), dict) else [])
        ]
        conf_list = [
            item.get("data", {}).get("confidence", 0.0)
            for item in synthesis_input
            if isinstance(item.get("data"), dict)
        ]
        avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0

        metrics["total_ms"] = round((time.perf_counter() - started) * 1000)
        LOGGER.info("[RID:%s] METRICS %s", request_id, json.dumps(metrics))
        LOGGER.info(
            "[RID:%s] ══ Pipeline DONE %.0fms | parts=%d | tables=%s ══",
            request_id, metrics["total_ms"], len(synthesis_input), all_tables,
        )

        result = _build_response(
            final_answer, all_tables, all_sqls, avg_conf, started,
            layer="complex", sub_results=synthesis_input, metrics=metrics,
        )
        if memory and all_tables:
            memory.update(original_query, result)
        return result

    except Exception as exc:
        LOGGER.error("[RID:%s] COMPLEX path failed: %s", request_id, exc, exc_info=True)
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
