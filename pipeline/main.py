"""main.py — Pipeline orchestrator with 3-agent routing.

Query flow:

    User Query
        ↓
    [GUARD]       Greeting / Identity / Blocked / Vague — 0ms, no LLM
        ↓ pass
    [LAYER 1]     Fast Path — regex SQL templates, NO LLM (~300ms)
                  Controlled by FAST_PATH_ENABLED in .env
        ↓ miss
    [CLASSIFY]    LLM classifier (Groq 8b → Gemini → Ollama)
                  Returns: SIMPLE | MEDIUM | COMPLEX
        ↓
    [ROUTE]       → simple_agent  (CrewAI, target <5s)
                  → medium_agent  (LangGraph, target <15s)
                  → complex_agent (LangGraph extended, target <30s)
        ↓
    [RESPONSE]    Standard dict: answer, sql_queries, tables_used, confidence, latency_ms

Public API (unchanged — compatible with agent.py and api.py):
    run(agent, user_query, memory=None, request_id="") -> Dict[str, Any]
    ConversationMemory  ← alias for ChatMemory
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from config import settings
from pipeline import fast_path
from pipeline.chat_memory import ChatMemory
from pipeline.classifier import classify
from pipeline.schema import build_text2sql_schema, discover_schema_links, get_table_names
from pipeline.utils import (
    Timer,
    _IDENTITY_ANSWER,
    _SYSTEM_CONFIG_ANSWER,
    format_final_answer,
    is_blocked,
    is_greeting,
    is_identity_question,
    is_system_config_query,
    is_user_input_unsafe,
    is_vague_query,
    sanitize_user_input,
)

LOGGER = logging.getLogger("sql_chatbot")

# Legacy alias — kept for backward compat with api.py / agent.py
ConversationMemory = ChatMemory

_SCHEMA_WARMED = False
_SCHEMA_WARM_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _warm_schema(agent) -> None:
    """Pre-warm schema caches on first call (once per process, thread-safe)."""
    global _SCHEMA_WARMED
    if _SCHEMA_WARMED:
        return
    with _SCHEMA_WARM_LOCK:
        if _SCHEMA_WARMED:  # double-checked locking
            return
        try:
            tables = get_table_names(agent)
            if tables:
                discover_schema_links(agent)
                build_text2sql_schema(agent)
                _SCHEMA_WARMED = True
                LOGGER.info("Schema pre-warmed: %d tables", len(tables))
        except Exception as exc:
            LOGGER.warning("Schema warmup non-fatal: %s", exc)


def _guard_response(answer: str, started: float, layer: str = "guard") -> Dict[str, Any]:
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    return {
        "answer":      answer,
        "latency_ms":  elapsed,
        "confidence":  1.0,
        "tables_used": [],
        "sql_queries": [],
        "layer":       layer,
    }


def _normalize_response(result: Dict[str, Any], started: float) -> Dict[str, Any]:
    """Ensure every response has all fields expected by api.py."""
    if "latency_ms" not in result or not result["latency_ms"]:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    if "answer" not in result:
        result["answer"] = "No response generated."
    if "tables_used" not in result:
        result["tables_used"] = []
    if "sql_queries" not in result:
        result["sql_queries"] = []
    if "confidence" not in result:
        result["confidence"] = 0.80
    if "layer" not in result:
        result["layer"] = "unknown"
    # Wrap answer with tables footer (kept for UI compatibility)
    result["answer"] = format_final_answer(
        result["answer"], result.get("tables_used", [])
    )
    return result


def _narrate_fast_path(query: str, fp_result: Dict) -> str:
    """Add executive summary to a fast-path answer via the narrate LLM route."""
    from pipeline.synthesizer import narrate_response
    try:
        return narrate_response(
            query,
            fp_result.get("answer", ""),
            fp_result.get("tables_used", []),
        )
    except Exception:
        return fp_result.get("answer", "")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    agent,
    user_query:        str,
    memory:            Optional[ChatMemory] = None,
    request_id:        str = "",
    query_preresolved: bool = False,
) -> Dict[str, Any]:
    """Execute the full pipeline for one user query and return a result dict.

    This is the single entry point called by agent.py → api.py.
    All agent imports are lazy (inside the routing block) to avoid circular
    imports and to keep the module fast on startup.

    Args:
        agent:            LangChain AgentExecutor (kept for fast_path and schema tools)
        user_query:       Raw user input (or already-resolved query when query_preresolved=True)
        memory:           Optional ChatMemory for working-memory tracking (entity, tables, etc.)
        request_id:       Correlation ID for log tracing
        query_preresolved: When True, skip ChatMemory.resolve() — the query was already
                          resolved upstream by LLM-powered MemoryManager. Prevents the
                          regex resolver from corrupting words like "last", "same", "this"
                          in a fully self-contained query.

    Returns:
        Dict with: answer, latency_ms, confidence, tables_used, sql_queries, layer
    """
    started = time.perf_counter()

    # ── Input sanitization ────────────────────────────────────────────────────
    user_query, was_truncated = sanitize_user_input(user_query)
    if was_truncated:
        LOGGER.warning("[RID:%s] Input truncated to 500 chars", request_id)

    # ── Memory: resolve pronouns / bare-action references ─────────────────────
    # Skip if query was already resolved by LLM-powered MemoryManager upstream.
    # The regex resolver uses "last", "same", "this" as pronouns which would
    # incorrectly corrupt a fully self-contained query like "companies added last year".
    original_query = user_query
    if memory and not query_preresolved:
        user_query = memory.resolve(user_query)

    # ══════════════════════════════════════════════════════════════════════
    # GUARD — 0ms checks, no DB, no LLM
    # ══════════════════════════════════════════════════════════════════════
    if is_identity_question(user_query):
        LOGGER.info("[RID:%s] Guard: identity", request_id)
        result = _guard_response(_IDENTITY_ANSWER, started, "guard_identity")
        if memory:
            memory.update(original_query, result)
        return result

    if is_greeting(user_query):
        LOGGER.info("[RID:%s] Guard: greeting", request_id)
        result = _guard_response(
            "Hello! I can help you query your CRM data. Try asking:\n\n"
            "• **Counts**: \"How many deals do we have?\"\n"
            "• **Revenue**: \"Total revenue this month\"\n"
            "• **Reports**: \"Sales leaderboard\" / \"Executive summary\"\n"
            "• **Analysis**: \"Which rep has the best win rate?\"\n"
            "• **Search**: \"Find contact John Smith\"",
            started, "guard_greeting",
        )
        if memory:
            memory.update(original_query, result)
        return result

    if is_user_input_unsafe(user_query):
        LOGGER.warning("[RID:%s] Guard: blocked: %.60s", request_id, user_query)
        result = _guard_response(
            "This query contains operations that are not permitted. "
            "I can only run read-only (SELECT) queries on your CRM data.",
            started, "guard_blocked",
        )
        return result

    if is_system_config_query(user_query):
        LOGGER.info("[RID:%s] Guard: system-config: %.60s", request_id, user_query)
        result = _guard_response(_SYSTEM_CONFIG_ANSWER, started, "guard_system_config")
        if memory:
            memory.update(original_query, result)
        return result

    if is_vague_query(user_query):
        LOGGER.info("[RID:%s] Guard: vague: %.60s", request_id, user_query)
        result = _guard_response(
            "Your query seems a bit vague. Could you be more specific? For example:\n\n"
            "• \"How many **open deals** do we have?\"\n"
            "• \"Show **revenue this month**\"\n"
            "• \"List **pending tasks** for all users\"",
            started, "guard_vague",
        )
        result["confidence"] = 0.5
        return result

    LOGGER.info("[RID:%s] ══ Pipeline START | query: %.80s ══", request_id, user_query)

    # ── Schema pre-warm (runs once per process) ───────────────────────────────
    _warm_schema(agent)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 — FAST PATH (regex + SQL templates, NO LLM, ~300ms)
    # ══════════════════════════════════════════════════════════════════════
    if settings.fast_path_enabled:
        with Timer("fast_path") as fp_timer:
            try:
                fp_result = fast_path.run(user_query, agent, apply_delay=True)
            except Exception as exc:
                LOGGER.warning("[RID:%s] Fast-path error: %s", request_id, exc)
                fp_result = None

        if fp_result is not None:
            LOGGER.info("[RID:%s] ══ Layer 1 HIT (fast-path) %.0fms", request_id, fp_timer.elapsed_ms)
            narrated = _narrate_fast_path(original_query, fp_result)
            result   = {
                "answer":      narrated,
                "latency_ms":  round(fp_timer.elapsed_ms, 2),
                "confidence":  fp_result.get("confidence", 0.95),
                "tables_used": fp_result.get("tables_used", []),
                "sql_queries": fp_result.get("sql_queries", []),
                "layer":       "fast_path",
            }
            if memory:
                memory.update(original_query, fp_result)
            return _normalize_response(result, started)

        LOGGER.info("[RID:%s] ══ Layer 1 MISS → Classifier", request_id)
    else:
        LOGGER.info("[RID:%s] ══ Fast path DISABLED → Classifier", request_id)

    # ══════════════════════════════════════════════════════════════════════
    # CLASSIFY — LLM decides SIMPLE | MEDIUM | COMPLEX
    # ══════════════════════════════════════════════════════════════════════
    with Timer("classify") as cls_timer:
        classification = classify(user_query)

    intent_type = classification["type"]
    LOGGER.info(
        "[RID:%s] ══ Classifier: %s (%.0fms) | %s",
        request_id, intent_type, cls_timer.elapsed_ms, classification.get("reason", ""),
    )

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE TO AGENT
    # ══════════════════════════════════════════════════════════════════════
    result: Optional[Dict[str, Any]] = None

    if intent_type == "SIMPLE":
        LOGGER.info("[RID:%s] ══ Routing to SIMPLE agent", request_id)
        try:
            from agents.simple_agent import run_simple_agent
            result = run_simple_agent(user_query, classification)
        except Exception as exc:
            LOGGER.error("[RID:%s] Simple agent error: %s — falling back to MEDIUM", request_id, exc)
            intent_type = "MEDIUM"

    if intent_type == "MEDIUM":
        LOGGER.info("[RID:%s] ══ Routing to MEDIUM agent", request_id)
        try:
            from agents.medium_agent import run_medium_agent
            result = run_medium_agent(user_query, classification)
        except Exception as exc:
            LOGGER.error("[RID:%s] Medium agent error: %s — falling back to COMPLEX", request_id, exc)
            intent_type = "COMPLEX"

    if intent_type == "COMPLEX" and result is None:
        LOGGER.info("[RID:%s] ══ Routing to COMPLEX agent", request_id)
        try:
            from agents.complex_agent import run_complex_agent
            result = run_complex_agent(user_query, classification)
        except Exception as exc:
            LOGGER.error("[RID:%s] Complex agent error: %s", request_id, exc, exc_info=True)
            result = {
                "answer": (
                    "I encountered an error processing your query. "
                    "Please try rephrasing or breaking it into simpler questions."
                ),
                "sql_queries": [],
                "tables_used": [],
                "confidence":  0.0,
                "layer":       "error",
            }

    if result is None:
        result = {
            "answer":      "No agent was able to process this query.",
            "sql_queries": [],
            "tables_used": [],
            "confidence":  0.0,
            "layer":       "no_agent",
        }

    # Update memory with successful result
    if memory and result.get("tables_used"):
        memory.update(original_query, result)

    normalized = _normalize_response(result, started)
    LOGGER.info(
        "[RID:%s] ══ Pipeline DONE %.0fms | agent=%s | tables=%s ══",
        request_id,
        normalized.get("latency_ms", 0),
        normalized.get("agent_type", intent_type.lower()),
        normalized.get("tables_used", []),
    )
    return normalized
