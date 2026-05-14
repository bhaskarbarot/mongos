"""main.py — Pipeline orchestrator (dynamic LLM-driven, zero hardcoding).

Full flow:
    User Query
        ↓
    [Guard]      greeting / blocked / vague / identity  (0ms)
        ↓
    [Memory]     pronoun + bare-action resolution
        ↓
    [sql_agent]  LLM generates SQL → executes → LLM synthesizes response
                 (Groq 70b → Gemini → OpenRouter → Ollama 7b)
        ↓
    Return rich markdown answer

No regex-based routing. No hardcoded SQL templates.
The LLM understands ANY query and generates precise PostgreSQL.

Public API:
    run(agent, user_query, memory, request_id="") -> Dict[str, Any]
    ConversationMemory                             <- legacy alias for ChatMemory
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from pipeline import sql_agent
from pipeline.chat_memory import ChatMemory
from pipeline.schema import build_db_metadata, build_text2sql_schema, discover_schema_links, get_table_names
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

ConversationMemory = ChatMemory  # legacy alias

_SCHEMA_WARMED = False


def _warm_schema(agent) -> None:
    global _SCHEMA_WARMED
    if _SCHEMA_WARMED:
        return
    try:
        tables = get_table_names(agent)
        if tables:
            discover_schema_links(agent)
            build_text2sql_schema(agent)
            build_db_metadata(agent)   # pre-warm live enum values + row counts
            _SCHEMA_WARMED = True
            LOGGER.info("Schema + metadata pre-warmed: %d tables", len(tables))
    except Exception as exc:
        LOGGER.warning("Schema warmup failed (non-fatal): %s", exc)


def _guard_response(
    answer: str,
    started: float,
    confidence: float = 1.0,
    layer: str = "guard",
) -> Dict[str, Any]:
    return {
        "answer":      answer,
        "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
        "confidence":  confidence,
        "tables_used": [],
        "sql_queries": [],
        "layer":       layer,
    }


def run(
    agent,
    user_query: str,
    memory: Optional[ChatMemory] = None,
    request_id: str = "",
) -> Dict[str, Any]:
    """Execute the full dynamic NL-to-SQL pipeline for one user query."""
    started = time.perf_counter()

    # ── 0. Input sanitization ─────────────────────────────────────────────────
    user_query, was_truncated = sanitize_user_input(user_query)
    if was_truncated:
        LOGGER.warning("[%s] Input truncated to 500 chars", request_id)

    # ── 1. Memory: pronoun / reference resolution ─────────────────────────────
    original_query = user_query
    if memory:
        user_query = memory.resolve(user_query)

    # ── 2. Guard: identity question ───────────────────────────────────────────
    if is_identity_question(user_query):
        return _guard_response(_IDENTITY_ANSWER, started)

    # ── 3. Guard: greeting ────────────────────────────────────────────────────
    if is_greeting(user_query):
        greeting = (
            "Hello! I'm your **CRM AI Assistant** — I can answer any question about your business data.\n\n"
            "Just ask me anything in plain English. For example:\n\n"
            "• **Deals** — *Show all closed won deals this quarter*\n"
            "• **Invoices** — *Which invoices are overdue and by how much?*\n"
            "• **Revenue** — *Total revenue by customer this year*\n"
            "• **Tasks** — *What are Ketul's pending tasks?*\n"
            "• **Analysis** — *Which sales rep has the best conversion rate?*\n"
            "• **Anything** — I understand natural language, so just ask!"
        )
        return _guard_response(greeting, started)

    # ── 4. Guard: blocked query ───────────────────────────────────────────────
    if is_blocked(user_query):
        return _guard_response(
            "This query contains operations that are not permitted. "
            "I only run read-only (SELECT) queries on your CRM data.",
            started,
        )

    # ── 5. Guard: vague query ─────────────────────────────────────────────────
    if is_vague_query(user_query):
        return _guard_response(
            "Your question is a bit vague. Could you be more specific? For example:\n\n"
            "• *How many **open deals** do we have?*\n"
            "• *Show **revenue this month** by customer*\n"
            "• *List **pending tasks** assigned to the sales team*",
            started,
            confidence=0.5,
        )

    LOGGER.info("[%s] Pipeline START | query: %.80s", request_id, user_query)

    # ── 6. Schema pre-warm (runs once, cached) ────────────────────────────────
    _warm_schema(agent)

    # ── 7. Memory context ─────────────────────────────────────────────────────
    mem_context = memory.get_context(user_query) if memory else ""

    # ── 8. SQL Agent — LLM generates SQL, executes, synthesizes ──────────────
    with Timer("sql_agent") as sa_timer:
        try:
            result = sql_agent.run(user_query, agent, memory_context=mem_context)
        except Exception as exc:
            LOGGER.error("[%s] sql_agent raised: %s", request_id, exc, exc_info=True)
            result = None

    LOGGER.info("[%s] sql_agent: %.0fms", request_id, sa_timer.elapsed_ms)

    if result is not None:
        if memory:
            memory.update(original_query, result)

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        result["latency_ms"] = total_ms
        result["answer"] = format_final_answer(
            result.get("answer", ""),
            result.get("tables_used", []),
        )
        LOGGER.info(
            "[%s] Pipeline DONE | %.0fms | layer=%s | tables=%s",
            request_id, total_ms, result.get("layer"), result.get("tables_used"),
        )
        return result

    # ── 9. Fallback — all attempts failed ─────────────────────────────────────
    LOGGER.error("[%s] Pipeline FAILED — all layers exhausted", request_id)
    return {
        "answer": (
            "I wasn't able to process your request. Please try:\n\n"
            "• **Rephrasing** your question more specifically\n"
            "• Checking that the data you're asking about exists in the system\n"
            "• Breaking complex questions into simpler parts"
        ),
        "latency_ms":  round((time.perf_counter() - started) * 1000, 2),
        "confidence":  0.0,
        "tables_used": [],
        "sql_queries": [],
        "layer":       "fallback",
    }
