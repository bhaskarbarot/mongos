"""agent.py — LangChain agent setup + compatibility layer.

All business logic lives in pipeline/*. This file ONLY handles:
  • LangChain AgentExecutor creation (needed for SQL tool access)
  • Schema relationship discovery at startup
  • Agent health check (DB connectivity, tool availability)
  • Public API re-exports for app.py

Public API:
    get_sql_agent(db)                          -> AgentExecutor
    run_agent_query(agent, query, memory=None) -> Dict[str, Any]
    ConversationMemory                         (re-exported class)

Architecture:
    app.py
      └─ agent.py          ← you are here (setup + delegation)
           └─ pipeline/
                ├─ main.py        (orchestrator)
                ├─ fast_path.py   (rule-based, <300ms)
                ├─ classifier.py  (SIMPLE / COMPLEX)
                ├─ text2sql.py    (Ollama NL→SQL)
                ├─ decomposer.py  (Groq/Ollama query splitting)
                ├─ executor.py    (parallel sub-query runner)
                ├─ synthesizer.py (Groq/Ollama final answer)
                ├─ schema.py      (registry, SQL runner, caching)
                └─ utils.py       (shared helpers)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

from config import settings
from prompt import build_system_prompt
from pipeline.schema import (
    discover_schema_links,
    schema_links_prompt,
    get_table_names,
    build_text2sql_schema,
)
import pipeline.main as _pipeline

# ── Re-export for backward compatibility (app.py imports from here) ────────────
ConversationMemory = _pipeline.ConversationMemory

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_agent(
    db,
    llm,
    schema_hints: str = "",
    max_execution_time: int = 10,
):
    """Build a LangChain SQL AgentExecutor with configured tools and prompt.

    Args:
        db:                  LangChain SQLDatabase instance
        llm:                 LLM instance for the agent (used by toolkit)
        schema_hints:        Cross-table relationship hints for the system prompt
        max_execution_time:  Hard timeout for agent execution (seconds)

    Returns:
        Configured AgentExecutor with sql_db_query, sql_db_list_tables tools
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        handle_parsing_errors=True,
        prefix=build_system_prompt(schema_hints=schema_hints),
        max_iterations=3,
        max_execution_time=max_execution_time,
        early_stopping_method="force",
        agent_executor_kwargs={
            "return_intermediate_steps": True,
            "handle_parsing_errors":     True,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _verify_agent_tools(agent) -> bool:
    """Verify the agent has the required SQL tools attached."""
    tool_names = {getattr(t, "name", "") for t in getattr(agent, "tools", [])}
    required   = {"sql_db_query", "sql_db_list_tables"}
    missing    = required - tool_names

    if missing:
        LOGGER.error("Agent missing required tools: %s (has: %s)", missing, tool_names)
        return False

    LOGGER.debug("Agent tools verified: %s", tool_names)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: AGENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_sql_agent(db):
    """Build and return a production-ready LangChain AgentExecutor.

    Startup sequence:
      1. Validate langchain-ollama is installed
      2. Create a temporary agent for schema discovery
      3. Discover cross-table FK relationships (cached in pipeline.schema)
      4. Pre-warm text2sql schema cache
      5. Build final agent with schema hints in system prompt
      6. Verify all required tools are attached
      7. Log configuration summary

    Args:
        db: LangChain SQLDatabase instance (connected to PostgreSQL)

    Returns:
        Fully configured AgentExecutor ready for pipeline.main.run()

    Raises:
        RuntimeError: If langchain-ollama is not installed
    """
    t_start = time.perf_counter()

    # ── 1. Dependency check ────────────────────────────────────────────────
    if OllamaLLM is None:
        raise RuntimeError(
            "langchain-ollama is not installed. "
            "Run: pip install langchain-ollama"
        )

    # ── 2. Create LLM instance ────────────────────────────────────────────
    llm = OllamaLLM(
        model=settings.ollama_primary_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    # ── 3. Schema discovery (temp agent with short timeout) ────────────────
    LOGGER.info("Starting schema discovery...")
    temp_agent = _make_agent(db, llm, max_execution_time=5)

    links      = discover_schema_links(temp_agent)
    hints      = schema_links_prompt(links)
    link_count = sum(len(v) for v in links.values())
    LOGGER.info("Schema discovery: %d relationships found", link_count)

    # ── 4. Pre-warm text2sql schema cache ──────────────────────────────────
    tables = get_table_names(temp_agent)
    build_text2sql_schema(temp_agent)
    LOGGER.info("Schema cache warmed: %d tables", len(tables))

    # ── 5. Build production agent with schema context ──────────────────────
    agent = _make_agent(db, llm, schema_hints=hints, max_execution_time=8)

    # ── 6. Verify tools ───────────────────────────────────────────────────
    if not _verify_agent_tools(agent):
        LOGGER.warning(
            "Agent tool verification failed — pipeline may not work correctly"
        )

    # ── 7. Log startup summary ────────────────────────────────────────────
    startup_ms = round((time.perf_counter() - t_start) * 1000)
    LOGGER.info("╔══════════════════════════════════════════════╗")
    LOGGER.info("║  SQL Chatbot Agent Ready                     ║")
    LOGGER.info("╠══════════════════════════════════════════════╣")
    LOGGER.info("║  Tables       : %-28s ║", f"{len(tables)} discovered")
    LOGGER.info("║  FK Links     : %-28s ║", f"{link_count} relationships")
    LOGGER.info("║  Primary LLM  : %-28s ║", settings.ollama_primary_model)
    LOGGER.info("║  Fallback LLM : %-28s ║", settings.ollama_fallback_model)
    LOGGER.info("║  Reasoning    : %-28s ║",
                settings.groq_decompose_model if settings.groq_api_key
                else settings.ollama_reasoning_model)
    LOGGER.info("║  Groq API     : %-28s ║",
                "connected" if settings.groq_api_key else "not configured")
    LOGGER.info("║  Startup      : %-28s ║", f"{startup_ms}ms")
    LOGGER.info("╚══════════════════════════════════════════════╝")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: QUERY ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_query(
    agent,
    user_query: str,
    memory: Optional[ConversationMemory] = None,
    request_id: str = "",
) -> Dict[str, Any]:
    """Execute a user query through the full pipeline.

    This is the single entry point called by app.py.
    All logic is delegated to pipeline.main.run().

    Args:
        agent:      AgentExecutor from get_sql_agent()
        user_query: Raw user input string
        memory:     Optional ConversationMemory for session context
        request_id: Correlation ID for log tracing (optional)

    Returns:
        Dict with: answer, latency_ms, confidence, tables_used,
                   sql_queries, layer, cached (optional)
    """
    return _pipeline.run(agent, user_query, memory, request_id=request_id)