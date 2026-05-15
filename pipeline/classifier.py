"""classifier.py — LLM-driven query intent classifier.

Classifies every incoming query as SIMPLE, MEDIUM, or COMPLEX using a small
LLM (Groq 8b → Gemini flash-lite → Ollama 3b). No hardcoded regex rules for
classification — the LLM understands context and table relationships.

Classification types:
  SIMPLE  — single table, basic filter/count/aggregate, raw data output
  MEDIUM  — 2–3 table joins, or analysis/calculation on top of data,
             or simple period comparison
  COMPLEX — 3+ tables, KPI dashboard, trend/anomaly detection, full reports

Public API:
    classify(query: str) -> dict
        Returns: {
          type: "SIMPLE" | "MEDIUM" | "COMPLEX",
          reason: str,
          tables_needed: list[str],
          requires_calculation: bool,
          requires_multi_period: bool,
        }
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from pipeline.db_schema import get_all_crm_tables
from pipeline.llm import call as llm_call

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (given to the classifying LLM)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a CRM query intent classifier. Your ONLY job is to decide if a query
needs a SIMPLE lookup, MEDIUM analysis, or COMPLEX report.

OUTPUT RULE: Respond with ONLY a valid JSON object. No text outside the JSON.
Format:
{
  "type": "SIMPLE" | "MEDIUM" | "COMPLEX",
  "reason": "<one concise sentence explaining why>",
  "tables_needed": ["<table1>", "<table2>"],
  "requires_calculation": true | false,
  "requires_multi_period": true | false
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION GUIDE:

SIMPLE — ALL of these apply:
  • Targets one main table (may join one lookup table for a name)
  • Single action: list, count, filter, total, find, show
  • No cross-entity analysis or comparison between time periods
  • Answer is raw data — no trends, percentages, or growth rates needed
  Examples:
    "how many deals do we have"               → deals only, COUNT
    "list all open deals"                     → deals only, filter
    "total revenue this year"                 → invoices only, SUM
    "show pending tasks"                      → createtasks only
    "top 10 companies by deal count"          → deals + companies, GROUP BY
    "find contact John Smith"                 → contacts, lookup
    "all active vendors"                      → vendors only

MEDIUM — ANY of these apply:
  • Needs 2–3 tables joined for meaningful analysis
  • Needs LLM to interpret/analyze the data (risk assessment, categorize)
  • Needs comparison between exactly 2 time periods (this month vs last month)
  • Needs a derived metric (win rate, conversion %, growth rate)
  Examples:
    "which sales rep has the best win rate this quarter"   → users+deals+sales
    "show deals that have not moved in 30 days"           → deals + calculation
    "revenue this month compared to last month"           → invoices, 2 periods
    "deals pipeline grouped by owner with department"     → deals+users+departments
    "which companies have overdue invoices"               → companies+invoices

COMPLEX — ANY of these apply:
  • Needs 3+ tables joined together
  • Full KPI dashboard or executive summary with multiple metrics
  • Trend analysis across 3+ time periods
  • Anomaly detection or pattern finding across the dataset
  • Multi-section report (pipeline + revenue + tasks + targets)
  • Deep performance analysis (leaderboard with all metrics)
  • "360 view" or "full report" style queries
  Examples:
    "full sales leaderboard with all metrics"             → users+sales+deals+tasks+targets
    "quarterly performance report"                        → all major tables
    "360 view of company X"                               → companies+deals+invoices+tasks+contacts
    "why is deal conversion dropping last 3 months"       → deals, trend over 3 periods
    "executive summary of business health"                → all tables, multiple KPIs
    "compare performance of all sales reps this year"     → cross-rep, multi-metric

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT NOTES:
  • A query can be SIMPLE even if it's long or uses complex words
  • A very short query can be COMPLEX ("360 view of TechCorp")
  • "report of" or "full breakdown" = COMPLEX
  • "vs" or "compared to" with 2 periods = MEDIUM minimum
  • "trends" over 3+ periods = COMPLEX
  • Always list the specific CRM tables needed in tables_needed
"""


def _build_user_prompt(query: str, available_tables: List[str]) -> str:
    """Build the classification request with live table context."""
    tables_str = ", ".join(available_tables) if available_tables else "unknown"
    return (
        f"Available CRM tables: {tables_str}\n\n"
        f'Query to classify: "{query}"\n\n'
        "Respond ONLY with the JSON object."
    )


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse and validate the LLM's JSON classification response."""
    if not raw:
        return None

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    # Find the JSON object
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        parsed = json.loads(cleaned[start:end])
    except json.JSONDecodeError as exc:
        LOGGER.debug("Classifier JSON parse failed: %s | raw: %.150s", exc, raw)
        return None

    intent_type = str(parsed.get("type", "")).upper().strip()
    if intent_type not in ("SIMPLE", "MEDIUM", "COMPLEX"):
        LOGGER.warning("Classifier returned unexpected type=%s", intent_type)
        return None

    return {
        "type":                 intent_type,
        "reason":               str(parsed.get("reason", "LLM classified")).strip(),
        "tables_needed":        [str(t) for t in parsed.get("tables_needed", [])],
        "requires_calculation": bool(parsed.get("requires_calculation", False)),
        "requires_multi_period": bool(parsed.get("requires_multi_period", False)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def classify(query: str) -> Dict:
    """Classify a CRM query as SIMPLE, MEDIUM, or COMPLEX using an LLM.

    The LLM receives the query plus the live list of CRM tables so it can
    reason correctly about which tables are involved.

    Fallback chain: Groq 8b-instant → Gemini flash-lite → Ollama 3b
    If all LLMs fail, defaults to MEDIUM (safe middle ground).

    Args:
        query: Raw user query string.

    Returns:
        {type, reason, tables_needed, requires_calculation, requires_multi_period}
    """
    if not query or not query.strip():
        return {
            "type":                  "SIMPLE",
            "reason":                "Empty query — defaulting to SIMPLE",
            "tables_needed":         [],
            "requires_calculation":  False,
            "requires_multi_period": False,
        }

    # Get live table list to give the LLM accurate context
    available_tables: List[str] = []
    try:
        available_tables = get_all_crm_tables()
    except Exception as exc:
        LOGGER.debug("Classifier: could not load table list: %s", exc)

    user_prompt = _build_user_prompt(query, available_tables)

    # Call via LLM router: Groq 8b → Gemini → Ollama
    raw = llm_call("classify", _SYSTEM_PROMPT, user_prompt, max_tokens=200)

    if raw:
        result = _parse_llm_response(raw)
        if result:
            LOGGER.info(
                "Classifier → %s | reason: %s | query: %.60s",
                result["type"], result["reason"], query,
            )
            return result
        LOGGER.warning("Classifier: LLM returned unparseable output: %.100s", raw)

    # All LLMs failed — safe default
    LOGGER.warning("Classifier: all LLMs unavailable — defaulting to MEDIUM | query: %.60s", query)
    return {
        "type":                  "MEDIUM",
        "reason":                "Classifier unavailable — defaulting to MEDIUM (safe fallback)",
        "tables_needed":         [],
        "requires_calculation":  False,
        "requires_multi_period": False,
    }
