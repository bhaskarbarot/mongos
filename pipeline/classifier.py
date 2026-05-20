"""classifier.py — Agent-aware query classifier.

Two-layer classification:

  Layer 1 — Pre-classifier (instant, no LLM, regex rules)
    Catches clear-cut COMPLEX and MEDIUM queries before spending an LLM call.
    Based on agent capability specs: if a query type can ONLY be answered by a
    specific agent, route it there immediately.

  Layer 2 — LLM classifier (Groq 8b → Gemini flash-lite → Ollama 3b)
    Handles ambiguous queries the regex cannot confidently decide.
    LLM is given full agent capability map so it routes by both complexity
    AND agent specification.

Agent capability reference used by both layers:

  SIMPLE  — 1 SQL, < 5s
    • Counts, lists, filters on 1-2 tables
    • Ordinal lookups (1st, last N, 2nd)
    • Entity name search (ILIKE)
    • Basic aggregations per dimension (by stage, by currency, by status)
    • Single date-range filter (this year, last month)
    • Named lookups (invoice number, sales order, company name)

  MEDIUM  — 2-3 SQL sub-queries, < 15s
    • 2-period comparisons (this month vs last month)
    • Derived metrics: win rate %, conversion rate %, growth %
    • Cross-entity analysis (companies + invoices + overdue)
    • Rep/user performance (single metric)
    • Target vs achieved (users + targets + sales join)
    • Risk queries (deals stuck, customers no business in N months)
    • Funnel stage breakdown with counts + amounts

  COMPLEX — 4-8 SQL sub-queries, < 30s
    • KPI dashboards / KPI reports (revenue + deals + tasks + targets combined)
    • Executive summaries / business health reports
    • Full leaderboards (multiple reps, multiple metrics)
    • 360 company view (deals + invoices + tasks + contacts + activity)
    • Trend analysis across 3+ time periods
    • Anomaly / pattern detection
    • Statistical analysis (growth rates, outliers, std dev)
    • Multi-section reports combining pipeline + revenue + people + targets
    • Top performing analysis with monthly breakdowns

Public API:
    classify(query: str) -> dict
        Returns: {
          type: "SIMPLE" | "MEDIUM" | "COMPLEX",
          reason: str,
          tables_needed: list[str],
          requires_calculation: bool,
          requires_multi_period: bool,
          routed_by: "pre_classifier" | "llm",
        }
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

import os

from pipeline.db_schema import get_all_crm_tables
from pipeline.llm import call as llm_call

# Read once at import time; respects runtime .env reload via dotenv
_PRE_CLASSIFIER_ENABLED = os.getenv("PRE_CLASSIFIER_ENABLED", "true").strip().lower() != "false"

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (given to the classifying LLM)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PRE-CLASSIFIER  (Layer 1 — instant, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that ALWAYS go to COMPLEX — these queries are ONLY handled well by
# the complex agent (10-node LangGraph with stats, trend, 70b synthesis).
_FORCE_COMPLEX: List[re.Pattern] = [
    # KPI / dashboard
    re.compile(r"\bkpi\s*(report|summary|dashboard)?\b",                        re.I),
    re.compile(r"\b(executive|business)\s*(summary|health|report|overview)\b",  re.I),
    re.compile(r"\bfull\s*(dashboard|report|breakdown|leaderboard)\b",           re.I),
    re.compile(r"\bdashboard\b",                                                 re.I),
    # Leaderboard
    re.compile(r"\bleaderboard\b",                                               re.I),
    re.compile(r"\ball\s+metrics\b",                                             re.I),
    re.compile(r"\bperformance\s+(report|review|analysis|summary)\b",            re.I),
    # 360 / deep reports
    re.compile(r"\b360\s*(view|report|degree|analysis)?\b",                     re.I),
    re.compile(r"\b(deep|full|complete)\s*(analysis|report|summary|review)\b",  re.I),
    # Trend (3+ periods)
    re.compile(r"\btrend\s*(analysis|over|across|by\s+month|by\s+quarter)?\b",  re.I),
    re.compile(r"\b(monthly|quarterly|yearly)\s+trend\b",                        re.I),
    re.compile(r"\b(last\s+[3-9]|past\s+[3-9])\s+(month|quarter|year)s?\b",   re.I),
    # Anomaly / pattern
    re.compile(r"\b(anomaly|anomalies|outlier|pattern\s+find|root\s+cause)\b",  re.I),
    re.compile(r"\bwhy\s+is\s+.*(drop|fall|declin|low|down)\b",                 re.I),
    # Statistical analysis
    re.compile(r"\b(statistical|statistics|std\s*dev|growth\s+rate)\b",         re.I),
    # Todays / yesterdays status (multi-table activity summary)
    re.compile(r"\b(today|yesterday)(s|\'s)?\s+status\b",                       re.I),
    # Top performing with monthly breakdown
    re.compile(r"\btop\s+performing.*(monthly|trend|breakdown)\b",              re.I),
    # Multi-section reports
    re.compile(r"\b(pipeline\s*\+\s*revenue|all\s+department|full\s+pipeline)\b", re.I),
]

# Patterns that ALWAYS go to MEDIUM — these need multi-table calculation
# that simple agent cannot do reliably.
_FORCE_MEDIUM: List[re.Pattern] = [
    # Period comparisons (2 periods)
    re.compile(r"\bthis\s+(month|year|quarter)\s+(vs|versus|compared?\s+to|and)\s+"
               r"last\s+(month|year|quarter)\b",                                 re.I),
    re.compile(r"\bcompare\s+.*(revenue|sales|deal|invoice).*(month|year|quarter)\b", re.I),
    # Derived ratio/percentage metrics
    re.compile(r"\bwin\s+rate\b",                                                re.I),
    re.compile(r"\bconversion\s+rate\b",                                         re.I),
    re.compile(r"\bgrowth\s+(rate|percent|%)\b",                                 re.I),
    # Target vs achieved (requires users+targets+sales join)
    re.compile(r"\btarget\s+(vs|versus|vs\.|and)\s+(achiev|actual|real)\b",     re.I),
    re.compile(r"\bachiev(ed|ement)\s*(vs|versus|and)\s+target\b",              re.I),
    # Cross-entity risk analysis
    re.compile(r"\bcustomers?\s+(not|who\s+have\s+not)\s+given\b",              re.I),
    re.compile(r"\bno\s+business\s+in\s+\d+\s+(month|week|day)\b",              re.I),
    re.compile(r"\bhigh\s+activity.*(weak|low)\s+payment\b",                    re.I),
    # Rep-level analysis (single metric across reps)
    re.compile(r"\bwhich\s+(sales\s+rep|rep|owner).*(best|most|highest|lowest|top)\b", re.I),
    re.compile(r"\b(sales\s+rep|rep\s+performance|owner\s+performance)\b",      re.I),
    # Funnel performance (multi-stage analysis)
    re.compile(r"\bfunnel\s+performance\b",                                     re.I),
    # Deals stuck for N days
    re.compile(r"\b(stuck|not\s+moved?|stagnant).*(deal|pipeline)\b",           re.I),
    re.compile(r"\bdeal.*not\s+moved?\s+in\s+\d+\s+day\b",                     re.I),
]


def _pre_classify(query: str) -> Optional[Dict]:
    """Layer 1: instant regex routing — no LLM needed for clear-cut cases.

    Returns a full classification dict if the pattern is decisive,
    or None to let the LLM handle it.
    """
    # Check COMPLEX first (higher priority)
    for pat in _FORCE_COMPLEX:
        if pat.search(query):
            return {
                "type":                  "COMPLEX",
                "reason":                f"Pre-classifier: matched complex pattern '{pat.pattern[:40]}' — requires full report/stats engine",
                "tables_needed":         [],
                "requires_calculation":  True,
                "requires_multi_period": True,
                "routed_by":             "pre_classifier",
            }

    # Check MEDIUM
    for pat in _FORCE_MEDIUM:
        if pat.search(query):
            return {
                "type":                  "MEDIUM",
                "reason":                f"Pre-classifier: matched medium pattern '{pat.pattern[:40]}' — requires cross-table calculation",
                "tables_needed":         [],
                "requires_calculation":  True,
                "requires_multi_period": False,
                "routed_by":             "pre_classifier",
            }

    return None  # let LLM decide


# ══════════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT  (Layer 2 — enhanced with agent capability map)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an agent-aware CRM query classifier. Route each query to the RIGHT agent
based on BOTH complexity AND what each agent is designed to do.

OUTPUT RULE: Respond with ONLY a valid JSON object. No text outside the JSON.
Format:
{
  "type": "SIMPLE" | "MEDIUM" | "COMPLEX",
  "reason": "<one concise sentence: complexity + which agent capability this maps to>",
  "tables_needed": ["<table1>", "<table2>"],
  "requires_calculation": true | false,
  "requires_multi_period": true | false
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT CAPABILITY MAP — route to the agent that CAN do the work:

┌─ SIMPLE AGENT (< 5s, 1 SQL query, CrewAI) ──────────────────────────────────┐
│ USE FOR:                                                                      │
│  • Counts: "how many deals", "total contacts", "count invoices"              │
│  • Lists/filters: "open deals", "pending tasks", "paid invoices in INR"      │
│  • Ordinal: "1st invoice", "last 5 contacts", "2nd deal"                     │
│  • Entity search: "details of Wiegand LLC", "deals by ketul", "find ELSN"    │
│  • Aggregation per dimension: "deals by stage", "revenue by currency"        │
│  • Single date filter: "revenue this year", "deals lost in July"             │
│  • Named lookups: "status of SO00080", "invoice ELSN/2026/020"               │
│  • Simple status checks: "overdue invoices", "draft sales orders"            │
│                                                                               │
│ DO NOT SEND TO SIMPLE:                                                        │
│  ✗ KPI reports / dashboards / executive summaries                             │
│  ✗ Leaderboards with multiple metrics                                         │
│  ✗ Trend analysis across 3+ periods                                           │
│  ✗ Win rate %, conversion %, growth % calculations                            │
│  ✗ Target vs achieved (multi-table join + calculation)                        │
└───────────────────────────────────────────────────────────────────────────────┘

┌─ MEDIUM AGENT (< 15s, 2-3 SQL queries, LangGraph) ─────────────────────────┐
│ USE FOR:                                                                      │
│  • 2-period comparisons: "this month vs last month revenue"                  │
│  • Derived metrics: "win rate", "conversion rate %", "growth %"              │
│  • Cross-entity analysis: "companies with overdue invoices"                  │
│  • Target vs achieved: "target vs achieved for all users"                    │
│  • Risk analysis: "customers not given business in 3 months"                 │
│  • Rep performance (single metric): "which rep closed most deals"            │
│  • Funnel performance: stage-by-stage breakdown with conversion              │
│  • Pipeline health: deals stuck, not moved in N days                         │
│                                                                               │
│ DO NOT SEND TO MEDIUM:                                                        │
│  ✗ Full KPI dashboards (→ COMPLEX)                                            │
│  ✗ Trend analysis 3+ periods (→ COMPLEX)                                     │
│  ✗ Leaderboards with all metrics (→ COMPLEX)                                 │
└───────────────────────────────────────────────────────────────────────────────┘

┌─ COMPLEX AGENT (< 30s, 4-8 SQL queries, LangGraph + Stats) ────────────────┐
│ USE FOR:                                                                      │
│  • KPI report / KPI dashboard (revenue + deals + tasks + targets together)   │
│  • Executive summary / business health report                                │
│  • Full leaderboard (multiple reps × multiple metrics)                       │
│  • 360 company view (deals + invoices + tasks + contacts + activity)         │
│  • Trend analysis across 3+ months / quarters / years                        │
│  • Anomaly detection / pattern finding / "why is X dropping"                 │
│  • Statistical analysis (growth rates, outliers, rankings)                   │
│  • Multi-section report (pipeline + revenue + team + targets combined)       │
│  • Top performing analysis with monthly trend breakdown                      │
│  • Today/yesterday status (multi-table activity summary)                     │
└───────────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING EXAMPLES:

"how many deals are there"                    → SIMPLE  (count, deals only)
"list all open deals"                         → SIMPLE  (filter, deals only)
"total revenue this year"                     → SIMPLE  (SUM on invoices)
"give me 1st invoice details"                 → SIMPLE  (ordinal lookup)
"details of Wiegand LLC Systems"              → SIMPLE  (ILIKE entity search)
"pending invoices"                            → SIMPLE  (filter on invoices)
"deals by stage"                              → SIMPLE  (GROUP BY, 1 table)
"top 10 customers by revenue"                 → SIMPLE  (invoices+companies JOIN)
"overdue invoice aging by company"            → SIMPLE  (invoices+companies, filter+sum)

"revenue this month vs last month"            → MEDIUM  (2-period comparison)
"win rate this quarter"                       → MEDIUM  (derived metric, deals)
"target vs achieved for all users"            → MEDIUM  (users+targets+sales join)
"which sales rep closed most deals"           → MEDIUM  (cross-rep, single metric)
"customers not given business in 3 months"    → MEDIUM  (risk analysis)
"funnel performance by stage"                 → MEDIUM  (multi-stage analysis)
"high activity companies with weak payment"   → MEDIUM  (cross-entity)

"give me kpi report"                          → COMPLEX (multi-KPI dashboard)
"executive summary of business health"        → COMPLEX (full report)
"full sales leaderboard with all metrics"     → COMPLEX (multi-rep, multi-metric)
"top performing sales owners monthly trend"   → COMPLEX (trend + stats)
"360 view of our top company"                 → COMPLEX (all tables)
"why is conversion dropping last 3 months"    → COMPLEX (trend + anomaly)
"give me todays status"                       → COMPLEX (multi-table activity)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT:
  • Short query ≠ SIMPLE  ("KPI" = 1 word but → COMPLEX)
  • Long query ≠ COMPLEX  ("give me all deals with year wise stages count" → SIMPLE)
  • A "summary" of a specific entity (deals, contacts) = SIMPLE
  • A "summary" of the business / team / performance = COMPLEX
  • "report" on a specific filter = SIMPLE; "report" on the whole business = COMPLEX
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

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()

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
        "type":                  intent_type,
        "reason":                str(parsed.get("reason", "LLM classified")).strip(),
        "tables_needed":         [str(t) for t in parsed.get("tables_needed", [])],
        "requires_calculation":  bool(parsed.get("requires_calculation", False)),
        "requires_multi_period": bool(parsed.get("requires_multi_period", False)),
        "routed_by":             "llm",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def classify(query: str) -> Dict:
    """Classify a CRM query as SIMPLE, MEDIUM, or COMPLEX.

    Two-layer routing:
      Layer 1 — pre_classifier: instant regex, no LLM, catches clear-cut cases
      Layer 2 — LLM (Groq 8b → Gemini flash-lite → Ollama 3b) for ambiguous queries

    Returns:
        {type, reason, tables_needed, requires_calculation,
         requires_multi_period, routed_by}
    """
    if not query or not query.strip():
        return {
            "type":                  "SIMPLE",
            "reason":                "Empty query — defaulting to SIMPLE",
            "tables_needed":         [],
            "requires_calculation":  False,
            "requires_multi_period": False,
            "routed_by":             "pre_classifier",
        }

    # ── Layer 1: pre-classifier (0ms, no LLM) ────────────────────────────────
    if _PRE_CLASSIFIER_ENABLED:
        pre = _pre_classify(query)
        if pre:
            LOGGER.info(
                "Classifier [PRE] → %s | %s | query: %.60s",
                pre["type"], pre["reason"][:60], query,
            )
            return pre
    else:
        LOGGER.info("Classifier [PRE] disabled via PRE_CLASSIFIER_ENABLED=false — going straight to LLM")

    # ── Layer 2: LLM classifier ───────────────────────────────────────────────
    available_tables: List[str] = []
    try:
        available_tables = get_all_crm_tables()
    except Exception as exc:
        LOGGER.debug("Classifier: could not load table list: %s", exc)

    user_prompt = _build_user_prompt(query, available_tables)
    raw = llm_call("classify", _SYSTEM_PROMPT, user_prompt, max_tokens=200)

    if raw:
        result = _parse_llm_response(raw)
        if result:
            LOGGER.info(
                "Classifier [LLM] → %s | %s | query: %.60s",
                result["type"], result["reason"][:60], query,
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
        "routed_by":             "fallback",
    }
