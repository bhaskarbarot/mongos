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
    # Trend (3+ periods) — only when asking for trend/analysis, NOT simple counts
    re.compile(r"\btrend\s*(analysis|over|across|by\s+month|by\s+quarter)?\b",  re.I),
    re.compile(r"\b(monthly|quarterly|yearly)\s+trend\b",                        re.I),
    # "last N months" only triggers COMPLEX when paired with trend/analysis/compare words
    re.compile(r"\b(last|past)\s+[3-9]\s+(month|quarter|year)s?\b.{0,30}\b(trend|analysis|growth|pattern|anomaly|compare)\b", re.I),
    re.compile(r"\b(trend|analysis|growth|pattern|anomaly|compare)\b.{0,30}\b(last|past)\s+[3-9]\s+(month|quarter|year)s?\b", re.I),
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
    # Deals stuck / pipeline health
    re.compile(r"\b(stuck|not\s+moved?|stagnant).*(deal|pipeline)\b",           re.I),
    re.compile(r"\bdeal.*not\s+moved?\s+in\s+\d+\s+day\b",                     re.I),
    re.compile(r"\bdeal[s]?\s+stuck\b",                                         re.I),
    # Cross-entity risk analysis
    re.compile(r"\bcompan(y|ies)\s+(with|having)\s+(overdue|unpaid|pending)\b", re.I),
    re.compile(r"\boverdue.*(compan|customer|client)\b",                        re.I),
    re.compile(r"\bwhich\s+compan.*(invoice|payment|overdue)\b",                re.I),
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

_SYSTEM_PROMPT = """You are a CRM query classifier. Output ONLY valid JSON, no other text.

JSON format:
{"type":"SIMPLE"|"MEDIUM"|"COMPLEX","reason":"<10 words max>","tables_needed":["t1"],"requires_calculation":true|false,"requires_multi_period":true|false}

DECISION RULE — ask yourself these 3 questions in order:

Q1: Does the user want a COMPREHENSIVE REPORT covering multiple business areas?
  YES → COMPLEX (report/dashboard/summary/leaderboard/360/today's status/KPI)

Q2: Does this need a RATIO, COMPARISON, or RISK ANALYSIS across 2+ tables?
  YES → MEDIUM (win rate, month vs month, rep ranking, stuck deals, overdue by company, targets vs actual)
  ALWAYS MEDIUM — these can NEVER be answered by 1 SQL:
    • "who achieved targets" / "who hit targets" / "who met targets"
    • "who didn't achieve targets" / "who missed targets" / "who failed targets"
    • Any "target" query that compares target amount vs actual sales → needs targets + sales + users

Q3: Can ONE SQL query answer this?
  YES → SIMPLE (count, list, filter, single metric, lookup, basic date range)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT CAPABILITIES (what each agent actually does):

SIMPLE — 1 SQL query, 1-2 tables, single focused answer
  ✓ Counts: "how many deals", "total contacts"
  ✓ Lists: "show open deals", "list pending tasks"
  ✓ Filters: "paid invoices in INR", "deals by stage"
  ✓ Date range: "revenue this year", "deals last 6 months" (COUNT + date = still 1 SQL)
  ✓ Lookups: "details of company X", "status of SO-080"
  ✓ Simple JOIN: "top 10 customers by revenue" (invoices + companies, 1 SQL)
  ✗ Cannot do: percentages, 2-period compare, risk analysis, reports, trend

MEDIUM — 2-3 SQL queries run in parallel, calculations across tables
  ✓ 2-period compare: "this month vs last month revenue"
  ✓ Derived %: "win rate", "conversion rate", "growth %"
  ✓ Pipeline health: "deals stuck for 30 days", "deals not moved in 2 weeks"
  ✓ Cross-entity risk: "companies with overdue invoices" (which companies? = 2 SQL)
  ✓ Cross-entity risk: "customers no business in 3 months" (2-3 SQL with conditions)
  ✓ Rep ranking: "which rep closed most deals" (deals + users, ranked)
  ✓ Target achievement: "who achieved targets", "who didn't achieve targets",
                         "who hit/missed/failed targets" — ALWAYS MEDIUM (needs targets+sales+users)
  ✓ Target vs actual: "target vs achieved" (targets + sales + users)
  ✓ Funnel: "funnel performance by stage" (multi-stage counts)
  ✗ Cannot do: full dashboards, trend 3+ periods, full leaderboard all metrics

COMPLEX — 4-8 SQL queries, synthesis engine, multi-section reports
  ✓ Report/Dashboard: "deals report", "kpi report", "business health report"
  ✓ Executive summary: everything in one — revenue + deals + tasks + targets
  ✓ Full leaderboard: all reps × all metrics × all time periods
  ✓ Today's/yesterday's status: ALL tables combined (deals+invoices+tasks+sales)
  ✓ 360 view: "360 of top company" = deals+invoices+tasks+contacts+activity
  ✓ Trend 3+ periods: "monthly trend last 6 months", "quarterly breakdown"
  ✓ Anomaly: "why is X dropping last 3 months"
  KEY: user wants EVERYTHING about a topic or MULTI-SECTION report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL BOUNDARIES (common mistakes to avoid):

"who achieved targets" = MEDIUM (needs targets table + sales table comparison — NEVER SIMPLE)
"who didn't achieve targets" = MEDIUM (same — needs targets + sales join — NEVER SIMPLE)
"who hit/missed/failed targets" = MEDIUM (always cross-table: targets vs sales)

"overdue invoices" = SIMPLE (filter 1 table)
"companies WITH overdue invoices" = MEDIUM (which companies? cross-table risk)

"deals count last 6 months" = SIMPLE (1 SQL count + date filter)
"deals stuck for 30 days" = MEDIUM (pipeline health, needs activity date logic)

"deals report" = COMPLEX (open+won+lost+pipeline value = multi-section)
"list open deals" = SIMPLE (1 SQL filter)

"today's status" = COMPLEX (all tables combined)
"status of invoice X" = SIMPLE (1 record lookup)

"customers no business in 3 months" = MEDIUM (risk: companies + invoices join)
NOT COMPLEX — it's 2-3 SQL but no multi-section narrative needed
"""


def _build_user_prompt(query: str, available_tables: List[str]) -> str:
    """Build the classification request — minimal tokens, just the query."""
    return f'Classify: "{query}"'


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse and validate the LLM's JSON classification response.

    Handles truncated responses (rate-limit cutoffs) by extracting the
    'type' field even if the JSON is incomplete.
    """
    if not raw:
        return None

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    # ── Try full JSON parse first ─────────────────────────────────────────────
    end = cleaned.rfind("}") + 1
    parsed = None
    if end > 0:
        try:
            parsed = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # ── Fallback: extract "type" via regex when JSON is truncated ────────────
    if parsed is None:
        type_match = re.search(r'"type"\s*:\s*"(SIMPLE|MEDIUM|COMPLEX)"', cleaned, re.I)
        if type_match:
            intent_type = type_match.group(1).upper()
            LOGGER.info("Classifier: extracted type from truncated JSON: %s", intent_type)
            return {
                "type":                  intent_type,
                "reason":                "LLM classified (truncated response)",
                "tables_needed":         [],
                "requires_calculation":  intent_type != "SIMPLE",
                "requires_multi_period": intent_type == "COMPLEX",
                "routed_by":             "llm",
            }
        LOGGER.debug("Classifier JSON parse failed — no type found | raw: %.150s", raw)
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
    raw = llm_call("classify", _SYSTEM_PROMPT, user_prompt, max_tokens=120)

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
