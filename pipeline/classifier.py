"""classifier.py — Hybrid SIMPLE / COMPLEX intent classifier.

Stage 1: Rule-based pre-screen (0ms) — catches obvious COMPLEX patterns fast.
Stage 2: Ollama LLM (qwen2.5:5b) — accurate semantic classification for
         everything that passes the rule screen.

Output:
    {"type": "SIMPLE" | "COMPLEX", "reason": "<short explanation>"}

SIMPLE  = single intent — one list, one count, one aggregation, one lookup
COMPLEX = multiple intents, cross-entity analysis, KPIs, comparisons,
          report-style requests, anything needing decomposition + synthesis
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Dict

from pipeline.llm import call as llm_call
from pipeline.utils import normalize_text

LOGGER = logging.getLogger("sql_chatbot")

# ── Ollama config ──────────────────────────────────────────────────────────────
_OLLAMA_BASE_URL   = "http://localhost:11434"
_CLASSIFY_MODEL    = "qwen2.5:5b"
_CLASSIFY_TIMEOUT  = 12          # seconds — fast model, should respond well within this
_CLASSIFY_TEMP     = 0.0         # deterministic


# ── Shared system prompt for the LLM classifier ────────────────────────────────
_SYSTEM_PROMPT = """You are a CRM query intent classifier. Your ONLY job is to decide
if a user query needs a SIMPLE database lookup or COMPLEX multi-step reasoning.

OUTPUT RULE: Respond with ONLY valid JSON. No explanation outside JSON.
Format: {"type": "SIMPLE" | "COMPLEX", "reason": "<one sentence>"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIMPLE = ALL of these are true:
  • Single table or entity focus
  • One action: list / count / total / filter / show / find / get
  • No cross-entity analysis
  • No comparison between time periods or entities
  • No report generation
  • No trend analysis
  • No multi-step reasoning required

Examples of SIMPLE:
  "give me all closed won deals"              → list from deals table, one filter
  "how many invoices this month"              → one count, one filter
  "show all contacts for TechCorp"            → one lookup
  "list all pending tasks"                    → one filter
  "total revenue this year"                   → one aggregation
  "get all users"                             → simple list
  "what is the status of invoice INV-001"     → single record lookup
  "show me top 10 customers by revenue"       → one sorted aggregation
  "give me all companies in USA"              → one filter list
  "how many open deals do we have"            → one count

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEX = ANY of these are true:
  • Multiple entities needed (e.g. deals + invoices + users together)
  • Comparison between time periods ("last month vs this month")
  • Report with multiple sections or KPIs in one query
  • Trend / growth analysis
  • Cross-entity analysis ("which users have pending tasks and overdue deals")
  • Requires computed insights beyond raw data (growth %, conversion rate, etc.)
  • Ambiguous intent needing AI interpretation
  • Multi-step: "find X, then calculate Y based on X"
  • Executive summary or overview requests
  • "with report of that" / "give me analysis" / "full breakdown"

Examples of COMPLEX:
  "give me all closed lost deals with report of that"  → list + analysis report
  "which user have task is pending"                    → cross-entity: users + tasks + analysis
  "compare this month vs last month revenue"           → two time periods
  "give me executive summary of pipeline"              → multi-KPI report
  "which products are selling best and why"            → analysis + reasoning
  "show conversion rate and revenue trend this year"   → multiple KPIs + trend
  "which sales reps are underperforming vs target"     → targets + invoices + comparison
  "pipeline health check"                              → multi-entity overview
  "give me full customer 360 for TechCorp"             → many entities joined
  "how are we doing this quarter"                      → vague, needs AI interpretation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL TRAPS (these LOOK simple but are COMPLEX):
  • "which user/rep has [condition]" → needs JOIN + grouping = COMPLEX
  • "tell me who has [X] pending/overdue" → cross-entity analysis = COMPLEX
  • "performance of [team/user]" → needs targets + invoices compared = COMPLEX
  • "any [risks/issues/problems]" → vague, needs AI judgment = COMPLEX
  • "give me report of [anything]" → always COMPLEX
  • "full details of [entity]" with analysis = COMPLEX
  • queries with "and also", "as well as", "along with" for different entity types = COMPLEX
"""


def _call_llm_classify(query: str) -> Dict[str, str] | None:
    """
    Call LLM for semantic classification via llm.py router.
    Chain: Groq 8b (fast, ~0.5s) → Gemini → Ollama 1.5b (local fallback)
    Returns parsed dict or None if all providers fail / parse fails.
    """
    user_msg = f'Classify this CRM query:\n\n"{query}"\n\nRespond ONLY with JSON.'
    raw = llm_call("classify", _SYSTEM_PROMPT, user_msg, max_tokens=120)
    if not raw:
        return None

    try:
        content = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        content = re.sub(r"\s*```$", "", content).strip()

        parsed      = json.loads(content)
        intent_type = parsed.get("type", "").upper().strip()
        reason      = parsed.get("reason", "LLM classified").strip()

        if intent_type not in ("SIMPLE", "COMPLEX"):
            LOGGER.warning("Classifier LLM returned unexpected type=%s", intent_type)
            return None

        return {"type": intent_type, "reason": reason}

    except json.JSONDecodeError as exc:
        LOGGER.warning("Classifier LLM JSON parse failed: %s | raw: %.100s", exc, raw)
        return None


# ── Rule-based pre-screen (catches definitive COMPLEX signals instantly) ────────

def _rule_prescreen(text: str, words: list[str]) -> Dict[str, str] | None:
    """
    Fast rule-based check. Only returns a result for HIGH-CONFIDENCE COMPLEX
    signals that we'd never want to mis-classify. Returns None to let LLM decide
    for everything ambiguous.

    This is NOT a replacement for the LLM — it's only an early exit for
    patterns that are unambiguously complex.
    """

    # Vague / open-ended semantic queries — LLM interpretation needed
    if re.search(
        r"^(how are we|is (the )?business|any (risks?|issues?|problems?)|"
        r"what.?s the (overall|status|health)|how.?s (the )?business|"
        r"are we (growing|improving|doing well)|give me insights?|"
        r"what should (i|we) (focus|worry|look)|overall (health|status|performance))\b",
        text,
    ):
        return {
            "type":   "COMPLEX",
            "reason": "vague open-ended query — needs AI to interpret intent and pull relevant KPIs",
        }

    # Explicit report / analysis keywords
    if re.search(
        r"\b(report of (that|this)|full report|give.*report|analysis|"
        r"executive summary|pipeline health|360.?view|full breakdown|"
        r"performance review|kpi dashboard)\b",
        text,
    ):
        return {
            "type":   "COMPLEX",
            "reason": "explicit report/analysis request — multi-section output required",
        }

    # Direct time-period comparison
    if re.search(
        r"\b(vs|versus|compared? to|compare|month.?on.?month|year.?on.?year"
        r"|last.*vs.*this|this.*vs.*last|growth rate|trend over)\b",
        text,
    ):
        return {
            "type":   "COMPLEX",
            "reason": "comparison / trend query — requires multiple result sets",
        }

    # "which user/rep has X" — always cross-entity
    if re.search(
        r"\b(which (user|rep|person|employee|member)|who (has|have|is|are))\b.*"
        r"\b(task|deal|target|overdue|pending|assigned|performance)\b",
        text,
    ):
        return {
            "type":   "COMPLEX",
            "reason": "cross-entity user analysis — requires joins across users + task/deal tables",
        }

    # "and also" / "as well as" for different data requests
    if re.search(
        r"\band\s+(also|give me|show me|tell me|list|what|how many|find)\b", text
    ):
        return {
            "type":   "COMPLEX",
            "reason": "multi-part request — user wants data from multiple sources in one answer",
        }

    return None  # let LLM decide


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(query: str) -> Dict[str, str]:
    """
    Classify a user query as SIMPLE or COMPLEX.

    Pipeline:
      1. Rule pre-screen — catches obvious COMPLEX signals at 0ms
      2. Ollama qwen2.5:5b — accurate semantic classification
      3. Fallback to SIMPLE if Ollama unavailable (safe default — text2sql handles it)

    Args:
        query: Raw user query string

    Returns:
        {"type": "SIMPLE" | "COMPLEX", "reason": "<one sentence explanation>"}
    """
    t     = normalize_text(query)
    words = t.split()

    # Stage 1a — SIMPLE fast-screen (zero LLM cost for obvious single-intent queries)
    # Catches: counts, single-table lists, single aggregations, single filters
    _SIMPLE_PATTERNS = [
        r"^how many \w",                                          # "how many deals"
        r"^(give me|show me?|list|get|fetch|display) all \w",   # "give me all deals"
        r"^(give me|show|list) (closed |open |pending |overdue )?\w",
        r"^total (revenue|sales|amount|invoices?|deals?)",       # "total revenue this month"
        r"^(count|number of) \w",
        r"^(find|search for?) \w+",                              # "find John Smith"
        r"^\w{3,20} (this|last) (month|week|year|quarter)",      # "revenue this month"
        r"^show (me )?(all )?(pending|open|overdue|completed) \w",
        r"^(top|best) \d+ \w+",                                  # "top 5 customers"
        r"^get (all )?\w+ (by|for|in|with) \w+$",               # simple lookups
    ]
    if len(words) <= 8 and any(re.match(p, t) for p in _SIMPLE_PATTERNS):
        LOGGER.info("Classifier SIMPLE fast-screen hit | query: %.60s", query)
        return {"type": "SIMPLE", "reason": "clear single-intent pattern — no LLM needed"}

    # Stage 1b — rule pre-screen for definitive COMPLEX patterns
    rule_result = _rule_prescreen(t, words)
    if rule_result is not None:
        LOGGER.info(
            "Classifier RULE hit → %s | reason: %s | query: %.60s",
            rule_result["type"], rule_result["reason"], query,
        )
        return rule_result

    # Stage 2 — LLM semantic classification (Groq 8b → Gemini → Ollama 1.5b)
    llm_result = _call_llm_classify(query)
    if llm_result is not None:
        LOGGER.info(
            "Classifier LLM → %s | reason: %s | query: %.60s",
            llm_result["type"], llm_result["reason"], query,
        )
        return llm_result

    # Stage 3 — Fallback (all LLMs unavailable)
    LOGGER.warning(
        "Classifier fallback (all LLMs unavailable) → SIMPLE | query: %.60s", query
    )
    return {
        "type":   "SIMPLE",
        "reason": "classifier unavailable — defaulting to simple path (text2sql handles)",
    }