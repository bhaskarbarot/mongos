"""decomposer.py — LLM-based query decomposition engine.

Breaks a complex user query into atomic, SQL-friendly sub-queries.

Priority order:
  1. Groq API   — fast (~0.3–0.8s), used when GROQ_API_KEY is set
  2. Ollama     — local fallback (~5–15s), uses reasoning model
  3. Graceful   — returns original query as single sub-query (never crashes)

Public API:
    decompose(query, table_names, schema_context) -> List[{"sub_query": str, "intent": str}]
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Dict, List, Optional

from config import settings
from pipeline.llm import call as llm_call
from pipeline.utils import normalize_text

LOGGER = logging.getLogger("sql_chatbot")

_MAX_SUB_QUERIES = 6   # hard cap — prevents over-decomposition


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert CRM query decomposition engine for a PostgreSQL database.

Your ONLY job: break a complex natural language query into the MINIMUM number of
atomic, self-contained, SQL-friendly sub-queries needed to answer the original question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES (STRICTLY FOLLOW):
  • Respond with ONLY a valid JSON array. No text before or after.
  • Each element: {"sub_query": "<natural language>", "intent": "<one word verb>"}
  • Maximum 6 sub-queries. If fewer suffice, use fewer.
  • Each sub_query must be a standalone question answerable by ONE SQL query.
  • No sub-query should depend on results from another sub-query.
  • Write sub-queries in plain English, NOT SQL.
  • Keep each sub_query concise (under 20 words).
  • intent = one of: count | list | sum | average | filter | compare | lookup | rank | trend

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECOMPOSITION PRINCIPLES:

1. PARALLEL INDEPENDENCE — every sub-query must be answerable independently.
   BAD:  "get deals, then find their owners" (dependent)
   GOOD: "list all deals with owner names" (one query with join)

2. MINIMUM DECOMPOSITION — don't split what can be one query.
   BAD:  ["count all deals", "count won deals", "count lost deals", "count open deals"]
   GOOD: ["count deals grouped by status"]

3. TIME SYMMETRY — for comparisons, create parallel time-period sub-queries.
   Input: "compare revenue this month vs last month"
   Output: [
     {"sub_query": "total revenue this month", "intent": "sum"},
     {"sub_query": "total revenue last month", "intent": "sum"}
   ]

4. ENTITY ISOLATION — split only when different tables/entities are involved.
   Input: "show pipeline and overdue tasks and target vs achieved for all reps"
   Output: [
     {"sub_query": "list open deals grouped by stage with total value", "intent": "list"},
     {"sub_query": "list overdue tasks grouped by user", "intent": "list"},
     {"sub_query": "target vs achieved for all users this year", "intent": "compare"}
   ]

5. PRESERVE CONTEXT — each sub_query must include enough context to run alone.
   BAD:  "get the count"          (what count?)
   GOOD: "count all open deals"   (fully self-contained)

6. NO HALLUCINATED FIELDS — only use concepts from the available tables list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRM DATABASE SCHEMA (PostgreSQL with JSONB document column):
  IMPORTANT: All field access uses document->>'fieldName' syntax.
  Primary key for all tables: _id (text)
  JOIN pattern: table1.document->>'fieldRef' = table2._id

  KEY TABLES & FIELDS:
  • deals       : _id, document->>'name', document->>'stage', document->>'owner'
                  document->>'grand_total_in_usd', document->>'closeDate'
                  document->>'deleted', document->>'dealWonAt', document->>'dealLostAt'
                  Open deals = dealWonAt IS NULL AND dealLostAt IS NULL
                  Won deals  = dealWonAt IS NOT NULL  |  Lost = dealLostAt IS NOT NULL

  • invoices     : _id, document->>'invoice_number', document->>'payment_status'
                  document->>'grandtotal_in_usd', document->>'grand_total'
                  document->>'invoice_date', document->>'due_date', document->>'company'
                  document->>'deleted', document->>'currency'
                  Revenue = paid invoices (payment_status='paid') using grandtotal_in_usd

  • sales        : _id, document->>'sales_number', document->>'status'
                  document->>'grand_total_in_usd', document->>'sales_date'
                  document->>'salesOwner', document->>'company', document->>'deleted'
                  Confirmed sales = status='Confirm'

  • companies    : _id, document->>'companyName', document->>'lifecycleStage'
                  document->>'leadStatus', document->>'industry', document->>'region'
                  document->>'deleted', document->>'companyOwner'

  • contacts     : _id, document->>'firstName', document->>'lastName'
                  document->>'email', document->>'jobTitle', document->>'company'
                  document->>'lifecycleStage', document->>'deleted'

  • users        : _id, document->>'name', document->>'email'
                  document->>'department', document->>'isActive'

  • createtasks  : _id, document->>'Task' (task title), document->>'status'
                  document->>'priority', document->>'createdBy' (→ users._id)
                  document->>'due_date', document->>'deleted'
                  Status values: 'Pending' / 'Completed'
                  Priority values: 'Low' / 'Medium' / 'High'

  • targets      : _id, document->>'userId' (→ users._id), document->>'targetInUSD'
                  document->>'month', document->>'year', document->>'teamName'
                  Achievement = SUM of confirmed sales for that user in that month/year

  • meetings     : _id, document->>'title', document->>'date'
                  document->>'attendees', document->>'createdBy'

  • outreaches   : _id, document->>'status', document->>'campaign'
                  document->>'isDeleted' (NOT deleted — different field name!)

  • departments  : _id, document->>'name'
                  JOIN with users: users.document->>'department' = departments._id

  SOFT DELETE: COALESCE(document->>'deleted','false') != 'true'
               For outreaches: COALESCE(document->>'isDeleted','false') != 'true'

  JOIN EXAMPLES:
    Users → Deals (by owner):
      deals.document->>'owner' = users._id
    Invoices → Companies:
      invoices.document->>'company' = companies._id
    Tasks → Users (creator):
      createtasks.document->>'createdBy' = users._id
    Deals owner name:
      LEFT JOIN users u ON u._id = deals.document->>'owner'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:

Input: "give me all closed lost deals with a report showing which stage they were lost from and which reps lost the most"
Output:
[
  {"sub_query": "list all closed lost deals with stage, owner, value and lost date", "intent": "list"},
  {"sub_query": "count lost deals grouped by stage ordered by count descending", "intent": "count"},
  {"sub_query": "count lost deals grouped by sales rep ordered by count descending", "intent": "rank"}
]

Input: "which users have pending tasks and how many overdue deals are in their pipeline"
Output:
[
  {"sub_query": "count pending tasks grouped by user including overdue flag", "intent": "count"},
  {"sub_query": "count open overdue deals grouped by deal owner", "intent": "count"}
]

Input: "show me target vs achieved for all reps this quarter with growth potential"
Output:
[
  {"sub_query": "list all sales targets for all users this quarter", "intent": "list"},
  {"sub_query": "total paid invoice revenue per user this quarter", "intent": "sum"}
]

Input: "executive summary: pipeline value, revenue this month, overdue invoices, pending tasks count"
Output:
[
  {"sub_query": "total value of all open deals grouped by stage", "intent": "sum"},
  {"sub_query": "total paid revenue this month", "intent": "sum"},
  {"sub_query": "count overdue unpaid invoices with total outstanding amount", "intent": "count"},
  {"sub_query": "count all pending tasks grouped by priority", "intent": "count"}
]

Input: "compare this month vs last month revenue and deal count"
Output:
[
  {"sub_query": "total paid revenue this month", "intent": "sum"},
  {"sub_query": "total paid revenue last month", "intent": "sum"},
  {"sub_query": "count deals created this month", "intent": "count"},
  {"sub_query": "count deals created last month", "intent": "count"}
]
"""


def _build_user_prompt(query: str, table_names: List[str], schema_context: str = "") -> str:
    """Build the user-turn message sent to the LLM."""
    tables_str = ", ".join(table_names) if table_names else "unknown"
    ctx_block  = f"\nSchema context:\n{schema_context}\n" if schema_context else ""
    return (
        f"Available database tables: {tables_str}\n"
        f"{ctx_block}"
        f"\nDecompose this query:\n\"{query}\"\n\n"
        f"Respond ONLY with a JSON array."
    )


# ── LLM call (via centralized multi-provider router) ──────────────────────────

def _llm_decompose(user_prompt: str) -> Optional[str]:
    """Call best available LLM for decomposition via llm.py fallback chain."""
    return llm_call("decompose", _SYSTEM_PROMPT, user_prompt, max_tokens=800)


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_sub_queries(raw: str) -> List[Dict[str, str]]:
    """
    Robustly extract a JSON array of sub-queries from model output.
    Handles: bare arrays, objects wrapping arrays, markdown fences.
    Returns validated list of {sub_query, intent} or empty list.
    """
    if not raw:
        return []

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # If model returned a JSON object wrapping an array, unwrap it
    # e.g. {"sub_queries": [...]} or {"queries": [...]}
    if cleaned.startswith("{"):
        try:
            obj = json.loads(cleaned)
            # Find the first list value
            for v in obj.values():
                if isinstance(v, list):
                    cleaned = json.dumps(v)
                    break
        except Exception:
            pass

    # Try to find a JSON array anywhere in the text
    array_match = re.search(r"\[[\s\S]*?\]", cleaned)
    if not array_match:
        # Last resort: find all JSON objects and wrap them
        objects = re.findall(r'\{[^{}]+\}', cleaned)
        if objects:
            try:
                cleaned = "[" + ",".join(objects) + "]"
            except Exception:
                return []
    else:
        cleaned = array_match.group(0)

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        LOGGER.debug("Decompose JSON parse error: %s | cleaned: %.200s", exc, cleaned)
        return []

    if not isinstance(items, list):
        return []

    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sub_query = str(item.get("sub_query", "")).strip()
        intent    = str(item.get("intent", "general")).strip().lower()
        if not sub_query or len(sub_query) < 4:
            continue
        # Normalize intent to allowed values
        allowed_intents = {"count", "list", "sum", "average", "filter",
                           "compare", "lookup", "rank", "trend", "general"}
        if intent not in allowed_intents:
            intent = "general"
        validated.append({"sub_query": sub_query, "intent": intent})

    return validated[:_MAX_SUB_QUERIES]


# ── Deduplication ──────────────────────────────────────────────────────────────

def _deduplicate(sub_queries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove near-duplicate sub-queries (same first 40 chars after normalization)."""
    seen, result = set(), []
    for sq in sub_queries:
        key = normalize_text(sq["sub_query"])[:40]
        if key not in seen:
            seen.add(key)
            result.append(sq)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def decompose(
    query: str,
    table_names: List[str],
    schema_context: str = "",
) -> List[Dict[str, str]]:
    """
    Break a complex user query into atomic sub-queries using LLM.

    Strategy:
      1. Try Groq first (fast, ~0.3–0.8s) if GROQ_API_KEY configured
      2. Fall back to local Ollama reasoning model (~5–15s)
      3. Graceful fallback — return original query as single sub-query

    Args:
        query:          Original user query (any length)
        table_names:    Available DB table names (schema context for the LLM)
        schema_context: Optional additional schema hints (field names, enums, etc.)

    Returns:
        List of {"sub_query": str, "intent": str} — max 6 items, deduplicated.
        Always returns at least one item (the original query) even if LLMs fail.
    """
    if not query or not query.strip():
        return [{"sub_query": query, "intent": "general"}]

    user_prompt = _build_user_prompt(query, table_names, schema_context)

    # ── Call via multi-provider router (Groq → Gemini → OpenRouter → Ollama) ─
    raw = _llm_decompose(user_prompt)
    if raw:
        parts = _parse_sub_queries(raw)
        if parts:
            parts = _deduplicate(parts)
            LOGGER.info(
                "Decompose: %d sub-queries for query: %.60s",
                len(parts), query,
            )
            return parts
        LOGGER.debug("Ollama decompose returned unparseable output for query: %.60s", query)

    # ── Graceful fallback ─────────────────────────────────────────────────────
    LOGGER.warning(
        "Decompose: all LLMs unavailable — treating original query as single sub-query: %.60s",
        query,
    )
    return [{"sub_query": query, "intent": "general"}]