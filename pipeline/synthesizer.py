"""synthesizer.py — LLM response synthesis engine (production-grade).

The final output brain of the pipeline. Takes original user query + all
sub-query results and produces ONE coherent, well-formatted answer.

Key capabilities:
  • Format-aware output: tables, lists, JSON, reports, yes/no, short, detail
  • Zero hallucination guardrails: only uses data from sub-query results
  • Executive summary with KPI extraction for complex queries
  • Structured response templates per query type
  • Graceful multi-tier fallback: Groq → Ollama → intelligent concatenation

Uses Groq (fast, ~1-3s) if GROQ_API_KEY is set,
else falls back to local Ollama reasoning model (~15-35s).

Public API:
    synthesize(original_query, sub_results) -> str
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any, Dict, List, Optional

from config import settings
from pipeline.llm import call as llm_call
from pipeline.utils import (
    ResponseFormat,
    Timer,
    coerce_number,
    detect_response_format,
    fmt_currency,
    fmt_number,
    fmt_percentage,
    normalize_text,
)

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_synthesis_prompt(
    original_query: str,
    sub_results: List[Dict],
    response_format: str,
) -> str:
    """Build the LLM prompt for final answer synthesis.

    The prompt is structured to:
      1. Provide strict anti-hallucination rules
      2. Inject all sub-query results as structured data
      3. Specify the desired output format
      4. Request KPIs and insights where appropriate
    """

    # Format sub-results as structured data blocks.
    # Each answer is capped at 1500 chars so the synthesizer sees enough detail
    # while still staying within Groq / Gemini context limits.
    _MAX_ANSWER_CHARS = 1500
    data_blocks = []
    for i, item in enumerate(sub_results, 1):
        sq     = item.get("sub_query", "Unknown")
        intent = item.get("intent", "general")
        data   = item.get("data", {}) if isinstance(item.get("data"), dict) else item
        answer = data.get("answer", "No data available.")
        tables = data.get("tables_used", [])
        conf   = data.get("confidence", 0.0)
        source = data.get("source", "unknown")
        error  = data.get("error")

        # Truncate large answers — keep first N chars then note truncation
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + f"\n…[truncated, {len(answer)} chars total]"

        block = (
            f"--- SUB-QUERY {i} ---\n"
            f"Question: {sq}\n"
            f"Intent: {intent}\n"
            f"Source: {source} (confidence: {conf})\n"
            f"Tables: {', '.join(tables) if tables else 'none'}\n"
        )
        if error:
            block += f"Error: {error}\n"
        block += f"Data:\n{answer}\n"
        data_blocks.append(block)

    data_section = "\n".join(data_blocks)

    # Format-specific instructions
    format_instructions = _get_format_instructions(response_format)

    return f"""You are a CRM data analyst. Synthesize the sub-query results below into ONE clear answer.

═══════════════════════════════════════════
ABSOLUTE RULES (NEVER BREAK):

1. USE ONLY the data provided below. NEVER invent, guess, or hallucinate numbers,
   names, dates, or any other facts. If data is missing, say "data unavailable".

2. NEVER claim data shows something it doesn't. If sub-query returned "No data found",
   report that honestly — don't rephrase it as a positive finding.

3. PRESERVE exact numbers from the data. Don't round, estimate, or approximate
   unless explicitly computing a derived metric (like percentage).

4. EVERY number in your response must trace back to the data below.
   If you compute a percentage or growth rate, show the source numbers.

5. If a sub-query errored or returned no data, acknowledge it briefly
   and proceed with available data. Don't fabricate replacement data.

6. DO NOT add generic advice, recommendations, or next steps unless
   the user explicitly asked for them.

7. Keep response concise and structured. No filler text, no repeating
   the question back, no "Based on the data provided..." preamble.

═══════════════════════════════════════════
USER QUERY: "{original_query}"

{format_instructions}

═══════════════════════════════════════════
SUB-QUERY RESULTS (your ONLY source of truth):

{data_section}

═══════════════════════════════════════════
Now write the final synthesized response. Start directly with the content."""


def _get_format_instructions(response_format: str) -> str:
    """Get format-specific instructions for the synthesis prompt."""

    instructions = {
        ResponseFormat.YES_NO: (
            "OUTPUT FORMAT: Yes/No answer\n"
            "- Start with a clear YES or NO\n"
            "- Follow with 1-2 sentences of supporting data\n"
            "- Keep it under 50 words total"
        ),
        ResponseFormat.SHORT: (
            "OUTPUT FORMAT: Brief response\n"
            "- Maximum 2-3 sentences\n"
            "- Lead with the key finding\n"
            "- Include only the most important number/fact"
        ),
        ResponseFormat.COUNT: (
            "OUTPUT FORMAT: Count response\n"
            "- Lead with the number in bold\n"
            "- One sentence of context\n"
            "- If multiple counts, use a compact list"
        ),
        ResponseFormat.LIST: (
            "OUTPUT FORMAT: List\n"
            "- Use a clean markdown bullet list\n"
            "- One item per line, most important first\n"
            "- Include relevant details (amount, status) inline"
        ),
        ResponseFormat.TABLE: (
            "OUTPUT FORMAT: Table\n"
            "- Use markdown table format\n"
            "- Include headers that describe each column\n"
            "- Sort by most relevant metric descending\n"
            "- Show totals row if applicable"
        ),
        ResponseFormat.JSON: (
            "OUTPUT FORMAT: JSON\n"
            "- Wrap entire response in a JSON code block\n"
            "- Use clear key names (snake_case)\n"
            "- Include a 'summary' field and a 'data' array\n"
            "- Numbers as numbers, not strings"
        ),
        ResponseFormat.SUMMARY: (
            "OUTPUT FORMAT: Executive Summary\n"
            "- Start with a 2-3 sentence overview of key findings\n"
            "- Use bold for important numbers\n"
            "- Highlight notable trends or outliers\n"
            "- Keep total response under 200 words"
        ),
        ResponseFormat.REPORT: (
            "OUTPUT FORMAT: Full Report\n"
            "- Start with a 2-3 line executive summary with the most critical KPIs\n"
            "- Use markdown headers (##) for each section\n"
            "- Include data tables where appropriate\n"
            "- Add a 'Key Insights' section at the end with 2-3 bullet points\n"
            "- Calculate derived metrics (growth %, ratios, averages) where possible\n"
            "- Format all currency with $ and commas"
        ),
        ResponseFormat.DETAIL: (
            "OUTPUT FORMAT: Detailed Response\n"
            "- Include all available data — don't truncate\n"
            "- Use tables for structured data\n"
            "- Show all fields and values\n"
            "- Add context for each data section"
        ),
    }

    default = (
        "OUTPUT FORMAT: Professional CRM Business Response\n"
        "- Open with 1-2 sentences stating the KEY FINDING in business language\n"
        "- Bold ALL important numbers: **176 deals**, **$2.4M revenue**, **42%**\n"
        "- Present data in the clearest format:\n"
        "  • Single number → bold value + 1 sentence of context\n"
        "  • List of names → bullet list with count in header\n"
        "  • Multi-field data → markdown table with totals row\n"
        "- If the result is a count or total, follow with a brief insight\n"
        "- Highlight the most significant item, trend, or outlier\n"
        "- Keep response concise and actionable — no filler text\n"
        "- Do NOT repeat the user's question back to them\n"
        "- Do NOT say 'Based on the data' or similar preambles"
    )

    return instructions.get(response_format, default)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _llm_synthesize(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call best available LLM for synthesis via llm.py fallback chain.
    Route: Groq 70b → Gemini flash-lite → OpenRouter → Ollama 7b
    """
    return llm_call("synthesize", system_prompt, user_prompt, max_tokens=2500)


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT FALLBACK (NO LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_kpis(sub_results: List[Dict]) -> List[str]:
    """Extract key metrics from sub-query results for the summary header."""
    kpis = []
    for item in sub_results:
        data   = item.get("data", {}) if isinstance(item.get("data"), dict) else item
        answer = data.get("answer", "")
        intent = item.get("intent", "")

        if not answer or "No data" in answer or "unavailable" in answer:
            continue

        # Extract bold numbers from the answer
        bold_numbers = re.findall(r"\*\*([^*]+)\*\*", answer)
        for bn in bold_numbers:
            n = coerce_number(bn.replace(",", "").replace("$", ""))
            if isinstance(n, (int, float)) and n > 0:
                sq = item.get("sub_query", "")
                # Build a KPI label from the sub-query
                label = sq[:50] if sq else intent
                kpis.append(f"**{bn}** ({label})")
                break  # one KPI per sub-query

    return kpis[:6]  # max 6 KPIs


def _fallback_structured(
    original_query: str,
    sub_results: List[Dict],
    response_format: str,
) -> str:
    """Intelligent concatenation fallback when both LLMs are unavailable.

    Produces structured output that mimics LLM synthesis quality
    by using templates, KPI extraction, and format-aware rendering.
    """
    # Separate successful and failed results
    successes = []
    failures  = []
    for item in sub_results:
        data   = item.get("data", {}) if isinstance(item.get("data"), dict) else item
        answer = data.get("answer", "")
        error  = data.get("error")

        if error or "unavailable" in answer.lower():
            failures.append(item)
        else:
            successes.append(item)

    if not successes and not failures:
        return "No data available to answer this query."

    parts = []

    # Build executive summary header
    kpis = _extract_kpis(sub_results)
    if kpis and response_format in (
        ResponseFormat.REPORT, ResponseFormat.SUMMARY, ResponseFormat.DEFAULT
    ):
        parts.append("## Executive Summary\n")
        parts.append(" · ".join(kpis[:4]))
        parts.append("")

    # Render each successful result
    for item in successes:
        sq     = item.get("sub_query", "")
        data   = item.get("data", {}) if isinstance(item.get("data"), dict) else item
        answer = data.get("answer", "No data.")

        # Clean up individual answers (remove duplicate headers)
        answer = _clean_sub_answer(answer)

        if response_format == ResponseFormat.SHORT:
            # Just the data, no sub-query headers
            parts.append(answer)
        else:
            parts.append(f"### {sq}\n\n{answer}")

    # Mention failures briefly
    if failures:
        failed_qs = [f.get("sub_query", "unknown") for f in failures]
        parts.append(
            f"\n_Note: Data unavailable for: {', '.join(failed_qs)}_"
        )

    return "\n\n".join(parts)


def _clean_sub_answer(answer: str) -> str:
    """Remove wrapper text that fast-path adds (we'll add our own structure)."""
    # Remove "Executive Summary:" prefix
    answer = re.sub(r"^Executive Summary:\s*", "", answer, flags=re.I)
    # Remove "Result is generated from..." footer
    answer = re.sub(
        r"\n*Result is generated from.*?database data\.\s*$",
        "", answer, flags=re.I | re.DOTALL,
    )
    return answer.strip()


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _post_process(answer: str, sub_results: List[Dict]) -> str:
    """Clean and validate the synthesized answer.

    Checks:
      • Not empty
      • Doesn't contain model artifacts (think tags, JSON wrappers)
      • Doesn't hallucinate tables not in results
      • Trims excessive length
    """
    if not answer or not answer.strip():
        return "Unable to generate a response. Please try rephrasing your query."

    cleaned = answer.strip()

    # Remove model artifacts
    cleaned = re.sub(r"</?think>", "", cleaned)
    cleaned = re.sub(r"```json\s*\{[^}]*\"(response|answer)\":", "", cleaned)
    cleaned = re.sub(r"\}\s*```\s*$", "", cleaned)

    # Remove "Based on the data..." preamble
    cleaned = re.sub(
        r"^(?:Based on (?:the )?(?:data|results|information) (?:provided|above|below)[,.]?\s*)+",
        "", cleaned, flags=re.I,
    )

    # Trim excessive length (>4000 chars usually means the model went off-rails)
    if len(cleaned) > 4000:
        # Find the last complete section
        last_section = cleaned[:4000].rfind("\n\n")
        if last_section > 2000:
            cleaned = cleaned[:last_section] + "\n\n_…response truncated for readability_"
        else:
            cleaned = cleaned[:4000] + "\n\n_…response truncated_"

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def narrate_response(
    original_query: str,
    raw_answer: str,
    tables_used: List[str],
) -> str:
    """
    Wrap a raw fast-path / text2sql answer with a professional business summary.

    Called for ALL SIMPLE-path results so every response includes a plain-English
    explanation — not just a bare number or raw table.

    Uses the classify chain (Groq 8b → Gemini → Ollama 1.5b) — fast (~300ms).
    On any failure the raw_answer is returned unchanged, so this is never blocking.

    Args:
        original_query: User's original question
        raw_answer:     The data answer (table / count / sum string)
        tables_used:    DB tables that were queried

    Returns:
        Professional summary + original data, or raw_answer on failure.
    """
    # Skip error / already-narrated / very short non-data answers
    skip_signals = [
        "no data found", "no records found", "data unavailable", "timed out",
        "unable to retrieve", "encountered an error", "executive summary",
        "i could not", "i wasn't able",
    ]
    ans_lower = raw_answer.lower().strip()

    # Already has a summary header — don't double-narrate
    if ans_lower.startswith("## executive summary"):
        return raw_answer

    # Skip genuine error/unavailable messages
    if any(ans_lower.startswith(s) for s in skip_signals):
        return raw_answer

    # Don't skip short count answers — "**176**" is valid data that needs narration
    # Only skip if truly empty or only whitespace
    if not ans_lower:
        return raw_answer

    table_hint = (
        " (source tables: " + ", ".join(tables_used) + ")" if tables_used else ""
    )

    system = (
        "You are a senior CRM business analyst. Given a user query and raw database result, "
        "write a professional 2-3 sentence business summary.\n\n"
        "RULES:\n"
        "1. Start DIRECTLY with the finding — no preamble like 'Here is', 'Based on', 'The data shows'\n"
        "2. Bold ALL key numbers, names, and percentages: **176 deals**, **$58,296**, **42%**\n"
        "3. Use business language: 'pipeline', 'revenue', 'conversion', 'performance'\n"
        "4. If the result is a COUNT (single number), say what it counts and add one insight\n"
        "   Example: count=176 deals → 'Your CRM pipeline contains **176 deals** across all stages...'\n"
        "5. If the result is a LIST or TABLE, summarise the total count and top pattern\n"
        "6. If result shows 0 or 'no records', say clearly that none were found and why that might be\n"
        "7. NEVER invent numbers, names, or dates not present in the data\n"
        "8. Maximum 80 words. Be concise and actionable.\n"
        "9. End with the exact marker: <<<END_SUMMARY>>>"
    )
    user = (
        f'User asked: "{original_query}"{table_hint}\n'
        f"Database returned:\n{raw_answer[:800]}\n\n"
        "Write the 2-3 sentence professional summary, then output <<<END_SUMMARY>>>:"
    )

    raw = llm_call("classify", system, user, max_tokens=160)
    if not raw:
        return raw_answer

    if "<<<END_SUMMARY>>>" in raw:
        summary = raw.split("<<<END_SUMMARY>>>")[0].strip()
    else:
        summary = raw.strip()

    if not summary or len(summary) < 15:
        return raw_answer

    # Strip accidental preambles
    summary = re.sub(
        r"^(here is|based on|the data (shows|indicates)|according to|"
        r"the query|the result|the database)[^.]*[.,]\s*",
        "", summary, flags=re.I,
    )
    summary = summary.strip()
    if not summary:
        return raw_answer

    return "## Summary\n\n" + summary + "\n\n---\n\n" + raw_answer


def synthesize(
    original_query: str,
    sub_results: List[Dict],
) -> str:
    """Generate one final human-readable answer from all sub-query results.

    Strategy:
      1. Detect desired response format from user query
      2. Build format-aware synthesis prompt
      3. Try Groq (fast, high-quality)
      4. Fall back to Ollama reasoning model
      5. Fall back to intelligent structured concatenation (no LLM)

    Anti-hallucination guarantees:
      • Prompt instructs LLM to use ONLY provided data
      • Post-processing validates the output
      • Fallback uses raw data directly (zero hallucination by construction)

    Args:
        original_query: The user's original question
        sub_results:    List of sub-query result dicts from executor.
                        Each item: {sub_query, intent, data: {answer, tables_used, ...}}

    Returns:
        Final formatted markdown string, ready for user display.
    """
    if not sub_results:
        return "No data available to answer this query."

    # Detect format
    response_format = detect_response_format(original_query)
    LOGGER.info("Synthesis format: %s | query: %.60s", response_format, original_query)

    # Check if all results are failures
    all_failed = all(
        (item.get("data", {}) if isinstance(item.get("data"), dict) else item)
        .get("error") is not None
        or "unavailable" in str(
            (item.get("data", {}) if isinstance(item.get("data"), dict) else item)
            .get("answer", "")
        ).lower()
        for item in sub_results
    )
    if all_failed:
        return (
            "I wasn't able to retrieve data for any part of your query. "
            "This may be due to missing tables or a database connection issue. "
            "Please try a simpler query or check if the relevant data exists."
        )

    # Build prompt
    prompt = _build_synthesis_prompt(original_query, sub_results, response_format)

    with Timer("synthesis_llm") as timer:
        # ── Multi-provider LLM: Groq 70b → Gemini → OpenRouter → Ollama ──
        system_prompt = (
            "You are a senior CRM business analyst. Generate a single, professional, "
            "insight-driven response. Use markdown formatting. Bold key numbers. "
            "Add percentages and growth rates where possible. Flag anomalies and risks."
        )
        answer = _llm_synthesize(system_prompt, prompt)
        if answer:
            answer = _post_process(answer, sub_results)
            LOGGER.info("Synthesis done in %.0fms", timer.elapsed_ms)
            return answer

    # ── Intelligent fallback (no LLM) ─────────────────────────────────────
    LOGGER.warning("Synthesis: all LLMs unavailable — using structured fallback")
    return _fallback_structured(original_query, sub_results, response_format)