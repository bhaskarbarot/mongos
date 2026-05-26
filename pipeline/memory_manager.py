"""memory_manager.py — LLM-powered semantic conversation memory resolver.

Zero hardcoding: the LLM itself decides if a query is a continuation of the
conversation or a completely new topic. No regex, no keyword lists, no pronoun
patterns — pure semantic understanding via LLM.

Public API:
    MemoryResolution                   — dataclass for resolve() result
    MemoryManager.resolve(query, history) -> MemoryResolution
    memory_manager                     — singleton instance

Design:
    On every /chat request, the frontend sends the full conversation history.
    Before forwarding to the SQL pipeline, we call resolve() which:
      1. Extracts the last 3–5 user/assistant turns
      2. Builds a lightweight topic summary (no LLM — just concatenate user queries)
      3. Calls the LLM with the current query + history context
      4. LLM returns JSON: {is_continuation, resolved_query, reasoning}
      5. The resolved_query is the fully self-contained query passed to the pipeline
    If the LLM call fails for any reason, the original query is used unchanged.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("sql_chatbot")

# Max turns to include in the LLM resolution prompt (controls token cost)
_MAX_TURNS = 5


@dataclass
class MemoryResolution:
    """Result of LLM-powered query resolution."""
    original_query:  str
    resolved_query:  str
    is_continuation: bool
    reasoning:       str
    context_summary: str = ""


class MemoryManager:
    """
    LLM-powered semantic conversation memory.

    The LLM receives the last N turns of conversation and decides:
    - Is the current query a CONTINUATION (references prior context)?
    - If yes: what is the fully self-contained resolved query?
    - If no: return the query unchanged.

    No hardcoded rules, no pronoun lists, no keyword matching anywhere.
    """

    def resolve(self, user_query: str, history: List[Dict]) -> MemoryResolution:
        """
        Semantically resolve user_query against conversation history.

        Args:
            user_query: Raw user input string
            history:    List of {"role": str, "content": str} message dicts,
                        oldest first, newest last (same format as frontend sends)

        Returns:
            MemoryResolution — resolved_query is safe to pass directly to pipeline.
            On any failure, resolved_query == original_query (graceful degradation).
        """
        if not history or len(history) < 2:
            # First message in session — nothing to resolve against
            return MemoryResolution(
                original_query=user_query,
                resolved_query=user_query,
                is_continuation=False,
                reasoning="No prior conversation history.",
            )

        turns = self._extract_turns(history, _MAX_TURNS)
        if not turns:
            return MemoryResolution(
                original_query=user_query,
                resolved_query=user_query,
                is_continuation=False,
                reasoning="Could not extract history turns.",
            )

        context_summary = self._lightweight_summary(turns)
        result = self._llm_resolve(user_query, turns, context_summary)

        if result is None:
            # LLM call failed — degrade gracefully, never block the user
            LOGGER.warning(
                "MemoryManager: LLM resolution failed — using original query as-is"
            )
            return MemoryResolution(
                original_query=user_query,
                resolved_query=user_query,
                is_continuation=False,
                reasoning="LLM resolution unavailable — query used as-is.",
                context_summary=context_summary,
            )

        resolved = result.get("resolved_query") or user_query
        LOGGER.info(
            "MemoryManager: continuation=%s | '%s' → '%s' | %s",
            result.get("is_continuation"),
            user_query[:60],
            resolved[:60],
            result.get("reasoning", "")[:80],
        )

        return MemoryResolution(
            original_query=user_query,
            resolved_query=resolved,
            is_continuation=bool(result.get("is_continuation", False)),
            reasoning=result.get("reasoning", ""),
            context_summary=context_summary,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_turns(
        self, history: List[Dict], max_turns: int
    ) -> List[Dict[str, str]]:
        """
        Parse history list into user/assistant pairs.
        Returns list of {"user": str, "assistant": str}, newest last.
        """
        turns: List[Dict[str, str]] = []
        msgs = list(history)
        i = 0
        while i < len(msgs):
            if msgs[i].get("role") == "user":
                user_text = (msgs[i].get("content") or "").strip()
                asst_text = ""
                if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                    asst_text = (msgs[i + 1].get("content") or "").strip()
                    i += 2
                else:
                    i += 1
                if user_text:
                    turns.append({"user": user_text, "assistant": asst_text})
            else:
                i += 1

        return turns[-max_turns:]

    def _lightweight_summary(self, turns: List[Dict[str, str]]) -> str:
        """
        Build a fast (no-LLM) context summary from recent user queries.
        Used to give the LLM a quick overview without burning tokens.
        """
        topics = [t["user"][:100] for t in turns[-3:] if t["user"]]
        if not topics:
            return ""
        return "Recent topics: " + " | ".join(topics)

    def _clean_assistant(self, text: str) -> str:
        """Strip markdown and collapse whitespace from assistant answers."""
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"\*\*|__|\[([^\]]+)\]\([^)]+\)|#{1,6}\s", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:300]

    def _llm_resolve(
        self,
        user_query: str,
        turns: List[Dict[str, str]],
        context_summary: str,
    ) -> Optional[Dict]:
        """
        Call the LLM to determine continuation vs. new topic.

        Returns parsed dict {is_continuation, resolved_query, reasoning}
        or None on any failure (caller handles graceful degradation).
        """
        from pipeline.llm_router import call as llm_call

        # Build the conversation history block for the prompt
        history_block = ""
        for idx, turn in enumerate(turns, 1):
            user_part = turn["user"][:200]
            asst_part = self._clean_assistant(turn["assistant"]) if turn["assistant"] else "(no response yet)"
            history_block += (
                f"Turn {idx}:\n"
                f"  User: {user_part}\n"
                f"  Assistant returned: {asst_part}\n\n"
            )

        system = (
            "You are a query resolver for a CRM data assistant. "
            "Your ONLY job: decide if the user's current query refers to the prior conversation, "
            "and if so, rewrite it as a fully self-contained question.\n\n"
            "RULES (follow exactly):\n"
            "1. Output ONLY a valid JSON object — no explanation, no markdown, no preamble.\n"
            "2. A query IS a continuation if it uses vague references like 'that', 'those', "
            "'them', 'it', 'the same', 'above', 'previous', 'also get', 'filter that', "
            "'now show', 'what about', 'for those' — AND the reference clearly points to "
            "something specific in the prior turns.\n"
            "3. A query is NOT a continuation if it introduces a completely new subject "
            "with no reference to prior turns, even if it uses words like 'and' or 'also'.\n"
            "4. The resolved_query MUST be fully self-contained: someone reading ONLY that "
            "query (without conversation history) must understand exactly what is being asked.\n"
            "5. NEVER invent context. Only incorporate what is explicitly in the history.\n"
            "6. If uncertain, treat as a new topic (is_continuation: false) and return the "
            "query unchanged.\n\n"
            'Return exactly this JSON:\n'
            '{\n'
            '  "is_continuation": true or false,\n'
            '  "resolved_query": "the complete, self-contained question",\n'
            '  "reasoning": "one sentence why"\n'
            '}'
        )

        user_prompt = (
            f"Conversation history:\n{history_block}"
            f"Context summary: {context_summary}\n\n"
            f'Current user query: "{user_query}"\n\n'
            "Respond with JSON only:"
        )

        raw = llm_call("classify", system, user_prompt, max_tokens=220)
        if not raw:
            return None

        raw = raw.strip()
        # Remove markdown code fences if the LLM wrapped its JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        # Parse JSON
        try:
            parsed = json.loads(raw)
            if "resolved_query" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract JSON from mixed text
        m = re.search(r"\{[^{}]*\"resolved_query\"[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        LOGGER.warning("MemoryManager: JSON parse failed for: %.120s", raw)
        return None


# Module-level singleton — imported by api.py
memory_manager = MemoryManager()
