"""chat_memory.py — Three-layer conversation memory for CRM AI.

Layers:
  1. Working memory   — active session context (last_entity, last_count, last_tables)
  2. Episodic memory  — past exchanges with Jaccard similarity matching
  3. Summary          — LLM-compressed digest, regenerated every 5 exchanges

Similarity matching:
  - Uses token-overlap (Jaccard) — no embedding model or extra RAM needed
  - Threshold 0.40 is loose enough to catch paraphrases:
      "tell me all deals" ↔ "show all deals" ↔ "list deals"
  - Threshold 0.60 for episodic retrieval to avoid injecting wrong context

Public API:
    ChatMemory                        ← main class
    ChatMemory.resolve(query)         ← pronoun + bare-action resolution
    ChatMemory.update(query, result)  ← call after every pipeline response
    ChatMemory.get_context(query)     ← context string to inject into LLM prompts
    ChatMemory.find_similar(query)    ← closest past episode (or None)
    ChatMemory.history                ← list of {query, answer} dicts
    ChatMemory.working_memory         ← {last_entity, last_count, last_tables}
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.llm import call as llm_call
from pipeline.utils import normalize_text

LOGGER = logging.getLogger("sql_chatbot")

# ── Stop-words stripped before similarity comparison ──────────────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "of", "to", "for",
    "me", "my", "and", "or", "can", "you", "i", "we", "all", "please",
    "show", "give", "tell", "get", "how", "what", "which", "do", "does",
    "from", "by", "with", "on", "at", "be", "it", "this", "that", "those",
    "these", "them", "their", "its", "also", "any", "some", "many", "few",
    "just", "only", "want", "need", "find", "list", "display", "fetch",
}

_PRONOUN_RE = re.compile(
    r"\b(that|those|them|their|these|it|its|this|his|her|"
    r"the same|above|previous|last|similar|aforementioned|"
    r"such|same|said|given|respective)\b",
    re.IGNORECASE,
)
_BARE_ACTION_RE = re.compile(
    r"^(give me|show|list|get|display|fetch)\s+(the\s+)?(names?|list|details?|all|records?|data)\s*$",
)

_SUMMARY_EVERY_N = 5   # regenerate summary every N new exchanges
_EPISODIC_THRESHOLD = 0.40   # minimum Jaccard to return an episode


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """One recorded exchange stored in episodic memory."""
    query:          str
    layer:          str          # pipeline layer that handled it
    result_summary: str          # first 150 chars of the answer
    tables_used:    List[str]    = field(default_factory=list)
    confidence:     float        = 0.0
    success:        bool         = True
    ts:             float        = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _jaccard(q1: str, q2: str) -> float:
    """Token-overlap (Jaccard) similarity — no model needed."""
    t1 = set(normalize_text(q1).split()) - _STOPWORDS
    t2 = set(normalize_text(q2).split()) - _STOPWORDS
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _extract_summary(result: Dict[str, Any]) -> str:
    """Pull a short readable summary out of a pipeline result dict."""
    answer = result.get("answer", "")
    if not answer:
        return "No data returned."
    clean = re.sub(r"\*\*|__|\[.*?\]\(.*?\)|```.*?```", "", answer, flags=re.DOTALL)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:150]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ChatMemory:
    """
    Per-session three-layer conversation memory.

    Layer 1 — Working memory (active task state):
        Tracks last entity, count, query, and tables so that pronouns like
        "those", "them", "that" can be resolved to the right referent.

    Layer 2 — Episodic memory (records of prior runs):
        Stores past (query, answer, tables, success) pairs. When a new query
        is similar to a past one, the context is injected into the pipeline so
        the LLM doesn't repeat known mistakes.

    Layer 3 — History summary:
        Every 5 exchanges a tiny LLM call compresses the session into 1-2
        sentences. This summary is prepended to LLM prompts so the model
        understands what the user has been exploring.
    """

    def __init__(self, max_episodes: int = 20, max_history: int = 10) -> None:
        self._max_episodes = max_episodes
        self._max_history  = max_history

        # Layer 1: working memory
        self.last_entity:     Optional[str] = None
        self.last_query:      str           = ""
        self.last_count:      Optional[int] = None
        self.last_tables:     List[str]     = []
        self.last_identifier: Optional[str] = None  # e.g. "ELSN/2025/1" for single-record results

        # Layer 2: episodic memory
        self.episodes: List[Episode] = []

        # Layer 3: summary
        self._summary:          str = ""
        self._summary_at_count: int = 0

        # Raw history for summary generation
        self._history: List[Dict[str, str]] = []

    # ── Layer 1: working memory ops ───────────────────────────────────────────

    def update(self, query: str, result: Dict[str, Any], success: bool = True) -> None:
        """Update all memory layers after a pipeline response. Call after every query."""
        tables = result.get("tables_used", [])
        entity = result.get("_entity") or (tables[0] if tables else None)
        count  = result.get("_count")

        # Working memory
        if entity:
            self.last_entity = entity
        self.last_query = query
        if count is not None:
            self.last_count = count
        if tables:
            self.last_tables = tables

        # Extract last record identifier — ONLY for single-record results.
        # Pattern: "**ELSN/2025/1**" or "reference ELSN/2025/1"
        ans = result.get("answer", "")
        sql_list = result.get("sql_queries", [])
        _is_single = (
            result.get("_count") == 1
            or bool(re.search(r"LIMIT\s+1\b", " ".join(sql_list), re.I))
            or bool(re.search(r"\b1\s+(?:invoice|deal|contact|company|task|record)\b", ans, re.I))
        )
        if _is_single:
            _id_m = re.search(
                r"\b([A-Z]{2,}/\d{4}/\d+)\b"           # invoice numbers like ELSN/2025/1
                r"|\*\*([A-Z][A-Z0-9/\-]{3,20})\*\*",   # any bold reference code
                ans,
            )
            if _id_m:
                self.last_identifier = (_id_m.group(1) or _id_m.group(2)).strip()
        else:
            self.last_identifier = None  # clear when multiple records shown

        # Raw history
        summary_str = _extract_summary(result)
        self._history.append({"query": query, "answer": summary_str})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Episodic memory
        ep = Episode(
            query=query,
            layer=result.get("layer", "unknown"),
            result_summary=summary_str,
            tables_used=list(tables),
            confidence=float(result.get("confidence", 0.0)),
            success=success,
        )
        self.episodes.append(ep)
        if len(self.episodes) > self._max_episodes:
            self.episodes = self.episodes[-self._max_episodes:]

        # Refresh summary every N exchanges
        if len(self._history) - self._summary_at_count >= _SUMMARY_EVERY_N:
            self._refresh_summary()

    def resolve(self, query: str) -> str:
        """
        Multi-level context resolution:

        Level 1 — Specific field on last record:
            "give me that payment status" + last_identifier=ELSN/2025/1
            → "payment status of ELSN/2025/1 in invoices"

        Level 2 — Field on last entity:
            "give me that status" + last_entity=invoices
            → "status of invoices"

        Level 3 — Pronoun → entity:
            "show those" → "show those (referring to invoices)"

        Level 4 — Bare action → entity:
            "give me all" → "give me all of invoices"
        """
        if not self.last_entity:
            return query
        t = normalize_text(query)

        # All pronouns that can precede a field name to mean "that/this [field]"
        _FIELD_PRONOUNS = r"(?:that|this|its|his|her|their|these|those|the|such|said)"

        # Known field semantic words — prevent table names / stopwords from
        # accidentally triggering the field-resolution levels.
        _FIELD_WORDS = {
            "status", "payment", "payment status", "stage", "amount", "total",
            "date", "name", "email", "phone", "owner", "details", "info",
            "number", "invoice number", "type", "count", "value", "price",
            "address", "region", "department", "team", "priority", "source",
            "currency", "due date", "created", "updated", "assigned",
        }
        _tables_lower = {e.lower() for e in self.last_tables}

        _LEADING_STOPWORDS = {"in", "of", "for", "by", "at", "on", "all", "a", "an", "the"}

        def _is_valid_field(word: str) -> bool:
            """Return True only when the extracted word looks like a field, not a table."""
            w     = word.lower().strip()
            parts = w.split()
            # reject if the word itself is a stopword or table name
            if w in _tables_lower or w in _LEADING_STOPWORDS:
                return False
            # reject if first sub-word is a preposition/stopword (e.g. "in paid")
            if parts and parts[0] in _LEADING_STOPWORDS:
                return False
            if len(w) < 3:
                return False
            return True

        # Level 1: "give me that/this/its [field]" with known last record identifier
        if self.last_identifier:
            field_ref = re.search(
                r"\b(?:give\s+me|show|get|what(?:'?s|\s+is)|tell\s+me)?\s*"
                + _FIELD_PRONOUNS + r"\s+(\w+(?:\s+\w+)?)\s*\??$",
                t,
            )
            if field_ref:
                field_word = field_ref.group(1).strip()
                if _is_valid_field(field_word):
                    resolved = (
                        f"{field_word} of {self.last_identifier} "
                        f"from {self.last_entity}"
                    )
                    LOGGER.debug("Memory: field+id resolved '%s' → '%s'", query, resolved)
                    return resolved

        # Level 2: "give me that/this [field]" — field breakdown on last entity
        field_ref2 = re.search(
            r"\b(?:give\s+me|show|get)?\s*" + _FIELD_PRONOUNS
            + r"\s+(\w+(?:\s+\w+)?)\s*\??$",
            t,
        )
        if field_ref2 and self.last_entity:
            field_word = field_ref2.group(1).strip()
            if _is_valid_field(field_word):
                resolved = f"show {self.last_entity} by {field_word}"
                LOGGER.debug("Memory: field resolved '%s' → '%s'", query, resolved)
                return resolved

        # Level 3: pronoun → entity
        if _PRONOUN_RE.search(t):
            resolved = f"{query} (referring to {self.last_entity})"
            LOGGER.debug("Memory: pronoun resolved '%s' → '%s'", query, resolved)
            return resolved

        # Level 4: bare action → entity
        if _BARE_ACTION_RE.match(t) and self.last_entity:
            resolved = f"{query} of {self.last_entity}"
            LOGGER.debug("Memory: bare action resolved '%s' → '%s'", query, resolved)
            return resolved

        return query

    # ── Layer 2: episodic memory ops ──────────────────────────────────────────

    def find_similar(self, query: str, threshold: float = _EPISODIC_THRESHOLD) -> Optional[Episode]:
        """
        Return the most similar past episode above `threshold`.

        High threshold (0.60): retrieves only very similar past queries.
        Low threshold (0.40):  catches paraphrases. Use for context injection.
        Returns None when no episode is similar enough.
        """
        best_ep:  Optional[Episode] = None
        best_sim: float             = 0.0

        for ep in reversed(self.episodes):
            sim = _jaccard(query, ep.query)
            if sim > best_sim:
                best_sim = sim
                best_ep  = ep

        if best_sim >= threshold and best_ep is not None:
            LOGGER.debug(
                "Memory: episodic match %.2f | '%s' ↔ '%s'",
                best_sim, query[:50], best_ep.query[:50],
            )
            return best_ep
        return None

    # ── Layer 3: context for LLM ──────────────────────────────────────────────

    def get_context(self, query: str) -> str:
        """
        Build a context string to inject into LLM prompts.

        Combines:
          - Session summary (what user has been exploring)
          - Most similar past episode (what worked / what entity was discussed)

        Returns empty string when no useful context exists.
        """
        parts: List[str] = []

        if self._summary:
            parts.append(f"Session context: {self._summary}")

        ep = self.find_similar(query)
        if ep and ep.success:
            entity_hint = ", ".join(ep.tables_used) or "CRM data"
            parts.append(
                f"User previously asked '{ep.query[:60]}' about {entity_hint}: "
                f"{ep.result_summary[:80]}"
            )

        return " | ".join(parts)

    # ── Layer 3: summary generation ───────────────────────────────────────────

    def _refresh_summary(self) -> None:
        """Compress the last N exchanges into a 1-2 sentence summary via tiny LLM."""
        if not self._history:
            return

        recent = self._history[-_SUMMARY_EVERY_N:]
        lines  = [
            f"Q: {h['query'][:60]} → A: {h['answer'][:80]}"
            for h in recent
        ]
        raw = "\n".join(lines)

        system = (
            "You are a session summarizer for a CRM assistant. "
            "Write 1-2 sentences summarizing what topics the user has been asking about "
            "(entities, filters, time periods). Do NOT include data values."
        )
        user = f"Recent exchanges:\n{raw}\n\nSummary (1-2 sentences max):"

        result = llm_call("classify", system, user, max_tokens=80)
        if result and result.strip():
            self._summary          = result.strip()
            self._summary_at_count = len(self._history)
            LOGGER.debug("Memory: summary updated → %s", self._summary[:80])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def history(self) -> List[Dict[str, str]]:
        """Conversation history as list of {query, answer} dicts."""
        return list(self._history)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def working_memory(self) -> Dict[str, Any]:
        return {
            "last_entity": self.last_entity,
            "last_query":  self.last_query,
            "last_count":  self.last_count,
            "last_tables": self.last_tables,
        }
