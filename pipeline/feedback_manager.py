"""feedback_manager.py — Persistent learning from user feedback.

Every time a user approves or corrects a response, that knowledge is stored
and injected into future SQL generation prompts — making the agent smarter
with every interaction.

Storage:
    logs/feedback_store.json  — list of example dicts (golden + corrections)

Public API:
    feedback_manager.save_positive(query, sql, result_summary) -> None
    feedback_manager.save_correction(query, bad_sql, user_feedback, good_sql) -> None
    feedback_manager.regenerate_sql(query, previous_sql, user_feedback) -> Optional[str]
    feedback_manager.get_injection_prompt(query) -> str   ← injected into SQL prompts
    feedback_manager                               ← singleton instance
"""
from __future__ import annotations

import fcntl
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("sql_chatbot")

_STORE_PATH  = Path("logs/feedback_store.json")
_MAX_SIZE    = 500   # entries; oldest evicted when exceeded
_MAX_INJECT  = 3     # max examples injected per SQL call (keep prompt lean)
_MAX_SCAN    = 30    # max examples sent to LLM for relevance ranking


class FeedbackManager:
    """
    Manages user feedback to continuously improve SQL generation.

    Two example types stored:
        "golden"     — user clicked 👍; SQL produced the right result
        "correction" — user clicked 👎 and typed what they wanted;
                       the corrected SQL was generated and approved

    On each SQL generation call, get_injection_prompt(query) is called.
    It:
        1. Loads the store
        2. Uses LLM to pick the most semantically relevant golden + correction examples
        3. Returns a formatted string to prepend to the SQL generation prompt
    If no examples exist or LLM fails → returns "" (pipeline proceeds normally).
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Storage
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        """Load feedback store. Returns [] on any error."""
        if not _STORE_PATH.exists():
            return []
        try:
            return json.loads(_STORE_PATH.read_text())
        except Exception:
            return []

    def _append(self, entry: Dict) -> None:
        """Append one entry to the feedback store with file locking."""
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(str(_STORE_PATH), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                data: List[Dict] = json.loads(raw) if raw.strip() else []
            except Exception:
                data = []
            data.append(entry)
            # FIFO eviction
            if len(data) > _MAX_SIZE:
                data = data[-_MAX_SIZE:]
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(data, indent=2))
            fcntl.flock(fh, fcntl.LOCK_UN)

    # ──────────────────────────────────────────────────────────────────────────
    # Write operations (called from API endpoints)
    # ──────────────────────────────────────────────────────────────────────────

    def has_positive(self, query: str) -> bool:
        """Return True if a golden entry already exists for this exact query."""
        normalized = query.strip().lower()
        for entry in self._load():
            if entry.get("type") == "golden" and entry.get("query", "").strip().lower() == normalized:
                return True
        return False

    def save_positive(
        self,
        query: str,
        sql: str,
        result_summary: str,
    ) -> bool:
        """Save a user-approved (👍) golden example.

        Returns True if saved, False if a golden entry for this query already exists
        (duplicate — caller should tell the user they already liked this).
        """
        if self.has_positive(query):
            LOGGER.info("FeedbackManager: duplicate golden skipped | query=%.80s", query)
            return False
        self._append({
            "type":           "golden",
            "query":          query,
            "sql":            sql,
            "result_summary": result_summary[:300],
            "ts":             time.time(),
        })
        LOGGER.info("FeedbackManager: golden saved | query=%.80s", query)
        return True

    def save_correction(
        self,
        query:         str,
        bad_sql:       str,
        user_feedback: str,
        good_sql:      str,
    ) -> None:
        """Save a user-rejected (👎) + corrected example."""
        self._append({
            "type":          "correction",
            "query":         query,
            "bad_sql":       bad_sql,
            "sql":           good_sql,
            "user_feedback": user_feedback[:500],
            "ts":            time.time(),
        })
        LOGGER.info("FeedbackManager: correction saved | query=%.80s", query)

    # ──────────────────────────────────────────────────────────────────────────
    # SQL re-generation (called by /feedback/correction endpoint)
    # ──────────────────────────────────────────────────────────────────────────

    def regenerate_sql(
        self,
        original_query: str,
        previous_sql:   str,
        user_feedback:  str,
        db_error:       Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a corrected SQL based on the user's rejection + correction note.

        The LLM receives:
        - What the user asked (and what the user now wants — which may expand the query)
        - The SQL that was generated and rejected
        - What the user said was wrong / what they want added/changed
        - Optional DB error from a previous attempt (for self-heal)

        Returns corrected SQL string, or None if generation failed.
        """
        from pipeline.llm_router import call as llm_call
        from pipeline.schema_compact import COMPACT_SYSTEM_PROMPT

        # Pull live schema from cache (warmed at startup, never empty after first query)
        schema_context = ""
        try:
            import pipeline.schema as _schema_mod
            cached = _schema_mod._TEXT2SQL_SCHEMA_CACHE
            if cached:
                schema_context = f"\n\nSCHEMA CONTEXT (use ONLY these real columns):\n{cached}"
        except Exception:
            pass

        error_hint = ""
        if db_error:
            error_hint = (
                f"\n\nPREVIOUS ATTEMPT FAILED WITH:\n{db_error}\n"
                "Fix this error in the new SQL. Use ONLY real column names from the schema above."
            )

        correction_system = (
            COMPACT_SYSTEM_PROMPT
            + schema_context
            + "\n\n"
            "CORRECTION MODE — CRITICAL:\n"
            "The user rejected the previous SQL. The user's feedback may expand the query "
            "(add columns, change filters, add a LIMIT, include a summary, etc.).\n"
            "Treat the user's correction as the NEW requirement for the same original question.\n"
            "Build a single SQL that satisfies BOTH the original question AND the user's correction.\n"
            "Use ONLY real columns from the SCHEMA CONTEXT above — NEVER guess column names.\n"
            "Output ONLY the corrected SQL. No explanation, no markdown."
            + error_hint
        )

        correction_user = (
            f'User originally asked: "{original_query}"\n\n'
            f"Previously generated SQL (REJECTED by user):\n{previous_sql}\n\n"
            f'User correction / extra requirement: "{user_feedback}"\n\n'
            "Generate corrected SQL that satisfies both the original question and the user's "
            "correction. Return ONLY the SQL:"
        )

        raw = llm_call("sql", correction_system, correction_user, max_tokens=600)
        if not raw:
            return None

        from agents.simple_agent import _extract_sql
        return _extract_sql(raw)

    # ──────────────────────────────────────────────────────────────────────────
    # Store management (called by API endpoints)
    # ──────────────────────────────────────────────────────────────────────────

    def list_all(self) -> List[Dict]:
        """Return all feedback entries sorted newest-first."""
        data = self._load()
        return list(reversed(data))

    def delete_entry(self, index: int) -> bool:
        """
        Delete the entry at `index` in the newest-first list returned by list_all().
        Returns True if deleted, False if index is out of range.
        """
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(str(_STORE_PATH), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                data: List[Dict] = json.loads(raw) if raw.strip() else []
            except Exception:
                data = []
            # Convert newest-first index back to list index
            real_idx = len(data) - 1 - index
            if real_idx < 0 or real_idx >= len(data):
                fcntl.flock(fh, fcntl.LOCK_UN)
                return False
            data.pop(real_idx)
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(data, indent=2))
            fcntl.flock(fh, fcntl.LOCK_UN)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Injection into SQL generation prompts
    # ──────────────────────────────────────────────────────────────────────────

    def get_injection_prompt(self, query: str) -> str:
        """
        Build a few-shot injection string from relevant past feedback.

        Returns "" when no feedback exists or relevance ranking fails —
        callers always proceed normally in that case.
        """
        data = self._load()
        if not data:
            return ""

        golden      = [d for d in data if d.get("type") == "golden"]
        corrections = [d for d in data if d.get("type") == "correction"]

        # If very few examples, skip LLM ranking (would cost more than benefit)
        if len(golden) + len(corrections) <= _MAX_INJECT * 2:
            best_golden      = golden[:_MAX_INJECT]
            best_corrections = corrections[:_MAX_INJECT]
        else:
            best_golden      = self._pick_relevant(query, golden,      _MAX_INJECT)
            best_corrections = self._pick_relevant(query, corrections, _MAX_INJECT)

        parts: List[str] = []

        if best_golden:
            parts.append("-- User-approved SQL examples (follow this style):")
            for i, ex in enumerate(best_golden, 1):
                q   = ex.get("query", "")[:100]
                sql = ex.get("sql", "")[:400]
                parts.append(f"-- ✓ Example {i}: User asked '{q}'\n-- SQL: {sql}")

        if best_corrections:
            parts.append("-- User-rejected patterns to AVOID:")
            for i, ex in enumerate(best_corrections, 1):
                q      = ex.get("query", "")[:100]
                badsql = ex.get("bad_sql", "")[:300]
                note   = ex.get("user_feedback", "")[:150]
                godsql = ex.get("sql", "")[:400]
                parts.append(
                    f"-- ✗ Avoid {i}: User asked '{q}'\n"
                    f"-- BAD SQL: {badsql}\n"
                    f"-- Reason rejected: {note}\n"
                    f"-- CORRECT SQL instead: {godsql}"
                )

        return "\n".join(parts) if parts else ""

    def _pick_relevant(
        self,
        query:    str,
        examples: List[Dict],
        n:        int,
    ) -> List[Dict]:
        """
        Use LLM to select the top-n most semantically relevant examples.
        Falls back to the most recent n if LLM fails.
        """
        if len(examples) <= n:
            return examples

        from pipeline.llm_router import call as llm_call

        # Build numbered list of past queries for the LLM
        lines = ""
        scan = examples[-_MAX_SCAN:]   # only scan the most recent N
        for i, ex in enumerate(scan, 1):
            lines += f"{i}. {ex.get('query','')[:120]}\n"

        system = (
            "You are a relevance ranker. Return ONLY a JSON integer array. "
            "No explanation, no markdown."
        )
        user_prompt = (
            f'Current query: "{query}"\n\n'
            f"Past queries (numbered):\n{lines}\n"
            f"Return the 1-based indices of the top {n} most semantically similar "
            f"past queries as a JSON array, e.g. [2, 5, 1]. "
            f"If none are similar, return []."
        )

        raw = llm_call("classify", system, user_prompt, max_tokens=60)
        if not raw:
            return scan[:n]

        raw = raw.strip()
        m = re.search(r"\[[\d,\s]*\]", raw)
        if not m:
            return scan[:n]

        try:
            indices = json.loads(m.group(0))
            picked: List[Dict] = []
            for idx in indices[:n]:
                if isinstance(idx, int) and 1 <= idx <= len(scan):
                    picked.append(scan[idx - 1])
            return picked if picked else scan[:n]
        except Exception:
            return scan[:n]


# Singleton — imported by api.py and by simple_agent / medium_agent
feedback_manager = FeedbackManager()
