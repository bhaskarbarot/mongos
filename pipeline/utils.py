"""utils.py — Shared utility functions used across all pipeline modules.

Production-grade utilities for:
  • Text normalization and sanitization
  • Number formatting and coercion
  • Markdown table rendering (with column headers from SQL)
  • Response format detection (table / list / json / summary / yes-no / short)
  • Final answer formatting with intelligent wrapping
  • Security: SQL injection guard, blocked pattern detection
  • Query hashing for caching
  • Latency tracking helpers

All functions are pure (no side effects) unless noted.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("sql_chatbot")

# ── Compiled regex patterns ────────────────────────────────────────────────────

_OID_RE = re.compile(r"\b([0-9a-f]{19})[0-9a-f]{5}\b", re.IGNORECASE)

BLOCKED_SQL_PATTERNS = [
    re.compile(r"\bDROP\b",       re.IGNORECASE),
    re.compile(r"\bDELETE\b",     re.IGNORECASE),
    re.compile(r"\bTRUNCATE\b",   re.IGNORECASE),
    re.compile(r"\bALTER\b",      re.IGNORECASE),
    re.compile(r"\bINSERT\b",     re.IGNORECASE),
    re.compile(r"\bUPDATE\b",     re.IGNORECASE),
    re.compile(r"\bGRANT\b",      re.IGNORECASE),
    re.compile(r"\bREVOKE\b",     re.IGNORECASE),
    re.compile(r"\bCREATE\b",     re.IGNORECASE),
    re.compile(r"\bEXEC\b",       re.IGNORECASE),
    re.compile(r"--",             re.IGNORECASE),
    re.compile(r";\s*SELECT\b",   re.IGNORECASE),  # stacked queries
]

GREETING_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|hola|namaste)\b[!.\s]*$",                     re.IGNORECASE),
    re.compile(r"^\s*good\s*(morning|afternoon|evening|day)\b[!.\s]*$",           re.IGNORECASE),
    re.compile(r"^\s*(how are you|what'?s up|howdy|sup)\b[?.!\s]*$",              re.IGNORECASE),
    re.compile(r"^\s*(thanks?|thank you|thx)\b[!.\s]*$",                          re.IGNORECASE),
]

_REVENUE_FIELD_KEYWORDS = [
    "total", "amount", "price", "revenue", "cost", "value",
    "fee", "tax", "subtotal", "grand", "usd", "billing", "payment",
]
_REVENUE_TABLE_KEYWORDS = [
    "invoice", "order", "sale", "payment", "transaction", "billing", "revenue",
]
_SUM_QUERY_KEYWORDS = [
    "revenue", "sales", "income", "amount", "billing", "earning",
    "total value", "total amount", "sum", "collection",
]


# ── Text normalization ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip edges."""
    return re.sub(r"\s+", " ", text.strip().lower())


def mask_ids(text: str) -> str:
    """Mask MongoDB-style ObjectIDs for cleaner user-facing output."""
    return _OID_RE.sub(r"\1…", text)


def sanitize_for_sql(value: str) -> str:
    """Basic SQL injection prevention for user-supplied values in fast-path SQL."""
    return value.replace("'", "''").replace(";", "").replace("--", "").strip()


def query_hash(query: str) -> str:
    """Deterministic hash for query caching. Normalizes before hashing."""
    return hashlib.sha256(normalize_text(query).encode("utf-8")).hexdigest()[:16]


# ── Number handling ────────────────────────────────────────────────────────────

def coerce_number(v: Any) -> Any:
    """Coerce a value to numeric type. Returns 0 for None, original for non-numeric."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return v
    try:
        s = str(v).replace(",", "").strip()
        if "." in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return v


def fmt_number(v: Any) -> str:
    """Format a number with commas and 2 decimal places for floats."""
    n = coerce_number(v)
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e15:
            return f"{int(n):,}"
        return f"{n:,.2f}"
    if isinstance(n, int):
        return f"{n:,}"
    return str(v) if v is not None else "—"


def fmt_currency(v: Any, symbol: str = "$") -> str:
    """Format a number as currency."""
    n = coerce_number(v)
    if isinstance(n, (int, float)):
        return f"{symbol}{fmt_number(n)}"
    return str(v) if v is not None else "—"


def fmt_percentage(v: Any, decimals: int = 1) -> str:
    """Format a number as percentage."""
    n = coerce_number(v)
    if isinstance(n, (int, float)):
        return f"{round(n, decimals)}%"
    return str(v) if v is not None else "—"


# ── Response format detection ──────────────────────────────────────────────────

class ResponseFormat:
    """Detected desired response format from user query."""
    TABLE    = "table"
    LIST     = "list"
    JSON     = "json"
    SUMMARY  = "summary"
    REPORT   = "report"
    YES_NO   = "yes_no"
    SHORT    = "short"
    COUNT    = "count"
    DETAIL   = "detail"
    DEFAULT  = "default"


def detect_response_format(query: str) -> str:
    """
    Detect what kind of response format the user expects.
    Used by synthesizer to shape the final output.
    """
    t = normalize_text(query)

    # Yes/No questions
    if re.match(r"^(is |are |do |does |did |has |have |was |were |can |should |will )", t):
        if not re.search(r"\b(list|show|give|report|table|detail|all)\b", t):
            return ResponseFormat.YES_NO

    # JSON output
    if re.search(r"\b(json|api format|json format|as json)\b", t):
        return ResponseFormat.JSON

    # Report / analysis
    if re.search(
        r"\b(report|analysis|executive summary|dashboard|overview|breakdown|"
        r"full breakdown|performance review|kpi|health check|360)\b", t
    ):
        return ResponseFormat.REPORT

    # Table format
    if re.search(r"\b(table|tabular|spreadsheet|grid|matrix)\b", t):
        return ResponseFormat.TABLE

    # List format
    if re.search(r"\b(list all|list of|names of|give me all|show all)\b", t):
        return ResponseFormat.LIST

    # Count only
    if re.match(r"^(how many|count|total number)\b", t):
        if not re.search(r"\b(list|show|give|detail|name|with)\b", t):
            return ResponseFormat.COUNT

    # Short response
    if re.search(r"\b(quick|brief|short|one line|one word|just tell|just give)\b", t):
        return ResponseFormat.SHORT

    # Detailed
    if re.search(r"\b(detail|full|complete|everything|all info|all information)\b", t):
        return ResponseFormat.DETAIL

    # Summary
    if re.search(r"\b(summary|summarize|sum up|wrap up|gist)\b", t):
        return ResponseFormat.SUMMARY

    return ResponseFormat.DEFAULT


# ── Markdown table formatting ──────────────────────────────────────────────────

def format_rows_as_markdown_table(
    rows: List[Any],
    headers: Optional[List[str]] = None,
    max_rows: int = 30,
    show_overflow: bool = True,
) -> str:
    """
    Render rows as a Markdown table.

    Args:
        rows:          List of tuples/lists from SQL result
        headers:       Column headers (auto-generated as col_1..col_N if None)
        max_rows:      Max rows to display (default 30)
        show_overflow: Whether to show "…and N more" when truncated

    Returns:
        Markdown table string
    """
    if not rows:
        return "_No data returned._"

    # Determine column count from first row
    first = rows[0]
    if isinstance(first, (list, tuple)):
        col_count = len(first)
    else:
        col_count = 1

    # Build headers
    if headers and len(headers) >= col_count:
        hdrs = headers[:col_count]
    else:
        hdrs = [f"col_{i + 1}" for i in range(col_count)]

    # Clean up header names
    hdrs = [h.replace("_", " ").strip().title() for h in hdrs]

    lines = [
        "| " + " | ".join(hdrs) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]

    display_count = min(len(rows), max_rows)
    for row in rows[:display_count]:
        vals = list(row) if isinstance(row, (list, tuple)) else [row]
        # Pad to col_count if short
        vals = vals + [""] * (col_count - len(vals))
        # Format each cell
        clean = []
        for v in vals[:col_count]:
            if v is None:
                clean.append("—")
            elif isinstance(v, float):
                clean.append(fmt_number(v))
            else:
                s = str(v).strip()
                # Truncate very long cell values
                clean.append(s[:120] + "…" if len(s) > 120 else s)
        lines.append("| " + " | ".join(clean) + " |")

    if show_overflow and len(rows) > max_rows:
        lines.append(f"\n_…and {len(rows) - max_rows} more rows (showing {max_rows} of {len(rows)})_")

    return "\n".join(lines)


def format_rows_as_list(
    rows: List[Any],
    name_index: int = 0,
    max_items: int = 50,
) -> str:
    """Render rows as a simple bullet list using the first column."""
    if not rows:
        return "_No items found._"
    items = []
    for row in rows[:max_items]:
        val = row[name_index] if isinstance(row, (list, tuple)) else row
        if val is not None:
            items.append(f"- {str(val).strip()}")
    if len(rows) > max_items:
        items.append(f"- _…and {len(rows) - max_items} more_")
    return "\n".join(items) if items else "_No items found._"


# ── Final answer formatting ────────────────────────────────────────────────────

def format_final_answer(answer: str, tables_used: List[str]) -> str:
    """
    Format the final answer for user display.

    Rules:
      • Strip iteration/timeout error messages
      • Mask internal IDs
      • If answer already has "Executive Summary" header, don't double-wrap
      • Add source attribution footer
    """
    cleaned = answer.strip()

    # Handle error/timeout messages
    if re.search(r"stopped due to|iteration limit|unable to complete", cleaned, re.I):
        cleaned = (
            "I was unable to complete this query within the time limit. "
            "Please try a more specific question."
        )

    # Mask internal object IDs
    cleaned = mask_ids(cleaned)

    # Don't double-wrap if already formatted
    if cleaned.lower().startswith("executive summary"):
        return cleaned

    # Build source attribution
    if tables_used:
        table_text = ", ".join(f"`{t}`" for t in tables_used)
        footer = f"\n\n---\n_Source: {table_text} · Live database query_"
    else:
        footer = ""

    return f"{cleaned}{footer}"


# ── Guard checks ───────────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    """Check if query is a greeting (no DB query needed)."""
    return any(p.match(text.strip()) for p in GREETING_PATTERNS)


def is_blocked(text: str) -> bool:
    """Check if query contains destructive SQL patterns."""
    return any(p.search(text) for p in BLOCKED_SQL_PATTERNS)


def is_vague_query(text: str) -> bool:
    """Check if query is too vague to produce meaningful results."""
    t = normalize_text(text)
    if len(t.split()) <= 2 and not re.search(r"\b(count|list|show|get|total)\b", t):
        return True
    return False


# ── Timing helper ──────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer for latency tracking."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if self.label:
            LOGGER.debug("[Timer] %s: %.1fms", self.label, self.elapsed_ms)

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0