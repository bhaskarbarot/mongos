"""currency_format.py — Currency-aware result formatter for all 3 agents.

Logic mirrors fast_path.py exactly:
  1. Detect currency column + native amount column in SQL result
  2. Group rows by currency → (currency, total_amount) pairs
  3. Convert each pair to USD via currency_converter (Frankfurter API, 1-hr cache)
  4. Supported currencies  → show original amount + USD equivalent in a table
  5. Unsupported currencies → shown as-is in a separate table (honest, no guessing)
  6. Grand total = sum of all supported-currency USD values (bold, prominent)

Column detection (no hardcoding — reads actual column names from query result):
  - Currency column:  any column whose name is exactly or contains 'currency'
  - Amount columns:   grand_total, grandtotal, amount, revenue, total, subtotal,
                      netpayableamount, value  (case-insensitive, no usd suffix)
  - Already-USD cols: any column whose name contains 'usd' → no conversion needed

Public API:
    apply_currency_conversion(rows, columns, sql) -> str | None
      Returns formatted currency-aware string, or None if no currency column found
      (caller falls back to standard table formatting)

    is_money_query(sql) -> bool
      Quick check if an SQL query involves financial amounts
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("sql_chatbot")

# Amount column name keywords (lowercase) — must NOT contain 'usd' to need conversion
_AMOUNT_KEYWORDS = {
    "grand_total", "grandtotal", "amount", "revenue", "total",
    "subtotal", "netpayableamount", "value", "sales", "amt",
}

# Columns that are already in USD — no conversion needed
_USD_KEYWORDS = {"usd", "in_usd"}

# Currency symbols for formatting
_CUR_SYMBOLS: Dict[str, str] = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹",
                                  "AUD": "A$", "CAD": "C$", "SGD": "S$", "AED": "AED "}


def _fmt(amount: float, currency: str = "USD") -> str:
    cur = (currency or "USD").upper().strip()
    sym = _CUR_SYMBOLS.get(cur, f"{cur} ")
    return f"{sym}{amount:,.2f}"


def _find_currency_col(columns: List[str]) -> Optional[int]:
    """Return index of the currency column, or None."""
    for i, c in enumerate(columns):
        if c.lower() in ("currency", "cur") or "currency" in c.lower():
            return i
    return None


def _find_amount_col(columns: List[str]) -> Optional[int]:
    """Return index of the best native-amount column (no USD suffix). None if not found."""
    for i, c in enumerate(columns):
        cl = c.lower().replace('"', '').replace(" ", "_")
        # Skip already-USD columns
        if any(kw in cl for kw in _USD_KEYWORDS):
            continue
        if any(kw in cl for kw in _AMOUNT_KEYWORDS):
            return i
    return None


def _already_usd_col(columns: List[str]) -> Optional[int]:
    """Return index of an already-USD column (e.g. grandtotal_in_usd)."""
    for i, c in enumerate(columns):
        cl = c.lower()
        if any(kw in cl for kw in _USD_KEYWORDS) and any(kw in cl for kw in _AMOUNT_KEYWORDS):
            return i
    return None


def is_money_query(sql: str) -> bool:
    """Quick heuristic: does this SQL involve financial amounts?"""
    low = sql.lower()
    return any(kw in low for kw in [
        "grand_total", "grandtotal", "revenue", "amount", "subtotal",
        "netpayable", "payment", "invoice", "sales", "billing",
    ])


def apply_currency_conversion(
    rows:    List[Any],
    columns: List[str],
    sql:     str = "",
) -> Optional[str]:
    """Try to apply currency-aware formatting to SQL result rows.

    Returns a formatted markdown string if the result contains currency data,
    or None if no currency column was found (caller uses standard formatting).

    Handles 3 result shapes:
      A. (currency, amount)            — already grouped by currency
      B. (label, currency, amount)     — per-entity rows with currency
      C. (amount,) already USD column  — just format as USD, no conversion
    """
    if not rows or not columns:
        return None

    # ── Shape C: already-USD column, no conversion needed ───────────────────
    usd_idx = _already_usd_col(columns)
    if usd_idx is not None and _find_currency_col(columns) is None:
        # Single scalar or aggregate already in USD
        if len(rows) == 1 and len(rows[0]) == 1:
            try:
                val = float(str(rows[0][0]).replace(",", ""))
                return f"**${val:,.2f} USD**"
            except (TypeError, ValueError):
                pass
        # Multi-row, already USD — return None to use standard table formatting
        return None

    cur_idx = _find_currency_col(columns)
    amt_idx = _find_amount_col(columns)

    if cur_idx is None or amt_idx is None:
        return None  # no currency data — use standard formatting

    # ── Build (currency, amount) groups ──────────────────────────────────────
    # Shape A: (currency, amount) — exactly 2 columns
    # Shape B: (label, ..., currency, ..., amount) — more columns
    currency_totals: Dict[str, float] = {}
    entity_rows: List[Tuple] = []     # for shape B multi-entity display

    for row in rows:
        try:
            cur = str(row[cur_idx] or "USD").upper().strip()
            amt = float(str(row[amt_idx]).replace(",", "") or "0")
        except (TypeError, ValueError, IndexError):
            continue
        currency_totals[cur] = currency_totals.get(cur, 0.0) + amt
        entity_rows.append(row)

    if not currency_totals:
        return None

    items: List[Tuple[str, float]] = list(currency_totals.items())

    # ── Convert to USD ────────────────────────────────────────────────────────
    try:
        from pipeline.currency_converter import convert_multi_to_usd
        grand_usd, converted = convert_multi_to_usd(items)
        converted_count = sum(1 for _, _, u in converted if u is not None)
    except Exception as exc:
        LOGGER.warning("Currency conversion failed: %s — showing originals", exc)
        grand_usd  = 0.0
        converted  = [(c, a, None) for c, a in items]
        converted_count = 0

    supported   = [(c, o, u) for c, o, u in converted if u is not None]
    unsupported = [(c, o)    for c, o, u in converted if u is None]

    lines: List[str] = []

    # ── Case: single currency ─────────────────────────────────────────────────
    if len(items) == 1:
        cur0, amt0 = items[0]
        usd0 = converted[0][2] if converted else None
        if cur0 == "USD":
            lines.append(f"**Total: ${amt0:,.2f} USD**")
        else:
            usd_note = f"  ≈ **${usd0:,.2f} USD** at live rates" if usd0 else " *(conversion unavailable)*"
            lines.append(f"**Total: {_fmt(amt0, cur0)}**{usd_note}")

    # ── Case: multiple currencies ─────────────────────────────────────────────
    else:
        lines.append(f"**Grand Total: ${grand_usd:,.2f} USD**")
        if converted_count < len(items):
            lines.append(f"*({converted_count} of {len(items)} currencies converted at live rates)*")
        else:
            lines.append("*(all currencies converted at live rates)*")

        if supported:
            lines.append("")
            lines.append("**Revenue by Currency:**")
            lines.append("| Currency | Original Amount | ≈ USD |")
            lines.append("| --- | --- | --- |")
            for cur, orig, usd_amt in supported:
                lines.append(f"| {cur} | {_fmt(orig, cur)} | ${usd_amt:,.2f} |")
            lines.append(f"| **Total** | | **${grand_usd:,.2f}** |")

        if unsupported:
            lines.append("")
            lines.append("*Could not convert (currency not supported by converter):*")
            lines.append("| Currency | Original Amount |")
            lines.append("| --- | --- |")
            for cur, orig in unsupported:
                lines.append(f"| {cur} | {_fmt(orig, cur)} |")

    # ── Shape B: per-entity rows — show detail table below the totals ─────────
    non_cur_non_amt = [i for i, c in enumerate(columns)
                       if i != cur_idx and i != amt_idx]

    if non_cur_non_amt and len(rows) > 1:
        lines.append("")
        lines.append("**Detail:**")
        detail_cols = [columns[i] for i in non_cur_non_amt] + [columns[cur_idx], columns[amt_idx]]
        lines.append("| " + " | ".join(str(c).replace("_", " ").title() for c in detail_cols) + " |")
        lines.append("| " + " | ".join("---" for _ in detail_cols) + " |")
        for row in rows[:50]:
            cells = [str(row[i]) if row[i] is not None else "—" for i in non_cur_non_amt]
            cur   = str(row[cur_idx] or "USD").upper()
            try:
                amt = float(str(row[amt_idx]).replace(",", "") or "0")
                cells.extend([cur, _fmt(amt, cur)])
            except (TypeError, ValueError):
                cells.extend([cur, "—"])
            lines.append("| " + " | ".join(cells) + " |")
        if len(rows) > 50:
            lines.append(f"*...and {len(rows)-50} more rows*")

    return "\n".join(lines)
