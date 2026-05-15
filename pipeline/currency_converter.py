"""currency_converter.py — Live currency conversion via Frankfurter API.

Free, no API key, live rates, 30+ currencies supported.
Rates cached in-memory for 1 hour — one API call per hour maximum.

Public API:
    get_rates()                          -> Dict[str, float]   # all rates vs USD
    to_usd(amount, currency)             -> Optional[float]    # single conversion
    convert_multi_to_usd(items)          -> (grand_usd, details)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("sql_chatbot")

_BASE_URL = "https://api.frankfurter.app"
_TIMEOUT  = 5  # seconds per request

# ── In-memory rate cache (base: USD) ─────────────────────────────────────────
_CACHE_LOCK  = threading.Lock()
_RATES_CACHE: Dict[str, float] = {}  # {CURRENCY: units_per_1_USD}
_CACHE_TS    = 0.0
_CACHE_TTL   = 3600  # 1 hour


def _fetch_all_rates() -> Dict[str, float]:
    """
    Fetch all exchange rates from Frankfurter (base: USD).

    Response format:
        {"base": "USD", "date": "2026-05-15", "rates": {"INR": 86.24, "AUD": 1.57, ...}}

    Each rate tells how many units of that currency equal 1 USD.
    To convert X INR → USD:  X / rates["INR"]
    """
    try:
        resp = requests.get(f"{_BASE_URL}/latest?from=USD", timeout=_TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        rates = {k.upper(): float(v) for k, v in data.get("rates", {}).items()}
        rates["USD"] = 1.0  # base
        LOGGER.info(
            "CurrencyConverter: loaded %d rates via Frankfurter (date=%s)",
            len(rates), data.get("date", "?"),
        )
        return rates
    except Exception as exc:
        LOGGER.warning("CurrencyConverter: Frankfurter fetch failed: %s", exc)
        return {}


def get_rates(force: bool = False) -> Dict[str, float]:
    """Return live USD-base rates, refreshing the cache only when stale (1 h TTL)."""
    global _RATES_CACHE, _CACHE_TS
    with _CACHE_LOCK:
        if force or not _RATES_CACHE or (time.time() - _CACHE_TS > _CACHE_TTL):
            fresh = _fetch_all_rates()
            if fresh:                      # only overwrite cache on success
                _RATES_CACHE = fresh
                _CACHE_TS    = time.time()
    return dict(_RATES_CACHE)


def to_usd(amount: float, currency: str) -> Optional[float]:
    """
    Convert *amount* of *currency* to USD.

    Returns None when the currency is unsupported (not in Frankfurter's list).

    Examples:
        to_usd(15025, "INR")  →  174.22
        to_usd(2000,  "USD")  →  2000.0
        to_usd(100,   "AOA")  →  None   (if not supported)
    """
    currency = (currency or "USD").upper().strip()
    if currency == "USD":
        return round(float(amount), 2)
    rates = get_rates()
    rate  = rates.get(currency)
    if not rate:
        return None
    return round(float(amount) / rate, 2)


def convert_multi_to_usd(
    items: List[Tuple[str, float]],
) -> Tuple[float, List[Tuple[str, float, Optional[float]]]]:
    """
    Convert multiple (currency, amount) pairs to USD and compute grand total.

    Unsupported currencies contribute 0 to the grand total (marked as None).
    Rates are fetched once per call (from cache — instant if warm).

    Args:
        items: list of (currency_code, amount)

    Returns:
        (grand_total_usd, [(currency, original_amount, usd_equiv_or_None), ...])

    Example:
        items = [("INR", 15025), ("USD", 2000)]
        → (2174.22, [("INR", 15025, 174.22), ("USD", 2000, 2000.0)])
    """
    rates       = get_rates()
    grand_total = 0.0
    result: List[Tuple[str, float, Optional[float]]] = []

    for currency, amount in items:
        currency = (currency or "USD").upper().strip()
        amount   = float(amount)

        if currency == "USD":
            usd = round(amount, 2)
        else:
            rate = rates.get(currency)
            usd  = round(amount / rate, 2) if rate else None

        result.append((currency, amount, usd))
        if usd is not None:
            grand_total += usd

    return round(grand_total, 2), result
