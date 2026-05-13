"""record_lookup.py — Fast record lookup by business number (SO, ELSN, ELS).

Handles: SO01241, ELSN/2026/018, ELS042
Expands JSONB items arrays as markdown tables.
Resolves company/user IDs to names via JOINs.
"""

import json
import re
import logging
from typing import Optional, Dict, Any

from pipeline.schema import run_sql

LOGGER = logging.getLogger("sql_chatbot")


def _expand_items(items_raw, currency: str = "USD") -> str:
    """Convert JSONB items array to readable markdown table."""
    if not items_raw:
        return ""
    try:
        items = json.loads(items_raw) if isinstance(items_raw, str) else items_raw
        if not items:
            return ""
        lines = [
            "\n**Line Items:**\n",
            "| Product | Qty | Unit Price | Total ({}) |".format(currency),
            "| --- | --- | --- | --- |",
        ]
        for item in items:
            name      = item.get("product_name", "—")
            qty       = item.get("quantity", 0)
            unit      = item.get("unit_price", 0)
            total_usd = item.get("total_price_in_usd", item.get("total_price", 0))
            try:
                lines.append(
                    f"| {name} | {qty} | {float(unit):,.2f} | USD {float(total_usd):,.2f} |"
                )
            except (TypeError, ValueError):
                lines.append(f"| {name} | {qty} | {unit} | {total_usd} |")
        return "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("_expand_items failed: %s", exc)
        return ""


def _so_lookup(agent, so_number: str) -> Optional[Dict[str, Any]]:
    """Look up a sales order by SO number with company and owner."""
    digits = re.search(r"\d+", so_number)
    if not digits:
        return None
    padded = "SO" + digits.group(0).zfill(5)
    bare   = "SO" + digits.group(0)

    sql = f"""
        SELECT s.sales_number, s.status, s.grand_total, s.grand_total_in_usd,
               s.currency, s.sales_date, s.items,
               c."companyName", u.name AS sales_rep,
               s.tax_amount, s.discount_amount, s.subtotal
        FROM sales s
        LEFT JOIN companies c ON c._id = s.company::text
        LEFT JOIN users     u ON u._id = s."salesOwner"::text
        WHERE s.sales_number IN ('{padded}', '{bare}')
          AND (s.deleted = false OR s.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row  = res.rows[0]
    cols = ["sales_number", "status", "grand_total", "grand_total_in_usd",
            "currency", "sales_date", "items", "companyName", "sales_rep",
            "tax_amount", "discount_amount", "subtotal"]
    data = dict(zip(cols, row))

    def _num(v, default=0):
        try:
            return float(v or default)
        except (TypeError, ValueError):
            return default

    lines = [
        f"**Sales Order: {data.get('sales_number', bare)}**\n",
        f"- **Company:** {data.get('companyName') or '—'}",
        f"- **Status:** {data.get('status') or '—'}",
        f"- **Sales Rep:** {data.get('sales_rep') or '—'}",
        f"- **Date:** {str(data.get('sales_date', ''))[:10] or '—'}",
        f"- **Currency:** {data.get('currency') or 'USD'}",
        f"- **Subtotal:** {_num(data.get('subtotal')):,.2f}",
        f"- **Tax:** {_num(data.get('tax_amount')):,.2f}",
        f"- **Discount:** {_num(data.get('discount_amount')):,.2f}",
        f"- **Grand Total:** {_num(data.get('grand_total')):,.2f}"
        f" (USD {_num(data.get('grand_total_in_usd')):,.2f})",
    ]

    items_text = _expand_items(data.get("items"), data.get("currency", "USD"))
    if items_text:
        lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["sales", "companies", "users"],
        "confidence":  0.98,
        "sql_queries": [sql.strip()],
    }


def _invoice_lookup(agent, inv_number: str) -> Optional[Dict[str, Any]]:
    """Look up an invoice by invoice_number (partial/ILIKE match)."""
    safe = inv_number.replace("'", "''")
    sql = f"""
        SELECT i.invoice_number, i.payment_status, i.approval_status,
               i.grandtotal_in_usd, i.currency, i.invoice_date, i.due_date,
               i.payment_date, i.items,
               COALESCE(c."companyName", i."companyName") AS company,
               u.name AS owner_name, i.notes
        FROM invoices i
        LEFT JOIN companies c ON c._id = i.company::text
        LEFT JOIN users     u ON u._id = i."invoiceOwner"::text
        WHERE i.invoice_number ILIKE '%{safe}%'
          AND (i.deleted = false OR i.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row  = res.rows[0]
    cols = ["invoice_number", "payment_status", "approval_status", "grandtotal_in_usd",
            "currency", "invoice_date", "due_date", "payment_date", "items",
            "company", "owner_name", "notes"]
    data = dict(zip(cols, row))

    def _num(v, default=0):
        try:
            return float(v or default)
        except (TypeError, ValueError):
            return default

    lines = [
        f"**Invoice: {data.get('invoice_number', safe)}**\n",
        f"- **Company:** {data.get('company') or '—'}",
        f"- **Payment Status:** {data.get('payment_status') or '—'}",
        f"- **Approval Status:** {data.get('approval_status') or '—'}",
        f"- **Amount (USD):** {_num(data.get('grandtotal_in_usd')):,.2f}",
        f"- **Currency:** {data.get('currency') or 'USD'}",
        f"- **Invoice Date:** {str(data.get('invoice_date', ''))[:10] or '—'}",
        f"- **Due Date:** {str(data.get('due_date', ''))[:10] or '—'}",
        f"- **Payment Date:** {str(data.get('payment_date') or 'Not paid')[:10]}",
        f"- **Owner:** {data.get('owner_name') or '—'}",
    ]
    if data.get("notes"):
        lines.append(f"- **Notes:** {str(data['notes'])[:150]}")

    items_text = _expand_items(data.get("items"), data.get("currency", "USD"))
    if items_text:
        lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["invoices", "companies", "users"],
        "confidence":  0.98,
        "sql_queries": [sql.strip()],
    }


def _deal_lookup(agent, deal_ref: str) -> Optional[Dict[str, Any]]:
    """Look up a deal by ELS number (e.g. ELS042 → sequence_number=42)."""
    digits = re.search(r"\d+", deal_ref)
    if not digits:
        return None
    seq_num = int(digits.group(0))

    sql = f"""
        SELECT d.name, d.stage, d.grand_total_in_usd, d.currency,
               d."closeDate", d."dealWonAt", d."dealLostAt", d.sequence_number,
               c."companyName", u.name AS owner_name, d."lineItems"
        FROM deals d
        LEFT JOIN companies c ON c._id = d.company::text
        LEFT JOIN users     u ON u._id = d.owner::text
        WHERE d.sequence_number = {seq_num}
          AND (d.deleted = false OR d.deleted IS NULL)
        LIMIT 1
    """
    res = run_sql(agent, sql)
    if res.error or not res.rows:
        return None

    row  = res.rows[0]
    cols = ["name", "stage", "grand_total_in_usd", "currency", "closeDate",
            "dealWonAt", "dealLostAt", "sequence_number", "companyName",
            "owner_name", "lineItems"]
    data = dict(zip(cols, row))

    def _num(v, default=0):
        try:
            return float(v or default)
        except (TypeError, ValueError):
            return default

    deal_num = "ELS" + str(seq_num).zfill(3)
    status = (
        "Won"  if data.get("dealWonAt")  else
        "Lost" if data.get("dealLostAt") else
        "Open"
    )

    lines = [
        f"**Deal {deal_num}: {data.get('name') or '—'}**\n",
        f"- **Company:** {data.get('companyName') or '—'}",
        f"- **Stage:** {data.get('stage') or '—'}",
        f"- **Status:** {status}",
        f"- **Owner:** {data.get('owner_name') or '—'}",
        f"- **Value (USD):** {_num(data.get('grand_total_in_usd')):,.2f}",
        f"- **Close Date:** {str(data.get('closeDate', ''))[:10] or '—'}",
    ]

    items_text = _expand_items(data.get("lineItems"), data.get("currency", "USD"))
    if items_text:
        lines.append(items_text)

    return {
        "answer":      "\n".join(lines),
        "tables_used": ["deals", "companies", "users"],
        "confidence":  0.98,
        "sql_queries": [sql.strip()],
    }


def run(query: str, agent) -> Optional[Dict[str, Any]]:
    """
    Try to handle the query as a record-by-number lookup.
    Returns None if no pattern matched — caller continues to next pipeline layer.

    Patterns recognized:
        SO01241   SO 01241   so1241     → sales order
        ELSN/2026/018  ELSN2026018     → invoice
        ELS042    ELS42                → deal by sequence_number
    """
    text = query.strip()

    # SO number: SO01241, SO 01241, so1241
    so_m = re.search(r'\bSO\s*(\d{4,6})\b', text, re.IGNORECASE)
    if so_m:
        result = _so_lookup(agent, "SO" + so_m.group(1))
        if result:
            LOGGER.info("RecordLookup HIT: SO%s", so_m.group(1))
            return result

    # Invoice number: ELSN/2026/018, ELSN2026018, ELSN/2025/24
    inv_m = re.search(r'\b(ELSN[/\s]?\d{4}[/\s]?\d+)\b', text, re.IGNORECASE)
    if inv_m:
        result = _invoice_lookup(agent, inv_m.group(1))
        if result:
            LOGGER.info("RecordLookup HIT: %s", inv_m.group(1))
            return result

    # Deal number: ELS042, ELS42 — must NOT match ELSN (invoice prefix)
    deal_m = re.search(r'\bELS(\d{2,4})\b', text, re.IGNORECASE)
    if deal_m and not re.search(r'\bELSN', text, re.IGNORECASE):
        result = _deal_lookup(agent, "ELS" + deal_m.group(1))
        if result:
            LOGGER.info("RecordLookup HIT: ELS%s", deal_m.group(1))
            return result

    return None
