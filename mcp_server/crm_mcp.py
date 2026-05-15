"""crm_mcp.py — Model Context Protocol server for CRM database access.

Exposes live database schema, SQL execution, and query examples as MCP tools.
Claude's MCP client and all project agents can call this server.

Start server:   python mcp_server/crm_mcp.py
              OR ./mcp_server/start_mcp.sh

The 6 tools exposed:
  get_live_schema          — live column metadata from information_schema
  find_column              — fuzzy-match a wrong column name to the real one
  execute_sql              — direct psycopg2 SQL execution (500-row max)
  get_table_relationships  — FK relationships between all CRM tables
  get_query_examples       — seed + saved natural-language→SQL pairs
  save_successful_query    — persist a working query to the examples store
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Add project root to path so we can import pipeline modules ─────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from config import settings
from pipeline.db_schema import (
    _FK_MAP,
    _SOFT_DELETE,
    _get_columns,
    _get_enum_values,
    find_correct_column,
    get_all_crm_tables,
    get_column_map,
)
from pipeline.schema import _run_sql_direct

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("crm_mcp")

# ── MCP server instance ────────────────────────────────────────────────────────
crm_mcp = FastMCP("CRM Database")

# ── Query examples store ───────────────────────────────────────────────────────
_EXAMPLES_FILE = Path(__file__).parent / "query_examples.json"
_examples_lock = threading.Lock()

# Seed examples that ship with the system (5 as specified)
_SEED_EXAMPLES: List[Dict] = [
    {
        "natural_query": "how many deals are there",
        "sql": 'SELECT COUNT(*) AS deal_count FROM "deals" WHERE NOT deleted',
        "tables": ["deals"],
    },
    {
        "natural_query": "list all open deals",
        "sql": (
            'SELECT name, stage, grand_total_in_usd FROM "deals" '
            'WHERE "dealWonAt" IS NULL AND "dealLostAt" IS NULL AND NOT deleted '
            "ORDER BY grand_total_in_usd DESC NULLS LAST LIMIT 50"
        ),
        "tables": ["deals"],
    },
    {
        "natural_query": "total revenue from paid invoices this year",
        "sql": (
            "SELECT COALESCE(SUM(grandtotal_in_usd), 0) AS total_revenue "
            'FROM "invoices" WHERE payment_status = \'paid\' AND NOT deleted '
            "AND EXTRACT(YEAR FROM NULLIF(payment_date,'')::timestamptz) "
            "= EXTRACT(YEAR FROM CURRENT_DATE)"
        ),
        "tables": ["invoices"],
    },
    {
        "natural_query": "top 5 sales reps by confirmed sales amount",
        "sql": (
            'SELECT u.name AS rep_name, COALESCE(SUM(s.grand_total_in_usd), 0) AS total_sales '
            'FROM "users" u '
            'LEFT JOIN "sales" s ON s."salesOwner" = u._id '
            "AND s.status = 'Confirm' AND NOT s.deleted "
            'WHERE u."isActive" = true '
            "GROUP BY u._id, u.name "
            "ORDER BY total_sales DESC LIMIT 5"
        ),
        "tables": ["users", "sales"],
    },
    {
        "natural_query": "count pending tasks by priority",
        "sql": (
            "SELECT priority, COUNT(*) AS task_count "
            'FROM "createtasks" '
            "WHERE status = 'Pending' AND NOT deleted "
            "GROUP BY priority ORDER BY task_count DESC"
        ),
        "tables": ["createtasks"],
    },
]


def _load_examples() -> List[Dict]:
    """Load seed examples merged with any saved successful queries."""
    with _examples_lock:
        if not _EXAMPLES_FILE.exists():
            return list(_SEED_EXAMPLES)
        try:
            saved = json.loads(_EXAMPLES_FILE.read_text(encoding="utf-8"))
            return list(_SEED_EXAMPLES) + saved
        except Exception:
            return list(_SEED_EXAMPLES)


def _save_example(natural_query: str, sql: str, tables: List[str]) -> bool:
    """Persist a working query→SQL pair; skip exact duplicates."""
    with _examples_lock:
        saved: List[Dict] = []
        if _EXAMPLES_FILE.exists():
            try:
                saved = json.loads(_EXAMPLES_FILE.read_text(encoding="utf-8"))
            except Exception:
                saved = []

        existing_keys = {e.get("natural_query", "").lower().strip() for e in saved}
        if natural_query.lower().strip() in existing_keys:
            return True  # already saved

        saved.append({
            "natural_query": natural_query,
            "sql":           sql,
            "tables":        tables,
            "saved_at":      datetime.utcnow().isoformat(),
        })
        _EXAMPLES_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        return True


# ── Schema cache (10-minute TTL) ───────────────────────────────────────────────
_SCHEMA_CACHE: Dict[str, Any] = {}
_SCHEMA_TS:    float = 0.0
_SCHEMA_LOCK   = threading.Lock()
_SCHEMA_TTL    = 600  # seconds


def _build_full_schema(tables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read live column metadata for every requested table from the real DB."""
    global _SCHEMA_CACHE, _SCHEMA_TS

    all_crm = get_all_crm_tables()
    target  = [t for t in (tables or all_crm) if t in all_crm]

    result: Dict[str, Any] = {}
    for tbl in target:
        try:
            raw_cols = _get_columns(tbl)
            col_list = []
            for col_name, data_type, nullable in raw_cols:
                needs_q = any(c.isupper() for c in col_name)
                col_list.append({
                    "name":        col_name,
                    "type":        data_type,
                    "nullable":    nullable == "YES",
                    "needs_quotes": needs_q,
                    "sql_ref":     f'"{col_name}"' if needs_q else col_name,
                })

            # Detect soft-delete column from live introspection
            col_names = {c[0] for c in raw_cols}
            if "deleted" in col_names:
                soft_delete = "NOT deleted"
            elif "isDeleted" in col_names:
                soft_delete = 'NOT "isDeleted"'
            else:
                soft_delete = None  # no soft-delete (vendors table etc.)

            # Sample enum-like values for text columns
            enum_samples: Dict[str, List[str]] = {}
            text_cols = [c[0] for c in raw_cols if "char" in c[1] or c[1] == "text"]
            for col_name in text_cols[:6]:
                vals = _get_enum_values(tbl, col_name, limit=10)
                if vals:
                    enum_samples[col_name] = vals

            # Row count estimate from pg_class (fast, no full scan)
            rc_res = _run_sql_direct(
                f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{tbl}'"
            )
            row_estimate = 0
            if rc_res and rc_res.rows:
                try:
                    row_estimate = int(rc_res.rows[0][0])
                except Exception:
                    pass

            result[tbl] = {
                "columns":           col_list,
                "soft_delete_col":   soft_delete,
                "enum_samples":      enum_samples,
                "row_count_estimate": row_estimate,
            }
        except Exception as exc:
            LOGGER.debug("Schema build error for %s: %s", tbl, exc)
            result[tbl] = {"columns": [], "error": str(exc)}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@crm_mcp.tool()
def get_live_schema(tables: list[str] | None = None) -> dict:
    """Get live CRM database schema from PostgreSQL information_schema.

    Returns column names with correct casing, types, soft-delete column,
    enum value samples, and row count estimates. Cached for 10 minutes.

    Args:
        tables: Optional subset of table names. If None, returns all CRM tables.
    """
    global _SCHEMA_CACHE, _SCHEMA_TS

    now = time.monotonic()
    cache_key = ",".join(sorted(tables or []))

    with _SCHEMA_LOCK:
        cached_entry = _SCHEMA_CACHE.get(cache_key)
        if cached_entry and (now - _SCHEMA_TS) < _SCHEMA_TTL:
            return cached_entry

    schema = _build_full_schema(tables)

    with _SCHEMA_LOCK:
        _SCHEMA_CACHE[cache_key] = schema
        _SCHEMA_TS = time.monotonic()

    return schema


@crm_mcp.tool()
def find_column(approximate_name: str, tables: list[str]) -> dict:
    """Fuzzy-match a wrong/approximate column name to the real DB column.

    Use this when SQL fails with 'column does not exist' to auto-heal the query.
    Search order:
      1. Exact lowercase match in specified tables
      2. Exact lowercase match across all CRM tables
      3. Suffix match  — 'wonAt'  matches 'dealWonAt'
      4. Substring match — 'owner' matches 'salesOwner', 'companyOwner'

    Args:
        approximate_name: Wrong or lowercased column name (e.g., 'salesowner', 'wonAt')
        tables: Tables to search in first (from the failing SQL)
    """
    def _make_result(actual: str, tbl: str) -> dict:
        needs_q = any(c.isupper() for c in actual)
        quoted  = f'"{actual}"' if needs_q else actual
        return {"found": True, "real_name": quoted,
                "needs_quotes": needs_q, "found_in_table": tbl, "searched": tables}

    norm       = approximate_name.lower().strip('"')
    all_tables = get_all_crm_tables()
    search_order = list(dict.fromkeys(tables + all_tables))  # hint tables first, deduped

    # Pass 1 — exact lowercase match
    result = find_correct_column(approximate_name, search_order)
    if result:
        clean = result.strip('"')
        return {"found": True, "real_name": result,
                "needs_quotes": any(c.isupper() for c in clean), "searched": tables}

    # Pass 2 — suffix match: 'wonAt' matches 'dealWonAt', 'lostAt' matches 'dealLostAt'
    for tbl in search_order:
        for actual in get_column_map(tbl).values():
            if actual.lower().endswith(norm):
                return _make_result(actual, tbl)

    # Pass 3 — substring match: 'owner' matches 'salesOwner', 'companyOwner', etc.
    for tbl in search_order:
        for actual in get_column_map(tbl).values():
            if norm in actual.lower() and len(norm) >= 4:  # min 4 chars to avoid false hits
                return _make_result(actual, tbl)

    return {"found": False, "real_name": None, "searched": tables}


@crm_mcp.tool()
def execute_sql(sql: str) -> dict:
    """Execute a SELECT SQL query directly via psycopg2. Hard limit: 500 rows.

    Returns rows, column names, row count, execution time, and any error.
    All values are JSON-safe (JSONB→str, Decimal→float, datetime→ISO).

    Args:
        sql: The SELECT SQL query to execute (read-only)
    """
    import datetime as _dt
    import psycopg2
    from decimal import Decimal

    t0 = time.monotonic()

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            connect_timeout=8,
        )
        conn.set_session(readonly=True, autocommit=True)

        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in (cur.description or [])]
            rows    = cur.fetchall()
        conn.close()

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        def _norm(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, dict):
                return json.dumps(v, default=str)
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, (_dt.datetime, _dt.date)):
                return v.isoformat()
            return v

        normalized   = [[_norm(v) for v in row] for row in rows[:500]]
        total_rows   = len(rows)

        return {
            "rows":         normalized,
            "columns":      columns,
            "row_count":    len(normalized),
            "total_rows":   total_rows,
            "truncated":    total_rows > 500,
            "error":        None,
            "execution_ms": elapsed_ms,
        }

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {
            "rows":         [],
            "columns":      [],
            "row_count":    0,
            "total_rows":   0,
            "truncated":    False,
            "error":        str(exc),
            "execution_ms": elapsed_ms,
        }


@crm_mcp.tool()
def get_table_relationships() -> dict:
    """Return FK relationships between all CRM tables.

    Derived from column naming conventions (owner→users, company→companies, etc.)
    and cross-validated with actual data during schema discovery.

    Returns: {source_table: {column_name: target_table}}
    """
    all_tables = get_all_crm_tables()
    table_set  = set(all_tables)
    result: Dict[str, Dict[str, str]] = {}

    for tbl in all_tables:
        try:
            col_map = get_column_map(tbl)
            tbl_rels: Dict[str, str] = {}
            for col_lower, col_actual in col_map.items():
                target = _FK_MAP.get(col_lower)
                if target and target in table_set:
                    tbl_rels[col_actual] = target
            if tbl_rels:
                result[tbl] = tbl_rels
        except Exception:
            pass

    return result


@crm_mcp.tool()
def get_query_examples() -> list:
    """Return natural language → SQL example pairs for few-shot SQL generation.

    Includes 5 seed examples that ship with the system, plus any previously
    saved successful queries. Agents should use these examples to improve
    SQL generation accuracy.
    """
    return _load_examples()


@crm_mcp.tool()
def save_successful_query(natural_query: str, sql: str, tables: list[str]) -> bool:
    """Persist a working natural language → SQL pair to the examples store.

    Call this after a query succeeds so future similar queries benefit
    from it as a few-shot example. Skips exact-duplicate queries.

    Args:
        natural_query: The user's natural language question
        sql:           The SQL that successfully answered it
        tables:        List of tables referenced in the SQL
    """
    return _save_example(natural_query, sql, tables)


# ══════════════════════════════════════════════════════════════════════════════
# Callable Python API (agents call these directly — no MCP round-trip needed)
# ══════════════════════════════════════════════════════════════════════════════

def mcp_get_live_schema(tables: Optional[List[str]] = None) -> Dict:
    return get_live_schema(tables)  # type: ignore[call-arg]


def mcp_find_column(approximate_name: str, tables: List[str]) -> Dict:
    return find_column(approximate_name, tables)  # type: ignore[call-arg]


def mcp_execute_sql(sql: str) -> Dict:
    return execute_sql(sql)  # type: ignore[call-arg]


def mcp_get_examples() -> List[Dict]:
    return _load_examples()


def mcp_save_query(natural_query: str, sql: str, tables: List[str]) -> bool:
    return _save_example(natural_query, sql, tables)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    LOGGER.info("Starting CRM MCP server (stdio transport)...")
    crm_mcp.run()
