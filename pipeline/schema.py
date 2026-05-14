"""schema.py — Schema registry, discovery, caching, and SQL execution.

Production-grade schema layer providing:
  • Thread-safe schema caching with TTL-based invalidation
  • SchemaRegistry for semantic field role resolution (name→field mapping)
  • Cross-table FK relationship auto-discovery with validation
  • Text2SQL schema prompt builder (compact, model-optimized)
  • Centralized SQL execution with retry, error normalization, and logging
  • Revenue table/field resolution helpers for fast-path handlers

All caches are module-level singletons, populated lazily on first access.

Public API:
    run_sql(agent, sql)                     -> SqlResult
    get_table_names(agent)                  -> List[str]
    get_document_fields(agent, table)       -> List[str]
    discover_schema_links(agent)            -> Dict[str, Dict[str, str]]
    build_text2sql_schema(agent)            -> str
    find_revenue_table(table_names)         -> Optional[str]
    build_revenue_coalesce(fields)          -> str
    resolve_entity_table(entity, tables, q) -> Optional[str]
    schema_links_prompt(links)              -> str
    REGISTRY                                -> SchemaRegistry instance
"""

from __future__ import annotations

import ast
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("sql_chatbot")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHES (thread-safe via _CACHE_LOCK)
# ══════════════════════════════════════════════════════════════════════════════

# E7: RLock (re-entrant) so discover_schema_links can call get_table_names/get_document_fields
# without deadlocking while holding the lock.
_CACHE_LOCK = threading.RLock()

_TABLE_NAMES_CACHE:    List[str]                = []
_TABLE_FIELDS_CACHE:   Dict[str, List[str]]     = {}
_SCHEMA_LINKS:         Dict[str, Dict[str, str]] = {}
_TEXT2SQL_SCHEMA_CACHE: str                      = ""
_SCHEMA_TIMESTAMP:     float                    = 0.0
_SCHEMA_TTL_SECONDS:   int                      = 600  # 10 min cache TTL

# FK field patterns → probable target table
_FK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^company(?:_?id)?$",                 re.I), "companies"),
    (re.compile(r"^(?:created_?by|createdBy)$",        re.I), "users"),
    (re.compile(r"^(?:sales|contact|company|deal)?_?[Oo]wner$", re.I), "users"),
    (re.compile(r"^(?:assigned_?to|assignedTo)$",      re.I), "users"),
    (re.compile(r"^reporting_?manager$",               re.I), "users"),
    (re.compile(r"^(?:user_?id|userId)$",              re.I), "users"),
    (re.compile(r"^department(?:_?id)?$",              re.I), "departments"),
    (re.compile(r"^contact(?:_?id)?$",                 re.I), "contacts"),
    (re.compile(r"^region(?:_?id)?$",                  re.I), "regions"),
    (re.compile(r"^product(?:_?id)?$",                 re.I), "products"),
    (re.compile(r"^deal(?:_?id)?$",                    re.I), "deals"),
    (re.compile(r"^invoice(?:_?id)?$",                 re.I), "invoices"),
]

_REVENUE_TABLE_KEYWORDS = [
    "invoice", "order", "sale", "payment", "transaction", "billing", "revenue",
]
_CLEAN_REVENUE_FIELDS = [
    "grand_total", "grandtotal_in_usd", "grand_total_in_usd",
    "total", "amount", "totalAmount",
]

# Entity aliases — maps common query words to actual table names
_ENTITY_ALIASES: Dict[str, str] = {
    "deal":       "deals",
    "opportunity": "deals",
    "opportunities": "deals",
    "invoice":    "invoices",
    "bill":       "invoices",
    "billing":    "invoices",
    "contact":    "contacts",
    "person":     "contacts",
    "people":     "contacts",
    "company":    "companies",
    "account":    "companies",
    "client":     "companies",
    "customer":   "companies",
    "user":       "users",
    "employee":   "users",
    "rep":        "users",
    "representative": "users",
    "member":     "users",
    "task":       "createtasks",
    "todo":       "createtasks",
    "follow-up":  "createtasks",
    "followup":   "createtasks",
    "target":     "targets",
    "goal":       "targets",
    "department": "departments",
    "team":       "departments",
    "product":    "products",
    "item":       "products",
    "region":     "regions",
    "territory":  "regions",
    "outreach":   "outreaches",
    "campaign":   "outreaches",
    "sale":       "sales",
    "order":      "sales",
    "sales order": "sales",
}


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA REGISTRY — Semantic role → field name mapping
# ══════════════════════════════════════════════════════════════════════════════

class SchemaRegistry:
    """Maps semantic roles (name, email, amount…) to actual JSONB field names.

    The registry tries exact matches first, then substring matches,
    and caches results per (table, role) pair for O(1) subsequent lookups.
    """

    _ROLE_PATTERNS: Dict[str, List[str]] = {
        "name":       ["companyname", "companyName", "name", "fullname",
                       "full_name", "username", "dealname", "dealName"],
        "first_name": ["firstname", "firstName", "first_name"],
        "last_name":  ["lastname", "lastName", "last_name"],
        "email":      ["email", "emailaddress", "emailAddress", "email_address"],
        "phone":      ["phonenumber", "phoneNumber", "phone_number", "phone", "mobile"],
        "amount":     ["grand_total", "grandtotal_in_usd", "grand_total_in_usd",
                       "totalAmount", "total", "amount", "value", "dealValue"],
        "date":       ["invoice_date", "closedate", "closeDate", "due_date",
                       "dueDate", "sales_date", "salesDate", "start", "date",
                       "createdAt", "created_at"],
        "status":     ["payment_status", "paymentStatus", "status",
                       "leadstatus", "leadStatus", "lifecyclestage", "lifecycleStage"],
        "identifier": ["invoice_number", "invoiceNumber", "so_number", "soNumber",
                       "sales_number", "number", "name", "title", "task", "subject"],
        "owner":      ["owner", "salesowner", "salesOwner", "contactowner",
                       "contactOwner", "companyowner", "companyOwner",
                       "createdby", "createdBy", "assignedTo"],
        "title":      ["jobtitle", "jobTitle", "job_title", "title",
                       "position", "designation"],
        "currency":   ["currency", "currencyCode", "currency_code"],
        "deleted":    ["deleted", "isDeleted", "is_deleted"],
        "stage":      ["stage", "dealStage", "deal_stage", "pipelineStage"],
        "priority":   ["priority", "taskPriority"],
        "source":     ["source", "leadSource", "lead_source"],
        "region":     ["region", "regionId", "territory"],
        "company_ref": ["company", "companyId", "company_id"],
    }

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Optional[str]]] = {}

    def get(self, table: str, role: str, fields: List[str]) -> Optional[str]:
        """Resolve a semantic role to an actual field name for a given table."""
        key = f"{table}:{role}"
        tbl_cache = self._cache.setdefault(table, {})
        if role in tbl_cache:
            return tbl_cache[role]

        patterns = self._ROLE_PATTERNS.get(role, [role])
        fl = {f.lower(): f for f in fields}

        # Exact match first (case-insensitive)
        for p in patterns:
            if p.lower() in fl:
                tbl_cache[role] = fl[p.lower()]
                return fl[p.lower()]

        # Substring match
        for p in patterns:
            for field_lower, field_orig in fl.items():
                if p.lower() in field_lower:
                    tbl_cache[role] = field_orig
                    return field_orig

        tbl_cache[role] = None
        return None

    def display_name_expr(self, table: str, fields: List[str]) -> str:
        """Build a SQL expression for the 'display name' of a record."""
        name = self.get(table, "name", fields)
        if name:
            return f"document->>'{name}'"

        first = self.get(table, "first_name", fields)
        last  = self.get(table, "last_name", fields)
        if first and last:
            return (f"TRIM(CONCAT(COALESCE(document->>'{first}',''), ' ',"
                    f" COALESCE(document->>'{last}','')))")
        if first:
            return f"document->>'{first}'"

        ident = self.get(table, "identifier", fields)
        if ident:
            return f"document->>'{ident}'"

        return "_id"  # fallback: PK column

    def get_all_roles(self, table: str, fields: List[str]) -> Dict[str, str]:
        """Resolve all known roles for a table. Returns {role: field_name}."""
        result = {}
        for role in self._ROLE_PATTERNS:
            field = self.get(table, role, fields)
            if field:
                result[role] = field
        return result

    def invalidate(self, table: Optional[str] = None) -> None:
        """Clear cache for a table or all tables."""
        if table:
            self._cache.pop(table, None)
        else:
            self._cache.clear()


REGISTRY = SchemaRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# E6: TYPED SQL RESULT — callers can distinguish empty vs error
# ══════════════════════════════════════════════════════════════════════════════

class SqlResult:
    """Typed wrapper for run_sql() output.

    Attributes:
        rows:  Result rows (empty list if no data or on error).
        error: None on success, error string on failure.
    """

    def __init__(self, rows: Optional[List[Any]] = None, error: Optional[str] = None) -> None:
        self.rows  = rows if rows is not None else []
        self.error = error

    def ok(self) -> bool:
        """True if the query completed without error."""
        return self.error is None

    def is_empty(self) -> bool:
        """True if the query succeeded but returned no rows."""
        return self.ok() and len(self.rows) == 0


# ══════════════════════════════════════════════════════════════════════════════
# SQL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def _run_sql_direct(sql: str) -> Optional["SqlResult"]:
    """Execute SQL directly via psycopg2, bypassing LangChain entirely.

    This is the RELIABLE fallback path. LangChain's sql_db_query tool
    serialises results to a Python repr string and then TRUNCATES it
    at max_string_length (default 10 000 chars). Any row containing a
    large JSONB `document` column exceeds that limit, so the string is
    cut mid-token and ast.literal_eval() fails — silently returning
    empty rows even though the DB returned real data.

    Direct psycopg2 fetches native Python objects (dicts, lists, etc.)
    so no string serialisation / truncation / ast.literal_eval() happens.
    """
    try:
        from config import settings
        import psycopg2
        import psycopg2.extras  # enables dict/JSON cursor

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
            rows = cur.fetchall()

        conn.close()

        # Normalise: convert each value to JSON-safe Python primitives.
        # psycopg2 returns JSONB as Python dicts — convert to string for
        # uniform downstream handling (format_rows_as_markdown_table etc.)
        def _norm(v):
            if v is None:
                return None
            if isinstance(v, dict):
                return json.dumps(v, default=str)
            # psycopg2 Decimal → float for display
            try:
                from decimal import Decimal
                if isinstance(v, Decimal):
                    return float(v)
            except ImportError:
                pass
            import datetime
            if isinstance(v, (datetime.datetime, datetime.date)):
                return v.isoformat()
            return v

        normalised = [tuple(_norm(v) for v in row) for row in rows]
        return SqlResult(rows=normalised)

    except ImportError:
        # psycopg2 not available — return None so caller falls back to LangChain
        LOGGER.debug("psycopg2 not available for direct execution")
        return None
    except Exception as exc:
        exc_str = str(exc)
        # Distinguish: DB connection errors → return None (try LangChain fallback)
        #              SQL errors → return SqlResult(error=...) so auto-repair can fix the SQL
        is_connection_error = any(k in exc_str.lower() for k in [
            "connection refused", "could not connect", "connection timed out",
            "password authentication", "database", "host", "connect timeout",
        ])
        if is_connection_error:
            LOGGER.debug("Direct psycopg2 connection error: %s", exc)
            return None  # fall back to LangChain tool
        # SQL-level error (AmbiguousColumn, syntax, permission) → propagate as error
        # This triggers auto-repair in run_sql() / text2sql.run()
        LOGGER.warning("Direct psycopg2 SQL error: %s | sql=%.100s", exc, sql)
        return SqlResult(error=exc_str)


def run_sql(agent, sql: str, max_retries: int = 1) -> SqlResult:
    """Execute raw SQL, with direct psycopg2 as the primary path.

    Strategy (in order):
      1. Direct psycopg2 — bypasses LangChain serialisation/truncation bug.
         This is the PRIMARY path. It never truncates and never silently
         drops rows because of ast.literal_eval() failures.
      2. LangChain sql_db_query tool — kept as the legacy fallback.

    The LangChain tool has a critical silent-failure bug:
      • tool.run() serialises the result to a Python repr string
      • LangChain truncates that string at max_string_length (10 000 chars)
      • Any row with a large JSONB `document` column exceeds the limit
      • The truncated string causes ast.literal_eval() to raise SyntaxError
      • The original code caught that silently and returned rows=[]
      → "No data found" for queries that DID return real data

    Args:
        agent:       LangChain AgentExecutor (used for LangChain fallback only)
        sql:         Raw SQL string to execute
        max_retries: Retry attempts on transient failure

    Returns:
        SqlResult — callers check .ok() / .error to distinguish empty vs failed.
    """
    # ── PRIMARY: direct psycopg2 ──────────────────────────────────────────────
    direct = _run_sql_direct(sql)
    if direct is not None:
        LOGGER.debug(
            "run_sql(direct): %d rows | sql=%.100s",
            len(direct.rows), sql,
        )
        return direct

    # ── FALLBACK: LangChain sql_db_query tool ─────────────────────────────────
    LOGGER.warning("Direct psycopg2 unavailable — falling back to LangChain tool")
    tool = next(
        (t for t in getattr(agent, "tools", [])
         if getattr(t, "name", "") == "sql_db_query"),
        None,
    )
    if not tool:
        LOGGER.error("sql_db_query tool not found on agent")
        return SqlResult(error="sql_db_query tool not found on agent")

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            raw = tool.run(sql)

            if isinstance(raw, list):
                return SqlResult(rows=raw)

            if isinstance(raw, str):
                # LangChain serialises Python objects into the result string.
                # Normalise the most common non-literal forms before eval.
                sanitized = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", raw)
                sanitized = re.sub(
                    r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*\d+)?\)",
                    lambda m: f"'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d} "
                              f"{int(m.group(4)):02d}:{int(m.group(5)):02d}'",
                    sanitized,
                )
                sanitized = re.sub(
                    r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)",
                    lambda m: f"'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'",
                    sanitized,
                )
                try:
                    parsed = ast.literal_eval(sanitized)
                    return SqlResult(rows=parsed if isinstance(parsed, list) else [])
                except (ValueError, SyntaxError) as exc:
                    # Log at WARNING — this is a data-loss event, not debug noise
                    LOGGER.warning(
                        "LangChain result parse failed (truncation likely): %s | "
                        "raw_len=%d | sql=%.100s",
                        exc, len(raw), sql,
                    )
                    # Try treating the raw string as an error message from the DB
                    if raw.strip().lower().startswith("error"):
                        return SqlResult(error=raw.strip())
                    # Result truncated — return empty rather than crash
                    return SqlResult(rows=[])

            return SqlResult(rows=[])

        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            transient = any(k in err_str for k in [
                "connection", "timeout", "reset", "broken pipe", "eof",
            ])
            if transient and attempt < max_retries:
                LOGGER.warning(
                    "SQL transient error (retry %d/%d): %s | SQL: %.80s",
                    attempt + 1, max_retries, exc, sql,
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            LOGGER.warning("SQL execution failed: %s | SQL: %.100s", exc, sql)
            return SqlResult(error=str(exc))

    return SqlResult(error=str(last_error) if last_error else "unknown error")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE / FIELD DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _is_cache_stale() -> bool:
    """Check if schema cache has expired."""
    if not _TABLE_NAMES_CACHE:
        return True
    return (time.time() - _SCHEMA_TIMESTAMP) > _SCHEMA_TTL_SECONDS



def get_table_names(agent) -> List[str]:
    """Return sorted list of all table names. Cached with TTL.

    E7: double-checked locking — fast read outside lock, safe write inside lock.
    SQL tool.run() is intentionally outside the lock to avoid blocking other threads.
    """
    global _TABLE_NAMES_CACHE, _SCHEMA_TIMESTAMP

    # Fast path: no lock needed for a stale check
    if _TABLE_NAMES_CACHE and not _is_cache_stale():
        return _TABLE_NAMES_CACHE

    tool = next(
        (t for t in getattr(agent, "tools", [])
         if getattr(t, "name", "") == "sql_db_list_tables"),
        None,
    )
    if not tool:
        LOGGER.error("sql_db_list_tables tool not found")
        return []

    # Execute outside lock — slow I/O must not hold the cache lock
    raw = tool.run("")
    if not isinstance(raw, str):
        return _TABLE_NAMES_CACHE or []

    tables = sorted([p.strip() for p in raw.split(",") if p.strip()])

    # Safe path: write under lock, double-check to avoid duplicate writes
    with _CACHE_LOCK:
        if not _TABLE_NAMES_CACHE or _is_cache_stale():
            _TABLE_NAMES_CACHE = tables
            _SCHEMA_TIMESTAMP  = time.time()
            LOGGER.info("Schema: discovered %d tables", len(tables))

    return _TABLE_NAMES_CACHE


def get_document_fields(agent, table: str) -> List[str]:
    """Return field names for a table.  Cached per table.

    Strategy (in order):
      1. JSONB document keys  — fast when document column is populated
      2. information_schema   — reliable fallback for tables whose document
                                column has not been populated yet

    E7: double-checked locking — SQL execution is outside the lock.
    """
    # Fast path: no lock needed for read
    if table in _TABLE_FIELDS_CACHE:
        return _TABLE_FIELDS_CACHE[table]

    # ── 1. JSONB document keys ────────────────────────────────────────────────
    fields: List[str] = []
    try:
        sql = (
            f'SELECT DISTINCT key FROM "{table}",'
            f" jsonb_object_keys(document) AS key LIMIT 300"
        )
        _res   = run_sql(agent, sql)
        fields = sorted([r[0] for r in _res.rows if r and r[0]])
    except Exception as exc:
        LOGGER.warning("Schema: JSONB field discovery failed for %s: %s", table, exc)

    # ── 2. information_schema fallback (when document column is empty) ────────
    _SYSTEM_COLS = frozenset({"document", "updated_at", "_synced_at"})
    if not fields:
        try:
            _col_sql = (
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}' AND table_schema = 'public' "
                "ORDER BY ordinal_position"
            )
            _c_res = run_sql(agent, _col_sql)
            fields = [
                r[0] for r in _c_res.rows
                if r and r[0] and r[0] not in _SYSTEM_COLS
            ]
            LOGGER.info("Schema: %s — using info_schema (%d cols)", table, len(fields))
        except Exception as exc2:
            LOGGER.warning("Schema: info_schema fallback failed for %s: %s", table, exc2)
            fields = []

    # Safe path: write under lock
    with _CACHE_LOCK:
        if table not in _TABLE_FIELDS_CACHE:
            _TABLE_FIELDS_CACHE[table] = fields
            LOGGER.debug("Schema: %s has %d fields", table, len(fields))

    return _TABLE_FIELDS_CACHE[table]



# ══════════════════════════════════════════════════════════════════════════════
# CROSS-TABLE RELATIONSHIP DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_schema_links(agent) -> Dict[str, Dict[str, str]]:
    """Auto-discover FK relationships between tables via field patterns + data validation.

    Strategy:
      1. Scan each table's fields for FK-like names (company, createdBy, etc.)
      2. Sample one value and check if it looks like a MongoDB ObjectID
      3. Verify the ID exists in the candidate target table
      4. Cache the validated link

    Returns:
        Nested dict: {source_table: {field_name: target_table}}
    """
    global _SCHEMA_LINKS

    if _SCHEMA_LINKS:
        return _SCHEMA_LINKS

    with _CACHE_LOCK:
        if _SCHEMA_LINKS:
            return _SCHEMA_LINKS

        table_names = get_table_names(agent)
        table_lower = {t.lower(): t for t in table_names}
        links: Dict[str, Dict[str, str]] = {}

        for table in table_names:
            fields = get_document_fields(agent, table)
            for field in fields:
                # Match field name against FK patterns
                target_name = None
                for pattern, candidate in _FK_PATTERNS:
                    if pattern.match(field):
                        target_name = table_lower.get(candidate)
                        break

                if not target_name or target_name == table:
                    continue

                # Validate with a sample value (SQL runs outside lock via RLock re-entrancy)
                try:
                    _s_res = run_sql(
                        agent,
                        f"SELECT document->>'{field}' FROM \"{table}\""
                        f" WHERE document->>'{field}' IS NOT NULL"
                        f" AND document->>'{field}' != '' LIMIT 1",
                    )
                    val = _s_res.rows[0][0] if _s_res.rows and _s_res.rows[0] else None
                    if not val:
                        continue

                    val_str = str(val).strip()
                    # Check for MongoDB ObjectID pattern (24 hex chars)
                    if not re.match(r"^[0-9a-f]{24}$", val_str):
                        continue

                    # Verify the ID exists in the target table
                    _v_res = run_sql(
                        agent,
                        f"SELECT 1 FROM \"{target_name}\""
                        f" WHERE _id = '{val_str}' LIMIT 1",
                    )
                    if _v_res.rows:
                        links.setdefault(table, {})[field] = target_name
                        LOGGER.info("Schema link: %s.%s → %s", table, field, target_name)

                except Exception as exc:
                    LOGGER.debug("Schema link check failed: %s.%s: %s", table, field, exc)

        _SCHEMA_LINKS = links
        return links


def schema_links_prompt(links: Dict[str, Dict[str, str]]) -> str:
    """Generate a human-readable prompt section for cross-table relationships."""
    if not links:
        return ""
    lines = ["## Cross-Table Relationships (always use these for JOINs):"]
    for tbl, flds in sorted(links.items()):
        for fld, tgt in sorted(flds.items()):
            lines.append(
                f"  JOIN: \"{tbl}\" t1 → \"{tgt}\" t2"
                f"  ON t1.document->>'{fld}' = t2.document->>'_id'"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT2SQL SCHEMA PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_text2sql_schema(agent) -> str:
    """Build a compact schema description optimized for Text2SQL model prompts.

    Includes:
      • Critical JSONB access rules
      • Table names with their top fields
      • Cross-table JOIN relationships
      • CRM domain hints (revenue fields, soft delete, etc.)
    """
    global _TEXT2SQL_SCHEMA_CACHE

    if _TEXT2SQL_SCHEMA_CACHE:
        return _TEXT2SQL_SCHEMA_CACHE

    with _CACHE_LOCK:
        if _TEXT2SQL_SCHEMA_CACHE:
            return _TEXT2SQL_SCHEMA_CACHE

        table_names = get_table_names(agent)
        lines = [
            "PostgreSQL CRM database — CRITICAL SQL RULES:",
            "  • Primary key: _id (TEXT) — use _id, NOT id",
            "  • ALL field access via JSONB: document->>'fieldName'",
            "  • Numbers: NULLIF(document->>'field','')::numeric",
            "  • Dates: NULLIF(document->>'field','')::timestamptz",
            "  • Soft-delete: WHERE COALESCE(document->>'deleted','false')!='true'",
            "  • For outreaches: WHERE COALESCE(document->>'isDeleted','false')!='true'",
            "  • Table names MUST be in double quotes: FROM \"tableName\"",
            "  • Use ILIKE for case-insensitive text matching",
            "  • JOIN syntax: LEFT JOIN \"users\" u ON u._id = t.document->>'owner'",
            "",
            "KEY DOMAIN RULES:",
            "  • Revenue = invoices WHERE payment_status='paid', field: grandtotal_in_usd",
            "  • Confirmed sales revenue = sales WHERE status='Confirm', field: grand_total_in_usd",
            "  • Open deals: document->>'dealWonAt' IS NULL AND document->>'dealLostAt' IS NULL",
            "  • Won deals: document->>'dealWonAt' IS NOT NULL",
            "  • Tasks table name: createtasks (NOT tasks). Status: 'Pending'/'Completed'",
            "  • createtasks.document->>'Task' = task title (capital T)",
            "  • createtasks JOIN users: createtasks.document->>'createdBy' = users._id",
            "  • Targets: targets.document->>'userId' = users._id, field: targetInUSD",
            "  • Companies lifecycle: document->>'lifecycleStage' = 'Lead'/'Customer'/'Partner'",
            "",
            "TABLES & FIELDS:",
        ]

        for table in table_names:
            fields = get_document_fields(agent, table)
            if fields:
                # Show up to 25 fields for completeness
                field_str = ", ".join(fields[:25])
                if len(fields) > 25:
                    field_str += f" (+{len(fields) - 25} more)"
                lines.append(f"  \"{table}\": [{field_str}]")

        # Add relationship context
        links = _SCHEMA_LINKS or discover_schema_links(agent)
        if links:
            lines.extend(["", "JOIN RELATIONSHIPS:"])
            for tbl, flds in sorted(links.items()):
                for fld, tgt in sorted(flds.items()):
                    lines.append(
                        f"  \"{tbl}\".document->>'{fld}' = \"{tgt}\".document->>'_id'"
                    )

        # CRM domain hints
        lines.extend([
            "",
            "CRM DOMAIN QUICK REFERENCE:",
            "  • Revenue:  SELECT SUM(NULLIF(document->>'grandtotal_in_usd','')::numeric)",
            "              FROM \"invoices\" WHERE document->>'payment_status'='paid'",
            "  • Open deals:  WHERE document->>'dealWonAt' IS NULL",
            "                 AND document->>'dealLostAt' IS NULL",
            "  • Won deals:   WHERE document->>'dealWonAt' IS NOT NULL",
            "  • Targets:     targets table — targetInUSD per userId per month/year",
            "  • Task title:  createtasks.document->>'Task'  (capital T)",
            "  • Task JOIN:   createtasks.document->>'createdBy' = users._id",
            "  • Outreaches:  use isDeleted field (NOT deleted)",
            "  • Sales owner: sales.document->>'salesOwner' = users._id",
            "  • Deals owner: deals.document->>'owner' = users._id",
        ])

        _TEXT2SQL_SCHEMA_CACHE = "\n".join(lines)
        return _TEXT2SQL_SCHEMA_CACHE


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA HELPERS (used by fast-path and text2sql)
# ══════════════════════════════════════════════════════════════════════════════

def find_revenue_table(table_names: List[str]) -> Optional[str]:
    """Find the most likely revenue/invoice table."""
    tl_map = {t.lower(): t for t in table_names}

    # Exact matches first
    for kw in _REVENUE_TABLE_KEYWORDS:
        if kw in tl_map:
            return tl_map[kw]
        if f"{kw}s" in tl_map:
            return tl_map[f"{kw}s"]

    # Prefix matches
    for kw in _REVENUE_TABLE_KEYWORDS:
        for t_lower, t_orig in tl_map.items():
            if t_lower.startswith(kw):
                return t_orig

    # Substring matches
    for kw in _REVENUE_TABLE_KEYWORDS:
        for t_lower, t_orig in tl_map.items():
            if kw in t_lower:
                return t_orig

    return None


def build_revenue_coalesce(fields: List[str]) -> str:
    """Build a COALESCE expression for revenue amount fields."""
    fl = {f.lower(): f for f in fields}

    # Try known clean fields first
    candidates = [fl[k] for k in [c.lower() for c in _CLEAN_REVENUE_FIELDS] if k in fl]

    # Fallback: any field with "grand_total" in name
    if not candidates:
        candidates = [f for f in fields if re.search(r"\bgrand_?total\b", f, re.I)]

    # Fallback: any field with "total" or "amount"
    if not candidates:
        candidates = [f for f in fields if re.search(r"\b(total|amount)\b", f, re.I)]

    if not candidates:
        candidates = ["grand_total"]

    # Use JSONB document access (document column is now populated from flat cols)
    parts = [f"NULLIF(document->>'{f}','')::numeric" for f in candidates]
    return "COALESCE(" + ", ".join(parts) + ", 0)"


def col(field: str) -> str:
    """Return the JSONB document access expression for a field.

    Centralised so a future schema change only needs updating here.
    """
    return f"document->>'{field}'"


def col_num(field: str) -> str:
    """Return a numeric JSONB access expression (safe cast)."""
    return f"NULLIF(document->>'{field}','')::numeric"


def col_ts(field: str) -> str:
    """Return a timestamptz JSONB access expression (safe cast)."""
    return f"NULLIF(document->>'{field}','')::timestamptz"


def resolve_entity_table(
    entity: str,
    table_names: List[str],
    query_text: str,
) -> Optional[str]:
    """Resolve a natural-language entity name to an actual table name.

    Resolution order:
      1. Alias map (deal→deals, customer→companies, etc.)
      2. Exact match (case-insensitive)
      3. Singular/plural variants
      4. Suffix match
      5. Query text scan for any table name mention
    """
    ec = re.sub(r"[^a-zA-Z0-9_]", "", (entity or "").lower())
    low = re.sub(r"\s+", " ", query_text.strip().lower())
    tm = {t.lower(): t for t in table_names}

    # 1. Alias map
    if ec in _ENTITY_ALIASES:
        alias_target = _ENTITY_ALIASES[ec]
        if alias_target in tm:
            return tm[alias_target]

    # 2. Exact match
    if ec in tm:
        return tm[ec]

    # 3. Singular/plural
    if ec.endswith("s") and ec[:-1] in tm:
        return tm[ec[:-1]]
    if ec and f"{ec}s" in tm:
        return tm[f"{ec}s"]
    if ec.endswith("ies"):
        singular = ec[:-3] + "y"
        if singular in tm:
            return tm[singular]

    # 4. Suffix match
    for tl, t in tm.items():
        if ec and tl.endswith(ec):
            return t

    # 5. Scan query text for table name mentions
    for table in table_names:
        t = table.lower()
        if re.search(rf"\b{re.escape(t)}\b", low):
            return table
        if t.endswith("s") and re.search(rf"\b{re.escape(t[:-1])}\b", low):
            return table

    # 6. Alias scan on full query
    for alias, target in _ENTITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low) and target in tm:
            return tm[target]

    return None