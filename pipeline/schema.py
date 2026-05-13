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
    "grandtotal_in_usd", "grand_total_in_usd",  # USD fields first (cross-currency safe)
    "grand_total", "total", "amount", "totalAmount",
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
            return f'"{name}"'

        first = self.get(table, "first_name", fields)
        last  = self.get(table, "last_name", fields)
        if first and last:
            return (f'TRIM(CONCAT(COALESCE("{first}"::text,\'\'), \' \','
                    f' COALESCE("{last}"::text,\'\')))')
        if first:
            return f'"{first}"'

        ident = self.get(table, "identifier", fields)
        if ident:
            return f'"{ident}"'

        return '"_id"'

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


def invalidate_all_caches() -> None:
    """Clear all schema caches — call after DB schema changes (e.g. column migration)."""
    global _TABLE_FIELDS_CACHE, _SCHEMA_LINKS, _TEXT2SQL_SCHEMA_CACHE
    _TABLE_FIELDS_CACHE.clear()
    _SCHEMA_LINKS = {}
    _TEXT2SQL_SCHEMA_CACHE = ""
    REGISTRY.invalidate()
    LOGGER.info("Schema: all caches invalidated")


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

def run_sql(agent, sql: str, max_retries: int = 1) -> SqlResult:
    """Execute raw SQL via the LangChain sql_db_query tool.

    Features:
      • Finds sql_db_query tool from agent's tool list
      • Handles Decimal('...') strings in output (LangChain quirk)
      • Retries once on transient errors (connection reset, timeout)
      • Never raises — always returns SqlResult (success or failure)

    Args:
        agent:       LangChain AgentExecutor with sql_db_query tool
        sql:         Raw SQL string to execute
        max_retries: Number of retry attempts on transient failure

    Returns:
        SqlResult — callers check .ok() / .error to distinguish empty vs failed.
    """
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
                # We normalise the three most common non-literal forms:
                #   Decimal('123.45')        → '123.45'
                #   datetime.date(Y, M, D)   → 'YYYY-MM-DD'
                #   datetime.datetime(...)   → 'YYYY-MM-DD HH:MM:SS'
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
                    LOGGER.debug("SQL result parse failed (attempt %d): %s", attempt, exc)
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
    """Return column names for a table via information_schema. Cached per table."""
    if table in _TABLE_FIELDS_CACHE:
        return _TABLE_FIELDS_CACHE[table]

    try:
        sql = (
            f"SELECT column_name FROM information_schema.columns"
            f" WHERE table_schema = 'public' AND table_name = '{table}'"
            f" AND column_name NOT IN ('_synced_at')"
            f" ORDER BY ordinal_position"
        )
        _res   = run_sql(agent, sql)
        fields = [r[0] for r in _res.rows if r and r[0]]
    except Exception as exc:
        LOGGER.warning("Schema: field discovery failed for %s: %s", table, exc)
        fields = []

    with _CACHE_LOCK:
        if table not in _TABLE_FIELDS_CACHE:
            _TABLE_FIELDS_CACHE[table] = fields
            LOGGER.debug("Schema: %s has %d columns", table, len(fields))

    return _TABLE_FIELDS_CACHE[table]


def get_column_types(agent, table: str) -> Dict[str, str]:
    """Return {column_name: pg_data_type} for a table via information_schema."""
    try:
        sql = (
            f"SELECT column_name, data_type FROM information_schema.columns"
            f" WHERE table_schema = 'public' AND table_name = '{table}'"
            f" AND column_name NOT IN ('_synced_at')"
            f" ORDER BY ordinal_position"
        )
        _res = run_sql(agent, sql)
        return {r[0]: r[1] for r in _res.rows if r and r[0]}
    except Exception:
        return {}



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

                # Validate with a sample value — direct column access (no JSONB)
                try:
                    _s_res = run_sql(
                        agent,
                        f'SELECT "{field}"::text FROM "{table}"'
                        f' WHERE "{field}" IS NOT NULL'
                        f" AND \"{field}\"::text != '' LIMIT 1",
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
                        f'SELECT 1 FROM "{target_name}"'
                        f" WHERE \"_id\" = '{val_str}' LIMIT 1",
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
    lines = ["## Cross-Table Relationships (use these for JOINs):"]
    for tbl, flds in sorted(links.items()):
        for fld, tgt in sorted(flds.items()):
            lines.append(
                f'  "{tbl}"."{fld}" = "{tgt}"."_id"'
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT2SQL SCHEMA PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_text2sql_schema(agent) -> str:
    """Build a compact schema description optimized for Text2SQL model prompts.

    New column-per-field format — each MongoDB field is a typed PostgreSQL column.
    Includes: SQL rules, table+column list with types, JOINs, CRM hints.
    """
    global _TEXT2SQL_SCHEMA_CACHE

    if _TEXT2SQL_SCHEMA_CACHE:
        return _TEXT2SQL_SCHEMA_CACHE

    with _CACHE_LOCK:
        if _TEXT2SQL_SCHEMA_CACHE:
            return _TEXT2SQL_SCHEMA_CACHE

        table_names = get_table_names(agent)

        # Short type labels for readability
        _TYPE_SHORT = {
            "text": "TEXT", "character varying": "TEXT",
            "numeric": "NUM", "double precision": "NUM", "integer": "NUM", "bigint": "NUM",
            "boolean": "BOOL",
            "jsonb": "JSONB", "json": "JSONB",
            "timestamp with time zone": "TS", "timestamp without time zone": "TS",
        }

        lines = [
            "PostgreSQL column-per-field database — CRITICAL SQL RULES:",
            "  • Use DIRECT column names — NO document->>'field' syntax",
            "  • Strings  (TEXT):    WHERE name ILIKE '%value%'",
            "  • Numbers  (NUM):     WHERE grand_total_in_usd > 1000",
            "  • Booleans (BOOL):    WHERE deleted = false OR deleted IS NULL",
            "  • Date strings(TEXT): WHERE \"closeDate\" > '2025-01-01'",
            "  • Nested objects(JSONB): WHERE \"lastActivity\"->>'type' = 'email'",
            "  • Primary key: \"_id\" TEXT (24-char hex MongoDB ObjectId)",
            "  • Soft-delete: WHERE deleted = false OR deleted IS NULL",
            "  • Always double-quote mixed-case column names: \"closeDate\", \"grandTotal\"",
            "  • Always double-quote table names: FROM \"deals\"",
            "  • Use ILIKE for case-insensitive text matching",
            "",
            "TABLES (column: TYPE):",
        ]

        for table in table_names:
            col_types = get_column_types(agent, table)
            if not col_types:
                continue
            # Show _id first, then up to 30 other columns with types
            cols = []
            if "_id" in col_types:
                cols.append("_id:TEXT")
            for col, dtype in col_types.items():
                if col == "_id":
                    continue
                short = _TYPE_SHORT.get(dtype, dtype[:4].upper())
                cols.append(f'"{col}":{short}' if col[0].isupper() or " " in col else f"{col}:{short}")
                if len(cols) >= 32:
                    cols.append(f"(+{len(col_types) - 32} more)")
                    break
            lines.append(f'  "{table}": {", ".join(cols)}')

        # JOIN relationships
        links = _SCHEMA_LINKS or discover_schema_links(agent)
        if links:
            lines.extend(["", "JOIN RELATIONSHIPS:"])
            for tbl, flds in sorted(links.items()):
                for fld, tgt in sorted(flds.items()):
                    lines.append(f'  "{tbl}"."{fld}" = "{tgt}"."_id"')

        # Rich domain knowledge from DATABASE_METADATA.md
        lines.extend([
            "",
            "═══ CRITICAL DOMAIN RULES ═══",
            "",
            "REVENUE & INVOICES:",
            "  • Revenue = invoices WHERE payment_status='paid', col: grandtotal_in_usd",
            "  • NEVER use sales table for revenue — invoices is the source of truth",
            "  • invoices.grandtotal_in_usd   ← NO underscore before 'in' (revenue field)",
            "  • deals.grand_total_in_usd      ← WITH underscore before 'in' (deal value)",
            "  • payment_status values: 'draft'|'paid'|'confirmed'|'cancelled'|'partial_payment'",
            "  • invoice_date=when issued, due_date=payment deadline, payment_date=when received",
            "  • Pending invoices: payment_status IN ('draft','confirmed','partial_payment')",
            "  • Overdue: \"due_date\"::timestamptz < NOW() AND payment_status NOT IN ('paid','cancelled')",
            "",
            "DEALS:",
            "  • Closed Won = stage='Closed Won' OR \"dealWonAt\" IS NOT NULL",
            "  • Closed Lost = stage='Closed Lost' OR \"dealLostAt\" IS NOT NULL",
            "  • Open deals = stage NOT IN ('Closed Won','Closed Lost')",
            "  • Deal stages: 'Analysis - To be Quoted'|'Negotiation'|'Closed Won'|'Closed Lost'|'On Hold'|'Contract Under Review'|'Quotation Sent'",
            "  • Deal number = 'ELS' || LPAD(sequence_number::text,3,'0')  e.g. ELS001",
            "",
            "COMPANIES:",
            "  • Customers: lifecycleStage='Customer'",
            "  • Leads: lifecycleStage='Lead'",
            "  • Inactive: \"inActiveSince\" IS NOT NULL",
            "  • Won: \"leadWonAt\" IS NOT NULL",
            "",
            "SOFT DELETE (PER TABLE):",
            "  • Most tables: WHERE deleted=false OR deleted IS NULL",
            "  • outreaches: WHERE \"isDeleted\"=false OR \"isDeleted\" IS NULL  ← DIFFERENT!",
            "",
            "SORTING PATTERNS:",
            "  • First/1st/oldest: ORDER BY \"createdAt\" ASC NULLS LAST LIMIT 1",
            "  • Last/latest/recent: ORDER BY \"createdAt\" DESC NULLS LAST LIMIT 1",
            "  • Top N invoices by amount: ORDER BY \"grandtotal_in_usd\" DESC LIMIT N",
            "  • Top N deals by value: ORDER BY \"grand_total_in_usd\" DESC LIMIT N",
            "  • Top N companies by revenue: JOIN invoices, GROUP BY company, ORDER BY SUM DESC",
            "",
            "GROUPING PATTERNS:",
            "  • By stage: GROUP BY stage ORDER BY COUNT(*) DESC",
            "  • By owner: GROUP BY owner → JOIN users ON users._id = owner",
            "  • By month: GROUP BY DATE_TRUNC('month',\"invoice_date\"::timestamptz)",
            "  • By status: GROUP BY payment_status",
            "",
            "KEY FIELD NAMES (exact column names):",
            "  • contacts name: \"firstName\" || ' ' || \"lastName\"",
            "  • companies name: \"companyName\"",
            "  • tasks description: \"Task\" (capital T)",
            "  • targets month: 0-indexed (Jan=0, Feb=1, ..., Dec=11)",
            "  • outreaches status: 'Unassigned'|'Not Contacted'|'Contacted'|'Converted to Deal'",
            "  • tasks status: 'Pending'|'Completed', priority: 'Low'|'Medium'|'High'",
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

    parts = [f'COALESCE("{f}", 0)' for f in candidates]
    return "COALESCE(" + ", ".join(parts) + ", 0)"


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