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
    run_sql(agent, sql)                     -> List[Any]
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
# MODULE-LEVEL CACHES (thread-safe via _LOCK)
# ══════════════════════════════════════════════════════════════════════════════

_LOCK = threading.Lock()

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

        return "id::text"

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
# SQL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_sql(agent, sql: str, max_retries: int = 1) -> List[Any]:
    """Execute raw SQL via the LangChain sql_db_query tool.

    Features:
      • Finds sql_db_query tool from agent's tool list
      • Handles Decimal('...') strings in output (LangChain quirk)
      • Retries once on transient errors (connection reset, timeout)
      • Returns empty list on all errors (never raises to caller)

    Args:
        agent:       LangChain AgentExecutor with sql_db_query tool
        sql:         Raw SQL string to execute
        max_retries: Number of retry attempts on transient failure

    Returns:
        List of tuples/rows, or empty list on failure
    """
    tool = next(
        (t for t in getattr(agent, "tools", [])
         if getattr(t, "name", "") == "sql_db_query"),
        None,
    )
    if not tool:
        LOGGER.error("sql_db_query tool not found on agent")
        return []

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = tool.run(sql)

            if isinstance(raw, list):
                return raw

            if isinstance(raw, str):
                # Handle LangChain Decimal('...') serialization
                sanitized = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", raw)
                try:
                    parsed = ast.literal_eval(sanitized)
                    return parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError) as exc:
                    LOGGER.debug("SQL result parse failed (attempt %d): %s", attempt, exc)
                    return []

            return []

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
            return []

    return []


# ══════════════════════════════════════════════════════════════════════════════
# TABLE / FIELD DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _is_cache_stale() -> bool:
    """Check if schema cache has expired."""
    if not _TABLE_NAMES_CACHE:
        return True
    return (time.time() - _SCHEMA_TIMESTAMP) > _SCHEMA_TTL_SECONDS


def invalidate_schema_cache() -> None:
    """Force refresh of all schema caches on next access."""
    global _TABLE_NAMES_CACHE, _TABLE_FIELDS_CACHE, _SCHEMA_LINKS
    global _TEXT2SQL_SCHEMA_CACHE, _SCHEMA_TIMESTAMP
    with _LOCK:
        _TABLE_NAMES_CACHE    = []
        _TABLE_FIELDS_CACHE   = {}
        _SCHEMA_LINKS         = {}
        _TEXT2SQL_SCHEMA_CACHE = ""
        _SCHEMA_TIMESTAMP     = 0.0
        REGISTRY.invalidate()
    LOGGER.info("Schema caches invalidated")


def get_table_names(agent) -> List[str]:
    """Return sorted list of all table names. Cached with TTL.
    NOTE: does NOT hold _LOCK while calling tool.run() to avoid deadlock.
    """
    global _TABLE_NAMES_CACHE, _SCHEMA_TIMESTAMP

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

    raw = tool.run("")
    if not isinstance(raw, str):
        return _TABLE_NAMES_CACHE or []

    tables = sorted([p.strip() for p in raw.split(",") if p.strip()])
    _TABLE_NAMES_CACHE = tables
    _SCHEMA_TIMESTAMP  = time.time()
    LOGGER.info("Schema: discovered %d tables", len(tables))
    return tables


def get_document_fields(agent, table: str) -> List[str]:
    """Return JSONB field names for a table. Cached per table.
    NOTE: does NOT hold _LOCK while running SQL to avoid deadlock with discover_schema_links.
    """
    if table in _TABLE_FIELDS_CACHE:
        return _TABLE_FIELDS_CACHE[table]

    try:
        sql = (
            f"SELECT DISTINCT key FROM \"{table}\","
            f" jsonb_object_keys(document) AS key LIMIT 300"
        )
        rows = run_sql(agent, sql)
        fields = sorted([r[0] for r in rows if r and r[0]])
        _TABLE_FIELDS_CACHE[table] = fields
        LOGGER.debug("Schema: %s has %d fields", table, len(fields))
        return fields
    except Exception as exc:
        LOGGER.warning("Schema: field discovery failed for %s: %s", table, exc)
        _TABLE_FIELDS_CACHE[table] = []
        return []


def get_table_row_count(agent, table: str) -> int:
    """Quick row count for a table (used for schema context)."""
    try:
        rows = run_sql(agent, f'SELECT COUNT(*)::int FROM "{table}"')
        return int(rows[0][0]) if rows and rows[0] else 0
    except Exception:
        return 0


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

    with _LOCK:
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

                # Validate with a sample value
                try:
                    sample = run_sql(
                        agent,
                        f"SELECT document->>'{field}' FROM \"{table}\""
                        f" WHERE document->>'{field}' IS NOT NULL"
                        f" AND document->>'{field}' != '' LIMIT 1",
                    )
                    val = sample[0][0] if sample and sample[0] else None
                    if not val:
                        continue

                    val_str = str(val).strip()
                    # Check for MongoDB ObjectID pattern (24 hex chars)
                    if not re.match(r"^[0-9a-f]{24}$", val_str):
                        continue

                    # Verify the ID exists in the target table
                    verify = run_sql(
                        agent,
                        f"SELECT 1 FROM \"{target_name}\""
                        f" WHERE document->>'_id' = '{val_str}' LIMIT 1",
                    )
                    if verify:
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

    with _LOCK:
        if _TEXT2SQL_SCHEMA_CACHE:
            return _TEXT2SQL_SCHEMA_CACHE

        table_names = get_table_names(agent)
        lines = [
            "PostgreSQL JSONB database — CRITICAL SQL RULES:",
            "  • ALL field access: document->>'fieldName'",
            "  • Numbers: NULLIF(document->>'field','')::numeric",
            "  • Dates: NULLIF(document->>'field','')::timestamptz",
            "  • NEVER use bare column names — always document->>''",
            "  • Soft-delete: add COALESCE(document->>'deleted','false')!='true'",
            "  • Table names in double quotes: FROM \"tableName\"",
            "  • Use COALESCE for nullable fields",
            "  • Use ILIKE for case-insensitive text matching",
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
            "CRM DOMAIN:",
            "  • Revenue = invoices WHERE payment_status='paid', field: grandtotal_in_usd",
            "  • Open deals = dealWonAt IS NULL AND dealLostAt IS NULL",
            "  • Won deals = dealWonAt IS NOT NULL",
            "  • Targets: targetInUSD per userId per month/year",
            "  • Tasks: createtasks (status: Pending/Completed, priority: Low/Medium/High)",
            "  • Outreaches: isDeleted field (not deleted)",
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

    parts = [f"NULLIF(document->>'{f}','')::numeric" for f in candidates]
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