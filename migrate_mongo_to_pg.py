"""
migrate_mongo_to_pg.py — One-time full sync: MongoDB ecrm → PostgreSQL mongos_sync

Creates PostgreSQL tables with REAL individual columns (not JSONB blobs).
Every MongoDB field becomes its own PostgreSQL column, typed correctly.
Also keeps `document JSONB` + `updated_at` for automation backward-compat.

Run:
    python3 migrate_mongo_to_pg.py

Tables synced: all 50 in _CRM_TABLES (db_schema.py)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
import psycopg2.extras
from bson import ObjectId
from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGODB_URI")
MONGO_DB    = os.getenv("MONGODB_DB", "ecrm")
PG_HOST     = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT     = int(os.getenv("POSTGRES_PORT", 5433))
PG_USER     = os.getenv("POSTGRES_USER", "postgres")
PG_PASS     = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_DB       = os.getenv("POSTGRES_DB", "mongos_sync")

# All 50 CRM tables to migrate (matches db_schema._CRM_TABLES)
CRM_TABLES = [
    "deals", "invoices", "sales", "companies", "contacts", "users",
    "createtasks", "targets", "outreaches", "vendors", "bills",
    "departments", "regions", "products", "sources", "technologies",
    "taxes", "categories", "campaigns", "lead_statuses", "lifecycle_stages",
    "dealstagesettings", "payments", "technologycategories", "status",
    "emails", "mails", "commonnotes", "activitylogs", "activityevents",
    "activities", "notifications", "conversations",
    "companynotes", "contactsnotes", "dealsnotes", "salesnotes",
    "remotejobnotes", "ai_notes", "notes",
    "countryregions", "projecttypes", "publicleads", "meetings",
    "tasks", "remotejobs", "vendormagiclinks",
    "outreachactivities", "deletedcompanies", "prompts",
]

# ── Colour helpers ─────────────────────────────────────────────────────────────
GRN = "\033[0;32m"; YLW = "\033[1;33m"; RED = "\033[0;31m"; NC = "\033[0m"
def ok(m):   print(f"  {GRN}✓{NC}  {m}")
def warn(m): print(f"  {YLW}⚠{NC}  {m}")
def err(m):  print(f"  {RED}✗{NC}  {m}")


# ══════════════════════════════════════════════════════════════════════════════
# VALUE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm(value: Any) -> Any:
    """Recursively normalise a MongoDB value to a JSON-safe Python value."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm(i) for i in value]
    # fallback
    return str(value)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE INFERENCE  (MongoDB value → PostgreSQL column type)
# ══════════════════════════════════════════════════════════════════════════════

def _pg_type(value: Any) -> str:
    """Infer the best PostgreSQL type for a MongoDB field value."""
    if value is None:
        return "TEXT"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "NUMERIC"
    if isinstance(value, Decimal128):
        return "NUMERIC"
    if isinstance(value, ObjectId):
        return "TEXT"
    if isinstance(value, datetime):
        return "TIMESTAMPTZ"
    if isinstance(value, (dict, list)):
        return "JSONB"
    return "TEXT"


def _discover_schema(docs: List[dict]) -> Dict[str, str]:
    """
    Scan all docs to discover every field name and the best PostgreSQL type.
    Later docs can upgrade a type (TEXT → BOOLEAN if a bool is found, etc.).
    """
    TYPE_RANK = {"TEXT": 0, "BOOLEAN": 1, "BIGINT": 2, "NUMERIC": 3,
                 "TIMESTAMPTZ": 4, "JSONB": 5}

    schema: Dict[str, str] = {}
    for doc in docs:
        for key, val in doc.items():
            if key == "_id":
                continue  # handled separately as PK
            t = _pg_type(val)
            # If conflict: keep the more permissive type
            existing = schema.get(key, "TEXT")
            if TYPE_RANK.get(t, 0) < TYPE_RANK.get(existing, 0):
                # existing is more permissive → keep it
                pass
            elif TYPE_RANK.get(t, 0) > TYPE_RANK.get(existing, 0):
                schema[key] = t
            else:
                schema[key] = existing if existing else t
            if key not in schema:
                schema[key] = t
    return schema


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _q(name: str) -> str:
    """Double-quote an identifier."""
    return f'"{name.replace(chr(34), chr(34)+chr(34))}"'


def _get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASS, dbname=PG_DB, connect_timeout=10,
    )


def _ensure_table(cur, table: str, schema: Dict[str, str]) -> None:
    """
    Create or update a PostgreSQL table so it has:
      - _id TEXT PRIMARY KEY
      - one column per MongoDB field
    """
    qt = _q(table)

    # Create with _id PK if not exists
    cur.execute(f"CREATE TABLE IF NOT EXISTS {qt} (_id TEXT PRIMARY KEY)")

    # Add individual columns
    for col, pg_type in schema.items():
        qc = _q(col)
        cur.execute(f"ALTER TABLE {qt} ADD COLUMN IF NOT EXISTS {qc} {pg_type}")


def _insert_rows(cur, table: str, docs: List[dict], schema: Dict[str, str]) -> int:
    """
    Bulk-upsert all documents into the table.
    Returns number of rows inserted/updated.
    """
    if not docs:
        return 0

    qt = _q(table)
    cols      = list(schema.keys())
    col_names = ["_id"] + cols
    col_sql   = ", ".join(_q(c) for c in col_names)
    placeholders = ", ".join(["%s"] * len(col_names))

    # Build SET clause for ON CONFLICT update (all cols except _id)
    update_parts = [f"{_q(c)} = EXCLUDED.{_q(c)}" for c in cols]
    update_sql = ", ".join(update_parts)

    upsert_sql = f"""
        INSERT INTO {qt} ({col_sql})
        VALUES ({placeholders})
        ON CONFLICT (_id) DO UPDATE SET {update_sql}
    """

    now = datetime.now(timezone.utc)
    rows_data = []
    for doc in docs:
        _id = str(doc.get("_id", "")) or None
        if not _id:
            continue

        # Build normalised flat values
        flat: Dict[str, Any] = {}
        for col in cols:
            raw = doc.get(col)
            v   = _norm(raw)
            pg  = schema[col]
            # Serialise complex types to JSON string for JSONB columns
            if pg == "JSONB" and v is not None and not isinstance(v, str):
                v = json.dumps(v, default=str)
            flat[col] = v

        row = [_id] + [flat.get(c) for c in cols]
        rows_data.append(row)

    if rows_data:
        psycopg2.extras.execute_batch(cur, upsert_sql, rows_data, page_size=200)

    return len(rows_data)


# ══════════════════════════════════════════════════════════════════════════════
# PER-COLLECTION MIGRATION
# ══════════════════════════════════════════════════════════════════════════════

def migrate_collection(
    mongo_db,
    pg_conn,
    collection: str,
    mongo_collections: Set[str],
) -> Tuple[int, int]:
    """
    Migrate one MongoDB collection → PostgreSQL table.
    Returns (rows_inserted, columns_created).
    """
    if collection not in mongo_collections:
        warn(f"{collection:25s} — not in MongoDB, skipping")
        return 0, 0

    coll = mongo_db[collection]
    total = coll.estimated_document_count()

    if total == 0:
        warn(f"{collection:25s} — 0 documents, creating empty table")
        with pg_conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_q(collection)} (
                    _id        TEXT PRIMARY KEY,
                    document   JSONB,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        pg_conn.commit()
        return 0, 0

    print(f"  → {collection:25s} ({total:5d} docs) … ", end="", flush=True)
    t0 = time.monotonic()

    # Fetch all docs
    docs = list(coll.find({}))

    # Discover schema from all docs
    schema = _discover_schema(docs)

    # Apply to PostgreSQL
    with pg_conn.cursor() as cur:
        _ensure_table(cur, collection, schema)
        inserted = _insert_rows(cur, collection, docs, schema)
    pg_conn.commit()

    elapsed = round(time.monotonic() - t0, 1)
    print(f"{GRN}done{NC}  {inserted} rows, {len(schema)} cols  [{elapsed}s]")
    return inserted, len(schema)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print(f"{GRN}╔══════════════════════════════════════════════════════╗{NC}")
    print(f"{GRN}║   MongoDB → PostgreSQL Full Migration                ║{NC}")
    print(f"{GRN}║   Source : ecrm (MongoDB)                            ║{NC}")
    print(f"{GRN}║   Target : mongos_sync (PostgreSQL :5433)            ║{NC}")
    print(f"{GRN}╚══════════════════════════════════════════════════════╝{NC}")
    print()

    # ── Connect MongoDB ────────────────────────────────────────────────────────
    print("Connecting to MongoDB …", end=" ", flush=True)
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        mongo_db     = mongo_client[MONGO_DB]
        mongo_client.admin.command("ping")
        mongo_collections = set(mongo_db.list_collection_names())
        print(f"{GRN}OK{NC} ({len(mongo_collections)} collections)")
    except Exception as e:
        err(f"MongoDB connection failed: {e}")
        sys.exit(1)

    # ── Connect PostgreSQL ─────────────────────────────────────────────────────
    print("Connecting to PostgreSQL …", end=" ", flush=True)
    try:
        pg_conn = _get_pg_conn()
        print(f"{GRN}OK{NC}")
    except Exception as e:
        err(f"PostgreSQL connection failed: {e}")
        sys.exit(1)

    print()
    print(f"Migrating {len(CRM_TABLES)} tables …")
    print()

    total_rows = 0
    total_tables = 0
    failed = []

    t_start = time.monotonic()

    for table in CRM_TABLES:
        try:
            rows, cols = migrate_collection(mongo_db, pg_conn, table, mongo_collections)
            total_rows   += rows
            total_tables += 1
        except Exception as exc:
            pg_conn.rollback()
            err(f"{table:25s} — FAILED: {exc}")
            failed.append((table, str(exc)))

    elapsed_total = round(time.monotonic() - t_start, 1)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print(f"{GRN}╔══════════════════════════════════════════════════════╗{NC}")
    print(f"{GRN}║   Migration Complete                                 ║{NC}")
    print(f"{GRN}╚══════════════════════════════════════════════════════╝{NC}")
    print(f"  Tables  : {total_tables}/{len(CRM_TABLES)}")
    print(f"  Rows    : {total_rows:,}")
    print(f"  Time    : {elapsed_total}s")
    if failed:
        print(f"  {RED}Failed  : {len(failed)}{NC}")
        for t, e in failed:
            print(f"    {RED}✗{NC} {t}: {e}")
    else:
        print(f"  {GRN}No failures{NC}")
    print()

    pg_conn.close()
    mongo_client.close()


if __name__ == "__main__":
    main()
