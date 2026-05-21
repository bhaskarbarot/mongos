"""db_indexes.py — Create performance indexes for CRM query speed.

Safe to re-run at any time — uses CONCURRENTLY IF NOT EXISTS.
CONCURRENTLY builds indexes without locking the table for reads/writes.

Usage:
    python3 -m pipeline.db_indexes
"""
import logging

import psycopg2

from config import settings

LOGGER = logging.getLogger("sql_chatbot")

_INDEXES = [
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_targets_user_month_year ON "targets" ("userId", month, year)',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activitylogs_createdat ON "activitylogs" ("createdAt")',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activitylogs_userid ON "activitylogs" ("userId")',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_deals_owner ON "deals" (owner)',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_deals_wonAt ON "deals" ("dealWonAt")',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_invoices_payment_date ON "invoices" (payment_date)',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sales_salesOwner ON "sales" ("salesOwner")',
]


def create_indexes() -> None:
    """Create all performance indexes. Safe to call multiple times."""
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    # CONCURRENTLY requires autocommit — cannot run inside a transaction block
    conn.autocommit = True
    cur = conn.cursor()
    for ddl in _INDEXES:
        try:
            cur.execute(ddl)
            LOGGER.info("Index OK: %s", ddl.split("idx_")[1].split(" ")[0])
        except Exception as exc:
            LOGGER.warning("Index skipped (%s): %s", type(exc).__name__, exc)
    cur.close()
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_indexes()
    print("Done.")
