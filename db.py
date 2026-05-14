from langchain_community.utilities import SQLDatabase
import psycopg2

from config import settings


def _non_empty_tables() -> list:
    """Return tables that actually contain rows (pg_class estimate > 0)."""
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            connect_timeout=8,
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT t.tablename
            FROM pg_tables t
            JOIN pg_class c
              ON c.relname = t.tablename
             AND c.relnamespace = 'public'::regnamespace
            WHERE t.schemaname = 'public'
              AND c.reltuples > 0
            ORDER BY t.tablename
        """)
        tables = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return tables
    except Exception:
        return []


def get_database() -> SQLDatabase:
    tables = _non_empty_tables()
    kwargs = {
        "sample_rows_in_table_info": 2,
        # Raise the result string limit so LangChain doesn't silently truncate
        # rows that contain large JSONB `document` columns. Truncation causes
        # ast.literal_eval() to fail, which was silently returning empty rows
        # even when the DB had real data matching the query.
        "max_string_length": 50_000,
    }
    if tables:
        kwargs["include_tables"] = tables
    return SQLDatabase.from_uri(settings.postgres_uri, **kwargs)
