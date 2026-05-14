/**
 * sqlHandler.js — PostgreSQL upsert / delete helpers for MongoDB→PG sync.
 *
 * Table schema expected:
 *   _id        TEXT PRIMARY KEY   (MongoDB ObjectId as string)
 *   document   JSONB              (full normalised MongoDB document)
 *   updated_at TIMESTAMPTZ        (sync timestamp)
 *
 * ensureTable creates this schema for NEW tables.
 * For EXISTING tables that already have _id as PK (created by older sync tool),
 * it idempotently adds any missing columns without touching existing data.
 */

const db = require("./postgres");

const ensuredTables = new Set();

function quoteIdentifier(identifier) {
  return `"${String(identifier).replace(/"/g, '""')}"`;
}

async function ensureTable(table) {
  if (ensuredTables.has(table)) return;

  const q = quoteIdentifier(table);

  // 1. Create table with correct schema if it does not exist yet
  await db.query(`
    CREATE TABLE IF NOT EXISTS ${q} (
      _id        TEXT PRIMARY KEY,
      document   JSONB,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  // 2. Idempotently add required columns for tables that already exist
  //    with a different schema (older sync tool created them without these).
  await db.query(
    `ALTER TABLE ${q} ADD COLUMN IF NOT EXISTS document   JSONB`
  );
  await db.query(
    `ALTER TABLE ${q} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
  );

  ensuredTables.add(table);
}

async function insertRow(table, data) {
  await ensureTable(table);

  const _id        = data._id;
  const document   = data.document !== undefined ? data.document : null;
  const updated_at = data.updated_at || new Date();

  if (!_id) return; // skip rows that have no usable primary key

  const q = quoteIdentifier(table);

  try {
    await db.query(
      `INSERT INTO ${q} (_id, document, updated_at)
       VALUES ($1, $2, $3)
       ON CONFLICT (_id) DO UPDATE
         SET document   = EXCLUDED.document,
             updated_at = EXCLUDED.updated_at`,
      [_id, JSON.stringify(document), updated_at]
    );
  } catch (err) {
    // Fallback for tables whose PK is not on _id yet (e.g. counters).
    // Try UPDATE first; if nothing matched, attempt a plain INSERT.
    if (
      err.message &&
      (err.message.includes("ON CONFLICT") ||
        err.message.includes("constraint") ||
        err.message.includes("unique"))
    ) {
      const res = await db.query(
        `UPDATE ${q} SET document = $2, updated_at = $3 WHERE _id = $1`,
        [_id, JSON.stringify(document), updated_at]
      );
      if (res.rowCount === 0) {
        await db.query(
          `INSERT INTO ${q} (_id, document, updated_at) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
          [_id, JSON.stringify(document), updated_at]
        );
      }
    } else {
      throw err;
    }
  }
}

async function deleteRow(table, id) {
  if (!id) return;
  await ensureTable(table);
  const q = quoteIdentifier(table);
  await db.query(`DELETE FROM ${q} WHERE _id = $1`, [id]);
}

async function deleteMissingRows(table, validIds) {
  await ensureTable(table);

  if (!Array.isArray(validIds)) {
    throw new Error("validIds must be an array");
  }

  const q = quoteIdentifier(table);

  if (validIds.length === 0) {
    await db.query(`DELETE FROM ${q}`);
    return;
  }

  await db.query(
    `DELETE FROM ${q} WHERE _id <> ALL($1::text[])`,
    [validIds]
  );
}

module.exports = { insertRow, deleteRow, deleteMissingRows };
