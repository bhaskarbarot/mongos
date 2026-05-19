/**
 * sqlHandler.js — Column-wise PostgreSQL upsert / delete for MongoDB→PG sync.
 *
 * Every top-level MongoDB field becomes its own PostgreSQL column.
 * Columns are created / added automatically on first insert (no manual schema).
 *
 * Column type mapping from JavaScript typeof:
 *   boolean → BOOLEAN
 *   number  → NUMERIC
 *   object  → JSONB  (plain object or array)
 *   string  → TEXT
 *   null    → TEXT   (safe default; overridden if a non-null value arrives later)
 */

const db = require("./postgres");

const _ensuredTables = new Set();
const _colCache = {}; // { tableName: Set<colName> }

function quote(name) {
  return '"' + String(name).replace(/"/g, '""') + '"';
}

function pgType(value) {
  if (value === null || value === undefined) return "TEXT";
  if (typeof value === "boolean") return "BOOLEAN";
  if (typeof value === "number")  return "NUMERIC";
  if (typeof value === "object")  return "JSONB";   // array or plain object
  return "TEXT";
}

function toPgValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "object") return JSON.stringify(value);
  return value;
}

async function _loadCols(table) {
  const res = await db.query(
    `SELECT column_name FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = $1`,
    [table]
  );
  _colCache[table] = new Set(res.rows.map((r) => r.column_name));
  return _colCache[table];
}

async function ensureTable(table) {
  if (_ensuredTables.has(table)) return;
  const q = quote(table);
  await db.query(`
    CREATE TABLE IF NOT EXISTS ${q} (
      _id        TEXT PRIMARY KEY,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  _ensuredTables.add(table);
  await _loadCols(table);
}

async function ensureColumns(table, row) {
  const cols = _colCache[table] || (await _loadCols(table));
  const q    = quote(table);

  for (const [key, value] of Object.entries(row)) {
    if (key === "_id" || key === "updated_at" || cols.has(key)) continue;

    const type = pgType(value);
    const colQ = quote(key);

    try {
      await db.query(`ALTER TABLE ${q} ADD COLUMN IF NOT EXISTS ${colQ} ${type}`);
      cols.add(key);
    } catch (_) {
      // Fallback to TEXT if typed column fails (e.g. reserved word edge cases)
      try {
        await db.query(`ALTER TABLE ${q} ADD COLUMN IF NOT EXISTS ${colQ} TEXT`);
        cols.add(key);
      } catch (__) { /* skip this column */ }
    }
  }
}

async function insertRow(table, data) {
  if (!data || !data._id) return;

  await ensureTable(table);
  await ensureColumns(table, data);

  const cols = _colCache[table];
  const q    = quote(table);

  // Build ordered key list: _id first, updated_at last, all others in between
  const keys = [
    "_id",
    ...Object.keys(data).filter((k) => k !== "_id" && k !== "updated_at" && cols.has(k)),
    "updated_at",
  ];

  const colList     = keys.map(quote).join(", ");
  const placeholders = keys.map((_, i) => `$${i + 1}`).join(", ");
  const values       = keys.map((k) => toPgValue(data[k]));

  const updateSet = keys
    .filter((k) => k !== "_id")
    .map((k) => `${quote(k)} = EXCLUDED.${quote(k)}`)
    .join(", ");

  await db.query(
    `INSERT INTO ${q} (${colList}) VALUES (${placeholders})
     ON CONFLICT (_id) DO UPDATE SET ${updateSet}`,
    values
  );
}

async function deleteRow(table, id) {
  if (!id) return;
  await ensureTable(table);
  const q = quote(table);
  await db.query(`DELETE FROM ${q} WHERE _id = $1`, [id]);
}

async function deleteMissingRows(table, validIds) {
  await ensureTable(table);
  if (!Array.isArray(validIds)) throw new Error("validIds must be an array");

  const q = quote(table);
  if (validIds.length === 0) {
    await db.query(`DELETE FROM ${q}`);
    return;
  }
  await db.query(`DELETE FROM ${q} WHERE _id <> ALL($1::text[])`, [validIds]);
}

module.exports = { insertRow, deleteRow, deleteMissingRows };
