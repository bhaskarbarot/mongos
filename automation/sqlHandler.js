/**
 * sqlHandler.js — Column-wise PostgreSQL upsert / delete for MongoDB→PG sync.
 *
 * Table schema (created automatically):
 *   _id        TEXT PRIMARY KEY   (MongoDB ObjectId as string)
 *   <field>    TEXT | BOOLEAN | BIGINT | NUMERIC | JSONB  (one col per MongoDB field)
 *
 * On every upsert:
 *   1. Ensure table exists (CREATE TABLE IF NOT EXISTS)
 *   2. For each field in the document, ADD COLUMN IF NOT EXISTS (auto-schema)
 *   3. INSERT … ON CONFLICT (_id) DO UPDATE  (upsert all columns)
 *
 * On delete:
 *   DELETE FROM <table> WHERE _id = $1
 *
 * On reconcile (polling):
 *   DELETE FROM <table> WHERE _id <> ALL($1)
 */

"use strict";

const db = require("./postgres");

// Caches — avoid repeated DDL round-trips
const _ensuredTables = new Set();          // tables confirmed to exist
const _knownCols     = new Map();          // table → Set<colName>

function quoteIdentifier(name) {
  return `"${String(name).replace(/"/g, '""')}"`;
}

// ── Type inference (JS value → PostgreSQL type) ───────────────────────────────
function _pgType(value) {
  if (value === null || value === undefined) return "TEXT";
  if (typeof value === "boolean")            return "BOOLEAN";
  if (typeof value === "number")             return Number.isInteger(value) ? "BIGINT" : "NUMERIC";
  if (typeof value === "string") {
    const t = value.trimStart();
    if (t.startsWith("{") || t.startsWith("[")) return "JSONB";
    return "TEXT";
  }
  return "TEXT";
}

// ── Ensure table + individual columns exist ───────────────────────────────────
async function ensureTable(table, fields) {
  const qt = quoteIdentifier(table);

  if (!_ensuredTables.has(table)) {
    await db.query(`
      CREATE TABLE IF NOT EXISTS ${qt} (
        _id TEXT PRIMARY KEY
      )
    `);
    // Load existing columns into cache
    const res = await db.query(
      `SELECT column_name FROM information_schema.columns
       WHERE table_schema = 'public' AND table_name = $1`,
      [table]
    );
    _knownCols.set(table, new Set(res.rows.map(r => r.column_name)));
    _ensuredTables.add(table);
  }

  // Add any new fields as individual columns
  const known = _knownCols.get(table);
  for (const [col, val] of Object.entries(fields || {})) {
    if (known.has(col)) continue;
    try {
      await db.query(
        `ALTER TABLE ${qt} ADD COLUMN IF NOT EXISTS ${quoteIdentifier(col)} ${_pgType(val)}`
      );
      known.add(col);
    } catch (_) {
      // Concurrent add — safe to ignore
    }
  }
}

// ── Upsert one row ────────────────────────────────────────────────────────────
async function insertRow(table, data) {
  const { _id, fields = {} } = data;
  if (!_id) return;

  await ensureTable(table, fields);

  const qt        = quoteIdentifier(table);
  const fieldNames = Object.keys(fields);
  const allCols   = ["_id", ...fieldNames];
  const allVals   = [_id, ...fieldNames.map(k => fields[k])];

  const colsSql      = allCols.map(quoteIdentifier).join(", ");
  const placeholders = allCols.map((_, i) => `$${i + 1}`).join(", ");
  const updateSql    = fieldNames
    .map(c => `${quoteIdentifier(c)} = EXCLUDED.${quoteIdentifier(c)}`)
    .join(", ");

  if (!updateSql) {
    // Only _id — just ensure the row exists
    await db.query(
      `INSERT INTO ${qt} (_id) VALUES ($1) ON CONFLICT (_id) DO NOTHING`,
      [_id]
    );
    return;
  }

  try {
    await db.query(
      `INSERT INTO ${qt} (${colsSql})
       VALUES (${placeholders})
       ON CONFLICT (_id) DO UPDATE SET ${updateSql}`,
      allVals
    );
  } catch (err) {
    if (err.message && (
      err.message.includes("ON CONFLICT") ||
      err.message.includes("constraint") ||
      err.message.includes("unique")
    )) {
      const setParts  = fieldNames.map((c, i) => `${quoteIdentifier(c)} = $${i + 2}`).join(", ");
      const updateVals = [_id, ...fieldNames.map(k => fields[k])];
      const res = await db.query(
        `UPDATE ${qt} SET ${setParts} WHERE _id = $1`, updateVals
      );
      if (res.rowCount === 0) {
        await db.query(
          `INSERT INTO ${qt} (${colsSql}) VALUES (${placeholders}) ON CONFLICT DO NOTHING`,
          allVals
        );
      }
    } else {
      throw err;
    }
  }
}

// ── Delete one row ────────────────────────────────────────────────────────────
async function deleteRow(table, id) {
  if (!id) return;
  const qt = quoteIdentifier(table);
  await db.query(`CREATE TABLE IF NOT EXISTS ${qt} (_id TEXT PRIMARY KEY)`);
  await db.query(`DELETE FROM ${qt} WHERE _id = $1`, [id]);
}

// ── Reconcile (polling): remove rows no longer in MongoDB ─────────────────────
async function deleteMissingRows(table, validIds) {
  if (!Array.isArray(validIds)) throw new Error("validIds must be an array");
  const qt = quoteIdentifier(table);
  await db.query(`CREATE TABLE IF NOT EXISTS ${qt} (_id TEXT PRIMARY KEY)`);

  if (validIds.length === 0) {
    await db.query(`DELETE FROM ${qt}`);
  } else {
    await db.query(
      `DELETE FROM ${qt} WHERE _id <> ALL($1::text[])`,
      [validIds]
    );
  }
}

module.exports = { insertRow, deleteRow, deleteMissingRows };
