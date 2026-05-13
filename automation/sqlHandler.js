"use strict";

/**
 * sqlHandler.js — Dynamic PostgreSQL schema manager + row upsert engine
 *
 * Architecture:
 *   • Each MongoDB collection → one PostgreSQL table
 *   • Each top-level document field → one typed column
 *   • "_id" TEXT PRIMARY KEY  (from MongoDB ObjectId, converted to string)
 *   • "_synced_at" TIMESTAMPTZ  (set on every upsert, managed by sync engine)
 *   • Schema is DYNAMIC: new fields → ALTER TABLE ADD COLUMN automatically
 *   • schemaCache avoids hitting information_schema on every row insert
 *
 * Column type mapping:
 *   boolean  → BOOLEAN
 *   integer  → BIGINT
 *   float    → NUMERIC
 *   array    → JSONB
 *   object   → JSONB
 *   string   → TEXT
 *   null     → TEXT (default; column exists, value is NULL)
 */

const db = require("./postgres");

// ── In-memory schema cache ─────────────────────────────────────────────────────
// Stores: { tableName: Map<columnName, pgType> }
// pgType is the CURRENT PostgreSQL type (from information_schema), used to
// detect type conflicts and trigger ALTER COLUMN TYPE when needed.
const _schemaCache = Object.create(null);

// Tables confirmed to exist (avoids repeated CREATE TABLE IF NOT EXISTS)
const _knownTables = new Set();

// ── Type compatibility ─────────────────────────────────────────────────────────
// Returns the new type to ALTER to, or null if no ALTER is needed.
// Upgrade hierarchy: TEXT beats all | NUMERIC beats BIGINT | others stay.
const _PG_TYPE_MAP = {
  "bigint":            "BIGINT",
  "integer":           "BIGINT",
  "smallint":          "BIGINT",
  "numeric":           "NUMERIC",
  "double precision":  "NUMERIC",
  "real":              "NUMERIC",
  "boolean":           "BOOLEAN",
  "jsonb":             "JSONB",
  "json":              "JSONB",
  "text":              "TEXT",
  "character varying": "TEXT",
  "character":         "TEXT",
};

function _needsUpgrade(existingRaw, incomingNorm) {
  const existing = _PG_TYPE_MAP[existingRaw] || existingRaw.toUpperCase();
  if (existing === incomingNorm) return null;                    // same → no change
  if (existing === "TEXT") return null;                          // TEXT handles anything
  if (existing === "JSONB" && incomingNorm !== "JSONB") return "TEXT"; // conflict → TEXT
  if (existing === "BOOLEAN" && incomingNorm !== "BOOLEAN") return "TEXT";
  if (existing === "BIGINT" && incomingNorm === "NUMERIC") return "NUMERIC"; // widen
  if (existing === "BIGINT" && incomingNorm === "TEXT") return "TEXT";
  if (existing === "NUMERIC" && incomingNorm === "TEXT") return "TEXT";
  return null; // default: trust current type
}

// ── SQL identifier quoting ─────────────────────────────────────────────────────
function Q(name) {
  return `"${String(name).replace(/"/g, '""')}"`;
}

// ── Schema helpers ─────────────────────────────────────────────────────────────

/**
 * Load all column names + types for a table from information_schema into the cache.
 * Cache stores Map<columnName, rawDataType> so type-conflict detection works.
 */
async function _loadSchema(tableName) {
  const res = await db.query(
    `SELECT column_name, data_type
     FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = $1`,
    [tableName]
  );
  const map = new Map();
  for (const row of res.rows) {
    map.set(row.column_name, row.data_type);
  }
  _schemaCache[tableName] = map;
}

/**
 * ensureTable — create the table with _id PK if it does not exist.
 * Loads schema cache if not already populated.
 */
async function ensureTable(tableName) {
  if (_knownTables.has(tableName)) return;

  await db.query(`
    CREATE TABLE IF NOT EXISTS ${Q(tableName)} (
      "_id"        TEXT        NOT NULL,
      "_synced_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CONSTRAINT ${Q(tableName + "_pkey")} PRIMARY KEY ("_id")
    )
  `);
  _knownTables.add(tableName);

  if (!_schemaCache[tableName]) {
    await _loadSchema(tableName);
  }
}

/**
 * ensureColumns — add missing columns AND upgrade conflicting column types.
 *
 * Two passes per call:
 *   1. ADD COLUMN IF NOT EXISTS — batched into one ALTER TABLE
 *   2. ALTER COLUMN TYPE        — one statement per conflict (rare, after initial sync)
 */
async function ensureColumns(tableName, flatDoc) {
  if (!_schemaCache[tableName]) {
    await _loadSchema(tableName);
  }

  const cache   = _schemaCache[tableName]; // Map<colName, rawDataType>
  const toAdd   = [];
  const toAlter = [];

  for (const [col, { pgType }] of Object.entries(flatDoc)) {
    if (!cache.has(col)) {
      toAdd.push({ col, pgType });
    } else {
      // Column exists — check if type needs widening
      const upgradeType = _needsUpgrade(cache.get(col), pgType);
      if (upgradeType) {
        toAlter.push({ col, upgradeType });
      }
    }
  }

  // ── 1. Batch-add all missing columns ────────────────────────────────────────
  if (toAdd.length > 0) {
    const clauses = toAdd
      .map(({ col, pgType }) => `ADD COLUMN IF NOT EXISTS ${Q(col)} ${pgType}`)
      .join(",\n  ");
    await db.query(`ALTER TABLE ${Q(tableName)}\n  ${clauses}`);
    for (const { col, pgType } of toAdd) {
      cache.set(col, pgType);
    }
  }

  // ── 2. Widen conflicting column types one at a time ──────────────────────────
  // (rare: only happens when the same field holds mixed types across documents)
  for (const { col, upgradeType } of toAlter) {
    await db.query(
      `ALTER TABLE ${Q(tableName)} ALTER COLUMN ${Q(col)} TYPE ${upgradeType} USING ${Q(col)}::text::${upgradeType}`
    );
    cache.set(col, upgradeType);
  }
}

// ── Core DML ───────────────────────────────────────────────────────────────────

/**
 * upsertRow — insert or update a document as individual columns.
 *
 * @param {string} tableName   - PostgreSQL table (= MongoDB collection name)
 * @param {Object} flatDoc     - output of flattenDocument(): { col: { value, pgType } }
 */
async function upsertRow(tableName, flatDoc) {
  if (!flatDoc || !flatDoc._id || flatDoc._id.value === null) {
    throw new Error(`upsertRow: document has no _id — table=${tableName}`);
  }

  await ensureTable(tableName);
  await ensureColumns(tableName, flatDoc);

  const entries = Object.entries(flatDoc);
  if (entries.length === 0) return;

  const cols   = entries.map(([c]) => c);
  const values = entries.map(([, { value }]) => value);

  const quotedCols   = cols.map(Q).join(", ");
  const placeholders = cols.map((_, i) => `$${i + 1}`).join(", ");
  const updateSet = cols
    .filter((c) => c !== "_id") // never overwrite PK
    .map((c) => `${Q(c)} = EXCLUDED.${Q(c)}`)
    .join(",\n      ");

  await db.query(
    `INSERT INTO ${Q(tableName)} (${quotedCols}, "_synced_at")
     VALUES (${placeholders}, NOW())
     ON CONFLICT ("_id") DO UPDATE SET
       ${updateSet},
       "_synced_at" = NOW()`,
    values
  );
}

/**
 * deleteRow — delete a single row by _id.
 */
async function deleteRow(tableName, id) {
  if (!id) return;
  await ensureTable(tableName);
  await db.query(`DELETE FROM ${Q(tableName)} WHERE "_id" = $1`, [String(id)]);
}

/**
 * deleteMissingRows — delete rows whose _id is NOT in validIds (reconciliation).
 * Called at end of each poll pass to remove docs deleted from MongoDB.
 */
async function deleteMissingRows(tableName, validIds) {
  await ensureTable(tableName);

  if (!Array.isArray(validIds)) {
    throw new Error("deleteMissingRows: validIds must be an array");
  }

  if (validIds.length === 0) {
    await db.query(`DELETE FROM ${Q(tableName)}`);
    return;
  }

  await db.query(
    `DELETE FROM ${Q(tableName)} WHERE "_id" <> ALL($1::text[])`,
    [validIds.map(String)]
  );
}

// ── Schema reset ───────────────────────────────────────────────────────────────

/**
 * detectOldFormat — returns true if any table still has the old JSONB "document" column.
 * Used to auto-detect when a clean rebuild is needed.
 */
async function detectOldFormat() {
  const res = await db.query(`
    SELECT COUNT(*)::int AS cnt
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND column_name  = 'document'
      AND data_type    = 'jsonb'
  `);
  return ((res.rows[0] && res.rows[0].cnt) || 0) > 0;
}

/**
 * resetAllTables — DROP every table in the public schema and clear caches.
 * Returns the list of dropped table names.
 */
async function resetAllTables(logger) {
  const res = await db.query(`
    SELECT tablename
    FROM   pg_tables
    WHERE  schemaname = 'public'
    ORDER  BY tablename
  `);

  const tables = res.rows.map((r) => r.tablename);
  if (tables.length === 0) {
    logger.info("Reset: no tables found — nothing to drop");
    return [];
  }

  logger.info(`Reset: dropping ${tables.length} tables → ${tables.join(", ")}`);
  const dropList = tables.map(Q).join(", ");
  await db.query(`DROP TABLE IF EXISTS ${dropList} CASCADE`);

  // Clear all caches
  Object.keys(_schemaCache).forEach((k) => delete _schemaCache[k]);
  _knownTables.clear();

  logger.info(`Reset: all ${tables.length} tables dropped ✓`);
  return tables;
}

module.exports = {
  upsertRow,
  deleteRow,
  deleteMissingRows,
  resetAllTables,
  detectOldFormat,
};
