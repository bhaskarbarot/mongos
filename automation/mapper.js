/**
 * mapper.js — Maps a raw MongoDB document to individual PostgreSQL columns.
 *
 * Returns:
 *   - _id:    string (MongoDB ObjectId as hex)
 *   - fields: { colName: normalisedValue, … }  ← one entry per MongoDB field
 *
 * Type mapping (MongoDB → PostgreSQL):
 *   ObjectId      → TEXT  (hex string)
 *   Date          → ISO-8601 string (stored as TIMESTAMPTZ)
 *   boolean       → BOOLEAN
 *   integer       → BIGINT
 *   float/Decimal → NUMERIC
 *   object/array  → JSON string (stored as JSONB)
 *   string        → TEXT
 *   null          → NULL
 */

"use strict";

// Columns that must never become individual PG columns
const _SKIP_KEYS = new Set(["_id", "__v", "updated_at", "_synced_at", "document"]);

/**
 * Normalise a single MongoDB value to a JS-safe scalar or JSON string.
 * - ObjectId / toHexString  → hex string
 * - Date                    → ISO-8601 string
 * - object / array          → JSON string (for JSONB columns)
 * - everything else         → as-is
 */
function normalizeValue(value) {
  if (value === null || value === undefined) return null;

  if (typeof value === "boolean") return value;
  if (typeof value === "number")  return value;
  if (typeof value === "string")  return value;

  if (typeof value === "object") {
    // MongoDB ObjectId
    if (value._bsontype === "ObjectId" || typeof value.toHexString === "function") {
      return value.toHexString ? value.toHexString() : String(value);
    }
    // JS Date or MongoDB Date
    if (value instanceof Date) {
      return value.toISOString();
    }
    // Nested object or array → serialise to JSON string (stored as JSONB)
    try {
      return JSON.stringify(value, replacer);
    } catch (_) {
      return String(value);
    }
  }

  return String(value);
}

/** JSON.stringify replacer that converts ObjectIds and Dates recursively. */
function replacer(key, val) {
  if (val === null || val === undefined) return val;
  if (typeof val === "object") {
    if (val._bsontype === "ObjectId" || typeof val.toHexString === "function") {
      return val.toHexString ? val.toHexString() : String(val);
    }
    if (val instanceof Date) return val.toISOString();
  }
  return val;
}

/**
 * mapDocumentToSQL(doc)
 *
 * Returns:
 * {
 *   _id:      string,
 *   fields:   { colName: normalisedValue, … },  ← individual columns
 *   document: object,                            ← full normalised doc (JSONB)
 *   updated_at: Date,
 * }
 */
function mapDocumentToSQL(doc) {
  if (!doc) {
    return { _id: null, fields: {} };
  }

  const _id = doc._id ? String(doc._id) : null;

  // Build individual field map (skip internal keys)
  const fields = {};
  for (const [key, val] of Object.entries(doc)) {
    if (_SKIP_KEYS.has(key)) continue;
    fields[key] = normalizeValue(val);
  }

  return { _id, fields };
}

module.exports = mapDocumentToSQL;
