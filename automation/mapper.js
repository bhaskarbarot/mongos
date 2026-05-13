"use strict";

/**
 * mapper.js — MongoDB document → flat PostgreSQL columns
 *
 * Rules:
 *   • Top-level scalar fields  → individual typed columns
 *   • Nested objects            → JSONB column (preserves all data)
 *   • Arrays                    → JSONB column (preserves all data)
 *   • null / undefined          → NULL (column stays, value is null)
 *   • MongoDB _id (ObjectId)    → TEXT column named "_id"
 *   • Booleans                  → BOOLEAN
 *   • Integers                  → BIGINT
 *   • Floats                    → NUMERIC
 *   • Strings                   → TEXT
 */

// ── Type inference ─────────────────────────────────────────────────────────────

/**
 * Infer the best PostgreSQL type for a raw MongoDB value.
 * Returns a PostgreSQL type string.
 */
function inferPgType(value) {
  if (value === null || value === undefined) return "TEXT";
  if (typeof value === "boolean") return "BOOLEAN";
  // Always use NUMERIC (not BIGINT) — some collections store the same field
  // as integer in one document and decimal in another. NUMERIC handles both.
  if (typeof value === "number") return "NUMERIC";
  if (Array.isArray(value)) return "JSONB";
  if (typeof value === "object") return "JSONB"; // nested document
  return "TEXT"; // string fallback
}

/**
 * Convert a raw MongoDB value to a PostgreSQL-ready value.
 * - JSONB fields   → JSON string with safe fallback (pg driver handles the rest)
 * - ObjectId-like  → string
 * - null           → null
 * - primitives     → as-is (pg driver handles boolean, number, string)
 */
function toPgValue(value, pgType) {
  if (value === null || value === undefined) return null;
  if (pgType === "JSONB") {
    try {
      return JSON.stringify(value);
    } catch (_) {
      // Fallback for MongoDB Binary / circular / non-serializable types
      return String(value);
    }
  }
  // MongoDB ObjectId, Binary, or other object types → string
  if (typeof value === "object" && typeof value.toString === "function") {
    return String(value);
  }
  return value;
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * flattenDocument(doc) → { colName: { value, pgType }, ... }
 *
 * Returns a flat map of all top-level MongoDB fields, ready for insertion.
 * The "_id" field is always included and always TEXT (converted from ObjectId).
 */
function flattenDocument(doc) {
  if (!doc || typeof doc !== "object") return {};

  const result = {};

  for (const [key, rawValue] of Object.entries(doc)) {
    let value = rawValue;

    // Always stringify MongoDB ObjectId (the primary _id field)
    if (key === "_id") {
      value = String(rawValue);
      result[key] = { value, pgType: "TEXT" };
      continue;
    }

    // MongoDB Date objects → ISO string TEXT (not JSONB)
    if (value instanceof Date) {
      result[key] = { value: value.toISOString(), pgType: "TEXT" };
      continue;
    }

    // MongoDB ObjectId references (e.g. owner, company, contact fields)
    // The driver returns them as ObjectId objects; store as hex TEXT
    if (value !== null && typeof value === "object" &&
        (value._bsontype === "ObjectId" || value._bsontype === "ObjectID" ||
         (typeof value.toHexString === "function"))) {
      result[key] = { value: String(value), pgType: "TEXT" };
      continue;
    }

    const pgType = inferPgType(value);
    const pgValue = toPgValue(value, pgType);
    result[key] = { value: pgValue, pgType };
  }

  return result;
}

module.exports = { flattenDocument, inferPgType, toPgValue };
