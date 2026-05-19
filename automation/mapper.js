/**
 * mapper.js — Maps a raw MongoDB document to a flat PostgreSQL row.
 *
 * Each top-level MongoDB field becomes its own column:
 *   - Scalars (string, number, boolean) → native TEXT / NUMERIC / BOOLEAN column
 *   - Objects and arrays                → JSONB column
 *   - ObjectId values                   → plain hex string (TEXT)
 *   - Date instances                    → ISO-8601 string (TEXT)
 *
 * sqlHandler.js creates/adds columns automatically on first insert.
 */

function normalizeValue(key, value) {
  if (value === null || value === undefined) return value;
  if (value && typeof value === "object") {
    if (value._bsontype === "ObjectId" || typeof value.toHexString === "function") {
      return value.toHexString ? value.toHexString() : String(value);
    }
    if (value instanceof Date) {
      return value.toISOString();
    }
  }
  return value;
}

function mapDocumentToSQL(doc) {
  if (!doc) return { _id: null, updated_at: new Date() };

  const _id = doc._id ? String(doc._id) : null;

  // Deep-clone with normalization (ObjectIds → hex strings, Dates → ISO strings)
  let normalized;
  try {
    normalized = JSON.parse(
      JSON.stringify(doc, function (key, value) {
        return normalizeValue(key, value);
      })
    );
  } catch (_) {
    normalized = { ...doc };
  }

  // Remove fields that belong to PostgreSQL meta or are duplicated
  delete normalized._id;
  delete normalized.updated_at;
  delete normalized._synced_at;

  // Spread every MongoDB field as its own key — sqlHandler will column-ify them
  return { _id, ...normalized, updated_at: new Date() };
}

module.exports = mapDocumentToSQL;
