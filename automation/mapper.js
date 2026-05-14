/**
 * mapper.js — Maps a raw MongoDB document to the PostgreSQL row shape.
 *
 * DB schema: _id TEXT PRIMARY KEY, document JSONB, updated_at TIMESTAMPTZ
 *
 * Normalisation applied to `document`:
 *   - ObjectId instances  → plain hex string
 *   - Date instances      → ISO-8601 string
 *   - Removes system keys (document, updated_at, _synced_at) to keep it clean
 */

function normalizeValue(key, value) {
  if (value === null || value === undefined) return value;

  // MongoDB ObjectId (has _bsontype or is an object with toHexString)
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
  if (!doc) return { _id: null, document: null, updated_at: new Date() };

  const _id = doc._id ? String(doc._id) : null;

  // Deep-clone with normalization (handles nested ObjectIds and Dates)
  let normalized;
  try {
    normalized = JSON.parse(
      JSON.stringify(doc, function (key, value) {
        return normalizeValue(key, value);
      })
    );
  } catch (_) {
    normalized = doc;
  }

  // Remove PostgreSQL meta columns from the JSONB document
  delete normalized.updated_at;
  delete normalized._synced_at;

  return {
    _id,
    document: normalized,
    updated_at: new Date(),
  };
}

module.exports = mapDocumentToSQL;
