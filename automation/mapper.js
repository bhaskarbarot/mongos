function mapDocumentToSQL(doc) {
  return {
    id: doc._id ? String(doc._id) : null,
    document: doc,
    updated_at: new Date(),
  };
}

module.exports = mapDocumentToSQL;
