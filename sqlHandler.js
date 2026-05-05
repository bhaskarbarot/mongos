const db = require("./postgres");
const ensuredTables = new Set();

function quoteIdentifier(identifier) {
  return `"${String(identifier).replace(/"/g, '""')}"`;
}

async function ensureTable(table) {
  if (ensuredTables.has(table)) {
    return;
  }

  const query = `
    CREATE TABLE IF NOT EXISTS ${quoteIdentifier(table)} (
      id TEXT PRIMARY KEY,
      document JSONB,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `;

  await db.query(query);
  await db.query(`
    ALTER TABLE ${quoteIdentifier(table)}
    ADD COLUMN IF NOT EXISTS document JSONB
  `);
  await db.query(`
    ALTER TABLE ${quoteIdentifier(table)}
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  `);
  ensuredTables.add(table);
}

async function insertRow(table, data) {
  await ensureTable(table);
  const keys = Object.keys(data);
  const values = Object.values(data);

  if (!keys.length) {
    throw new Error("Cannot insert empty data object");
  }

  const query = `
    INSERT INTO ${quoteIdentifier(table)} (${keys.map(quoteIdentifier).join(",")})
    VALUES (${keys.map((_, index) => `$${index + 1}`).join(",")})
    ON CONFLICT (id) DO UPDATE SET
    ${keys.map((k) => `${quoteIdentifier(k)}=EXCLUDED.${quoteIdentifier(k)}`).join(",")}
  `;

  await db.query(query, values);
}

async function deleteRow(table, id) {
  await ensureTable(table);
  const query = `DELETE FROM ${quoteIdentifier(table)} WHERE id = $1`;
  await db.query(query, [id]);
}

async function deleteMissingRows(table, validIds) {
  await ensureTable(table);
  if (!Array.isArray(validIds)) {
    throw new Error("validIds must be an array");
  }

  if (validIds.length === 0) {
    await db.query(`DELETE FROM ${quoteIdentifier(table)}`);
    return;
  }

  const query = `
    DELETE FROM ${quoteIdentifier(table)}
    WHERE id <> ALL($1::text[])
  `;
  await db.query(query, [validIds]);
}

module.exports = { insertRow, deleteRow, deleteMissingRows };
