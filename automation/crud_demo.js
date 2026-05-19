require("dotenv").config({ path: require("path").join(__dirname, "..", ".env") });
const { MongoClient } = require("mongodb");
const { Pool } = require("pg");

const mongo = new MongoClient(process.env.MONGODB_URI);
const pg = new Pool({
  host: process.env.POSTGRES_HOST, port: Number(process.env.POSTGRES_PORT),
  user: process.env.POSTGRES_USER, password: process.env.POSTGRES_PASSWORD,
  database: process.env.POSTGRES_DB,
});

const WAIT_MS = 35000; // 35s — one full poll pass across 58 collections takes ~30s

async function pgRow(id) {
  const r = await pg.query(`SELECT _id, "companyName", industry, deleted FROM companies WHERE _id = $1`, [id]);
  return r.rows[0] || null;
}
async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function banner(msg) { console.log(`\n${"=".repeat(55)}\n  ${msg}\n${"=".repeat(55)}`); }

async function main() {
  await mongo.connect();
  const col = mongo.db(process.env.MONGODB_DB).collection("companies");

  // ── 1. INSERT ───────────────────────────────────────────────────────────────
  banner("1. INSERT — adding demo company to MongoDB");
  const doc = { companyName: "LiveSync Demo Co", industry: "Technology", deleted: false, createdAt: new Date().toISOString() };
  const ins = await col.insertOne(doc);
  const id = String(ins.insertedId);
  console.log(`  MongoDB _id: ${id}`);
  console.log(`  Waiting ${WAIT_MS/1000}s for automation poll...`);
  await sleep(WAIT_MS);
  const afterInsert = await pgRow(id);
  console.log(afterInsert
    ? `  ✅ PostgreSQL: companyName="${afterInsert.companyName}", deleted=${afterInsert.deleted}`
    : "  ❌ NOT FOUND in PostgreSQL");

  // ── 2. UPDATE ───────────────────────────────────────────────────────────────
  banner("2. UPDATE — changing companyName + industry in MongoDB");
  await col.updateOne({ _id: ins.insertedId }, { $set: { companyName: "LiveSync Demo Co — UPDATED", industry: "AI/ML" } });
  console.log(`  MongoDB updated companyName → "LiveSync Demo Co — UPDATED"`);
  console.log(`  Waiting ${WAIT_MS/1000}s for automation poll...`);
  await sleep(WAIT_MS);
  const afterUpdate = await pgRow(id);
  console.log(afterUpdate
    ? `  ✅ PostgreSQL: companyName="${afterUpdate.companyName}", industry="${afterUpdate.industry}"`
    : "  ❌ NOT FOUND in PostgreSQL");

  // ── 3. DELETE ───────────────────────────────────────────────────────────────
  banner("3. DELETE — removing document from MongoDB");
  await col.deleteOne({ _id: ins.insertedId });
  console.log(`  MongoDB deleted _id: ${id}`);
  console.log(`  Waiting ${WAIT_MS/1000}s for automation poll...`);
  await sleep(WAIT_MS);
  const afterDelete = await pgRow(id);
  console.log(afterDelete
    ? `  ❌ Still found in PostgreSQL (unexpected)`
    : `  ✅ PostgreSQL: record deleted — NOT FOUND`);

  banner("RESULT");
  const updated = afterUpdate && afterUpdate.companyName && afterUpdate.companyName.includes("UPDATED");
  const pass = !afterDelete && updated && afterInsert;
  console.log(pass ? "  🟢 ALL 3 CRUD ops live-synced MongoDB → PostgreSQL" : "  🔴 Some ops failed — check above");
  await mongo.close(); await pg.end();
}
main().catch(e => { console.error(e); process.exit(1); });
