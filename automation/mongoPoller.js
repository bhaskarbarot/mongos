require("dotenv").config();
const { MongoClient } = require("mongodb");

function getEnv(nameA, nameB) {
  return process.env[nameA] || process.env[nameB];
}

function parseCollections() {
  const raw = getEnv("MONGO_COLLECTIONS", "MONGODB_COLLECTIONS");
  if (!raw) {
    return [];
  }
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

async function getCollections(db) {
  const configured = parseCollections();
  if (configured.length) {
    return configured;
  }

  const result = await db.listCollections({}, { nameOnly: true }).toArray();
  return result
    .map((item) => item.name)
    .filter((name) => name && !name.startsWith("system."));
}

async function startMongoPolling({ onUpsert, onDelete }) {
  const uri = getEnv("MONGO_URI", "MONGODB_URI");
  const dbName = getEnv("MONGO_DB", "MONGODB_DB");
  const intervalMs = Number(process.env.POLL_INTERVAL_MS || 5000);

  if (!uri) {
    throw new Error("Missing Mongo URI. Set MONGO_URI or MONGODB_URI in .env");
  }
  if (!dbName) {
    throw new Error("Missing Mongo DB name. Set MONGO_DB or MONGODB_DB in .env");
  }
  const client = new MongoClient(uri);
  await client.connect();
  console.log(`Polling mode active (${intervalMs}ms)`);

  const db = client.db(dbName);
  let busy = false;

  async function runOnePass() {
    if (busy) {
      return;
    }
    busy = true;
    try {
      const collections = await getCollections(db);
      if (!collections.length) {
        console.warn("No collections found for polling.");
      }

      for (const collectionName of collections) {
        const collection = db.collection(collectionName);
        const docs = await collection.find({}).toArray();
        const currentIds = new Set();

        for (const doc of docs) {
          const id = String(doc._id);
          currentIds.add(id);
          await onUpsert(collectionName, doc);
        }

        if (typeof onDelete === "function") {
          await onDelete(collectionName, Array.from(currentIds));
        }
      }
    } catch (error) {
      console.error("Polling pass failed:", error);
    } finally {
      busy = false;
    }
  }

  await runOnePass();
  setInterval(runOnePass, intervalMs);
}

module.exports = startMongoPolling;
