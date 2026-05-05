require("dotenv").config();
const { MongoClient } = require("mongodb");

async function watchMongoChanges(onChange, onError) {
  const uri = process.env.MONGO_URI || process.env.MONGODB_URI;
  const dbName = process.env.MONGO_DB || process.env.MONGODB_DB;

  if (!uri) {
    throw new Error("Missing Mongo URI. Set MONGO_URI or MONGODB_URI in .env");
  }
  if (!dbName) {
    throw new Error("Missing Mongo DB name. Set MONGO_DB or MONGODB_DB in .env");
  }

  const client = new MongoClient(uri);
  await client.connect();
  console.log("Connected to MongoDB");

  const db = client.db(dbName);
  const changeStream = db.watch([], { fullDocument: "updateLookup" });

  changeStream.on("change", async (change) => {
    try {
      console.log(`Change detected: ${change.operationType}`);
      await onChange(change);
    } catch (error) {
      console.error("Failed to process Mongo change:", error);
    }
  });

  changeStream.on("error", (error) => {
    if (typeof onError === "function") {
      onError(error);
      return;
    }
    console.error("Mongo change stream error:", error);
  });
}

module.exports = watchMongoChanges;
