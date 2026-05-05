const watchMongoChanges = require("./mongoWatcher");
const startMongoPolling = require("./mongoPoller");
const mapDocumentToSQL = require("./mapper");
const { insertRow, deleteRow, deleteMissingRows } = require("./sqlHandler");

async function startSync() {
  let fallbackStarted = false;

  async function syncUpsert(collection, doc) {
    const mapped = mapDocumentToSQL(doc || {});
    await insertRow(collection, mapped);
  }

  async function syncDelete(collection, id) {
    if (!id) {
      return;
    }
    await deleteRow(collection, id);
  }

  async function startPollingFallback(error) {
    if (fallbackStarted) {
      return;
    }
    fallbackStarted = true;
    const isReplicaSetRequired = error && error.code === 40573;
    if (isReplicaSetRequired) {
      console.warn("Change streams need a replica set. Falling back to polling mode.");
    } else {
      console.warn("Change stream unavailable. Falling back to polling mode.");
    }
    await startMongoPolling({
      onUpsert: async (collection, doc) => {
        await syncUpsert(collection, doc);
        console.log("Polled INSERT/UPDATE synced");
      },
      onDelete: async (collection, idsInMongo) => {
        await deleteMissingRows(collection, idsInMongo);
        console.log("Polled DELETE reconciliation synced");
      },
    });
  }

  await watchMongoChanges(async (change) => {
    const collection = change.ns && change.ns.coll;
    if (!collection) {
      return;
    }

    if (change.operationType === "insert" || change.operationType === "update") {
      await syncUpsert(collection, change.fullDocument || {});
      console.log("Synced INSERT/UPDATE");
      return;
    }

    if (change.operationType === "delete") {
      const id = change.documentKey && change.documentKey._id ? String(change.documentKey._id) : null;
      await syncDelete(collection, id);
      console.log("Synced DELETE");
    }
  }, async (error) => {
    console.error("Mongo change stream error:", error);
    await startPollingFallback(error);
  });
}

startSync().catch((error) => {
  console.error("Sync startup failed:", error);
  process.exit(1);
});
