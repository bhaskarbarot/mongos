/**
 * sync.js — MongoDB → PostgreSQL live sync engine
 *
 * Strategy:
 *   1. Try MongoDB change streams (real-time, instant)
 *   2. If replica set not available → fall back to polling every POLL_INTERVAL_MS
 *
 * Logs: logs/automation_log.txt (rotating, max 50 MB)
 * Location: automation/sync.js → logs/ is at ../logs/ relative to this file
 */

"use strict";

// Load .env from project root (one level up from automation/)
require("dotenv").config({ path: require("path").join(__dirname, "..", ".env") });

const fs   = require("fs");
const path = require("path");

// ── Logger ────────────────────────────────────────────────────────────────────
// Logs directory lives at project root: ../logs/
const LOG_DIR  = path.join(__dirname, "..", "logs");
const LOG_FILE = path.join(LOG_DIR, "automation_log.txt");
const MAX_LOG_BYTES = 50 * 1024 * 1024; // 50 MB rotate

if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

function rotateLogs() {
  try {
    const stat = fs.statSync(LOG_FILE);
    if (stat.size > MAX_LOG_BYTES) {
      fs.renameSync(LOG_FILE, LOG_FILE + ".old");
    }
  } catch (_) { /* file doesn't exist yet — OK */ }
}

function log(level, message, extra) {
  const ts   = new Date().toISOString();
  const line = `[${ts}] [${level.padEnd(5)}] ${message}${extra ? " | " + JSON.stringify(extra) : ""}\n`;
  process.stdout.write(line);
  try {
    rotateLogs();
    fs.appendFileSync(LOG_FILE, line);
  } catch (_) { /* non-fatal */ }
}

const logger = {
  info:  (msg, extra) => log("INFO",  msg, extra),
  warn:  (msg, extra) => log("WARN",  msg, extra),
  error: (msg, extra) => log("ERROR", msg, extra),
  debug: (msg, extra) => log("DEBUG", msg, extra),
};

// ── Error flood guard ──────────────────────────────────────────────────────────
// During DB outages, polling can touch thousands of docs and generate massive
// repeated ECONNREFUSED lines. We aggregate and print a compact periodic summary.
const _errorBuckets = Object.create(null);
const ERROR_FLUSH_MS = 10000;

function _errorKey(collection, err) {
  const msg = String(err?.message || err || "unknown error");
  if (msg.includes("ECONNREFUSED")) return `${collection}|ECONNREFUSED`;
  return `${collection}|${msg.slice(0, 120)}`;
}

function logSyncErrorOnce(collection, err, context = "sync") {
  const key = _errorKey(collection, err);
  const now = Date.now();
  const bucket = _errorBuckets[key] || {
    collection,
    context,
    firstTs: now,
    lastTs: now,
    count: 0,
    sample: String(err?.message || err || "unknown error"),
  };
  bucket.count += 1;
  bucket.lastTs = now;
  _errorBuckets[key] = bucket;
}

setInterval(() => {
  const now = Date.now();
  for (const [key, b] of Object.entries(_errorBuckets)) {
    if (now - b.lastTs >= ERROR_FLUSH_MS) {
      logger.error(`${b.context} failed | ${b.collection} | ${b.sample} (repeated ${b.count}x)`);
      delete _errorBuckets[key];
    }
  }
}, 2000).unref();

// ── Modules ───────────────────────────────────────────────────────────────────
const watchMongoChanges  = require("./mongoWatcher");
const startMongoPolling  = require("./mongoPoller");
const mapDocumentToSQL   = require("./mapper");
const { insertRow, deleteRow, deleteMissingRows } = require("./sqlHandler");

// ── Stats tracking ────────────────────────────────────────────────────────────
const stats = {
  inserts:   0,
  updates:   0,
  deletes:   0,
  errors:    0,
  startedAt: new Date(),
};

function printStats() {
  const upMs  = Date.now() - stats.startedAt.getTime();
  const upMin = Math.floor(upMs / 60000);
  logger.info("Stats snapshot", {
    uptime:  `${upMin}m`,
    inserts: stats.inserts,
    updates: stats.updates,
    deletes: stats.deletes,
    errors:  stats.errors,
  });
}

// Print stats every 5 minutes
setInterval(printStats, 5 * 60 * 1000).unref();

// ── Per-collection poll counter (reset each poll pass) ───────────────────────
// Used by polling mode to batch per-doc noise into one summary line per collection.
const _pollCounts = {};   // { collectionName: number }

// ── Core sync helpers ─────────────────────────────────────────────────────────

/**
 * syncUpsert — called per document during polling or per change-stream event.
 *
 * For POLLING: silently counts docs; the summary is logged by syncPollDone().
 * For CHANGE STREAMS: logs one line immediately (real CRUD action by the user).
 */
async function syncUpsert(collection, doc, opType, silent = false) {
  try {
    const mapped = mapDocumentToSQL(doc || {});
    await insertRow(collection, mapped);
    if (opType === "insert") stats.inserts++;
    else                     stats.updates++;

    if (!silent) {
      // Change stream path — log one line per real event
      const verb = opType === "insert" ? "INSERT" : "UPDATE";
      logger.info(`${collection} === ${verb} done in MongoDB → updated in PostgreSQL | id: ${mapped.id}`);
    } else {
      // Polling path — just count, summary logged by syncPollDone()
      _pollCounts[collection] = (_pollCounts[collection] || 0) + 1;
    }
  } catch (err) {
    stats.errors++;
    logSyncErrorOnce(collection, err, "syncUpsert");
  }
}

/**
 * syncPollDone — called once per collection after the poll pass completes.
 * Logs a single summary line for that collection.
 */
function syncPollDone(collection, totalDocs) {
  const n = _pollCounts[collection] || 0;
  delete _pollCounts[collection];
  // Only log if anything was actually written
  if (n > 0) {
    logger.info(`${collection} === synced in MongoDB → updated in PostgreSQL (${n} of ${totalDocs} records)`);
  }
}

async function syncDelete(collection, id) {
  if (!id) return;
  try {
    await deleteRow(collection, id);
    stats.deletes++;
    // Change stream DELETE — one line per real event
    logger.info(`${collection} === DELETE done in MongoDB → removed from PostgreSQL | id: ${id}`);
  } catch (err) {
    stats.errors++;
    logSyncErrorOnce(collection, err, "syncDelete");
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function startSync() {
  logger.info("═══ CRM MongoDB→PostgreSQL Sync Starting ═══");
  logger.info("MongoDB      : " + (process.env.MONGODB_DB || process.env.MONGO_DB));
  logger.info("PostgreSQL   : " + process.env.POSTGRES_HOST + ":" + process.env.POSTGRES_PORT + "/" + process.env.POSTGRES_DB);
  logger.info("Poll interval: " + (process.env.POLL_INTERVAL_MS || 5000) + "ms");
  logger.info("Log file     : " + LOG_FILE);

  let fallbackStarted = false;
  let initialSyncCompleted = false;

  async function startPollingFallback(error) {
    if (fallbackStarted) return;
    fallbackStarted = true;

    if (error && error.code === 40573) {
      logger.warn("Change streams need replica set — using POLLING mode (every " + (process.env.POLL_INTERVAL_MS || 5000) + "ms)");
    } else {
      logger.warn("Change stream unavailable — using POLLING mode", { reason: error?.message });
    }

    await startMongoPolling({
      // Called per-document — silent, just count
      onUpsert: async (collection, doc) => {
        await syncUpsert(collection, doc, "update", true /* silent */);
      },
      // Called once per collection after all docs — log the summary line
      onDelete: async (collection, idsInMongo) => {
        try {
          await deleteMissingRows(collection, idsInMongo);
          syncPollDone(collection, idsInMongo.length);
        } catch (err) {
          stats.errors++;
          logSyncErrorOnce(collection, err, "reconcile");
        }
      },
      // Called once per full poll pass.
      onPassComplete: async ({ collections }) => {
        if (!initialSyncCompleted) {
          initialSyncCompleted = true;
          logger.info("🟢 INITIAL SYNC COMPLETE — polling is now in live watch mode", {
            collections,
            poll_interval_ms: Number(process.env.POLL_INTERVAL_MS || 5000),
          });
        }
      },
    });
  }

  // Try change streams first (instant real-time, one log per real event)
  try {
    await watchMongoChanges(
      async (change) => {
        const collection = change.ns?.coll;
        if (!collection) return;

        if (change.operationType === "insert") {
          await syncUpsert(collection, change.fullDocument || {}, "insert", false);
        } else if (change.operationType === "update" || change.operationType === "replace") {
          await syncUpsert(collection, change.fullDocument || {}, "update", false);
        } else if (change.operationType === "delete") {
          const id = change.documentKey?._id ? String(change.documentKey._id) : null;
          await syncDelete(collection, id);
        }
      },
      async (error) => {
        logger.error("Change stream error", { error: error?.message, code: error?.code });
        await startPollingFallback(error);
      }
    );
    logger.info("Change stream ACTIVE — real-time sync running ✓");
    if (!initialSyncCompleted) {
      initialSyncCompleted = true;
      logger.info("🟢 INITIAL SYNC READY — change stream connected and live updates active");
    }
  } catch (err) {
    logger.warn("Change stream failed — using polling fallback", { error: err.message });
    await startPollingFallback(err);
  }
}

// Graceful error handling
process.on("uncaughtException", (err) => {
  logger.error("Uncaught exception", { error: err.message });
});
process.on("unhandledRejection", (reason) => {
  logger.error("Unhandled rejection", { reason: String(reason) });
});
process.on("SIGTERM", () => { logger.info("SIGTERM — stopping sync"); printStats(); process.exit(0); });
process.on("SIGINT",  () => { logger.info("SIGINT — stopping sync");  printStats(); process.exit(0); });

startSync().catch((err) => {
  logger.error("Sync startup failed", { error: err.message });
  process.exit(1);
});
