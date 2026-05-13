/**
 * sync.js — MongoDB → PostgreSQL live sync engine
 *
 * Strategy:
 *   1. Auto-detect old JSONB format → drop all tables → rebuild with proper columns
 *   2. Try MongoDB change streams (real-time, instant)
 *   3. If replica set not available → fall back to polling every POLL_INTERVAL_MS
 *
 * Schema: every MongoDB field becomes its own typed PostgreSQL column.
 *   boolean → BOOLEAN | integer → BIGINT | float → NUMERIC
 *   object  → JSONB   | array   → JSONB  | string → TEXT
 *
 * Logs: logs/automation_log.txt (rotating, max 50 MB)
 */

"use strict";

require("dotenv").config({ path: require("path").join(__dirname, "..", ".env") });

const fs   = require("fs");
const path = require("path");

// ── Logger ─────────────────────────────────────────────────────────────────────
const LOG_DIR      = path.join(__dirname, "..", "logs");
const LOG_FILE     = path.join(LOG_DIR, "automation_log.txt");
const MAX_LOG_BYTES = 50 * 1024 * 1024; // 50 MB

if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

function rotateLogs() {
  try {
    if (fs.statSync(LOG_FILE).size > MAX_LOG_BYTES) {
      fs.renameSync(LOG_FILE, LOG_FILE + ".old");
    }
  } catch (_) { /* file may not exist yet */ }
}

function log(level, message, extra) {
  const ts   = new Date().toISOString();
  const line = `[${ts}] [${level.padEnd(5)}] ${message}${extra ? " | " + JSON.stringify(extra) : ""}\n`;
  process.stdout.write(line);
  try { rotateLogs(); fs.appendFileSync(LOG_FILE, line); } catch (_) { /* non-fatal */ }
}

const logger = {
  info:  (msg, extra) => log("INFO",  msg, extra),
  warn:  (msg, extra) => log("WARN",  msg, extra),
  error: (msg, extra) => log("ERROR", msg, extra),
  debug: (msg, extra) => log("DEBUG", msg, extra),
};

// ── Error flood guard ──────────────────────────────────────────────────────────
const _errorBuckets = Object.create(null);

function _errorKey(collection, err) {
  const msg = String((err && err.message) || err || "unknown error");
  if (msg.includes("ECONNREFUSED")) return `${collection}|ECONNREFUSED`;
  return `${collection}|${msg.slice(0, 120)}`;
}

function logSyncErrorOnce(collection, err, context = "sync") {
  const key = _errorKey(collection, err);
  const now = Date.now();
  const b   = _errorBuckets[key] || {
    collection, context,
    firstTs: now, lastTs: now, count: 0,
    sample: String((err && err.message) || err || "unknown error"),
  };
  b.count++;
  b.lastTs = now;
  _errorBuckets[key] = b;
}

setInterval(() => {
  const now = Date.now();
  for (const [key, b] of Object.entries(_errorBuckets)) {
    if (now - b.lastTs >= 10000) {
      logger.error(`${b.context} failed | ${b.collection} | ${b.sample} (×${b.count})`);
      delete _errorBuckets[key];
    }
  }
}, 2000).unref();

// ── Modules ────────────────────────────────────────────────────────────────────
const watchMongoChanges = require("./mongoWatcher");
const startMongoPolling = require("./mongoPoller");
const { flattenDocument } = require("./mapper");
const {
  upsertRow,
  deleteRow,
  deleteMissingRows,
  resetAllTables,
  detectOldFormat,
} = require("./sqlHandler");

// ── Stats ──────────────────────────────────────────────────────────────────────
const stats = { inserts: 0, updates: 0, deletes: 0, errors: 0, startedAt: new Date() };

function printStats() {
  const upMin = Math.floor((Date.now() - stats.startedAt.getTime()) / 60000);
  logger.info("Stats snapshot", {
    uptime:  `${upMin}m`,
    inserts:  stats.inserts,
    updates:  stats.updates,
    deletes:  stats.deletes,
    errors:   stats.errors,
  });
}

setInterval(printStats, 5 * 60 * 1000).unref();

// ── Per-collection poll counters ───────────────────────────────────────────────
const _pollCounts = {};

// ── Core sync helpers ──────────────────────────────────────────────────────────

/**
 * syncUpsert — flatten a MongoDB doc and upsert all its fields as columns.
 *
 * @param {string}  collection  - MongoDB collection name (= PG table name)
 * @param {Object}  doc         - raw MongoDB document
 * @param {string}  opType      - "insert" | "update"
 * @param {boolean} silent      - if true, just count (polling mode)
 */
async function syncUpsert(collection, doc, opType, silent = false) {
  try {
    const flat = flattenDocument(doc || {});
    if (!flat._id || flat._id.value === null) {
      logger.warn(`Skipping doc with no _id in ${collection}`);
      return;
    }

    await upsertRow(collection, flat);

    if (opType === "insert") stats.inserts++;
    else                     stats.updates++;

    if (!silent) {
      const verb = opType === "insert" ? "INSERT" : "UPDATE";
      logger.info(`${collection} === ${verb} | _id: ${flat._id.value}`);
    } else {
      _pollCounts[collection] = (_pollCounts[collection] || 0) + 1;
    }
  } catch (err) {
    stats.errors++;
    logSyncErrorOnce(collection, err, "syncUpsert");
  }
}

/**
 * syncPollDone — log a single summary line after a collection poll pass.
 */
function syncPollDone(collection, totalDocs) {
  const n = _pollCounts[collection] || 0;
  delete _pollCounts[collection];
  if (n > 0) {
    logger.info(`${collection} === polled ${n}/${totalDocs} docs synced to PostgreSQL`);
  }
}

/**
 * syncDelete — delete a single document by _id.
 */
async function syncDelete(collection, id) {
  if (!id) return;
  try {
    await deleteRow(collection, id);
    stats.deletes++;
    logger.info(`${collection} === DELETE | _id: ${id}`);
  } catch (err) {
    stats.errors++;
    logSyncErrorOnce(collection, err, "syncDelete");
  }
}

// ── Auto-reset: detect and rebuild on old JSONB format ────────────────────────

async function maybeReset() {
  const needsRebuild = await detectOldFormat();
  if (!needsRebuild) {
    logger.info("Schema check: column-per-field format detected ✓ — no reset needed");
    return;
  }

  logger.warn("═══ SCHEMA MIGRATION DETECTED ═══");
  logger.warn("Old JSONB 'document' column format found → dropping all tables and rebuilding with proper columns...");

  const dropped = await resetAllTables(logger);
  logger.info(`Migration: ${dropped.length} tables dropped — full re-sync will now run`);
}

// ── Main ───────────────────────────────────────────────────────────────────────

async function startSync() {
  logger.info("═══ CRM MongoDB→PostgreSQL Sync Starting ═══");
  logger.info("Format     : column-per-field (proper typed columns, no JSONB blob)");
  logger.info("MongoDB    : " + (process.env.MONGODB_DB || process.env.MONGO_DB || "(env not set)"));
  logger.info("PostgreSQL : " + process.env.POSTGRES_HOST + ":" + process.env.POSTGRES_PORT + "/" + process.env.POSTGRES_DB);
  logger.info("Interval   : " + (process.env.POLL_INTERVAL_MS || 5000) + "ms");

  // Step 1: Auto-detect and reset old JSONB format if needed
  await maybeReset();

  let fallbackStarted     = false;
  let initialSyncCompleted = false;

  // ── Change stream failure handler ────────────────────────────────────────────
  async function startPollingFallback(error) {
    if (fallbackStarted) return;
    fallbackStarted = true;

    if (error && error.code === 40573) {
      logger.warn("Change streams need replica set — switching to POLLING mode (" + (process.env.POLL_INTERVAL_MS || 5000) + "ms)");
    } else {
      logger.warn("Change stream unavailable — switching to POLLING mode", { reason: error && error.message });
    }

    await startMongoPolling({
      onUpsert: async (collection, doc) => {
        await syncUpsert(collection, doc, "update", true /* silent, count only */);
      },
      onDelete: async (collection, idsInMongo) => {
        try {
          await deleteMissingRows(collection, idsInMongo);
          syncPollDone(collection, idsInMongo.length);
        } catch (err) {
          stats.errors++;
          logSyncErrorOnce(collection, err, "reconcile");
        }
      },
      onPassComplete: async ({ collections }) => {
        if (!initialSyncCompleted) {
          initialSyncCompleted = true;
          logger.info("🟢 INITIAL SYNC COMPLETE — all collections synced, live polling active", {
            collections,
            poll_interval_ms: Number(process.env.POLL_INTERVAL_MS || 5000),
          });
        }
      },
    });
  }

  // ── Try change streams (real-time) ───────────────────────────────────────────
  try {
    await watchMongoChanges(
      async (change) => {
        const collection = change.ns && change.ns.coll;
        if (!collection) return;

        if (change.operationType === "insert") {
          await syncUpsert(collection, change.fullDocument || {}, "insert", false);
        } else if (change.operationType === "update" || change.operationType === "replace") {
          await syncUpsert(collection, change.fullDocument || {}, "update", false);
        } else if (change.operationType === "delete") {
          const dk = change.documentKey;
          const id = (dk && dk._id) ? String(dk._id) : null;
          await syncDelete(collection, id);
        }
      },
      async (error) => {
        logger.error("Change stream error", { error: error && error.message, code: error && error.code });
        await startPollingFallback(error);
      }
    );
    logger.info("Change stream ACTIVE — real-time sync running ✓");
    if (!initialSyncCompleted) {
      initialSyncCompleted = true;
      logger.info("🟢 INITIAL SYNC READY — change stream connected, live updates active");
    }
  } catch (err) {
    logger.warn("Change stream failed — using polling fallback", { error: err && err.message });
    await startPollingFallback(err);
  }
}

// ── Process error handlers ─────────────────────────────────────────────────────
process.on("uncaughtException",  (err)    => logger.error("Uncaught exception",  { error: err.message }));
process.on("unhandledRejection", (reason) => logger.error("Unhandled rejection", { reason: String(reason) }));
process.on("SIGTERM", () => { logger.info("SIGTERM — stopping sync"); printStats(); process.exit(0); });
process.on("SIGINT",  () => { logger.info("SIGINT — stopping sync");  printStats(); process.exit(0); });

startSync().catch((err) => {
  logger.error("Sync startup failed", { error: err.message });
  process.exit(1);
});
