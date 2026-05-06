#!/usr/bin/env bash
# =============================================================================
# start.sh — Full CRM AI Assistant launcher
#
# Starts (in order):
#   1. PostgreSQL Docker container  (mongos-postgres → port 5433)
#   2. MongoDB→PostgreSQL automation  (automation/sync.js)
#   3. FastAPI backend               (port 8000)
#   4. React frontend                (port 5173, proxies API to 8000)
#
# Stops on Ctrl+C:
#   • Frontend → Backend → Automation → PostgreSQL container
#
# Logs:
#   logs/automation_log.txt  — MongoDB sync
#   logs/backend.log         — FastAPI
#   logs/frontend.log        — Vite dev server
#   logs/query.log           — AI query log
# =============================================================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Track PIDs ────────────────────────────────────────────────────────────────
CONTAINER_NAME="mongos-postgres"
SYNC_PID=""
BACKEND_PID=""
FRONTEND_PID=""

# ── Graceful shutdown ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Shutting down CRM AI Assistant…                    ║"
  echo "╚══════════════════════════════════════════════════════╝"

  [ -n "$FRONTEND_PID" ] && {
    echo "  [4/4] Stopping React frontend…"
    kill "$FRONTEND_PID" 2>/dev/null; wait "$FRONTEND_PID" 2>/dev/null || true
  }
  [ -n "$BACKEND_PID" ] && {
    echo "  [3/4] Stopping FastAPI backend…"
    kill "$BACKEND_PID" 2>/dev/null; wait "$BACKEND_PID" 2>/dev/null || true
  }
  [ -n "$SYNC_PID" ] && {
    echo "  [2/4] Stopping MongoDB sync automation…"
    kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null || true
  }
  echo "  [1/4] Stopping PostgreSQL container ($CONTAINER_NAME)…"
  docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true

  echo ""
  echo "  All services stopped cleanly."
  exit 0
}
trap cleanup INT TERM

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         CRM AI Assistant — Full Stack Launch        ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  1. PostgreSQL container  (port 5433)               ║"
echo "║  2. MongoDB→PG automation (logs/automation_log.txt) ║"
echo "║  3. FastAPI backend       (port 8000)               ║"
echo "║  4. React frontend        (port 5173)               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Ensure logs dir exists ────────────────────────────────────────────────────
mkdir -p "$ROOT/logs"

# ── Install dependencies (first-run only) ─────────────────────────────────────
if [ ! -d "$ROOT/automation/node_modules" ]; then
  echo ">>> Installing automation Node.js dependencies…"
  cd "$ROOT/automation" && npm install --silent && cd "$ROOT"
fi

if [ ! -d "$ROOT/chat-ui/node_modules" ]; then
  echo ">>> Installing React frontend dependencies…"
  cd "$ROOT/chat-ui" && npm install --silent && cd "$ROOT"
fi

# ── Step 1: PostgreSQL Docker container ──────────────────────────────────────
echo ">>> [1/4] PostgreSQL Docker container ($CONTAINER_NAME)…"

CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing")

if [ "$CONTAINER_STATUS" = "running" ]; then
  echo "      Already running — skipped ✓"
elif [ "$CONTAINER_STATUS" = "missing" ]; then
  echo "      ERROR: Container '$CONTAINER_NAME' not found."
  echo "      Run: docker run --name $CONTAINER_NAME -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=mongos_sync -p 5433:5432 -d postgres:16"
  exit 1
else
  echo "      Starting (was: $CONTAINER_STATUS)…"
  docker start "$CONTAINER_NAME" > /dev/null
  for i in $(seq 1 15); do
    docker exec "$CONTAINER_NAME" pg_isready -U postgres -q 2>/dev/null && break
    sleep 2
  done
  echo "      PostgreSQL ready ✓"
fi

# ── Step 2: MongoDB → PostgreSQL automation ───────────────────────────────────
echo ">>> [2/4] Starting MongoDB→PostgreSQL sync automation…"

node "$ROOT/automation/sync.js" >> "$ROOT/logs/automation_log.txt" 2>&1 &
SYNC_PID=$!
sleep 3

if kill -0 "$SYNC_PID" 2>/dev/null; then
  echo "      Sync running ✓  (PID $SYNC_PID)"
  echo "      Log → logs/automation_log.txt"
else
  echo "      WARNING: Sync exited early — check logs/automation_log.txt"
fi

# ── Step 3: FastAPI backend ───────────────────────────────────────────────────
echo ">>> [3/4] Starting FastAPI backend (port 8000)…"

fuser -k 8000/tcp 2>/dev/null || true
sleep 1

CACHE_DISABLED=true REDIS_CACHE_TTL=0 python3 -m uvicorn api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  > >(tee -a "$ROOT/logs/backend.log") \
  2> >(tee -a "$ROOT/logs/backend.log" >&2) &
BACKEND_PID=$!

READY=0
for i in $(seq 1 20); do
  curl -sf http://localhost:8000/health > /dev/null 2>&1 && READY=1 && break
  sleep 2
done

[ $READY -eq 1 ] && echo "      Backend ready ✓  (PID $BACKEND_PID)  Log → logs/backend.log" \
                 || echo "      Backend initialising — check logs/backend.log if queries fail"

# ── Step 4: React frontend ────────────────────────────────────────────────────
echo ">>> [4/4] Starting React frontend (port 5173)…"

fuser -k 5173/tcp 2>/dev/null || true
sleep 1

cd "$ROOT/chat-ui"
npm run dev >> "$ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$ROOT"
sleep 5

echo "      Frontend ready ✓  (PID $FRONTEND_PID)  Log → logs/frontend.log"

# ── All up ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  CRM AI Assistant is UP                         ║"
echo "║                                                      ║"
echo "║  Open in browser:  http://localhost:5173            ║"
echo "║                                                      ║"
echo "║  Logs (all in logs/):                               ║"
echo "║    automation_log.txt   MongoDB→PG sync             ║"
echo "║    backend.log          FastAPI                      ║"
echo "║    frontend.log         Vite dev server              ║"
echo "║    query.log            AI query log                 ║"
echo "║                                                      ║"
echo "║  Press Ctrl+C to stop everything cleanly            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

wait
