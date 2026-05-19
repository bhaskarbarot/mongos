#!/usr/bin/env bash
# =============================================================================
# start.sh — CRM AI Assistant full-stack launcher
#
# Services started (in order):
#   1. PostgreSQL Docker container  (mongos-postgres → port 5433)
#   2. pgAdmin Docker container     (pgadmin → port 5050)
#   3. MongoDB→PostgreSQL sync      (automation/sync.js — background)
#   4. FastAPI backend              (port 8000)
#   5. React frontend              (port 5173, proxies API to 8000)
#
# Stop everything: Ctrl+C
#
# Logs:
#   logs/automation_log.txt  — MongoDB sync
#   logs/backend.log         — FastAPI
#   logs/frontend.log        — Vite dev server
#   logs/query.log           — AI query log
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "      ${GREEN}✓${NC} $*"; }
warn() { echo -e "      ${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "      ${RED}✗${NC}  $*"; }

# ── Track PIDs for cleanup ─────────────────────────────────────────────────────
SYNC_PID=""
BACKEND_PID=""
FRONTEND_PID=""
PG_CONTAINER="mongos-postgres"
PGA_CONTAINER="pgadmin"

# ── Graceful shutdown ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo -e "${YELLOW}╔══════════════════════════════════════════════╗${NC}"
  echo -e "${YELLOW}║  Shutting down CRM AI Assistant…            ║${NC}"
  echo -e "${YELLOW}╚══════════════════════════════════════════════╝${NC}"

  [ -n "$FRONTEND_PID" ] && { echo "  [5] Stopping React frontend…"; kill "$FRONTEND_PID" 2>/dev/null; wait "$FRONTEND_PID" 2>/dev/null || true; }
  [ -n "$BACKEND_PID"  ] && { echo "  [4] Stopping FastAPI backend…"; kill "$BACKEND_PID"  2>/dev/null; wait "$BACKEND_PID"  2>/dev/null || true; }
  [ -n "$SYNC_PID"     ] && { echo "  [3] Stopping MongoDB sync…";   kill "$SYNC_PID"     2>/dev/null; wait "$SYNC_PID"     2>/dev/null || true; }
  echo "  [2] PostgreSQL container left running (use 'docker stop $PG_CONTAINER' to stop)"
  echo ""
  echo "  All services stopped. PostgreSQL kept running for fast restart."
  exit 0
}
trap cleanup INT TERM

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     CRM AI Assistant — Full Stack Launch     ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  1. PostgreSQL   port 5433                   ║${NC}"
echo -e "${GREEN}║  2. pgAdmin      port 5050  (optional)       ║${NC}"
echo -e "${GREEN}║  3. MongoDB sync (background)                ║${NC}"
echo -e "${GREEN}║  4. FastAPI      port 8000                   ║${NC}"
echo -e "${GREEN}║  5. React UI     port 5173                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

mkdir -p "$ROOT/logs"
: > "$ROOT/logs/automation_log.txt"

# ── Pre-flight checks ──────────────────────────────────────────────────────────
echo ">>> Pre-flight checks…"

# Check Docker
if ! docker info >/dev/null 2>&1; then
  err "Docker is not running. Start Docker first."; exit 1
fi
ok "Docker running"

# Check Ollama
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")
  ok "Ollama running ($MODEL_COUNT models)"
else
  warn "Ollama not detected at port 11434 — Text2SQL may fall back to Groq only"
fi

# Check Python deps
python3 -c "import fastapi, uvicorn, langchain_community, psycopg2" 2>/dev/null \
  && ok "Python dependencies OK" \
  || { err "Missing Python packages. Run: pip install -r requirements.txt"; exit 1; }

# Check Node.js — Vite requires Node 14+; use nvm Node 20 if available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"   # load nvm if installed

NODE_VER=$(node --version 2>/dev/null | sed 's/v//')
NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)

if [ -z "$NODE_VER" ]; then
  err "Node.js not found. Install Node.js 14+ from https://nodejs.org"; exit 1
fi

if [ "$NODE_MAJOR" -lt 14 ] 2>/dev/null; then
  warn "Node.js v$NODE_VER is too old for Vite (need 14+). Trying nvm Node 20…"
  if command -v nvm >/dev/null 2>&1; then
    nvm use 20 >/dev/null 2>&1 || nvm use 18 >/dev/null 2>&1
    NODE_VER=$(node --version 2>/dev/null | sed 's/v//')
    ok "Switched to Node.js v$NODE_VER via nvm"
  else
    err "nvm not found — install Node.js 20 from https://nodejs.org"; exit 1
  fi
else
  ok "Node.js v$NODE_VER"
fi

# Install automation deps if needed
if [ ! -d "$ROOT/automation/node_modules" ]; then
  echo ">>> Installing automation Node.js dependencies…"
  cd "$ROOT/automation" && npm install --silent && cd "$ROOT"
fi

# Install frontend deps if needed
if [ ! -d "$ROOT/chat-ui/node_modules" ]; then
  echo ">>> Installing React frontend dependencies…"
  cd "$ROOT/chat-ui" && npm install --silent && cd "$ROOT"
fi

echo ""

# ── Step 1: PostgreSQL (via Docker Compose) ────────────────────────────────────
echo ">>> [1/5] PostgreSQL Docker container ($PG_CONTAINER)…"

CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' "$PG_CONTAINER" 2>/dev/null || echo "missing")

if [ "$CONTAINER_STATUS" = "running" ]; then
  ok "Already running on port 5433"
else
  echo "      Starting via docker compose…"
  docker compose up -d postgres 2>/dev/null
  # Wait for postgres to be ready
  for i in $(seq 1 20); do
    docker exec "$PG_CONTAINER" pg_isready -U postgres -q 2>/dev/null && break
    sleep 2
  done
  ok "PostgreSQL ready on port 5433"
fi

# Verify DB connection
if PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d mongos_sync -c "SELECT 1" >/dev/null 2>&1; then
  TABLE_COUNT=$(PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d mongos_sync -tAc "SELECT COUNT(*) FROM pg_tables WHERE schemaname='public'" 2>/dev/null || echo "?")
  ok "DB connected — $TABLE_COUNT tables in mongos_sync"
else
  warn "DB connection check failed — backend will retry on first query"
fi

# ── Step 2: pgAdmin (via Docker Compose) ───────────────────────────────────────
echo ">>> [2/5] pgAdmin ($PGA_CONTAINER on port 5050)…"

PGA_STATUS=$(docker inspect -f '{{.State.Status}}' "$PGA_CONTAINER" 2>/dev/null || echo "missing")
if [ "$PGA_STATUS" = "running" ]; then
  ok "Already running — http://localhost:5050"
else
  docker compose up -d pgadmin 2>/dev/null
  ok "pgAdmin started — http://localhost:5050  (login: admin@admin.com / admin)"
fi

# ── Step 3: MongoDB → PostgreSQL sync ──────────────────────────────────────────
echo ">>> [3/5] MongoDB→PostgreSQL sync automation…"

# Kill any orphan sync processes
OLD_SYNC_PIDS="$(pgrep -f "$ROOT/automation/sync.js" 2>/dev/null || true)"
[ -n "$OLD_SYNC_PIDS" ] && { kill $OLD_SYNC_PIDS 2>/dev/null || true; sleep 1; }

node "$ROOT/automation/sync.js" >> "$ROOT/logs/automation_log.txt" 2>&1 &
SYNC_PID=$!
sleep 3

if kill -0 "$SYNC_PID" 2>/dev/null; then
  ok "Sync running (PID $SYNC_PID) — tail logs/automation_log.txt"
else
  warn "Sync exited early — check logs/automation_log.txt"
  SYNC_PID=""
fi

# ── Step 4: FastAPI backend ────────────────────────────────────────────────────
echo ">>> [4/5] FastAPI backend (port 8000)…"

fuser -k 8000/tcp 2>/dev/null || true
sleep 1

python3 -m uvicorn api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  2>&1 | tee "$ROOT/logs/backend.log" &
BACKEND_PID=${PIPESTATUS[0]}; BACKEND_PID=$!

echo "      Waiting for backend…"
READY=0
for i in $(seq 1 25); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then READY=1; break; fi
  sleep 2
done

if [ $READY -eq 1 ]; then
  ok "Backend ready (PID $BACKEND_PID) — http://localhost:8000"
  ok "API docs: http://localhost:8000/docs"
else
  warn "Backend slow to start — check logs/backend.log"
fi

# ── Step 5: React frontend ─────────────────────────────────────────────────────
echo ">>> [5/5] React frontend (port 5173)…"

fuser -k 5173/tcp 2>/dev/null || true
sleep 1

cd "$ROOT/chat-ui"
# Use nvm Node 20 if system node < 14
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
NODE_MAJOR=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
if [ "${NODE_MAJOR:-0}" -lt 14 ] 2>/dev/null; then
  nvm exec 20 npm run dev >> "$ROOT/logs/frontend.log" 2>&1 &
else
  npm run dev >> "$ROOT/logs/frontend.log" 2>&1 &
fi
FRONTEND_PID=$!
cd "$ROOT"
sleep 4

if kill -0 "$FRONTEND_PID" 2>/dev/null; then
  ok "Frontend ready (PID $FRONTEND_PID) — http://localhost:5173"
else
  warn "Frontend failed — check logs/frontend.log"
  FRONTEND_PID=""
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅  CRM AI Assistant is UP                              ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║  🌐  Chat UI   →  http://localhost:5173                  ║${NC}"
echo -e "${GREEN}║  ⚙️   API       →  http://localhost:8000                  ║${NC}"
echo -e "${GREEN}║  📊  pgAdmin   →  http://localhost:5050                  ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║  Backend logs → streaming below in terminal (+ backend.log)║${NC}"
echo -e "${GREEN}║  Other logs (tail in logs/):                             ║${NC}"
echo -e "${GREEN}║    automation_log.txt   MongoDB→PG sync                  ║${NC}"
echo -e "${GREEN}║    frontend.log         React/Vite                       ║${NC}"
echo -e "${GREEN}║    query.log            AI query log                     ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop                                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

wait
