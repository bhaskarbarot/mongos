#!/usr/bin/env bash
# =============================================================================
# start.sh — CRM AI Assistant — Full Stack Launcher
#
# Services started (in order):
#   1. PostgreSQL  Docker container  mongos-postgres  → port 5433
#   2. MongoDB→PG  automation        sync.js          → background
#   3. FastAPI     backend           api.py           → port 8000
#   4. React       frontend          chat-ui          → port 5173
#
# Stop cleanly with Ctrl+C — all 4 services shut down in reverse order.
#
# Logs written to logs/:
#   automation_log.txt   MongoDB → PostgreSQL sync
#   backend.log          FastAPI + pipeline
#   frontend.log         Vite dev server
#   query.log            AI query log (written by pipeline)
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Load .env so we can read POSTGRES_* and GROQ_API_KEY ─────────────────────
if [ -f "$ROOT/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +o allexport
fi

# ── Config (can be overridden via .env) ───────────────────────────────────────
CONTAINER_NAME="${CONTAINER_NAME:-mongos-postgres}"
PG_PORT="${POSTGRES_PORT:-5433}"
PG_USER="${POSTGRES_USER:-postgres}"
PG_PASS="${POSTGRES_PASSWORD:-postgres}"
PG_DB="${POSTGRES_DB:-mongos_sync}"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── PID tracking ──────────────────────────────────────────────────────────────
SYNC_PID=""
BACKEND_PID=""
FRONTEND_PID=""

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
BOLD="\033[1m"
RESET="\033[0m"

ok()   { echo -e "      ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "      ${YELLOW}⚠${RESET}  $*"; }
err()  { echo -e "      ${RED}✗${RESET}  $*"; }
info() { echo -e "      ${CYAN}→${RESET}  $*"; }

# ── Graceful shutdown ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
  echo -e "${BOLD}║  Shutting down CRM AI Assistant…                    ║${RESET}"
  echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"

  [ -n "$FRONTEND_PID" ] && {
    echo -e "  [4/4] Stopping React frontend (PID $FRONTEND_PID)…"
    kill "$FRONTEND_PID" 2>/dev/null
    wait "$FRONTEND_PID" 2>/dev/null || true
  }
  [ -n "$BACKEND_PID" ] && {
    echo -e "  [3/4] Stopping FastAPI backend (PID $BACKEND_PID)…"
    kill "$BACKEND_PID" 2>/dev/null
    wait "$BACKEND_PID" 2>/dev/null || true
  }
  [ -n "$SYNC_PID" ] && {
    echo -e "  [2/4] Stopping MongoDB sync (PID $SYNC_PID)…"
    kill "$SYNC_PID" 2>/dev/null
    wait "$SYNC_PID" 2>/dev/null || true
  }
  echo -e "  [1/4] Stopping PostgreSQL container ($CONTAINER_NAME)…"
  docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true

  echo ""
  echo -e "  ${GREEN}All services stopped cleanly.${RESET}"
  exit 0
}
trap cleanup INT TERM

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       CRM AI Assistant — Full Stack Launch          ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Pipeline : Dynamic LLM  (Groq 70b × 3 accounts)   ║${RESET}"
echo -e "${BOLD}║  1. PostgreSQL  container  (port $PG_PORT)               ║${RESET}"
echo -e "${BOLD}║  2. MongoDB→PG automation  (logs/automation_log.txt)║${RESET}"
echo -e "${BOLD}║  3. FastAPI backend        (port $BACKEND_PORT)               ║${RESET}"
echo -e "${BOLD}║  4. React frontend         (port $FRONTEND_PORT)               ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo -e "${BOLD}>>> Pre-flight checks…${RESET}"

# Docker
if ! command -v docker &> /dev/null; then
  err "Docker not found. Install Docker and retry."; exit 1
fi
ok "Docker found: $(docker --version | cut -d' ' -f3 | tr -d ',')"

# Python
if ! command -v python3 &> /dev/null; then
  err "python3 not found. Install Python 3.10+ and retry."; exit 1
fi
ok "Python: $(python3 --version)"

# Groq API keys
GROQ_COUNT=0
[ -n "${GROQ_API_KEY:-}"   ] && GROQ_COUNT=$((GROQ_COUNT+1))
[ -n "${GROQ_API_KEY_2:-}" ] && GROQ_COUNT=$((GROQ_COUNT+1))
[ -n "${GROQ_API_KEY_3:-}" ] && GROQ_COUNT=$((GROQ_COUNT+1))
if [ "$GROQ_COUNT" -eq 0 ]; then
  warn "No GROQ_API_KEY found — LLM will use Gemini/Ollama fallbacks only"
else
  ok "$GROQ_COUNT Groq API account(s) configured → ~$((GROQ_COUNT*12))K TPM on 70b"
fi

# Gemini
[ -n "${GEMINI_API_KEY:-}" ] && ok "Gemini API key configured" \
                              || warn "No GEMINI_API_KEY — Gemini fallback disabled"

echo ""

# ── Logs directory ────────────────────────────────────────────────────────────
mkdir -p "$ROOT/logs"
: > "$ROOT/logs/automation_log.txt"   # fresh each launch
: > "$ROOT/logs/backend.log"          # fresh each launch
: > "$ROOT/logs/frontend.log"         # fresh each launch

# ── Node.js detection (prefer NVM Node 20+, fall back to system node) ─────────
NODE_BIN=""
# Try nvm versions newest first
for ver_dir in $(ls -1d "$HOME/.nvm/versions/node"/v* 2>/dev/null | sort -rV); do
  major="${ver_dir##*/v}"
  major="${major%%.*}"
  if [ "$major" -ge 20 ] 2>/dev/null && [ -x "$ver_dir/bin/node" ]; then
    NODE_BIN="$ver_dir/bin/node"
    break
  fi
done
# Fall back to system node (warn if < v20)
if [ -z "$NODE_BIN" ]; then
  NODE_BIN="$(command -v node 2>/dev/null || true)"
  if [ -z "$NODE_BIN" ]; then
    err "Node.js not found. Install Node.js 20+ via nvm or system package."; exit 1
  fi
  NODE_VER="$("$NODE_BIN" --version 2>/dev/null | sed 's/v//')"
  NODE_MAJOR="${NODE_VER%%.*}"
  if [ "$NODE_MAJOR" -lt 20 ] 2>/dev/null; then
    warn "Node.js $NODE_VER found — Node 20+ recommended for MongoDB driver"
  fi
fi
ok "Node.js: $("$NODE_BIN" --version)  ($NODE_BIN)"

# ── Install Node dependencies (first run only) ────────────────────────────────
if [ ! -d "$ROOT/automation/node_modules" ]; then
  echo ""
  info "Installing automation Node.js dependencies (first run)…"
  cd "$ROOT/automation" && npm install --silent && cd "$ROOT"
  ok "automation/node_modules ready"
fi

if [ ! -d "$ROOT/chat-ui/node_modules" ]; then
  echo ""
  info "Installing React frontend dependencies (first run)…"
  cd "$ROOT/chat-ui" && npm install --silent && cd "$ROOT"
  ok "chat-ui/node_modules ready"
fi

# ── Install Python dependencies (first run only) ──────────────────────────────
if [ -f "$ROOT/requirements.txt" ]; then
  if ! python3 -c "import fastapi, uvicorn, langchain_community" &>/dev/null; then
    echo ""
    info "Installing Python dependencies (first run)…"
    pip3 install -q -r "$ROOT/requirements.txt"
    ok "Python dependencies ready"
  fi
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PostgreSQL Docker Container
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}>>> [1/4] PostgreSQL Docker container ($CONTAINER_NAME, port $PG_PORT)…${RESET}"

CONTAINER_STATUS="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing")"

if [ "$CONTAINER_STATUS" = "running" ]; then
  ok "Already running — skipped"

elif [ "$CONTAINER_STATUS" = "missing" ]; then
  info "Container not found — creating it now…"
  docker run \
    --name "$CONTAINER_NAME" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASS" \
    -e POSTGRES_DB="$PG_DB" \
    -p "${PG_PORT}:5432" \
    -d postgres:16 \
    > /dev/null
  info "Waiting for PostgreSQL to be ready…"
  for i in $(seq 1 20); do
    docker exec "$CONTAINER_NAME" pg_isready -U "$PG_USER" -q 2>/dev/null && break
    sleep 2
  done
  ok "PostgreSQL container created and started"

else
  info "Starting container (was: $CONTAINER_STATUS)…"
  docker start "$CONTAINER_NAME" > /dev/null
  for i in $(seq 1 15); do
    docker exec "$CONTAINER_NAME" pg_isready -U "$PG_USER" -q 2>/dev/null && break
    sleep 2
  done
  ok "PostgreSQL ready"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MongoDB → PostgreSQL Sync Automation
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}>>> [2/4] MongoDB→PostgreSQL sync automation…${RESET}"

# Kill any leftover sync process from a previous run
OLD_SYNC_PIDS="$(pgrep -f "$ROOT/automation/sync.js" 2>/dev/null || true)"
if [ -n "$OLD_SYNC_PIDS" ]; then
  info "Stopping leftover sync process(es): $OLD_SYNC_PIDS"
  kill $OLD_SYNC_PIDS 2>/dev/null || true
  sleep 1
fi

"$NODE_BIN" "$ROOT/automation/sync.js" \
  >> "$ROOT/logs/automation_log.txt" 2>&1 &
SYNC_PID=$!
sleep 3

if kill -0 "$SYNC_PID" 2>/dev/null; then
  ok "Sync running  (PID $SYNC_PID)   Log → logs/automation_log.txt"
else
  warn "Sync exited early — check logs/automation_log.txt"
  warn "Continuing without live sync (existing data still usable)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FastAPI Backend
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}>>> [3/4] FastAPI backend (port $BACKEND_PORT)…${RESET}"

# Free the port if something is already using it
fuser -k "${BACKEND_PORT}/tcp" 2>/dev/null || true
sleep 1

python3 -m uvicorn api:app \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" \
  --log-level info \
  >> "$ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!

# Wait up to 40s for the backend to respond — new pipeline does DB + schema warmup
info "Waiting for backend to start…"
READY=0
for i in $(seq 1 20); do
  if curl -sf "http://localhost:${BACKEND_PORT}/health" > /dev/null 2>&1; then
    READY=1; break
  fi
  sleep 2
done

if [ "$READY" -eq 1 ]; then
  ok "Backend ready   (PID $BACKEND_PID)   Log → logs/backend.log"
else
  warn "Backend did not respond in 40s — check logs/backend.log"
  warn "It may still be starting (LLM warmup) — wait 10s then try again"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — React Frontend
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}>>> [4/4] React frontend (port $FRONTEND_PORT)…${RESET}"

fuser -k "${FRONTEND_PORT}/tcp" 2>/dev/null || true
sleep 1

cd "$ROOT/chat-ui"
npm run dev >> "$ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$ROOT"

# Wait for Vite to print its "ready" line
for i in $(seq 1 15); do
  grep -q "Local:" "$ROOT/logs/frontend.log" 2>/dev/null && break
  sleep 1
done
ok "Frontend ready  (PID $FRONTEND_PID)   Log → logs/frontend.log"

# ══════════════════════════════════════════════════════════════════════════════
# ALL SERVICES UP
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║  ✅  CRM AI Assistant is UP and ready               ║${RESET}"
echo -e "${BOLD}${GREEN}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}${GREEN}║                                                      ║${RESET}"
echo -e "${BOLD}${GREEN}║  Open browser:  http://localhost:${FRONTEND_PORT}            ║${RESET}"
echo -e "${BOLD}${GREEN}║  API health:    http://localhost:${BACKEND_PORT}/health        ║${RESET}"
echo -e "${BOLD}${GREEN}║                                                      ║${RESET}"
echo -e "${BOLD}${GREEN}║  LLM chain:                                          ║${RESET}"
echo -e "${BOLD}${GREEN}║    SQL + Synthesis: Groq 70b (${GROQ_COUNT} accounts)           ║${RESET}"
echo -e "${BOLD}${GREEN}║    Fallback: Gemini → OpenRouter → Ollama llama8b    ║${RESET}"
echo -e "${BOLD}${GREEN}║                                                      ║${RESET}"
echo -e "${BOLD}${GREEN}║  Logs (logs/):                                       ║${RESET}"
echo -e "${BOLD}${GREEN}║    automation_log.txt  MongoDB→PG sync               ║${RESET}"
echo -e "${BOLD}${GREEN}║    backend.log         FastAPI + AI pipeline         ║${RESET}"
echo -e "${BOLD}${GREEN}║    frontend.log        Vite dev server               ║${RESET}"
echo -e "${BOLD}${GREEN}║    query.log           Query history                 ║${RESET}"
echo -e "${BOLD}${GREEN}║                                                      ║${RESET}"
echo -e "${BOLD}${GREEN}║  Press Ctrl+C to stop everything cleanly             ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# Keep script alive — trap above will clean up on Ctrl+C
wait
