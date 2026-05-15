#!/usr/bin/env bash
# start_mcp.sh — Launch the CRM MCP server
# Usage: ./mcp_server/start_mcp.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

echo "Starting CRM MCP server from $PROJECT_ROOT..."
exec python "$SCRIPT_DIR/crm_mcp.py"
