# ── MongoDB → PostgreSQL Sync (Node.js) ──────────────────────────────────────
# Build context is the project ROOT (not automation/) so sync.js can resolve
# ../logs/ correctly at runtime (maps to /app/logs/).
FROM node:20-alpine

WORKDIR /app

# Install only production dependencies
COPY automation/package.json automation/package-lock.json ./automation/
RUN cd automation && npm ci --only=production

# Copy automation source
COPY automation/ ./automation/

# Create logs dir — sync.js writes to path.join(__dirname, '..', 'logs')
# which resolves to /app/logs/ at runtime
RUN mkdir -p logs

CMD ["node", "automation/sync.js"]
