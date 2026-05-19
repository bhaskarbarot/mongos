# ── Python FastAPI Backend ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps: gcc + libpq for psycopg2, curl for healthcheck
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (everything not excluded by .dockerignore)
COPY . .

# Ensure logs directory exists
RUN mkdir -p logs

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
