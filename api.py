"""
api.py — FastAPI server bridging the React chat-ui to the CRM pipeline.

Endpoints:
  GET  /health              → liveness check
  GET  /cache/status        → Redis/cache status
  GET  /sources             → available DB tables
  GET  /feedback/learnings  → AI learning rules (stub)
  POST /chat                → main query handler
  POST /feedback            → submit rating (stub stored in file)
  POST /cache/clear         → clear in-memory cache
  POST /transcribe          → audio → text (not implemented)

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import asyncio
import fcntl
import functools

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
for noisy in ["transformers", "sentence_transformers", "httpx",
              "urllib3", "sqlalchemy.engine", "langchain"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

LOGGER = logging.getLogger("crm_api")

# ── Lazy pipeline init (initialised on first request, not at import time) ────
from agent import run_agent_query, ConversationMemory

_agent       = None
_agent_error = None   # stores last init error message
_agent_lock  = threading.Lock()

def _get_agent():
    """Return the shared agent, initialising it lazily on first call (thread-safe)."""
    global _agent, _agent_error
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is not None:  # double-checked locking
            return _agent
        try:
            LOGGER.info("Initialising CRM pipeline (DB + schema discovery)…")
            from db import get_database
            from agent import get_sql_agent
            db     = get_database()
            _agent = get_sql_agent(db)
            _agent_error = None
            LOGGER.info("Pipeline ready ✓")
            return _agent
        except Exception as exc:
            _agent_error = str(exc)
            LOGGER.error("Pipeline init failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Database unavailable — start the PostgreSQL service first.",
            )

def _get_table_names_safe() -> list:
    try:
        from pipeline.schema import get_table_names
        return get_table_names(_get_agent())
    except Exception:
        return []


# ── E2: Safe error response helper ───────────────────────────────────────────

def safe_error_response(exc: Exception, request_id: str = "") -> dict:
    """Return a sanitized error dict — never exposes raw exception text."""
    LOGGER.error("Internal error [RID:%s]: %s", request_id, exc, exc_info=True)
    return {"error": "An internal error occurred", "request_id": request_id, "success": False}


# ── E4: In-memory rate limiter (thread-safe, TTL=60s) ────────────────────────

_rate_store: Dict[str, List[float]] = {}
_rate_lock  = threading.Lock()

def _check_rate(request: Request, limit: int, request_id: str = "") -> None:
    """Raise HTTP 429 if the caller has exceeded `limit` requests in the last 60s.

    Stale entries (IPs with no activity in the last 60s) are deleted on access
    to prevent unbounded memory growth.
    """
    now = time.time()
    ip  = request.client.host if request.client else "unknown"
    key = f"{ip}:{request.url.path}"
    with _rate_lock:
        window = [t for t in _rate_store.get(key, []) if now - t < 60]
        if not window:
            # All timestamps expired — remove stale entry (memory leak prevention)
            _rate_store.pop(key, None)
        if len(window) >= limit:
            _rate_store[key] = window
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded, please slow down",
                    "request_id": request_id,
                    "success": False,
                },
            )
        window.append(now)
        _rate_store[key] = window


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CRM AI Assistant API",
    description="Backend for the chat-ui React frontend",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    """Run at startup: create indexes and pre-warm agent in background thread."""
    import concurrent.futures
    try:
        from pipeline.db_indexes import create_indexes
        create_indexes()
    except Exception as exc:
        LOGGER.warning("Index creation skipped: %s", exc)
    # Pre-warm agent so first user request doesn't pay cold-start cost
    concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_get_agent)


# E3: Hardened CORS — restrict origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int
    comment: Optional[str] = None
    query_plan: Optional[Dict[str, Any]] = None
    sources_used: Optional[List[str]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_answer(raw: str) -> str:
    """
    Strip pipeline wrapper text so the UI only gets clean markdown.
    Removes: 'Executive Summary:' prefix, 'Result is generated from …' footer,
             'The response is based on …' footer.
    """
    text = raw.strip()

    # Remove "Executive Summary:" header (keep the content after it)
    if text.lower().startswith("executive summary:"):
        text = text[len("executive summary:"):].strip()

    # Remove footer lines added by format_final_answer()
    for marker in [
        "\n\nResult is generated from",
        "\nResult is generated from",
        "Result is generated from",
        "\n\nThe response is based on",
        "\nThe response is based on",
        "The response is based on",
    ]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    return text


def _build_query_plan(result: dict) -> dict:
    """Map pipeline metadata to the UI's query_plan schema."""
    tables      = result.get("tables_used", [])
    sub_results = result.get("_sub_results", [])
    layer       = result.get("layer", "")
    agent_type  = result.get("agent_type", "")
    answer      = result.get("answer", "")

    # Detect output format from answer content
    if "|" in answer and "---" in answer:
        output_fmt = "table"
    elif answer.strip().startswith("#"):
        output_fmt = "report"
    else:
        output_fmt = "text"

    # Determine which agent/layer handled the query
    if "fast_path" in layer or "fast_path" in agent_type:
        intent = "FAST_PATH"
    elif "complex" in agent_type or "complex" in layer:
        intent = "COMPLEX"
    elif "medium" in agent_type or "medium" in layer:
        intent = "MEDIUM"
    elif "simple" in agent_type or "simple" in layer:
        intent = "SIMPLE"
    elif "guard" in layer:
        intent = "GUARD"
    else:
        intent = "COMPLEX" if sub_results else "SIMPLE"

    plan: dict = {
        "intent":        intent,
        "collection":    ", ".join(tables) if tables else "—",
        "output_format": output_fmt,
    }

    filters: dict = {}
    if sub_results:
        plan["sub_queries"] = [s.get("sub_query", "") for s in sub_results[:6]]
    plan["filters"] = filters

    return plan


def _extract_structured_data(result: dict) -> Optional[Any]:
    """
    Try to extract structured data for the UI's DataTable / MetricCard renderer.
    Returns None when the markdown answer is sufficient (tables already in text).
    """
    answer      = result.get("answer", "")
    sub_results = result.get("_sub_results")

    # If answer already has markdown tables, the UI renders them fine — skip structured data
    if "|" in answer and "---" in answer:
        return None

    # For COMPLEX multi-part results with no tables, expose as metrics
    if sub_results and len(sub_results) > 1:
        metrics: dict = {}
        for item in sub_results:
            sq   = item.get("sub_query", "")
            ans  = (item.get("data") or {}).get("answer", "")
            # Extract bold numbers from sub-answers
            nums = re.findall(r"\*\*([0-9,\.]+)\*\*", ans)
            if nums and sq:
                label = sq[:40].rstrip("?.")
                metrics[label] = nums[0]
        if metrics:
            return {"metrics": metrics}

    return None


# ── Memory rebuild helper ────────────────────────────────────────────────────

# Entity keyword → table name (for inferring entity from answer text)
_ANSWER_ENTITY_RE = [
    (re.compile(r"\binvoices?\b",              re.I), "invoices"),
    (re.compile(r"\bbills?\b",                 re.I), "bills"),
    (re.compile(r"\bdeals?\b",                 re.I), "deals"),
    (re.compile(r"\bcontacts?\b",              re.I), "contacts"),
    (re.compile(r"\bcompan(?:y|ies)\b",        re.I), "companies"),
    (re.compile(r"\btasks?\b",                 re.I), "createtasks"),
    (re.compile(r"\busers?\b",                 re.I), "users"),
    (re.compile(r"\btargets?\b",               re.I), "targets"),
    (re.compile(r"\bsales?\b",                 re.I), "sales"),
    (re.compile(r"\bdepartments?\b",           re.I), "departments"),
]


def _infer_entity(text: str) -> Optional[str]:
    """Return the most-mentioned entity name from an answer string."""
    counts: Dict[str, int] = {}
    for pat, entity in _ANSWER_ENTITY_RE:
        n = len(pat.findall(text))
        if n:
            counts[entity] = n
    return max(counts, key=counts.get) if counts else None


def _rebuild_memory(mem: ConversationMemory, history: List[ChatMessage]) -> None:
    """
    Replay conversation history into ChatMemory so pronoun resolution and
    episodic matching work across requests.

    The React frontend sends the full history on every POST /chat, so we
    reconstruct the memory state by iterating user/assistant pairs.
    We limit to the last 10 exchanges to keep it fast (no LLM call here).
    """
    # Pair up user → assistant messages
    pairs: List[tuple] = []
    i = 0
    msgs = list(history)
    while i < len(msgs):
        if msgs[i].role == "user":
            user_content = msgs[i].content
            asst_content = ""
            if i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                asst_content = msgs[i + 1].content
                i += 2
            else:
                i += 1
            pairs.append((user_content, asst_content))
        else:
            i += 1

    # Replay last 10 pairs into memory (oldest first so state is correct at end)
    for user_q, asst_a in pairs[-10:]:
        entity = _infer_entity(asst_a)
        # Build a synthetic result dict that mem.update() can consume
        synthetic = {
            "answer":      asst_a,
            "tables_used": [entity] if entity else [],
            "layer":       "history",
            "confidence":  0.85,
            "sql_queries": [],
        }
        mem.update(user_q, synthetic, success=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    # No rate limiting on /health
    db_ok = _agent is not None
    return {
        "status":  "ok",
        "service": "CRM AI Assistant",
        "version": "1.0.0",
        "db_ready": db_ok,
        "db_error": _agent_error,
    }


@app.get("/cache/status")
async def cache_status(request: Request):
    _check_rate(request, 60, request_id=str(uuid.uuid4())[:8])
    # Cache is disabled — report honestly
    return {"redis_connected": False, "cache_disabled": True, "status": "disabled"}


@app.get("/sources")
async def sources(request: Request):
    """Return available DB tables so the sidebar can show data sources."""
    _check_rate(request, 60, request_id=str(uuid.uuid4())[:8])
    tables = _get_table_names_safe()
    # UI iterates: Object.entries(sources).map(([stype, items]) => items?.length)
    # so the value must be a plain array, not an object.
    return {"PostgreSQL (CRM)": sorted(tables)} if tables else {}


@app.get("/feedback/learnings")
async def feedback_learnings(request: Request):
    """Return any saved feedback rules. Currently a stub."""
    _check_rate(request, 60, request_id=str(uuid.uuid4())[:8])
    feedback_file = Path("logs/feedback_log.json")
    count = 0
    if feedback_file.exists():
        try:
            data  = json.loads(feedback_file.read_text())
            count = len(data)
        except Exception:
            pass
    return {
        "total_feedback": count,
        "rules": {
            "intent_rules":        [],
            "format_preferences":  [],
            "never_do":            [],
        },
    }


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Main chat endpoint.
    Accepts {query, history} → returns full response matching UI contract.
    """
    # E5: generate correlation ID first so 429 responses also carry it
    request_id = str(uuid.uuid4())[:8]
    # E4: rate limit
    _check_rate(request, 30, request_id=request_id)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Rebuild ChatMemory from conversation history so pronoun resolution,
    # episodic matching, and "give me that X" all work correctly.
    # Without this, every request gets a fresh memory and context is lost.
    mem = ConversationMemory()
    _rebuild_memory(mem, req.history)

    LOGGER.info("[RID:%s] POST /chat | query: %.80s", request_id, req.query)
    t0 = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                functools.partial(run_agent_query, _get_agent(), req.query, memory=mem, request_id=request_id),
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={"error": "Query timed out — please simplify your question", "request_id": request_id, "success": False})
    except HTTPException:
        raise
    except Exception as exc:
        err = safe_error_response(exc, request_id)
        raise HTTPException(status_code=500, detail=err)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    raw_answer  = result.get("answer", "I could not generate an answer for this query.")
    clean       = _clean_answer(raw_answer)
    tables      = result.get("tables_used", [])
    sql_queries = result.get("sql_queries", [])
    confidence  = round(result.get("confidence", 0.0), 3)
    latency_ms  = result.get("latency_ms", elapsed_ms)

    # Format SQL queries for display (show up to 3, joined)
    query_used = None
    if sql_queries:
        query_used = "\n\n".join(sql_queries[:3])
        if len(sql_queries) > 3:
            query_used += f"\n\n… and {len(sql_queries)-3} more queries"

    agent_type = result.get("agent_type") or result.get("layer", "unknown")

    return {
        "answer":              clean,
        "data":                _extract_structured_data(result),
        "confidence":          confidence,
        "sources_used":        tables,
        "processing_time_ms":  round(latency_ms),
        "agent_time_ms":       round(latency_ms),
        "query_used":          query_used,
        "query_plan":          _build_query_plan(result),
        "agent_type":          agent_type,
        "request_id":          request_id,
        "metrics":             result.get("metrics"),
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    """Store feedback to a local JSON file."""
    request_id = str(uuid.uuid4())[:8]
    _check_rate(request, 30, request_id=request_id)

    try:
        feedback_file = Path("logs/feedback_log.json")
        feedback_file.parent.mkdir(exist_ok=True)
        with open(str(feedback_file), "a+") as _fh:
            fcntl.flock(_fh, fcntl.LOCK_EX)
            try:
                _fh.seek(0)
                raw = _fh.read()
                existing: list = json.loads(raw) if raw.strip() else []
            except Exception:
                existing = []
            existing.append({
                "ts":         time.time(),
                "query":      req.query,
                "rating":     req.rating,
                "comment":    req.comment or "",
                "sources":    req.sources_used or [],
                "query_plan": req.query_plan,
            })
            _fh.seek(0)
            _fh.truncate()
            _fh.write(json.dumps(existing, indent=2))
            fcntl.flock(_fh, fcntl.LOCK_UN)
        LOGGER.info("[RID:%s] Feedback saved: rating=%d | query=%.60s",
                    request_id, req.rating, req.query)
    except Exception as exc:
        LOGGER.warning("[RID:%s] Feedback save error: %s", request_id, exc)

    return {"status": "ok", "message": "Feedback received — thank you!", "request_id": request_id}


@app.post("/cache/clear")
async def cache_clear(request: Request):
    """Cache is disabled in this deployment."""
    request_id = str(uuid.uuid4())[:8]
    _check_rate(request, 30, request_id=request_id)
    return {
        "status": "ok",
        "cleared": False,
        "cache_disabled": True,
        "message": "Cache is disabled",
        "request_id": request_id,
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    """Audio transcription — not implemented in this deployment."""
    request_id = str(uuid.uuid4())[:8]
    _check_rate(request, 30, request_id=request_id)
    raise HTTPException(
        status_code=501,
        detail={
            "error": "Voice transcription is not available in this deployment. Use text input.",
            "request_id": request_id,
            "success": False,
        },
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
