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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

def _get_agent():
    """Return the shared agent, initialising it lazily on first call."""
    global _agent, _agent_error
    if _agent is not None:
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
        LOGGER.error("Pipeline init failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable — start the PostgreSQL service first. ({exc})",
        )

def _get_table_names_safe() -> list:
    try:
        from pipeline.schema import get_table_names
        return get_table_names(_get_agent())
    except Exception:
        return []

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CRM AI Assistant API",
    description="Backend for the chat-ui React frontend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Vite dev server on any port
    allow_methods=["*"],
    allow_headers=["*"],
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
    layer       = result.get("layer", "fast_path")
    answer      = result.get("answer", "")

    # Detect output format from answer content
    if "|" in answer and "---" in answer:
        output_fmt = "table"
    elif answer.strip().startswith("#"):
        output_fmt = "report"
    else:
        output_fmt = "text"

    plan: dict = {
        "intent":        "COMPLEX" if sub_results else "SIMPLE",
        "collection":    ", ".join(tables) if tables else "—",
        "output_format": output_fmt,
    }

    # Add active filters if detectable
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = _agent is not None
    return {
        "status":  "ok",
        "service": "CRM AI Assistant",
        "version": "1.0.0",
        "db_ready": db_ok,
        "db_error": _agent_error,
    }


@app.get("/cache/status")
async def cache_status():
    # Cache is disabled — report honestly
    return {"redis_connected": False, "cache_disabled": True, "status": "disabled"}


@app.get("/sources")
async def sources():
    """Return available DB tables so the sidebar can show data sources."""
    tables = _get_table_names_safe()
    return {
        "PostgreSQL (CRM)": {
            "type":  "database",
            "items": sorted(tables),
        }
    } if tables else {}


@app.get("/feedback/learnings")
async def feedback_learnings():
    """Return any saved feedback rules. Currently a stub."""
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
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Accepts {query, history} → returns full response matching UI contract.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Per-request ConversationMemory — history is passed each time so we
    # only need the last assistant message to seed the entity reference.
    mem = ConversationMemory()
    for msg in req.history[-6:]:
        if msg.role == "assistant":
            # Try to set last_entity from previous response content
            pass  # ConversationMemory resolves pronouns; full rebuild not needed

    LOGGER.info("POST /chat | query: %.80s", req.query)
    t0 = time.perf_counter()

    try:
        result = run_agent_query(_get_agent(), req.query, memory=mem)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

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

    return {
        "answer":              clean,
        "data":                _extract_structured_data(result),
        "confidence":          confidence,
        "sources_used":        tables,
        "processing_time_ms":  round(latency_ms),
        "agent_time_ms":       round(latency_ms),
        "query_used":          query_used,
        "query_plan":          _build_query_plan(result),
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Store feedback to a local JSON file."""
    try:
        feedback_file = Path("logs/feedback_log.json")
        existing: list = []
        if feedback_file.exists():
            try:
                existing = json.loads(feedback_file.read_text())
            except Exception:
                existing = []
        existing.append({
            "ts":          time.time(),
            "query":       req.query,
            "rating":      req.rating,
            "comment":     req.comment or "",
            "sources":     req.sources_used or [],
            "query_plan":  req.query_plan,
        })
        feedback_file.write_text(json.dumps(existing, indent=2))
        LOGGER.info("Feedback saved: rating=%d | query=%.60s", req.rating, req.query)
    except Exception as exc:
        LOGGER.warning("Feedback save error: %s", exc)

    return {"status": "ok", "message": "Feedback received — thank you!"}


@app.post("/cache/clear")
async def cache_clear():
    """Cache is disabled in this deployment."""
    return {
        "status": "ok",
        "cleared": False,
        "cache_disabled": True,
        "message": "Cache is disabled",
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    """Audio transcription — not implemented in this deployment."""
    raise HTTPException(
        status_code=501,
        detail="Voice transcription is not available in this deployment. Use text input.",
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
