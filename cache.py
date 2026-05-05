"""
cache.py — Redis-backed semantic query cache.

Architecture:
  Exact match  : hash(normalised_query) → cached JSON response (instant, no embedding)
  Semantic hit : cosine_similarity(query_embedding, cached_embedding)
                 >= SIMILARITY_THRESHOLD → return cached response
  Cache miss   : run the pipeline, store result in Redis with TTL

IMPORTANT: Semantic embedding is done in a background thread so it never
blocks the main query path or causes Ollama model-swap overhead.
"""

import hashlib
import json
import logging
import math
import threading
import urllib.request
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("sql_chatbot")

# ── Config ────────────────────────────────────────────────────────────────────
import os
CACHE_TTL            = int(os.getenv("REDIS_CACHE_TTL", "3600"))
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.70"))
CACHE_DISABLED       = os.getenv("CACHE_DISABLED", "false").lower() == "true"
EMBED_MODEL          = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT           = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB             = int(os.getenv("REDIS_DB", "0"))
MAX_SEMANTIC_SCAN    = 200

# ── Redis client (lazy init) ──────────────────────────────────────────────────
_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=True, socket_timeout=2, socket_connect_timeout=2,
        )
        client.ping()
        _redis = client
        LOGGER.info("Redis cache connected at %s:%s", REDIS_HOST, REDIS_PORT)
        return _redis
    except Exception as exc:
        LOGGER.warning("Redis unavailable (cache disabled): %s", exc)
        return None


# ── Embeddings ────────────────────────────────────────────────────────────────

def _embed(text: str) -> Optional[list]:
    """Get embedding vector from Ollama nomic-embed-text. Returns None on failure."""
    payload = {"model": EMBED_MODEL, "prompt": text[:2000]}
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("embedding")
    except Exception as exc:
        LOGGER.debug("Embedding failed: %s", exc)
        return None


def _cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ── Cache keys ────────────────────────────────────────────────────────────────

def _exact_key(query: str) -> str:
    return "crmbot:exact:" + hashlib.md5(query.lower().strip().encode()).hexdigest()


def _sem_key(query: str) -> str:
    return "crmbot:sem:" + hashlib.md5(query.lower().strip().encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached(query: str, skip_semantic: bool = False) -> Optional[Dict[str, Any]]:
    if CACHE_DISABLED:
        return None
    """
    Returns cached response dict if found (exact or semantic match >= threshold).

    Key design: semantic embedding is SKIPPED when the semantic cache is empty,
    preventing Ollama model-swap overhead on first-run queries.

    skip_semantic=True: only check exact match (use for filter queries to avoid
    wrong cache hits where "give me all deals" matches "give me all closed won deals").
    """
    r = _get_redis()
    if not r:
        return None

    # 1. Exact match — O(1), no embedding, no model load
    raw = r.get(_exact_key(query))
    if raw:
        try:
            result = json.loads(raw)
            result["from_cache"] = True
            result["cache_similarity"] = 100.0
            LOGGER.info("Cache HIT exact: %s", query[:60])
            return result
        except Exception:
            pass

    if skip_semantic:
        return None  # caller asked to skip semantic (filter-specific queries)

    # 2. Semantic match — only when cached entries exist
    # Avoids forcing nomic-embed-text to load (which triggers Ollama model swap)
    sem_keys = r.keys("crmbot:sem:*")
    if not sem_keys:
        return None   # nothing to compare — skip embedding entirely

    q_emb = _embed(query)
    if not q_emb:
        return None

    best_sim = 0.0
    best_raw = None

    for key in list(sem_keys)[:MAX_SEMANTIC_SCAN]:
        try:
            stored = json.loads(r.get(key) or "{}")
            s_emb  = stored.get("embedding")
            if s_emb:
                sim = _cosine(q_emb, s_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_raw = stored.get("response")
        except Exception:
            continue

    if best_sim >= SIMILARITY_THRESHOLD and best_raw:
        try:
            result = json.loads(best_raw)
            result["from_cache"]       = True
            result["cache_similarity"] = round(best_sim * 100, 1)
            LOGGER.info("Cache HIT semantic %.0f%%: %s", best_sim * 100, query[:60])
            return result
        except Exception:
            pass

    return None


def set_cache(query: str, response: Dict[str, Any]) -> None:
    if CACHE_DISABLED:
        return
    """
    Store response in Redis.
    - Exact key: stored immediately (fast, no embedding).
    - Semantic key: stored in a background thread — never blocks the query path.
    """
    r = _get_redis()
    if not r:
        return

    safe = {k: v for k, v in response.items()
            if k not in ("from_cache", "cache_similarity")}

    # Always store exact match synchronously (instant, no model needed)
    try:
        r.setex(_exact_key(query), CACHE_TTL, json.dumps(safe))
    except Exception as exc:
        LOGGER.debug("Redis set (exact) failed: %s", exc)
        return

    # Store semantic entry in background — best-effort, non-blocking
    def _background_embed():
        try:
            emb = _embed(query)
            if emb:
                r.setex(_sem_key(query), CACHE_TTL, json.dumps({
                    "embedding": emb,
                    "response":  json.dumps(safe),
                }))
        except Exception:
            pass

    threading.Thread(target=_background_embed, daemon=True).start()


def clear_cache() -> int:
    """Remove all crmbot cache entries. Returns count deleted."""
    r = _get_redis()
    if not r:
        return 0
    keys = r.keys("crmbot:*")
    if keys:
        return r.delete(*keys)
    return 0


def cache_stats() -> Dict[str, int]:
    """Return cache statistics."""
    r = _get_redis()
    if not r:
        return {"status": "unavailable"}
    exact = len(r.keys("crmbot:exact:*"))
    sem   = len(r.keys("crmbot:sem:*"))
    return {"exact_entries": exact, "semantic_entries": sem, "total": exact + sem}
