"""llm_router.py — Centralized model router for all LLM tasks.

Task → model routing:
  sql        Groq Scout-17b   → Gemini flash-lite → OpenRouter DeepSeek → Ollama
  classify   Groq 8b-instant  → Gemini flash-lite → Ollama 3b
  decompose  Groq Scout-17b   → Gemini flash-lite → OpenRouter DeepSeek → Ollama 8b
  synthesize Groq 70b         → Gemini flash-lite → OpenRouter DeepSeek → Ollama 8b
  narrate    Groq 8b-instant  → Gemini flash-lite → Ollama 3b

Key features:
  - Per-key exponential backoff (respects retry-after header, caps at 60s)
  - Groq key rotation (3 keys, skip blocked keys, no sleep between keys)
  - SQL result cache (SHA-256 key, 5-min TTL, 200-entry LRU)
  - Token estimation (~1.3 tok/word) for budget logging
  - temperature=0 + max_tokens=250 for deterministic SQL
  - Thread-safe state for concurrent sub-query execution

Public API:
  call(task, system, user, max_tokens=None) -> Optional[str]
  call_sql(query, compact_schema, system_prompt, table_names) -> Optional[str]
  sql_cache_get(query) -> Optional[str]
  sql_cache_set(query, sql)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from config import settings

LOGGER = logging.getLogger("sql_chatbot")


# ══════════════════════════════════════════════════════════════════════════════
# TASK CONFIG  (model chain per task)
# ══════════════════════════════════════════════════════════════════════════════

def _build_chains() -> Dict:
    """Build task→chain config. Called once at import time."""
    groq_sql_model   = getattr(settings, "groq_sql_model",        "meta-llama/llama-4-scout-17b-16e-instruct")
    groq_cls_model   = getattr(settings, "groq_classify_model",   "llama-3.1-8b-instant")
    groq_dec_model   = getattr(settings, "groq_decompose_model",  "meta-llama/llama-4-scout-17b-16e-instruct")
    groq_syn_model   = getattr(settings, "groq_synthesis_model",  "llama-3.3-70b-versatile")
    gem_model        = getattr(settings, "gemini_classify_model",  "gemini-2.0-flash-lite")
    cbr_sql          = getattr(settings, "cerebras_sql_model",     "llama-3.3-70b")
    cbr_syn          = getattr(settings, "cerebras_synthesis_model","llama-3.3-70b")
    cbr_cls          = getattr(settings, "cerebras_classify_model", "llama3.1-8b")
    snv_sql          = getattr(settings, "sambanova_sql_model",    "Meta-Llama-3.3-70B-Instruct")
    snv_syn          = getattr(settings, "sambanova_synthesis_model","Meta-Llama-3.3-70B-Instruct")
    snv_cls          = getattr(settings, "sambanova_classify_model","Meta-Llama-3.1-8B-Instruct")
    oai_sql          = getattr(settings, "openai_sql_model",        "gpt-4o-mini")
    oai_cls          = getattr(settings, "openai_classify_model",   "gpt-4o-mini")
    oai_dec          = getattr(settings, "openai_decompose_model",  "gpt-4o-mini")
    oai_syn          = getattr(settings, "openai_synthesis_model",  "gpt-4o")
    oai_nar          = getattr(settings, "openai_narrate_model",    "gpt-4o-mini")
    or_sql_model     = getattr(settings, "openrouter_sql_model",    "nvidia/nemotron-3-super-120b-a12b:free")
    or_syn_model     = getattr(settings, "openrouter_synthesis_model", "nvidia/nemotron-3-super-120b-a12b:free")
    oll_cls          = getattr(settings, "ollama_classify_model",  "qwen2.5:3b")
    oll_rsn          = getattr(settings, "ollama_reasoning_model", "llama3.1:8b")
    oll_sql          = getattr(settings, "ollama_fallback_model",  "a-kore/Arctic-Text2SQL-R1-7B:latest")
    oll_pri          = getattr(settings, "ollama_primary_model",   "debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned:latest")

    return {
        "sql": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens":  1200,
            "chain": (
                [
                    ("ollama_pri", oll_pri),      # debopam Qwen2.5-Coder 3B — SQL expert, first
                    ("ollama",     oll_sql),      # Arctic-Text2SQL-R1 7B   — SQL expert, second
                ] if getattr(settings, "ollama_sql_enabled", True) else []
            ) + [
                ("openai",     oai_sql),
                ("groq",       groq_sql_model),
                ("gemini",     gem_model),
                ("cerebras",   cbr_sql),
                ("sambanova",  snv_sql),
                ("openrouter", or_sql_model),
            ],
        },
        "classify": {
            "temperature": 0,
            "top_p": 0.9,
            "max_tokens":  80,
            "chain": [
                ("openai",    oai_cls),
                ("groq",      groq_cls_model),
                ("gemini",    gem_model),
                ("cerebras",  cbr_cls),
                ("sambanova", snv_cls),
                ("ollama",    oll_cls),
            ],
        },
        "decompose": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens":  1200,
            "chain": [
                ("openai",    oai_dec),
                ("groq",      groq_dec_model),
                ("gemini",    gem_model),
                ("cerebras",  cbr_sql),
                ("sambanova", snv_sql),
                ("openrouter",or_syn_model),
                ("ollama",    oll_rsn),
            ],
        },
        "synthesize": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens":  3000,
            "chain": [
                ("openai",    oai_syn),
                ("groq",      groq_syn_model),
                ("gemini",    gem_model),
                ("cerebras",  cbr_syn),
                ("sambanova", snv_syn),
                ("openrouter",or_syn_model),
                ("ollama",    oll_rsn),
            ],
        },
        "narrate": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens":  200,
            "chain": [
                ("openai",    oai_nar),
                ("groq",      groq_cls_model),
                ("gemini",    gem_model),
                ("cerebras",  cbr_cls),
                ("sambanova", snv_cls),
                ("ollama",    oll_cls),
            ],
        },
    }


_TASK_CONFIG: Dict = {}
_config_lock = threading.Lock()


def _get_task_config(task: str) -> Optional[Dict]:
    global _TASK_CONFIG
    if not _TASK_CONFIG:
        with _config_lock:
            if not _TASK_CONFIG:
                _TASK_CONFIG = _build_chains()
    return _TASK_CONFIG.get(task)


# ══════════════════════════════════════════════════════════════════════════════
# PER-KEY BACKOFF STATE  (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

# key_id → (blocked_until: float, next_backoff_s: float)
_KEY_STATE: Dict[str, Tuple[float, float]] = {}
_key_lock  = threading.Lock()


def _key_available(key_id: str) -> bool:
    with _key_lock:
        entry = _KEY_STATE.get(key_id)
        return (not entry) or (time.monotonic() >= entry[0])


def _mark_key_blocked(key_id: str, retry_after: float) -> None:
    """Block a key for retry_after seconds (or exponential back-off if 0)."""
    with _key_lock:
        prev_backoff = _KEY_STATE.get(key_id, (0, 1.0))[1]
        if retry_after > 0:
            wait = min(retry_after, 60.0)    # cap at 60s so test doesn't hang
        else:
            wait = min(prev_backoff * 2, 60.0)
        _KEY_STATE[key_id] = (time.monotonic() + wait, wait)
    LOGGER.warning("Key ...%s blocked %.1fs", key_id[-6:], wait)


def _clear_key_state(key_id: str) -> None:
    with _key_lock:
        _KEY_STATE.pop(key_id, None)


def _parse_retry_after(exc: urllib.error.HTTPError) -> float:
    """Read retry-after header or body message from a 429 response."""
    try:
        ra = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
        if ra:
            return max(1.0, float(ra))
        body = exc.read().decode(errors="ignore")
        m = re.search(r"try again in ([\d.]+)s", body)
        if m:
            return max(1.0, float(m.group(1)))
    except Exception:
        pass
    return 0.0  # 0 → caller uses exponential back-off


# ══════════════════════════════════════════════════════════════════════════════
# SQL RESULT CACHE  (thread-safe LRU, 5-min TTL)
# ══════════════════════════════════════════════════════════════════════════════

_SQL_CACHE: Dict[str, Tuple[str, float]] = {}   # cache_key → (sql, ts)
_SQL_CACHE_TTL = 300      # seconds
_SQL_CACHE_MAX = 200
_cache_lock    = threading.Lock()


def _cache_key(query: str) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


def sql_cache_get(query: str) -> Optional[str]:
    k = _cache_key(query)
    with _cache_lock:
        entry = _SQL_CACHE.get(k)
        if entry and (time.monotonic() - entry[1]) < _SQL_CACHE_TTL:
            LOGGER.info("SQL cache HIT: %.60s", query)
            return entry[0]
    return None


def sql_cache_set(query: str, sql: str) -> None:
    k = _cache_key(query)
    with _cache_lock:
        if len(_SQL_CACHE) >= _SQL_CACHE_MAX:
            oldest = min(_SQL_CACHE, key=lambda x: _SQL_CACHE[x][1])
            del _SQL_CACHE[oldest]
        _SQL_CACHE[k] = (sql, time.monotonic())


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Approximate token count: ~1.3 tokens per whitespace-separated word."""
    return max(1, int(len(text.split()) * 1.3))


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CALLERS
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_BASE      = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"
_OR_BASE        = "https://openrouter.ai/api/v1/chat/completions"
_CEREBRAS_BASE  = "https://api.cerebras.ai/v1/chat/completions"
_SAMBANOVA_BASE = "https://api.sambanova.ai/v1/chat/completions"


def _call_groq(
    key:         str,
    model:       str,
    system:      str,
    user:        str,
    max_tokens:  int,
    temperature: float,
) -> Optional[str]:
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        req = urllib.request.Request(
            _GROQ_BASE,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":  "application/json",
                "User-Agent":    "groq-python/0.9.0",
                "Accept":        "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                _clear_key_state(key)
            return text or None
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            _mark_key_blocked(key, _parse_retry_after(e))
        elif e.code == 403:
            # Cloudflare block — treat as long-ish backoff but don't hang
            _mark_key_blocked(key, 60.0)
            LOGGER.warning("Groq key ...%s Cloudflare 403 — blocking 60s", key[-6:])
        else:
            LOGGER.warning("Groq HTTP %d key ...%s model %s", e.code, key[-6:], model)
        return None
    except Exception as exc:
        LOGGER.debug("Groq call error: %s", exc)
        return None


def _call_gemini(
    model:       str,
    system:      str,
    user:        str,
    max_tokens:  int,
    temperature: float,
) -> Optional[str]:
    if not settings.gemini_api_key:
        return None
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    url = f"{_GEMINI_BASE}/{model}:generateContent"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "crm-ai/1.0",
                "x-goog-api-key": settings.gemini_api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip() or None
    except urllib.error.HTTPError as e:
        LOGGER.warning("Gemini HTTP %d model %s", e.code, model)
        return None
    except Exception as exc:
        LOGGER.debug("Gemini error: %s", exc)
        return None


def _call_openrouter(
    model:       str,
    system:      str,
    user:        str,
    max_tokens:  int,
    temperature: float,
) -> Optional[str]:
    if not settings.openrouter_api_key:
        return None
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        req = urllib.request.Request(
            _OR_BASE,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer":  "https://crm-assistant.local",
                "X-Title":       "CRM AI Assistant",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip() or None
    except urllib.error.HTTPError as e:
        LOGGER.warning("OpenRouter HTTP %d model %s", e.code, model)
        return None
    except Exception as exc:
        LOGGER.debug("OpenRouter error: %s", exc)
        return None


def _call_ollama(
    model:       str,
    system:      str,
    user:        str,
    max_tokens:  int,
    temperature: float,
    timeout:     int = 60,
) -> Optional[str]:
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip() or None
    except Exception as exc:
        LOGGER.warning("Ollama error model %s: %s", model, exc)
        return None


def _call_openai_compat(
    base_url:    str,
    api_key:     str,
    provider:    str,
    model:       str,
    system:      str,
    user:        str,
    max_tokens:  int,
    temperature: float,
) -> Optional[str]:
    """Generic caller for any OpenAI-compatible endpoint (Cerebras, SambaNova, etc.)."""
    if not api_key:
        return None
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip() or None
    except urllib.error.HTTPError as e:
        LOGGER.warning("%s HTTP %d model %s", provider, e.code, model)
        return None
    except Exception as exc:
        LOGGER.debug("%s error: %s", provider, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def call(
    task:       str,
    system:     str,
    user:       str,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Route a task to the best available provider.

    Tries each provider in the task's chain in order.
    Groq: all available keys are tried before moving to next provider.
    Returns the first successful response, or None if all fail.
    """
    config = _get_task_config(task)
    if not config:
        LOGGER.error("llm_router: unknown task '%s'", task)
        return None

    temperature = config["temperature"]
    tok         = max_tokens or config["max_tokens"]
    chain       = config["chain"]
    groq_keys   = settings.groq_api_keys

    t0 = time.monotonic()

    for provider, model in chain:
        if provider == "groq":
            available = [k for k in groq_keys if _key_available(k)]
            if not available:
                LOGGER.debug("All Groq keys blocked — skipping Groq for task=%s", task)
                continue
            for key in available:
                result = _call_groq(key, model, system, user, tok, temperature)
                if result:
                    LOGGER.info(
                        "Router [%s/groq/%s] %.0fms %dtok",
                        task, model[:30], (time.monotonic()-t0)*1000,
                        estimate_tokens(system + user),
                    )
                    return result

        elif provider == "gemini":
            result = _call_gemini(model, system, user, tok, temperature)
            if result:
                LOGGER.info("Router [%s/gemini] %.0fms", task, (time.monotonic()-t0)*1000)
                return result

        elif provider == "openai":
            if not getattr(settings, "openai_enabled", True):
                LOGGER.debug("Router [%s/openai] skipped — OPENAI_ENABLED=false", task)
                continue
            result = _call_openai_compat(
                "https://api.openai.com/v1/chat/completions",
                settings.openai_api_key, "OpenAI",
                model, system, user, tok, temperature,
            )
            if result:
                LOGGER.info("Router [%s/openai/%s] %.0fms", task, model, (time.monotonic()-t0)*1000)
                return result

        elif provider == "cerebras":
            result = _call_openai_compat(
                _CEREBRAS_BASE, settings.cerebras_api_key, "Cerebras",
                model, system, user, tok, temperature,
            )
            if result:
                LOGGER.info("Router [%s/cerebras/%s] %.0fms", task, model, (time.monotonic()-t0)*1000)
                return result

        elif provider == "sambanova":
            result = _call_openai_compat(
                _SAMBANOVA_BASE, settings.sambanova_api_key, "SambaNova",
                model, system, user, tok, temperature,
            )
            if result:
                LOGGER.info("Router [%s/sambanova/%s] %.0fms", task, model, (time.monotonic()-t0)*1000)
                return result

        elif provider == "openrouter":
            result = _call_openrouter(model, system, user, tok, temperature)
            if result:
                LOGGER.info("Router [%s/openrouter] %.0fms", task, (time.monotonic()-t0)*1000)
                return result

        elif provider in ("ollama", "ollama_pri"):
            timeout = 180 if task in ("synthesize", "decompose") else 45
            result  = _call_ollama(model, system, user, tok, temperature, timeout)
            if result:
                LOGGER.info("Router [%s/%s/%s] %.0fms", task, provider, model[:30], (time.monotonic()-t0)*1000)
                return result

    LOGGER.error(
        "Router: ALL providers failed for task=%s after %.0fms",
        task, (time.monotonic()-t0)*1000,
    )
    return None
