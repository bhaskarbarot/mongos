"""llm.py — Centralized multi-provider LLM client with automatic fallback.

Provider chain (fastest → most available):

  classify  → Groq 8b-instant   → Gemini flash-lite → Ollama classify-model
  decompose → Groq llama-4-scout → Gemini flash-lite → OpenRouter → Ollama 8b
  synthesize→ Groq 70b-versatile → Gemini flash-lite → OpenRouter → Ollama 8b

Groq key rotation:
  3 API keys are tried in sequence on each call.
  If key-1 gets a 429, key-2 is tried immediately (no sleep), then key-3.
  This gives 3× the effective rate-limit budget at zero extra latency for
  the common case where key-1 succeeds.

Public API:
    call(task, system, user, max_tokens) -> Optional[str]

Tasks: "classify" | "decompose" | "synthesize"
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from typing import List, Optional

from config import settings

LOGGER = logging.getLogger("sql_chatbot")

# ── Groq headers (User-Agent required to bypass Cloudflare 403) ───────────────
_GROQ_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent":   "groq-python/0.9.0",
    "Accept":       "application/json",
}

# ── OpenRouter headers ────────────────────────────────────────────────────────
_OR_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://crm-assistant.local",
    "X-Title":      "CRM AI Assistant",
}


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _groq_with_key(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    timeout: int = 20,
) -> Optional[str]:
    """Single Groq call with one API key. Returns None on any failure (incl. 429)."""
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
        "max_tokens":  max_tokens,
    }
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={**_GROQ_HEADERS, "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            LOGGER.debug("Groq [%s] OK — %d chars", model, len(text))
            return text
    except urllib.error.HTTPError as e:
        if e.code == 429:
            LOGGER.warning("Groq [%s] 429 on key ...%s — trying next key", model, api_key[-6:])
            return None  # caller rotates to next key
        body = e.read().decode()[:200]
        LOGGER.warning("Groq [%s] HTTP %d: %s", model, e.code, body)
        return None
    except Exception as exc:
        LOGGER.warning("Groq [%s] error: %s", model, exc)
        return None


def _groq(model: str, system: str, user: str, max_tokens: int, timeout: int = 20) -> Optional[str]:
    """Try all configured Groq API keys in order. Returns first successful result."""
    keys: List[str] = settings.groq_api_keys
    if not keys:
        return None

    for i, key in enumerate(keys):
        result = _groq_with_key(key, model, system, user, max_tokens, timeout)
        if result is not None:
            if i > 0:
                LOGGER.info("Groq [%s] succeeded on key %d/%d", model, i + 1, len(keys))
            return result

    LOGGER.warning("Groq [%s] all %d keys failed/rate-limited", model, len(keys))
    return None


def _gemini(model: str, system: str, user: str, max_tokens: int, timeout: int = 25) -> Optional[str]:
    """Call Gemini API (Google AI Studio). Returns text or None on any failure."""
    if not settings.gemini_api_key:
        return None
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": max_tokens,
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={settings.gemini_api_key}"
    )
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "crm-ai/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            LOGGER.debug("Gemini [%s] OK — %d chars", model, len(text))
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        LOGGER.warning("Gemini [%s] HTTP %d: %s", model, e.code, body)
        return None
    except Exception as exc:
        LOGGER.warning("Gemini [%s] error: %s", model, exc)
        return None


def _openrouter(model: str, system: str, user: str, max_tokens: int, timeout: int = 25) -> Optional[str]:
    """Call OpenRouter API (last-resort cloud fallback). Returns text or None."""
    if not settings.openrouter_api_key:
        return None
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
        "max_tokens":  max_tokens,
    }
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={**_OR_HEADERS, "Authorization": f"Bearer {settings.openrouter_api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            LOGGER.debug("OpenRouter [%s] OK — %d chars", model, len(text))
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        LOGGER.warning("OpenRouter [%s] HTTP %d: %s", model, e.code, body)
        return None
    except Exception as exc:
        LOGGER.warning("OpenRouter [%s] error: %s", model, exc)
        return None


def _ollama(model: str, system: str, user: str, max_tokens: int, timeout: int = 60) -> Optional[str]:
    """Call local Ollama (always available — final fallback). Returns text or None."""
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
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
            text = data.get("message", {}).get("content", "").strip()
            LOGGER.debug("Ollama [%s] OK — %d chars", model, len(text))
            return text
    except Exception as exc:
        LOGGER.warning("Ollama [%s] error: %s", model, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TASK-AWARE FALLBACK CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def call(
    task:       str,
    system:     str,
    user:       str,
    max_tokens: int = 1500,
) -> Optional[str]:
    """
    Call the best available LLM for the given task with automatic fallback.

    Args:
        task:       "classify" | "decompose" | "synthesize"
        system:     System prompt (role + rules)
        user:       User prompt (query / data to process)
        max_tokens: Max output tokens

    Provider chains:
        classify  → Groq 8b-instant    → Gemini flash-lite → Ollama classify-model
        decompose → Groq llama-4-scout  → Gemini flash-lite → OpenRouter → Ollama 8b
        synthesize→ Groq 70b-versatile  → Gemini flash-lite → OpenRouter → Ollama 8b

    Groq rotation: all 3 API keys are tried before moving to Gemini.
    Returns response string, or None if ALL providers failed.
    """
    t0 = time.perf_counter()

    if task == "classify":
        steps = [
            ("Groq-8b",       lambda: _groq(settings.groq_classify_model,      system, user, max_tokens, timeout=15)),
            ("Gemini",        lambda: _gemini(settings.gemini_classify_model,   system, user, max_tokens, timeout=20)),
            ("Ollama-classify",lambda: _ollama(settings.ollama_classify_model,  system, user, max_tokens, timeout=30)),
        ]

    elif task == "decompose":
        steps = [
            ("Groq-scout",    lambda: _groq(settings.groq_decompose_model,          system, user, max_tokens, timeout=20)),
            ("Gemini",        lambda: _gemini(settings.gemini_decompose_model,       system, user, max_tokens, timeout=25)),
            ("OpenRouter",    lambda: _openrouter(settings.openrouter_decompose_model, system, user, max_tokens, timeout=25)),
            ("Ollama-8b",     lambda: _ollama(settings.ollama_reasoning_model,       system, user, max_tokens, timeout=settings.ollama_reasoning_timeout)),
        ]

    elif task == "synthesize":
        steps = [
            ("Groq-70b",      lambda: _groq(settings.groq_synthesis_model,           system, user, max_tokens, timeout=30)),
            ("Gemini",        lambda: _gemini(settings.gemini_synthesis_model,        system, user, max_tokens, timeout=30)),
            ("OpenRouter",    lambda: _openrouter(settings.openrouter_synthesis_model, system, user, max_tokens, timeout=30)),
            ("Ollama-8b",     lambda: _ollama(settings.ollama_reasoning_model,        system, user, max_tokens, timeout=settings.ollama_reasoning_timeout)),
        ]

    else:
        LOGGER.error("llm.call: unknown task=%s", task)
        return None

    for provider_name, fn in steps:
        try:
            result = fn()
            if result:
                elapsed = round((time.perf_counter() - t0) * 1000)
                LOGGER.info("LLM [%s/%s] ✓ %dms", task, provider_name, elapsed)
                return result
        except Exception as exc:
            LOGGER.warning("LLM [%s/%s] exception: %s", task, provider_name, exc)

    elapsed = round((time.perf_counter() - t0) * 1000)
    LOGGER.error("LLM [%s] ALL providers failed after %dms", task, elapsed)
    return None
