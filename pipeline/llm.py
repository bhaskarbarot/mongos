"""llm.py — Thin wrapper that delegates to llm_router.py.

All pipeline modules (classifier, decomposer, synthesizer, etc.) import:
    from pipeline.llm import call as llm_call

This file keeps that interface intact while routing everything through
the centralized llm_router which handles backoff, key rotation, fallback,
and caching.

Public API (unchanged):
    call(task, system, user, max_tokens=None) -> Optional[str]
"""
from pipeline.llm_router import call  # re-export — all callers work unchanged

__all__ = ["call"]
