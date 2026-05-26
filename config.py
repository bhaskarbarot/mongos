import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host:     str = os.getenv("POSTGRES_HOST",     "localhost")
    postgres_port:     int = int(os.getenv("POSTGRES_PORT", "5433"))
    postgres_user:     str = os.getenv("POSTGRES_USER",     "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_db:       str = os.getenv("POSTGRES_DB",       "mongos_sync")

    # ── Fast Path ─────────────────────────────────────────────────────────────
    # FAST_PATH_ENABLED=true  → fast path runs before LLM pipeline (recommended)
    # FAST_PATH_ENABLED=false → bypass fast path entirely (debug / testing only)
    fast_path_enabled: bool = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"

    # ── Ollama — Text2SQL fine-tuned models ───────────────────────────────────
    ollama_base_url:         str = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
    ollama_primary_model:    str = os.getenv("OLLAMA_PRIMARY_MODEL",  "debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned:latest")
    ollama_primary_timeout:  int = int(os.getenv("OLLAMA_PRIMARY_TIMEOUT", "8"))
    ollama_fallback_model:   str = os.getenv("OLLAMA_FALLBACK_MODEL", "a-kore/Arctic-Text2SQL-R1-7B:latest")
    ollama_fallback_timeout: int = int(os.getenv("OLLAMA_FALLBACK_TIMEOUT", "30"))

    # ── Ollama — General reasoning (decompose / synthesize local fallback) ────
    ollama_reasoning_model:   str = os.getenv("OLLAMA_REASONING_MODEL",   "llama3.1:8b")
    ollama_reasoning_timeout: int = int(os.getenv("OLLAMA_REASONING_TIMEOUT", "180"))
    ollama_classify_model:    str = os.getenv("OLLAMA_CLASSIFY_MODEL",    "qwen2.5:3b")

    # ── Groq — 3 API keys for rate-limit rotation ─────────────────────────────
    # Priority: key1 → key2 → key3 → Gemini → OpenRouter → Ollama
    groq_api_key:   str = os.getenv("GROQ_API_KEY",   "")
    groq_api_key_2: str = os.getenv("GROQ_API_KEY_2", "")
    groq_api_key_3: str = os.getenv("GROQ_API_KEY_3", "")

    # classify  → fastest small model (binary output, tiny prompt)
    groq_classify_model:  str = os.getenv("GROQ_CLASSIFY_MODEL",  "llama-3.1-8b-instant")
    # decompose → scout 17B mid-size (smart, generous TPM budget)
    groq_decompose_model: str = os.getenv("GROQ_DECOMPOSE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    # synthesize→ 70B only for final answer (highest quality)
    groq_synthesis_model: str = os.getenv("GROQ_SYNTHESIS_MODEL", "llama-3.3-70b-versatile")
    # sql       → scout 17B (compact schema = ~500 tok/req, 60+ qpm per key)
    groq_sql_model:       str = os.getenv("GROQ_SQL_MODEL",       "meta-llama/llama-4-scout-17b-16e-instruct")

    # ── Gemini — secondary fallback (30 RPM / 1M TPD free) ───────────────────
    gemini_api_key:        str = os.getenv("GEMINI_API_KEY",        "")
    gemini_classify_model: str = os.getenv("GEMINI_CLASSIFY_MODEL", "gemini-2.0-flash-lite")
    gemini_decompose_model:str = os.getenv("GEMINI_DECOMPOSE_MODEL","gemini-2.0-flash-lite")
    gemini_synthesis_model:str = os.getenv("GEMINI_SYNTHESIS_MODEL","gemini-2.0-flash-lite")

    # ── OpenRouter — tertiary fallback (free tier) ────────────────────────────
    openrouter_api_key:        str = os.getenv("OPENROUTER_API_KEY",     "")
    openrouter_decompose_model:str = os.getenv("OR_DECOMPOSE_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    openrouter_synthesis_model:str = os.getenv("OR_SYNTHESIS_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    openrouter_sql_model:      str = os.getenv("OR_SQL_MODEL",       "deepseek/deepseek-chat-v3-0324:free")

    # ── Ollama SQL toggle ─────────────────────────────────────────────────────
    ollama_sql_enabled: bool = os.getenv("OLLAMA_SQL_ENABLED", "true").lower() == "true"

    # ── OpenAI — fallback 1 (paid, most reliable) ────────────────────────────
    openai_enabled:           bool = os.getenv("OPENAI_ENABLED", "true").lower() == "true"
    openai_api_key:           str = os.getenv("OPENAI_API_KEY",          "")
    openai_sql_model:         str = os.getenv("OPENAI_SQL_MODEL",        "gpt-4o-mini")
    openai_classify_model:    str = os.getenv("OPENAI_CLASSIFY_MODEL",   "gpt-4o-mini")
    openai_decompose_model:   str = os.getenv("OPENAI_DECOMPOSE_MODEL",  "gpt-4o-mini")
    openai_synthesis_model:   str = os.getenv("OPENAI_SYNTHESIS_MODEL",  "gpt-4o")
    openai_narrate_model:     str = os.getenv("OPENAI_NARRATE_MODEL",    "gpt-4o-mini")

    # ── Cerebras — fallback 3 (free, 800 tok/s, llama-3.3-70b) ──────────────
    cerebras_api_key:          str = os.getenv("CEREBRAS_API_KEY",          "")
    cerebras_sql_model:        str = os.getenv("CEREBRAS_SQL_MODEL",        "llama-3.3-70b")
    cerebras_synthesis_model:  str = os.getenv("CEREBRAS_SYNTHESIS_MODEL",  "llama-3.3-70b")
    cerebras_classify_model:   str = os.getenv("CEREBRAS_CLASSIFY_MODEL",   "llama3.1-8b")

    # ── SambaNova — fallback 4 (free, llama-3.3-70b) ─────────────────────────
    sambanova_api_key:          str = os.getenv("SAMBANOVA_API_KEY",          "")
    sambanova_sql_model:        str = os.getenv("SAMBANOVA_SQL_MODEL",        "Meta-Llama-3.3-70B-Instruct")
    sambanova_synthesis_model:  str = os.getenv("SAMBANOVA_SYNTHESIS_MODEL",  "Meta-Llama-3.3-70B-Instruct")
    sambanova_classify_model:   str = os.getenv("SAMBANOVA_CLASSIFY_MODEL",   "Meta-Llama-3.1-8B-Instruct")

    # ── Decomposer toggle ─────────────────────────────────────────────────────
    # DECOMPOSER_ENABLED=true  → medium/complex agents decompose query into sub-queries (default)
    # DECOMPOSER_ENABLED=false → skip decomposer, pass raw query as single sub-query directly to SQL
    decomposer_enabled: bool = os.getenv("DECOMPOSER_ENABLED", "true").strip().lower() == "true"

    # ── Plotly charts toggle ──────────────────────────────────────────────────
    # PLOTLY_CHARTS_ENABLED=true  → /api/chart active; chart_data returned in /chat
    # PLOTLY_CHARTS_ENABLED=false → charts off; chart_data=null; /api/chart → 503
    plotly_charts_enabled: bool = os.getenv("PLOTLY_CHARTS_ENABLED", "true").strip().lower() == "true"

    # ── Misc ──────────────────────────────────────────────────────────────────
    strict_grounded_mode: bool = os.getenv("STRICT_GROUNDED_MODE", "false").lower() == "true"
    log_file:             str  = os.getenv("QUERY_LOG_FILE", "logs/query.log")
    max_retries:          int  = int(os.getenv("AGENT_MAX_RETRIES", "0"))
    allowed_origins:      str  = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://localhost:3000,http://localhost:5173",
    )

    @property
    def postgres_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def groq_api_keys(self) -> list:
        """Return all configured Groq API keys (non-empty only)."""
        return [k for k in [self.groq_api_key, self.groq_api_key_2, self.groq_api_key_3] if k]


settings = Settings()
