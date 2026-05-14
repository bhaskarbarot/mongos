import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_db: str = os.getenv("POSTGRES_DB", "mongos_sync")

    # Primary LLM — fast Text2SQL model (local Ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_primary_model: str = os.getenv("OLLAMA_PRIMARY_MODEL", "debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned:latest")
    ollama_primary_timeout: int = int(os.getenv("OLLAMA_PRIMARY_TIMEOUT", "8"))

    # Fallback LLM — larger Text2SQL model (used when primary fails/times out)
    ollama_fallback_model: str = os.getenv("OLLAMA_FALLBACK_MODEL", "a-kore/Arctic-Text2SQL-R1-7B:latest")
    ollama_fallback_timeout: int = int(os.getenv("OLLAMA_FALLBACK_TIMEOUT", "30"))

    # Reasoning LLM — local fallback for decomposition + synthesis (when Groq not set)
    ollama_reasoning_model: str = os.getenv("OLLAMA_REASONING_MODEL", "qwen2.5:7b")
    ollama_reasoning_timeout: int = int(os.getenv("OLLAMA_REASONING_TIMEOUT", "60"))

    # ── Reasoning LLM providers (decompose + synthesize + classify) ──────────
    # Priority: Groq → Gemini → OpenRouter → Ollama (local, always available)
    # Text2SQL models above are NOT changed — they are fine-tuned for SQL only.

    # Groq — fastest cloud inference, high RPD
    # 8b-instant: 14,400 RPD, 6K TPM  → classify + decompose (small prompts)
    # 70b-versatile: 14,400 RPD, 12K TPM → synthesize (large output, best quality)
    groq_api_key: str        = os.getenv("GROQ_API_KEY",   "")
    groq_api_key_2: str      = os.getenv("GROQ_API_KEY_2", "")   # account 2 — 2× TPM budget
    groq_api_key_3: str      = os.getenv("GROQ_API_KEY_3", "")   # account 3 — 3× TPM budget = 36K TPM on 70b
    groq_classify_model: str = os.getenv("GROQ_CLASSIFY_MODEL",  "meta-llama/llama-4-scout-17b-16e-instruct")
    groq_decompose_model: str= os.getenv("GROQ_DECOMPOSE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    # Both SQL and synthesis use 70b — 2 accounts × 12K TPM = 24K TPM → supports 3 qpm easily
    groq_sql_model: str      = os.getenv("GROQ_SQL_MODEL",       "llama-3.3-70b-versatile")
    groq_synthesis_model: str= os.getenv("GROQ_SYNTHESIS_MODEL", "llama-3.3-70b-versatile")

    # Gemini — 250K TPM free, good for large context synthesis
    # 1,000 RPD limit → secondary provider, fallback for Groq TPM spikes
    gemini_api_key: str        = os.getenv("GEMINI_API_KEY", "")
    gemini_decompose_model: str= os.getenv("GEMINI_DECOMPOSE_MODEL", "gemini-2.5-flash-lite")
    gemini_synthesis_model: str= os.getenv("GEMINI_SYNTHESIS_MODEL", "gemini-2.5-flash-lite")
    gemini_classify_model: str = os.getenv("GEMINI_CLASSIFY_MODEL",  "gemini-2.5-flash-lite")

    # OpenRouter — last-resort cloud fallback (free tier, multiple models)
    openrouter_api_key: str        = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_decompose_model: str= os.getenv("OR_DECOMPOSE_MODEL", "meta-llama/llama-3.1-8b-instruct")
    openrouter_synthesis_model: str= os.getenv("OR_SYNTHESIS_MODEL", "meta-llama/llama-3.1-8b-instruct")

    # Ollama — local final fallback, always available (no API key needed)
    # qwen2.5:1.5b → classify (tiny, fast)  | qwen2.5:7b → decompose + synthesize
    ollama_classify_model: str = os.getenv("OLLAMA_CLASSIFY_MODEL", "qwen2.5:1.5b")

    strict_grounded_mode: bool = os.getenv("STRICT_GROUNDED_MODE", "false").lower() == "true"
    fast_path_enabled: bool    = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"
    log_file: str = os.getenv("QUERY_LOG_FILE", "logs/query.log")
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "0"))

    # E3: Allowed CORS origins — comma-separated, no spaces
    allowed_origins: str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://localhost:3000,http://localhost:5173",
    )

    @property
    def postgres_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
