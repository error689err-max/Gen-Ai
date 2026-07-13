from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ───────────────────────────────────────────────────────
    # Application
    # ───────────────────────────────────────────────────────
    app_name: str = Field("Document AI Agent", alias="APP_NAME")
    app_version: str = Field("1.0.0", alias="APP_VERSION")
    environment: Literal["development", "production"] = Field(
        "development", alias="ENVIRONMENT"
    )
    debug: bool = Field(False, alias="DEBUG")

    # ───────────────────────────────────────────────────────
    # Provider Selection
    # ───────────────────────────────────────────────────────
    llm_provider: Literal["gemini", "bedrock", "openai"] = Field(
        "gemini", alias="LLM_PROVIDER"
    )

    embedding_provider: Literal["gemini", "bedrock", "local"] = Field(
        "gemini", alias="EMBEDDING_PROVIDER"
    )

    EMBEDDING_BATCH_SIZE: int = Field(10, alias="EMBEDDING_BATCH_SIZE")
    # ───────────────────────────────────────────────────────
    # Gemini
    # ───────────────────────────────────────────────────────
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "models/text-embedding-004",
        alias="GEMINI_EMBEDDING_MODEL",
    )

    # ───────────────────────────────────────────────────────
    # OpenAI / Groq
    # ───────────────────────────────────────────────────────
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_url: str = Field("", alias="OPENAI_URL")
    openai_chat_model: str = Field("", alias="OPENAI_CHAT_MODEL")

    # ───────────────────────────────────────────────────────
    # Local Embeddings
    # ───────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        "", alias="LOCAL_EMBEDDING_MODEL"
    )

    embedding_dimensions: int = Field(
        768,
        alias="EMBEDDING_DIMENSIONS",
    )

    # ───────────────────────────────────────────────────────
    # Bedrock
    # ───────────────────────────────────────────────────────
    aws_region: str = Field("us-east-1", alias="AWS_REGION")

    bedrock_model_id: str = Field(
        "anthropic.claude-3-sonnet-20240229-v1:0",
        alias="BEDROCK_MODEL_ID",
    )

    bedrock_embedding_model_id: str = Field(
        "amazon.titan-embed-text-v2:0",
        alias="BEDROCK_EMBEDDING_MODEL_ID",
    )

    # ───────────────────────────────────────────────────────
    # Qdrant
    # ───────────────────────────────────────────────────────
    qdrant_host: str = Field("127.0.0.1", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        "company_documents",
        alias="QDRANT_COLLECTION",
    )
    qdrant_vector_size: int = Field(
        768,
        alias="QDRANT_VECTOR_SIZE",
    )

    # ───────────────────────────────────────────────────────
    # Redis
    # ───────────────────────────────────────────────────────
    redis_url: str = Field(
        "redis://127.0.0.1:6379",
        alias="REDIS_URL",
    )

    session_ttl_seconds: int = Field(
        3600,
        alias="SESSION_TTL_SECONDS",
    )

    # ───────────────────────────────────────────────────────
    # PostgreSQL
    # ───────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://docai_user:docai_password@127.0.0.1:5432/docai_db",
        alias="DATABASE_URL",
    )

    # ───────────────────────────────────────────────────────
    # Document Processing
    # ───────────────────────────────────────────────────────
    max_file_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 128
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.5

    # ───────────────────────────────────────────────────────
    # Agent
    # ───────────────────────────────────────────────────────
    agent_max_iterations: int = 5
    agent_timeout_seconds: int = 30

    # ───────────────────────────────────────────────────────
    # Logging
    # ───────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ───────────────────────────────────────────────────────
    # Feature Flags
    # ───────────────────────────────────────────────────────
    enable_streaming: bool = Field(True, alias="ENABLE_STREAMING")
    enable_citations: bool = Field(True, alias="ENABLE_CITATIONS")
    enable_conversation_memory: bool = Field(
        True,
        alias="ENABLE_CONVERSATION_MEMORY",
    )

    secret_key: str = Field(default="dev-secret", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()