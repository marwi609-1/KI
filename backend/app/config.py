from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings: "voyage" (echt) oder "hash" (nur zum Testen ohne API-Key, keine Semantik!)
    embedding_backend: str = "voyage"
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3-lite"

    # Vektordatenbank: URL des Qdrant-Containers oder ":memory:" für lokale Tests
    qdrant_url: str = "http://qdrant:6333"

    # LLM für Modul 4 (optional – ohne Key zeigt das Dashboard den fertigen Prompt)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"


settings = Settings()
