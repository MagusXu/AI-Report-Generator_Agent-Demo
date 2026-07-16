from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from local env files when present."""

    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    workspace_id: str = Field(default="", alias="WORKSPACE_ID")
    llm_base_url: str = Field(
        default="https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="qwen3.6-35b-a3b", alias="LLM_MODEL")
    enable_thinking: bool = Field(default=False, alias="ENABLE_THINKING")
    embedding_base_url: str = Field(
        default="https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        alias="EMBEDDING_BASE_URL",
    )
    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")
    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    export_dir: str = Field(default="./exports", alias="EXPORT_DIR")
    cors_origins: str = Field(
        default="http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )
    chunk_child_target_chars: int = Field(default=820, alias="CHUNK_CHILD_TARGET_CHARS")
    chunk_child_min_chars: int = Field(default=300, alias="CHUNK_CHILD_MIN_CHARS")
    chunk_child_max_chars: int = Field(default=1200, alias="CHUNK_CHILD_MAX_CHARS")
    chunk_child_overlap_chars: int = Field(default=100, alias="CHUNK_CHILD_OVERLAP_CHARS")
    chunk_parent_target_chars: int = Field(default=2200, alias="CHUNK_PARENT_TARGET_CHARS")
    chunk_group_max_chars: int = Field(default=3520, alias="CHUNK_GROUP_MAX_CHARS")
    table_rows_per_child: int = Field(default=18, alias="TABLE_ROWS_PER_CHILD")
    table_rows_per_parent: int = Field(default=45, alias="TABLE_ROWS_PER_PARENT")
    chunk_heading_prefix_enabled: bool = Field(default=True, alias="CHUNK_HEADING_PREFIX_ENABLED")
    chunk_fallback_pages_per_group: int = Field(default=3, alias="CHUNK_FALLBACK_PAGES_PER_GROUP")
    chunk_heading_chunks_enabled: bool = Field(default=True, alias="CHUNK_HEADING_CHUNKS_ENABLED")
    chunk_quality_gate_mode: str = Field(default="warn", alias="CHUNK_QUALITY_GATE_MODE")
    chunk_quality_min_score: int = Field(default=50, alias="CHUNK_QUALITY_MIN_SCORE")

    model_config = SettingsConfigDict(
        env_file=(".env", "config.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def expand_workspace_urls(self) -> "Settings":
        if self.workspace_id:
            self.llm_base_url = self.llm_base_url.replace("${WORKSPACE_ID}", self.workspace_id)
            self.embedding_base_url = self.embedding_base_url.replace("${WORKSPACE_ID}", self.workspace_id)
        return self

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def export_path(self) -> Path:
        return Path(self.export_dir)

    @property
    def chroma_data_path(self) -> Path:
        return Path(self.chroma_path)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
