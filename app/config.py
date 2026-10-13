"""Settings, read from environment variables (see .env.example)."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

DEFAULT_MODELS = {
    # provider: (LLM, embeddings)
    "ollama": ("qwen3.5:9b", "bge-m3"),
    "bedrock": ("", "amazon.titan-embed-text-v2:0"),
}


class Settings(BaseSettings):
    provider: Literal["ollama", "bedrock"] = "ollama"
    llm_model: str = ""                 # empty: the provider's default
    embed_model: str = ""               # empty: the provider's default
    thinking: Optional[bool] = False    # Ollama "thinking" mode; None for models without it
    temperature: Optional[float] = None  # empty: 0 on Ollama, the model's default on Bedrock
    num_ctx: int = 8192                 # context window given to an Ollama model, in tokens
    chunk_size: int = 1000              # characters
    chunk_overlap: int = 150
    top_k: int = 6                      # chunks given to the model per question
    data_dir: Path = Path("data")       # the documents; read only
    db_dir: Path = Path("chroma_db")    # the vector store is persisted here
    ollama_base_url: str = "http://localhost:11434"
    aws_region: str = "us-east-1"
    max_sessions: int = 200             # conversations kept in memory
    log_response_chars: int = 4000      # API responses are logged, cut to this length; 0: not logged

    @field_validator("thinking", "temperature", mode="before")
    @classmethod
    def empty_is_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}:
            return None
        return v

    @model_validator(mode="after")
    def fill_defaults(self):
        llm, embed = DEFAULT_MODELS[self.provider]
        self.llm_model = self.llm_model.strip() or llm
        self.embed_model = self.embed_model.strip() or embed
        if not self.llm_model:
            raise ValueError(
                "LLM_MODEL is required when PROVIDER=bedrock: set it to a Bedrock model ID "
                "or inference profile ID (see .env.example)."
            )
        if self.temperature is None and self.provider == "ollama":
            self.temperature = 0
        return self
