from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, HttpUrl, SecretStr, validator
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class AppSettings(BaseSettings):
    llm_provider: LLMProvider = Field(..., description="Select the LLM provider.")
    llm_server_url: HttpUrl = Field(..., description="Base URL for the LLM server.")
    llm_model_name: str = Field(..., min_length=1, description="Model name for the LLM provider.")
    llm_api_key: SecretStr = Field(..., description="API key for the selected LLM provider.")
    ping_timeout_seconds: int = Field(10, ge=1, le=60, description="Timeout for ping commands.")
    traceroute_timeout_seconds: int = Field(30, ge=5, le=120, description="Timeout for traceroute commands.")
    telnet_timeout_seconds: int = Field(10, ge=1, le=60, description="Timeout for telnet commands.")
    log_directory: Path = Field(default=Path("logs"), description="Directory to store application logs.")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        use_enum_values = True

    @validator("llm_server_url")
    def normalize_server_url(cls, value: HttpUrl) -> HttpUrl:
        normalized = str(value).rstrip("/")
        return HttpUrl.build(scheme=value.scheme, host=value.host, path=value.path or "", port=value.port)

    @validator("llm_model_name")
    def ensure_model_name(cls, value: str) -> str:
        return value.strip()

    @validator("log_directory", pre=True)
    def normalize_log_path(cls, value: Optional[str]) -> Path:
        return Path(value or "logs").expanduser().resolve()


def get_settings() -> AppSettings:
    return AppSettings()
