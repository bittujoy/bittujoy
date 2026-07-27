import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


def _resolve_env_file() -> Path:
    project_root = Path(__file__).resolve().parent
    for candidate in (project_root / ".env", project_root / "enviornment.env"):
        if candidate.exists():
            return candidate
    return project_root / ".env"


def _load_env_values(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        value = raw_value.split("#", 1)[0].strip()
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
            value = value[1:-1]
        values[key.strip()] = value
    return values


class AppSettings(BaseSettings):
    llm_provider: str = Field(default="openai", description="LLM provider configured in the environment.")
    llm_server_url: str = Field(default="https://api.openai.com", description="Base URL for the LLM server.")
    llm_model_name: str = Field(default="gpt-4o-mini", description="Model name for the LLM provider.")
    llm_api_key: str = Field(default="", description="API key for the selected LLM provider.")
    llm_auth_header: str = Field(default="Authorization", description="Auth header for enterprise LLM requests.")
    llm_auth_prefix: str = Field(default="Bearer", description="Prefix for the auth value for enterprise LLM requests.")
    ping_timeout_seconds: int = Field(10, ge=1, le=60, description="Timeout for ping commands.")
    traceroute_timeout_seconds: int = Field(30, ge=5, le=120, description="Timeout for traceroute commands.")
    telnet_timeout_seconds: int = Field(10, ge=1, le=60, description="Timeout for telnet commands.")
    log_directory: Path = Field(default=Path("logs"), description="Directory to store application logs.")

    class Config:
        env_file = str(_resolve_env_file())
        env_file_encoding = "utf-8"
        extra = "ignore"

    @validator("llm_server_url")
    def normalize_server_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return "https://api.openai.com"
        for suffix in ("/v1/chat/completions", "/chat/completions"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break
        return cleaned.rstrip("/")

    @validator("llm_model_name")
    def ensure_model_name(cls, value: str) -> str:
        return value.strip()

    @validator("llm_provider")
    def ensure_provider(cls, value: str) -> str:
        return value.strip().lower()

    @validator("llm_api_key", "llm_auth_header", "llm_auth_prefix")
    def ensure_non_empty_strings(cls, value: str) -> str:
        return value.strip()

    @validator("log_directory", pre=True)
    def normalize_log_path(cls, value: Optional[str]) -> Path:
        if value is None or value == "":
            return Path("logs")
        return Path(value).expanduser().resolve()


def get_settings() -> AppSettings:
    env_file = _resolve_env_file()
    values = _load_env_values(env_file)

    env_overrides = {
        "llm_provider": values.get("LLM_PROVIDER") or values.get("ENTERPRISE_PROVIDER") or os.getenv("LLM_PROVIDER") or os.getenv("ENTERPRISE_PROVIDER"),
        "llm_server_url": values.get("LLM_SERVER_URL") or values.get("ENTERPRISE_API_URL") or os.getenv("LLM_SERVER_URL") or os.getenv("ENTERPRISE_API_URL"),
        "llm_model_name": values.get("LLM_MODEL_NAME") or values.get("ENTERPRISE_MODEL_NAME") or os.getenv("LLM_MODEL_NAME") or os.getenv("ENTERPRISE_MODEL_NAME"),
        "llm_api_key": values.get("LLM_API_KEY") or values.get("ENTERPRISE_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("ENTERPRISE_API_KEY"),
        "llm_auth_header": values.get("ENTERPRISE_AUTH_HEADER") or os.getenv("ENTERPRISE_AUTH_HEADER") or os.getenv("LLM_AUTH_HEADER"),
        "llm_auth_prefix": values.get("ENTERPRISE_AUTH_PREFIX") or os.getenv("ENTERPRISE_AUTH_PREFIX") or os.getenv("LLM_AUTH_PREFIX"),
    }

    return AppSettings(**{key: value for key, value in env_overrides.items() if value is not None})
