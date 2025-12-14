# settings.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",  # Значение по умолчанию
        description="Redis connection URL"
    )
    OPENROUTER_KEY: str
    model_urls: Dict[str, str] = {
        "Mistral": "mistralai/devstral-2512:free",
        "Chimera": "tngtech/tng-r1t-chimera:free",
        "GPT-OSS": "openai/gpt-oss-120b:free"
    }
    modes: List[str] = ["Labels", "Summary"]
    class Config:
        env_file = ".env"

settings = Settings()
