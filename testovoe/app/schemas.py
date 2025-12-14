from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List
from app.settings import settings


class PromptRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze",
        examples=["I can't sleep..."]
    )
    mode: str = Field(
        ...,
        description="Operation mode",
        examples=["Summary"]
    )
    model: str = Field(
        ...,
        description="Model name",
        examples=["Mistral"]
    )

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, model_name, info):
        if not isinstance(model_name, str):
            raise ValueError("Model name must be a string")


        available_models = list(settings.model_urls.keys())

        if not available_models:
            raise ValueError("No models available in settings")

        model_name = model_name.strip()
        for allowed_name in available_models:
            if allowed_name.lower() == model_name.lower():
                return allowed_name

        raise ValueError(
            f"Model must be one of: {', '.join(available_models)}. "
            f"Got: {model_name}"
        )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, mode, info):
        if not isinstance(mode, str):
            raise ValueError("Mode must be a string")

        # Получаем текущие настройки
        from app.settings import settings
        available_modes = settings.modes

        if not available_modes:
            raise ValueError("No modes available in settings")

        mode = mode.strip()
        for allowed_mode in available_modes:
            if allowed_mode.lower() == mode.lower():
                return allowed_mode

        raise ValueError(
            f"Mode must be one of: {', '.join(available_modes)}. "
            f"Got: {mode}"
        )