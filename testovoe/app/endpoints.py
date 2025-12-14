from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Optional
import logging
import sys
from datetime import datetime
from pathlib import Path

from app.settings import settings
from app.providers import Model
from app.schemas import PromptRequest, ValidationError

app = FastAPI()
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
provider = Model()


# ==================== НАСТРОЙКА ЛОГГИРОВАНИЯ ====================
def setup_logging():
    """Настройка логгера с записью в файл и консоль"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Форматтер с подробной информацией
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Обработчик для записи в файл
    file_handler = logging.FileHandler(f'app_logs_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Инициализация логгера
logger = setup_logging()

# Логирование информации о доступных моделях и режимах
try:
    logger.info(f"Available models: {list(settings.model_urls.keys())}")
    logger.info(f"Available modes: {settings.modes}")
    logger.info(f"Number of models configured: {len(settings.model_urls)}")
except Exception as e:
    logger.error(f"Error loading settings on startup: {e}")


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    """Render the main form page with default values."""
    request_id = datetime.now().strftime("%H%M%S%f")[-8:]
    logger.info(f"[{request_id}] GET / - Rendering form")

    try:
        # Проверка наличия моделей
        if not settings.model_urls:
            logger.warning(f"[{request_id}] No models configured in settings")
            raise ValueError("No models configured in settings")

        # Get first model as default
        default_model = list(settings.model_urls.keys())[0]
        logger.debug(f"[{request_id}] Default model selected: {default_model}")

        # Default values for the form
        default_context = {
            "request": request,
            "result": None,
            "model_urls": settings.model_urls,
            "selected_model": default_model,
            "mode": "summary",
            "original_text": "I can't sleep...",
            "modes": settings.modes,
            "text_length": 16,
            "error": None
        }

        logger.info(f"[{request_id}] Form rendered successfully")
        return templates.TemplateResponse("index.html", default_context)

    except Exception as e:
        logger.error(f"[{request_id}] Error in form rendering: {str(e)}", exc_info=True)
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Service temporarily unavailable"
        }, status_code=500)


@app.post("/", response_class=HTMLResponse)
async def get_ai_response(
        request: Request,
        message: str = Form(...),
        mode: str = Form(...),
        model: str = Form(...)
):
    """Process the form and return AI analysis result."""
    request_id = datetime.now().strftime("%H%M%S%f")[-8:]

    # Логирование входящих данных (с обрезкой длинных сообщений)
    truncated_message = message[:50] + "..." if len(message) > 50 else message
    logger.info(
        f"[{request_id}] POST / - Processing request | "
        f"Model: {model}, Mode: {mode}, "
        f"Message length: {len(message)} chars, "
        f"Preview: '{truncated_message}'"
    )

    # Сохраняем исходные данные для отображения в форме
    form_data = {
        "selected_model": model,
        "mode": mode,
        "text_length": len(message),
        "original_text": message
    }

    try:
        # Create request object (валидация происходит здесь)
        logger.debug(f"[{request_id}] Creating PromptRequest object")
        req = PromptRequest(message=message, mode=mode, model=model)
        logger.info(f"[{request_id}] Validation successful")

        # Get AI analysis с таймаутом
        logger.info(f"[{request_id}] Calling AI provider: {model}/{mode}")
        start_time = datetime.now()

        try:
            ai_output, from_cache = await provider.analyze(
                message=req.message,
                mode=req.mode,
                model=req.model
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{request_id}] AI response received in {processing_time:.2f}s")
            logger.debug(f"[{request_id}] AI output length: {len(ai_output)} chars")

        except HTTPException as http_exc:
            logger.warning(
                f"[{request_id}] HTTPException from AI provider: "
                f"Status: {http_exc.status_code}, Detail: {http_exc.detail}"
            )
            raise http_exc
        except Exception as e:
            logger.error(f"[{request_id}] AI provider error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="AI service temporarily unavailable"
            )

        # Prepare context for template
        context = {
            "request": request,
            "result": ai_output,
            "model_urls": settings.model_urls,
            "selected_model": req.model,
            "from_cache": from_cache,
            "mode": req.mode,
            "modes": settings.modes,
            "original_text": form_data["original_text"],
            "text_length": len(req.message),
            "error": None
        }

        logger.info(f"[{request_id}] Request processed successfully")
        if from_cache:
            logger.info(f"[{request_id}] ⚡ Ответ получен из кэша")
        else:
            logger.info(f"[{request_id}] ♻️  Выполнен новый запрос к AI")
        return templates.TemplateResponse("index.html", context)

    except ValidationError as e:
        # Ошибки валидации (включая Pydantic)
        logger.warning(
            f"[{request_id}] Validation Error: {str(e)} | "
            f"Model: {model}, Mode: {mode}, "
            f"Message length: {len(message) if message else 0}"
        )

        # Формируем контекст с корректными данными
        context = {
            "request": request,
            "result": None,
            "model_urls": settings.model_urls,
            "selected_model": form_data["selected_model"],
            "mode": form_data["mode"],
            "modes": settings.modes,
            "from_cache": False,
            "text_length": form_data["text_length"],
            "original_text": form_data["original_text"],
            "error": f"Model must be one of: {', '.join(settings.model_urls)}\n"
                     f"Mode must be one of: {', '.join(settings.modes)}"
        }

        logger.debug(f"[{request_id}] Returning validation error context")
        return templates.TemplateResponse("index.html", context, status_code=422)

    except HTTPException as http_exc:
        # HTTP ошибки от провайдера
        logger.error(
            f"[{request_id}] HTTP error from provider | "
            f"Status: {http_exc.status_code}, Detail: {http_exc.detail} | "
            f"User model: {model}, mode: {mode}"
        )

        context = {
            "request": request,
            "result": None,
            "model_urls": settings.model_urls,
            "selected_model": form_data["selected_model"],
            "mode": form_data["mode"],
            "modes": settings.modes,
            "from_cache": False,
            "original_text": form_data["original_text"],
            "text_length": form_data["text_length"],
            "error": f"The server is busy. Try another model or try again shortly later"
        }

        return templates.TemplateResponse("index.html", context, status_code=http_exc.status_code)

    except Exception as e:
        # Общая обработка непредвиденных ошибок
        logger.error(
            f"[{request_id}] Unexpected error in get_ai_response: {str(e)} | "
            f"Model: {model}, Mode: {mode}",
            exc_info=True
        )

        context = {
            "request": request,
            "result": None,
            "model_urls": settings.model_urls,
            "selected_model": form_data["selected_model"],
            "mode": form_data["mode"],
            "modes": settings.modes,
            "from_cache": False,
            "original_text": form_data["original_text"],
            "text_length": form_data["text_length"],
            "error": "An unexpected error occurred. Please try again."
        }

        return templates.TemplateResponse("index.html", context, status_code=500)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработчик ошибок валидации запросов FastAPI"""
    request_id = datetime.now().strftime("%H%M%S%f")[-8:]

    # Детальное логирование ошибок валидации
    logger.warning(f"[{request_id}] RequestValidationError: {exc.errors()}")

    # Получаем данные формы из запроса
    try:
        form_data = await request.form()
        logger.debug(f"[{request_id}] Form data received: {dict(form_data)}")
    except Exception as e:
        logger.error(f"[{request_id}] Error reading form data: {str(e)}")
        form_data = {}

    # Получаем текущие настройки
    try:
        current_modes = settings.modes
        current_model_urls = settings.model_urls
        default_model = list(current_model_urls.keys())[0] if current_model_urls else "Mistral"
        logger.debug(f"[{request_id}] Settings loaded successfully")
    except Exception as e:
        logger.error(f"[{request_id}] Error loading settings in exception handler: {str(e)}")
        current_modes = ["summary", "labels"]
        current_model_urls = {}
        default_model = "Mistral"

    # Извлекаем значения из формы или используем значения по умолчанию
    message = form_data.get("message", "")
    mode = form_data.get("mode", "Summary")
    model = form_data.get("model", default_model)

    logger.debug(f"[{request_id}] Extracted values - Model: {model}, Mode: {mode}")

    # Проверяем, что модель существует в списке
    if model not in current_model_urls:
        logger.warning(f"[{request_id}] Model '{model}' not in available models. Using default: {default_model}")
        model = default_model

    # Формируем понятное сообщение об ошибке
    error_msg = "Please fill in all required fields correctly."
    for error in exc.errors():
        logger.debug(f"[{request_id}] Validation error detail: {error}")
        if error["loc"][-1] == "message" and error["type"] == "missing":
            error_msg = "Message cannot be empty."
            break

    context = {
        "request": request,
        "result": None,
        "model_urls": current_model_urls,
        "selected_model": model,
        "mode": mode,
        "modes": current_modes,
        "original_text": message if message else "",
        "text_length": len(message) if message else 0,
        "error": error_msg
    }

    logger.info(f"[{request_id}] Returning validation error page")
    return templates.TemplateResponse("index.html", context, status_code=422)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех входящих запросов"""
    request_id = datetime.now().strftime("%H%M%S%f")[-8:]
    request.state.request_id = request_id

    start_time = datetime.now()

    # Логирование входящего запроса
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )

    try:
        response = await call_next(request)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Логирование успешного ответа
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {processing_time:.3f}s"
        )

        return response

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Exception: {str(e)} - "
            f"Time: {processing_time:.3f}s",
            exc_info=True
        )
        raise




if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting FastAPI application")
    logger.info("=" * 50)

    uvicorn.run(
        "endpoints:app",
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "default",
                },
            },
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                },
            },
            "loggers": {
                "uvicorn": {"level": "INFO", "handlers": ["console"]},
            },
        }
    )
