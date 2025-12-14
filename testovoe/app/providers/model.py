import asyncio
import hashlib
from typing import Tuple

import httpx
from fastapi import HTTPException
from pydantic import ValidationError
import logging
from redis.asyncio import Redis

from app.settings import settings

logger = logging.getLogger(__name__)
redis = Redis.from_url(
    getattr(settings, "REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True
)


class Model:
    def __init__(self, cache_ttl: int = 3600):
        self.mode_prompts = {
            "Labels": """You are an emotion classifier. For a given input sentence, output a score for each of the 10 emotion classes in the exact format shown below (one label per line, label name followed by colon and a decimal score between 0.00 and 1.00). Scores should reflect the model's estimate of how strongly each emotion is present; they do not have to sum to 1.0 but should be comparable across labels. After the scores, include a short (one‑line) human‑readable description explaining the dominant emotion.

                    One-shot example (demonstrates all labels with numeric scores and a short description):

                    Input:
                    "I can't believe I got the promotion — I'm thrilled, a little nervous about the new role, and honestly a bit proud of myself."

                    Example output:
                    Joy: 0.40
                    Sadness: 0.00
                    Anger: 0.00
                    Fear: 0.10
                    Surprise: 0.15
                    Disgust: 0.00
                    Trust: 0.12
                    Anticipation: 0.15
                    Shame: 0.00
                    Neutral: 0.08

                    Dominant: Joy — excited and pleased about the promotion, with mild anticipation and nervousness.

                    Emotion descriptions (one line each):
                    Joy: positive feeling of pleasure, happiness, or satisfaction.
                    Sadness: low mood, loss, or sorrow.
                    Anger: irritation, hostility, or frustration.
                    Fear: anxiety, alarm, or sense of threat.
                    Surprise: brief reaction to something unexpected.
                    Disgust: strong aversion or rejection.
                    Trust: confidence or positive expectation toward someone/something.
                    Anticipation: focused expectation or readiness for a future event.
                    Shame: self‑conscious embarrassment or regret.
                    Neutral: absence of a clear emotional tone.

                    Task rule: For any new input sentence, follow the exact same output format: ten labeled scores (one per line) followed by a single-line "Dominant" explanation and the one-line description of the dominant emotion.
                    Make sure the labels' probabilities sum up to 1""",

            "Summary": """You are a concise summarizer and emotion analyst. For any input text, produce plain text output with exactly three short lines in this order:

                    1) Short summary of the situation (one sentence).
                    2) Emotion label and short description (one line), choose one of: Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation, Shame, Neutral.
                    3) One-line reason citing cues from the input that justify the assigned emotion.

                    Do NOT output JSON or extra commentary. Keep each line concise (one sentence or phrase).

                    One-shot example:

                    Input:
                    "I can't sleep; the deadline is tomorrow and I'm worried I won't finish."

                    Example output (exact plain-text format):
                    The speaker is anxious about an upcoming deadline and fears not completing the work.
                    Anticipation — focused expectation with concern about the near future.
                    Reason: mentions 'deadline is tomorrow' and explicit worry 'I'm worried I won't finish'.
                    """}
        self.timeout = 30.0
        self.max_retries = 2
        self.cache_ttl = cache_ttl  # seconds

    @staticmethod
    def _make_cache_key(message: str, model: str, mode: str) -> str:
        raw = f"{model}|{mode}|{message}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"websent:analyze:{h}"

    async def analyze(self, message: str, model: str, mode: str = "summary") -> Tuple[str, bool]:
        """
        Асинхронный метод, возвращает (result: str, from_cache: bool).
        Использует Redis (асинхронный клиент) для кэширования результатов.
        Сохраняет существующую логику логирования и обработки ошибок.
        """
        if not mode or not message or not model:
            raise ValidationError("Model, mode must be provided. Message must not be empty")
        if mode not in self.mode_prompts:
            raise ValueError(f"Mode '{mode}' not supported")

        context = self.mode_prompts[mode]

        if model not in settings.model_urls:
            raise ValueError(f"Model '{model}' not found in settings")

        model_url = settings.model_urls[model]

        if not settings.OPENROUTER_KEY:
            raise HTTPException(status_code=500, detail="API key not configured")

        cache_key = self._make_cache_key(message, model, mode)

        # Попытка получить результат из Redis
        try:
            cached = await redis.get(cache_key)
        except Exception as e:
            logger.warning("Redis GET failed, continuing without cache: %s", e)
            cached = None

        if cached is not None:
            return cached, True

        url = "https://openrouter.ai/api/v1/chat/completions"

        # Конфигурация клиента с ретраями
        transport = httpx.AsyncHTTPTransport(retries=self.max_retries)

        async with httpx.AsyncClient(transport=transport, timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url=url,
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_url,
                        "messages": [
                            {"role": "system", "content": context},
                            {"role": "user", "content": message}
                        ],
                        "reasoning": {"enabled": True}
                    }
                )

                status = response.status_code
                headers = dict(response.headers)
                # Попытка безопасно получить тело (json или текст)
                try:
                    data = response.json()
                    body_preview = str(data)
                except Exception:
                    try:
                        body_text = await response.aread()
                        body_preview = body_text.decode(errors="ignore")[:1000]
                    except Exception:
                        body_preview = "<unreadable body>"

                logger.debug("AI provider response | status=%s headers=%s body_preview=%s", status, headers,
                             body_preview)

                # Если провайдер вернул ошибку в HTTP статусе
                if status == 429:
                    retry_after = headers.get("Retry-After")
                    logger.warning("AI provider rate limited (429). Retry-After=%s body=%s", retry_after, body_preview)
                    # Пробрасываем 429 дальше с подсказкой
                    raise HTTPException(status_code=429,
                                        detail=f"AI provider rate limit. Retry-After: {retry_after or 'unknown'}")

                if 400 <= status < 500:
                    # Клиентская ошибка провайдера — пробрасываем её код и сообщение (если есть)
                    provider_msg = None
                    if isinstance(data, dict):
                        provider_msg = data.get("error") or data.get("message")
                    logger.error("AI provider client error: %s | body=%s", status, body_preview)
                    raise HTTPException(status_code=status, detail=f"AI service client error: {provider_msg or status}")

                if status >= 500:
                    logger.error("AI provider server error: %s | body=%s", status, body_preview)
                    raise HTTPException(status_code=502, detail=f"AI service returned server error: {status}")

                # На этом этапе статус 2xx — проверяем содержимое
                if not isinstance(data, dict):
                    logger.error("Unexpected response format from AI provider: %s", body_preview)
                    raise HTTPException(status_code=502, detail="Invalid response from AI service")

                # Если провайдер вернул структуру error внутри JSON
                if "error" in data:
                    err = data["error"]
                    provider_msg = err.get("message", "Provider returned an error") if isinstance(err, dict) else str(
                        err)
                    provider_code = err.get("code", "unknown") if isinstance(err, dict) else "unknown"
                    logger.error("Provider error in JSON: %s - %s", provider_code, provider_msg)

                    error_map = {
                        "rate_limit_exceeded": 429,
                        "invalid_api_key": 401,
                        "insufficient_quota": 402,
                        "model_not_found": 404,
                    }
                    status_code = error_map.get(provider_code, 502)
                    raise HTTPException(status_code=status_code, detail=f"AI service error: {provider_msg}")

                # Проверяем наличие choices
                if "choices" not in data or not data["choices"]:
                    logger.warning("AI provider returned no choices | body=%s", body_preview)
                    raise HTTPException(status_code=502, detail="Invalid response from AI service: no choices returned")

                result = data['choices'][0]['message']['content']

                # Сохраняем в Redis с TTL; при ошибке логируем, но не мешаем возврат результата
                try:
                    await redis.set(cache_key, result, ex=self.cache_ttl)
                except Exception as e:
                    logger.warning("Redis SET failed (result not cached): %s", e)

                return result, False

            except httpx.TimeoutException:
                logger.error("Request to AI service timed out")
                raise HTTPException(status_code=504, detail="AI service timeout")
            except httpx.RequestError as e:
                logger.error("Request error: %s", e)
                raise HTTPException(status_code=503, detail="AI service unavailable")
