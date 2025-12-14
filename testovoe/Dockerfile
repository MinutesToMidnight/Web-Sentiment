FROM python:3.10-slim

# Системные зависимости (минимум)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Копируем зависимости и устанавливаем их (кэшируем слой)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY . .

# Создаём непривилегированного пользователя и переключаемся на него
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]