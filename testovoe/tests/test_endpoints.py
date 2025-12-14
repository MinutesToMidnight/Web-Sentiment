# app/tests/test_endpoints.py
import importlib
import pytest
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates


@pytest.fixture
def client(monkeypatch, tmp_path):
    # 1) Устанавливаем env до импорта приложения
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")

    # 2) Создаём временную директорию для шаблонов и пишем минимальные шаблоны
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    index_html = templates_dir / "index.html"
    error_html = templates_dir / "error.html"

    # Минимальный index.html: использует переменные, которые рендерятся в коде
    index_html.write_text(
        """<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Index</title></head>
  <body>
    {% if error %}
      <div id="error">{{ error }}</div>
    {% endif %}

    <div id="models">
      {% for key in model_urls.keys() %}
        <span class="model">{{ key }}</span>
      {% endfor %}
    </div>

    <div id="result">
      {% if result %}
        <pre>{{ result }}</pre>
      {% endif %}
    </div>

    <form method="post">
      <input type="hidden" id="model-input" name="model" value="{{ selected_model or default_model or (model_urls.keys()|first) }}">
      <select name="mode">
        {% for m in modes %}
          <option value="{{ m }}" {% if m == mode %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>
      <textarea name="message">{{ original_text }}</textarea>
      <button type="submit">Analyze</button>
    </form>
  </body>
</html>"""
    )

    # Минимальный error.html
    error_html.write_text(
        """<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Error</title></head>
  <body>
    <h1>Error</h1>
    <p id="error">{{ error }}</p>
  </body>
</html>"""
    )

    # 3) Импортируем приложение после установки env
    endpoints = importlib.import_module("app.endpoints")
    settings = importlib.import_module("app.settings")

    # 4) Подменяем templates в модуле endpoints на Jinja2Templates с нашей временной папкой
    monkeypatch.setattr(endpoints, "templates", Jinja2Templates(directory=str(templates_dir)), raising=True)

    # 5) Подменяем настройки модели/режимов (если нужно)
    monkeypatch.setattr(settings, "model_urls", {
        "Mistral": "mistralai/devstral-2512:free",
        "Chimera": "tngtech/tng-r1t-chimera:free"
    }, raising=False)
    monkeypatch.setattr(settings, "modes", ["Labels", "Summary"], raising=False)

    # 6) Подменяем провайдера на объект с асинхронным методом analyze
    class FakeProvider:
        async def analyze(self, message: str, mode: str, model: str):
            return f"ANALYSIS: model={model}, mode={mode}, text={message[:20]}"

    monkeypatch.setattr(endpoints, "provider", FakeProvider(), raising=True)

    # 7) Создаём TestClient и возвращаем
    client = TestClient(endpoints.app)
    yield client


def test_get_index_contains_models(client):
    """GET / должен вернуть страницу с селектором моделей"""
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.text
    assert "Mistral" in text
    assert "Chimera" in text


def test_post_valid_form_returns_result(client):
    """POST с корректными полями возвращает страницу с результатом анализа"""
    data = {
        "message": "Hello world, test message",
        "mode": "Summary",
        "model": "Mistral"
    }
    resp = client.post("/", data=data)
    assert resp.status_code == 200
    assert "ANALYSIS: model=Mistral" in resp.text


def test_post_empty_message_returns_422_and_friendly_error(client):
    """Если message пустой, ожидаем 422 и дружелюбное сообщение об ошибке"""
    data = {
        "message": "",
        "mode": "Summary",
        "model": "Mistral"
    }
    resp = client.post("/", data=data)
    assert resp.status_code == 422
    assert ("empty" in resp.text) or ("Введите текст" in resp.text) or ("Please" in resp.text)


def test_post_invalid_model_triggers_validation_and_returns_422(client):
    """Если передан недопустимый model, PromptRequest должен провалиться и вернуть 422"""
    data = {
        "message": "Some text",
        "mode": "Summary",
        "model": "invalid-model-name"
    }
    resp = client.post("/", data=data)
    assert resp.status_code == 422
    assert "invalid" in resp.text.lower() or "ошиб" in resp.text.lower() or "Prompt" in resp.text
