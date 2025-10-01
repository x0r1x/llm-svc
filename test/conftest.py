import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import os


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Настройка тестового окружения."""
    # Устанавливаем тестовую конфигурацию ДО импорта любых модулей приложения
    os.environ["CONFIG_PATH"] = "config/config.test.yml"

    # Сбрасываем настройки перед началом тестов
    from app.core.config import reset_settings
    settings = reset_settings()

    yield settings

    # Очистка после тестов
    if "CONFIG_PATH" in os.environ:
        del os.environ["CONFIG_PATH"]


@pytest.fixture(scope="session", autouse=True)
def global_mock_llama():
    """Глобальный мок LlamaHandler на уровне сессии."""
    with patch('app.dependencies.LlamaHandler') as mock_handler:
        # Создаем асинхронный мок для экземпляра
        instance = MagicMock()
        instance.is_loaded.return_value = True
        instance.model_name = "test-model"
        instance.initialized = True

        # Мок для асинхронного метода generate_response, принимаем любые аргументы
        async def generate_response(*args, **kwargs):
            return {
                "id": "test-id",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Test response from mock handler"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }

        instance.generate_response = AsyncMock(side_effect=generate_response)
        instance.initialize = AsyncMock()
        instance.cleanup = AsyncMock()

        mock_handler.return_value = instance
        mock_handler.get_instance.return_value = instance

        yield instance

@pytest.fixture(scope="session")
def test_app():
    """Создание тестового приложения с переопределенными зависимостями."""
    from app.main import create_application
    app = create_application()
    yield app

@pytest.fixture
def test_client(test_app):
    """Создание тестового клиента."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def auth_enabled_config():
    """Временное включение аутентификации для тестов."""
    from app.core.config import get_settings

    settings = get_settings()

    # Сохраняем оригинальные значения
    original_enabled = settings.security.enabled
    original_api_key = settings.security.api_key

    # Включаем аутентификацию и устанавливаем тестовый ключ
    settings.security.enabled = True
    settings.security.api_key = "test-api-key"

    yield

    # Восстанавливаем оригинальные значения
    settings.security.enabled = original_enabled
    settings.security.api_key = original_api_key


@pytest.fixture
def auth_enabled_no_key_config():
    """Временное включение аутентификации для тестов."""
    from app.core.config import get_settings

    settings = get_settings()

    # Сохраняем оригинальные значения
    original_enabled = settings.security.enabled
    original_api_key = settings.security.api_key

    # Включаем аутентификацию и сбрасываем ключ
    settings.security.enabled = True
    settings.security.api_key = None

    yield

    # Восстанавливаем оригинальные значения
    settings.security.enabled = original_enabled
    settings.security.api_key = original_api_key