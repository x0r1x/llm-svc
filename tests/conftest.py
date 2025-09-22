import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import os


@pytest.fixture(scope="session", autouse=True)
def test_config():
    """Установка тестовой конфигурации."""
    os.environ["CONFIG_PATH"] = "config/config.test.yml"
    yield
    if "CONFIG_PATH" in os.environ:
        del os.environ["CONFIG_PATH"]


# @pytest.fixture(scope="session", autouse=True)
# def mock_llama_cpp():
#     """Мокирование llama_cpp на уровне сессии."""
#     with patch('llama_cpp.Llama') as mock_llama:
#         mock_instance = MagicMock()
#         mock_instance.create_chat_completion.return_value = {
#             "choices": [
#                 {
#                     "message": {
#                         "role": "assistant",
#                         "content": "Test response from mock model"
#                     },
#                     "finish_reason": "stop"
#                 }
#             ],
#             "usage": {
#                 "prompt_tokens": 10,
#                 "completion_tokens": 20,
#                 "total_tokens": 30
#             }
#         }
#         mock_llama.return_value = mock_instance
#         yield mock_instance


@pytest.fixture(scope="function")
def test_app():
    """Создание тестового приложения для каждого теста."""
    from app.main import create_application
    app = create_application()
    yield app


@pytest.fixture(scope="function")
def test_client(test_app):
    """Создание тестового клиента для каждого теста."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="function")
def mock_llama_handler():
    """Мок LlamaHandler для каждого теста."""
    with patch('app.services.llama_handler.LlamaHandler') as mock:
        instance = mock.return_value
        instance.is_loaded.return_value = True
        instance.model_name = "test-model"

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

        yield instance


@pytest.fixture(scope="function", autouse=True)
def override_dependencies(test_app, mock_llama_handler):
    """Автоматическое переопределение зависимостей для всех тестов."""
    from app.dependencies import get_llama_handler

    async def mock_get_llama_handler():
        return mock_llama_handler

    test_app.dependency_overrides[get_llama_handler] = mock_get_llama_handler
    yield
    test_app.dependency_overrides.clear()
