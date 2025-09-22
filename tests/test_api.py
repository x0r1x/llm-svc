import pytest
from fastapi import status


def test_health_check(test_client):
    """Тест эндпоинта health check."""
    response = test_client.get("/v1/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_list_models(test_client):
    """Тест эндпоинта list models."""
    response = test_client.get("/v1/models")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["id"] == "test-model"


def test_chat_completion(test_client):
    """Тест эндпоинта chat/completions."""
    request_data = {
        "model": "test-model",
        "stream": "false",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    response = test_client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["model"] == "test-model"
    assert response.json()["choices"][0]["message"]["content"] == "Test response from mock handler"
