# app/dependencies.py
from app.services.models_service import LlamaService

def get_llama_service() -> LlamaService:
    """Зависимость для получения сервиса LLM (синхронная)"""
    return LlamaService.get_instance()

get_llama_handler = get_llama_service

def cleanup_llama_service() -> None:
    """Очистка сервиса LLM"""
    service = LlamaService.get_instance()
    service.cleanup()