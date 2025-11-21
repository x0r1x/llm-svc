from fastapi import Depends, HTTPException, status
from app.services.models_service import LlamaService
from app.dependencies import get_llama_service
from app.core.security import verify_api_key
import logging

logger = logging.getLogger(__name__)


async def get_llama_service_handler_non_connection_pool(
        llama_service: LlamaService = Depends(get_llama_service),
) -> LlamaService:
    """Зависимость для проверки доступности сервиса LLM."""
    if not llama_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM model is not loaded",
        )
        
    return llama_service


async def get_llama_service_handler(
        llama_service: LlamaService = Depends(get_llama_service),
) -> LlamaService:
    """Зависимость для проверки доступности сервиса LLM."""
    if not llama_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM model is not loaded",
        )
        
    # Проверяем, не занята ли модель другим запросом
    if not llama_service.is_available:
        logger.warning("Service is busy processing another request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is busy processing another request. Please try again later."
        )
    
    return llama_service


async def require_api_key_handler(
    api_key_verified: bool = Depends(verify_api_key)
) -> bool:
    """Зависимость для проверки API ключа."""
    if not api_key_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return True