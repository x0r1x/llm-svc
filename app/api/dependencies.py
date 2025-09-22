from fastapi import Depends, HTTPException, status
from app.services.llama_handler import LlamaHandler
from app.dependencies import get_llama_handler

async def get_llama_service(
    llama_handler: LlamaHandler = Depends(get_llama_handler),
) -> LlamaHandler:
    """Зависимость для получения сервиса LLM."""
    if not llama_handler.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM model is not loaded",
        )
    return llama_handler