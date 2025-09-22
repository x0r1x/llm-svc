from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ModelsList
from app.services.llama_handler import LlamaHandler
from app.api.dependencies import get_llama_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat/completions")
async def chat_completion(
    request: ChatRequest,
    llama_service: LlamaHandler = Depends(get_llama_service),
):
    """
    Эндпоинт совместимый с OpenAI API для обработки запросов чата.
    Поддерживает как обычные запросы, так и потоковую передачу.
    """
    logger.info(f"Chat request Model: {request.model}, "
                f"Messages: {len(request.messages)}, "
                f"Temperature: {request.temperature}, "
                f"Stream: {request.stream}")
    try:
        if request.stream:
            # Потоковый режим - возвращаем StreamingResponse
            response_generator = await llama_service.generate_response(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            return StreamingResponse(
                response_generator,
                media_type="text/event-stream"
            )
        else:
            # Обычный режим - возвращаем обычный ответ
            response = await llama_service.generate_response(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            logger.info("Chat request: Response generated successfully")
            return response
    except Exception as e:
        logger.error(f"Chat request Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check(
    llama_service: LlamaHandler = Depends(get_llama_service),
):
    """Проверка статуса сервиса и загрузки модели."""
    logger.info("Health requested")
    return HealthResponse(
        status="healthy",
        model_loaded=llama_service.is_loaded(),
        model_name=llama_service.model_name if llama_service.is_loaded() else None,
    )

@router.get("/models", response_model=ModelsList)
async def list_models(
    llama_service: LlamaHandler = Depends(get_llama_service),
):
    """Список доступных моделей."""
    logger.info("Models list requested")
    return ModelsList(
        data=[
            {
                "id": llama_service.model_name,
                "object": "model",
                "owned_by": "local",
                "permissions": []
            }
        ]
    )
