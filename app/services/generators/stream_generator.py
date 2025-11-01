import json
import time
import uuid
from typing import List, Optional, AsyncGenerator, Dict, Any

from app.models.schemas import (
    Message,
    ToolDefinition,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
)
from .base_generator import BaseResponseGenerator
import logging

logger = logging.getLogger(__name__)


class StreamResponseGenerator(BaseResponseGenerator):
    """Генератор потоковых ответов"""

    async def generate(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
        tools: Optional[List[ToolDefinition]] = None,
        session_id: str = None,
    ) -> AsyncGenerator[str, None]:
        """Генерация потокового ответа"""
        logger.info("Starting stream generation")

        # Проверяем, не используются ли инструменты в потоковом режиме
        if self._should_use_tools(tools):
            yield self._create_error_chunk(
                f"chatcmpl-{uuid.uuid4().hex}",
                "Streaming is not supported when tools are provided. Use stream=false.",
            )
            return

        start_time = time.time()
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        try:
            # Подготавливаем параметры (без инструментов для потокового режима)
            params = self._prepare_generation_params(
                messages, temperature, max_tokens, frequency_penalty, presence_penalty, None
            )

            # Получаем потоковый результат с session_id
            async for chunk in self._completion_caller(session_id, **params):
                processed_chunk = self._process_chunk(chunk, response_id)
                if processed_chunk:
                    yield processed_chunk

            # Завершающий чанк
            yield "data: [DONE]\n\n"

            processing_time = time.time() - start_time
            logger.info(f"Stream response generated in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}", exc_info=True)
            yield self._create_error_chunk(response_id, str(e))

    def _process_chunk(self, chunk: Dict[str, Any], response_id: str) -> Optional[str]:
        """Обрабатывает отдельный чанк - прямая передача данных"""
        if not chunk:
            return None

        # Обновляем только необходимые метаданные
        chunk["id"] = response_id
        chunk["model"] = self.model_name

        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    def _create_error_chunk(self, response_id: str, error_message: str) -> str:
        """Создает чанк с ошибкой с использованием Pydantic моделей"""
        choice = ChatCompletionChunkChoice(
            index=0,
            delta={"content": f"Error: {error_message}"},
            finish_reason="error",
        )
        error_chunk = ChatCompletionChunk(
            id=response_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
        )
        return f"data: {error_chunk.model_dump_json()}\n\n"
