import json
import time
import uuid
from typing import List, Optional, AsyncGenerator, Dict, Any
from app.models.schemas import Message, ToolDefinition
from .base_generator import BaseResponseGenerator
import logging

logger = logging.getLogger(__name__)

class StreamResponseGenerator(BaseResponseGenerator):
    """Генератор потоковых ответов"""

    def __init__(self, model_name: str, completion_caller):
        super().__init__(model_name)
        self._completion_caller = completion_caller

    async def generate(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None
    ) -> AsyncGenerator[str, None]:
        """Генерация потокового ответа"""
        logger.info("Starting stream generation")

        start_time = time.time()
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        try:
            # Подготавливаем параметры
            params = self._prepare_generation_params(
                messages, temperature, max_tokens, tools
            )

            # Получаем потоковый результат
            stream_generator = self._completion_caller(**params)

            # Обрабатываем поток
            async for chunk in self._process_stream(stream_generator, response_id):
                yield chunk

            processing_time = time.time() - start_time
            logger.info(f"Stream response generated in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}", exc_info=True)
            yield self._create_error_chunk(response_id, str(e))

    async def _process_stream(
            self,
            stream_generator: AsyncGenerator[Dict[str, Any], None],
            response_id: str
    ) -> AsyncGenerator[str, None]:
        """Обрабатывает потоковый результат"""
        try:
            async for chunk in stream_generator:
                processed_chunk = self._process_chunk(chunk, response_id)
                if processed_chunk:
                    yield processed_chunk

            # Завершающий чанк
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            yield self._create_error_chunk(response_id, str(e))

    def _process_chunk(self, chunk: Dict[str, Any], response_id: str) -> Optional[str]:
        """Обрабатывает отдельный чанк"""
        if not chunk:
            return None

        # Обновляем метаданные чанка
        chunk['id'] = response_id
        chunk['model'] = self.model_name

        return f"data: {json.dumps(chunk)}\n\n"

    def _create_error_chunk(self, response_id: str, error_message: str) -> str:
        """Создает чанк с ошибкой"""
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Error: {error_message}"},
                    "finish_reason": "error"
                }
            ]
        }
        return f"data: {json.dumps(error_chunk)}\n\n"