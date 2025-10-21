import time
import uuid
from typing import List, Optional
from app.models.schemas import (
    Message, ToolDefinition, ChatCompletionResponse,
    UsageInfo, MessageRole, ChatCompletionResponseChoice
)
from .base_generator import BaseResponseGenerator
import logging

logger = logging.getLogger(__name__)


class NonStreamResponseGenerator(BaseResponseGenerator):
    """Генератор не-потоковых ответов"""

    def __init__(self, model_name: str, completion_caller):
        super().__init__(model_name)
        self._completion_caller = completion_caller

    async def generate(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None
    ) -> ChatCompletionResponse:
        """Генерация не-потокового ответа"""
        logger.info("Starting non-stream generation")

        start_time = time.time()
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        try:
            # Подготавливаем параметры
            params = self._prepare_generation_params(
                messages, temperature, max_tokens, tools
            )

            # Вызываем completion
            response = await self._completion_caller(**params, stream=False)

            # Обрабатываем ответ
            processed_response = self._process_response(response, tools)

            processing_time = time.time() - start_time
            logger.info(f"Non-stream response generated in {processing_time:.2f}s")

            return processed_response

        except Exception as e:
            logger.error(f"Non-stream generation error: {str(e)}", exc_info=True)
            return self._create_error_response(response_id, str(e))

    def _process_response(
            self,
            response: dict,
            tools: Optional[List[ToolDefinition]]
    ) -> ChatCompletionResponse:
        """Обрабатывает raw response от модели"""
        content = response['choices'][0]['message'].get('content', '')
        finish_reason = response['choices'][0].get('finish_reason', 'stop')

        # Обрабатываем tool calls если есть инструменты
        tool_calls = None
        if self._should_use_tools(tools) and content:
            cleaned_content, tool_calls = self.tool_processor.extract_tool_calls(content)
            if tool_calls:
                content = cleaned_content
                finish_reason = "tool_calls"

        return ChatCompletionResponse(
            id=response.get('id', f"chatcmpl-{uuid.uuid4().hex}"),
            object="chat.completion",
            created=response.get('created', int(time.time())),
            model=response.get('model', self.model_name),
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        tool_calls=tool_calls
                    ),
                    finish_reason=finish_reason
                )],
            usage=UsageInfo(
                prompt_tokens=response.get('usage', {}).get('prompt_tokens', 0),
                completion_tokens=response.get('usage', {}).get('completion_tokens', 0),
                total_tokens=response.get('usage', {}).get('total_tokens', 0)
            )
        )

    def _create_error_response(self, response_id: str, error_message: str) -> ChatCompletionResponse:
        """Создает ответ с ошибкой"""
        return ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Error: {error_message}"
                    ),
                    finish_reason="stop"
                )],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )