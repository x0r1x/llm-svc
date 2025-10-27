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

    async def generate(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None,
            session_id: str = None
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

            logger.info(f"Calling model completion for session {session_id}")

            # Вызываем completion с session_id через _completion_caller
            response = await self._completion_caller(session_id, **params)
            logger.info(f"Model response received, processing...")

            # Обрабатываем ответ
            processed_response = self._process_response(response, tools, response_id)

            processing_time = time.time() - start_time
            logger.info(f"Non-stream response generated in {processing_time:.2f}s")

            return processed_response

        except Exception as e:
            logger.error(f"Non-stream generation error: {str(e)}", exc_info=True)
            return self._create_error_response(response_id, str(e))

    def _process_response(
            self,
            response: dict,
            tools: Optional[List[ToolDefinition]],
            response_id: str
    ) -> ChatCompletionResponse:
        """Обрабатывает raw response от модели"""
        if not response or 'choices' not in response or not response['choices']:
            return self._create_error_response(response_id, "Invalid response from model")

        choices_data = response.get('choices', [{}])[0]
        message_data = choices_data.get('message', {})
        finish_reason = choices_data.get('finish_reason', 'stop')
        content = message_data.get('content', '')
        tool_calls = message_data.get('tool_calls')

        # Обрабатываем tool calls если есть инструменты
        if self._should_use_tools(tools) and content:
            cleaned_content, extracted_tool_calls = self.tool_processor.extract_tool_calls(content)
            if extracted_tool_calls:
                content = cleaned_content.strip() if cleaned_content else ''
                tool_calls = extracted_tool_calls
                finish_reason = "tool_calls"
                logger.info(f"Extracted {len(extracted_tool_calls)} tool calls")

        # Создаем сообщение ассистента
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=content if content else None,
            tool_calls=tool_calls
        )

        return ChatCompletionResponse(
            id=response.get('id', response_id),
            object="chat.completion",
            created=response.get('created', int(time.time())),
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=assistant_message,
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