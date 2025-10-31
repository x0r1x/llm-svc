from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from app.models.schemas import Message, ToolDefinition
from app.services.tool_call_processor import ToolCallProcessor
import logging

logger = logging.getLogger(__name__)


class BaseResponseGenerator(ABC):
    """Базовый класс генераторов ответов"""

    def __init__(self, model_name: str, completion_caller: Callable):
        self.model_name = model_name
        self.tool_processor = ToolCallProcessor()
        self._completion_caller = completion_caller

    @abstractmethod
    async def generate(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            frequency_penalty: float,
            presence_penalty: float,
            tools: Optional[List[ToolDefinition]] = None,
            session_id: str = None
    ):
        """Абстрактный метод генерации"""
        pass

    def _should_use_tools(self, tools: Optional[List[ToolDefinition]]) -> bool:
        """Проверка необходимости использования инструментов"""
        return tools is not None and len(tools) > 0

    def _convert_messages_to_dict(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Конвертация сообщений в словари"""
        result = []
        for message in messages:
            message_dict = {
                "role": message.role.value,
                "content": message.content
            }
            # Добавляем опциональные поля если они есть
            if message.name is not None:
                message_dict["name"] = message.name
            if message.tool_calls is not None:
                message_dict["tool_calls"] = [
                    tool_call.model_dump() for tool_call in message.tool_calls
                ]
            result.append(message_dict)
        return result

    def _prepare_generation_params(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            frequency_penalty: float,
            presence_penalty: float,
            tools: Optional[List[ToolDefinition]]
    ) -> Dict[str, Any]:
        """Подготовка параметров генерации"""
        logger.info(f"Preparing generation params for {self.__class__.__name__}")

        # Конвертируем сообщения в словари
        messages_dict = self._convert_messages_to_dict(messages)
        logger.info(f"Messages dict: {messages_dict}")

        params = {
            "messages": messages_dict,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # Добавляем инструменты, если они переданы
        if self._should_use_tools(tools):
            params["tools"] = [tool.model_dump() for tool in tools]
            params["tool_choice"] = "auto"
            logger.info(f"Tools enabled: {[tool.function.name for tool in tools]}")

        logger.info(f"Completion params: {params}")
        return params