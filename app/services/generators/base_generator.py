from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.schemas import Message, ToolDefinition
from app.services.tool_call_processor import ToolCallProcessor
import logging

logger = logging.getLogger(__name__)


class BaseResponseGenerator(ABC):
    """Абстрактный базовый класс для всех генераторов ответов"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tool_processor = ToolCallProcessor()

    @abstractmethod
    async def generate(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None
    ):
        """Абстрактный метод генерации ответа"""
        pass

    def _should_use_tools(self, tools: Optional[List[ToolDefinition]]) -> bool:
        """Определяет, нужно ли использовать инструменты"""
        return tools is not None and len(tools) > 0

    def _prepare_generation_params(self, messages, temperature, max_tokens, tools):
        """Подготавливает параметры для генерации (общая логика)"""
        logger.info(f"Preparing generation params for {self.__class__.__name__}")

        params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Преобразуем инструменты в словари, если они переданы
        tools_dict = None
        if tools:
            tools_dict = [tool.model_dump() for tool in tools]

        logger.info(f"Tools: {tools_dict}")

        # Добавляем инструменты, если они переданы
        if self._should_use_tools(tools):
            params["tools"] = tools_dict
            params["tool_choice"] = "auto"
            logger.info(f"Tools enabled: {[tool.function.name for tool in tools]}")

        logger.info(f"Completion params: {params}")

        return params