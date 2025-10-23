import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from app.core.config import settings
from app.models.schemas import Message, ToolDefinition, ChatCompletionResponse
from .generators.non_stream_generator import NonStreamResponseGenerator
from .generators.stream_generator import StreamResponseGenerator

logger = logging.getLogger(__name__)


class LlamaHandler:
    _instance = None

    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_path = settings.model.path
        self.model_name = settings.model.name
        self.chat_template_path = settings.model.chat_template_path
        self.chat_format = settings.model.chat_format
        self.n_ctx = settings.model.ctx_size
        self.n_gpu_layers = settings.model.gpu_layers
        self.verbose = settings.model.verbose
        self.is_initialized = False

        # Инициализируем генераторы
        self.non_stream_generator = NonStreamResponseGenerator(
            self.model_name, self._create_completion
        )
        self.stream_generator = StreamResponseGenerator(
            self.model_name, self._create_completion_stream
        )

    @classmethod
    def get_instance(cls):
        """Получение экземпляра синглтона."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_chat_handler(self, chat_template_path: str) -> Any:
        """
        Загрузка и настройка chat handler из шаблона.
        Отдельная функция для соблюдения принципа единственной ответственности.
        """
        if not chat_template_path:
            return

        template_path = Path(chat_template_path)

        if not template_path.exists():
            raise FileNotFoundError(f"Chat template not found at: {template_path}")

        logger.info(f"Loading chat template from: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()

        custom_chat_formatter = Jinja2ChatFormatter(
            template=template_str,
            eos_token="<|im_end|>",
            bos_token="<|im_start|>",
            add_generation_prompt=True
        ).to_chat_handler()

        logger.info("Chat handler loaded successfully")
        return custom_chat_formatter

    async def initialize(self):
        """Асинхронная инициализация модели."""
        if self.is_initialized:
            return

        try:
            logger.info(f"Loading model from {self.model_path}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=self.model_path,
                    n_threads=7,
                    n_threads_batch=7,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    chat_handler=self._load_chat_handler(self.chat_template_path), #custom_chat_formatter,
                    chat_format=self.chat_format,
                    verbose=self.verbose,
                )
            )
            self.is_initialized = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель."""
        return self.is_initialized and self.model is not None

    async def cleanup(self):
        """Очистка ресурсов модели."""
        if self.model:
            del self.model
        self.model = None
        self.is_initialized = False
        logger.info("Model resources cleaned up")

    async def _run_in_executor(self, func: Callable, *args, **kwargs):
        """Утилита для запуска блокирующих операций в executor'е."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _create_completion(self, **kwargs) -> Dict[str, Any]:
        """Создание non-stream completion"""
        if not self.model:
            await self.initialize()

        # Убираем stream параметр для non-stream вызовов
        kwargs.pop('stream', None)
        return await self._run_in_executor(
            self.model.create_chat_completion, **kwargs
        )

    async def _create_completion_stream(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Создание stream completion"""
        if not self.model:
            await self.initialize()

        # Убедимся, что stream=True
        kwargs['stream'] = True

        # Для потокового режима возвращаем генератор
        stream_result = await self._run_in_executor(
            self.model.create_chat_completion, **kwargs
        )

        # Обрабатываем разные типы возвращаемых значений
        if hasattr(stream_result, '__aiter__'):
            # Асинхронный генератор
            async for chunk in stream_result:
                yield chunk
        else:
            # Синхронный генератор
            for chunk in stream_result:
                yield chunk

    async def generate_response_non_stream(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None
    ) -> ChatCompletionResponse:
        """Не-потоковая генерация через соответствующий генератор"""
        return await self.non_stream_generator.generate(
            messages, temperature, max_tokens, tools
        )

    async def generate_response_stream(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            tools: Optional[List[ToolDefinition]] = None
    ) -> AsyncGenerator[str, None]:
        """Потоковая генерация через соответствующий генератор"""
        async for chunk in self.stream_generator.generate(
                messages, temperature, max_tokens, tools
        ):
            yield chunk