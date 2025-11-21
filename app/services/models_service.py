import uuid
from typing import List, Optional, Generator
from llama_cpp import Llama
from pathlib import Path
import logging
import threading

from app.models.schemas import Message, ToolDefinition, ChatCompletionResponse
from app.core.config import settings
from app.exceptions import ServiceUnavailableError
from .generators.non_stream_generator import NonStreamResponseGenerator
from .generators.stream_generator import StreamResponseGenerator

logger = logging.getLogger(__name__)


class LlamaService:
    """Сервис для работы с моделью LLM (синхронная версия)"""

    _instance = None
    _lock = threading.Lock()  # Блокировка для синглтона

    def __init__(self):
        self._model = None
        self.model_name = settings.model.name
        self._is_initialized = False
        self._is_processing = False  # Флаг занятости
        self._processing_lock = threading.Lock()  # Блокировка для обработки запросов
        
        # Синхронные генераторы
        self.non_stream_generator = NonStreamResponseGenerator(
            self.model_name, self._create_completion
        )
        self.stream_generator = StreamResponseGenerator(
            self.model_name, self._create_completion_stream
        )

    @classmethod
    def get_instance(cls) -> 'LlamaService':
        """Получение экземпляра сервиса"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        """Инициализация модели (синхронная)"""
        if self._is_initialized:
            return

        try:
            logger.info("Initializing Llama model...")
            
            model_params = {
                "model_path": settings.model.path,
                "n_thread": settings.model.n_thread,
                "n_threads_batch": settings.model.n_threads_batch,
                "n_batch": settings.model.n_batch,
                "n_ubatch": settings.model.n_ubatch,
                "n_ctx": settings.model.ctx_size,
                "n_gpu_layers": settings.model.gpu_layers,
                "verbose": settings.model.verbose,
            }

            if hasattr(settings.model, 'chat_format') and settings.model.chat_format:
                model_params["chat_format"] = settings.model.chat_format

            chat_handler = self._load_chat_handler()
            if chat_handler:
                model_params["chat_handler"] = chat_handler

            logger.info(f"Creating model with params: { {k: v for k, v in model_params.items() if k != 'model_path'} }")
            
            self._model = Llama(**model_params)
            self._is_initialized = True
            logger.info("LlamaService initialized successfully")
            
        except Exception as e:
            logger.error(f"LlamaService initialization failed: {e}")
            self._is_initialized = False
            raise ServiceUnavailableError(f"Service initialization failed: {str(e)}")

    def _load_chat_handler(self):
        """Загрузка chat handler из шаблона"""
        chat_template_path = getattr(settings.model, 'chat_template_path', None)
        if not chat_template_path:
            return None

        template_path = Path(chat_template_path)
        if not template_path.exists():
            logger.warning(f"Chat template not found at: {template_path}")
            return None

        try:
            logger.info(f"Loading chat template from: {template_path}")
            with open(template_path, 'r', encoding='utf-8') as f:
                template_str = f.read()

            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            chat_formatter = Jinja2ChatFormatter(
                template=template_str,
                eos_token="<|im_end|>",
                bos_token="<|im_start|>",
                add_generation_prompt=True
            ).to_chat_handler()

            logger.info("Chat handler loaded successfully")
            return chat_formatter
        except Exception as e:
            logger.error(f"Failed to load chat template: {e}")
            return None

    def _create_completion(self, session_id: str, **kwargs) -> dict:
        """Создание non-stream completion (синхронное)"""
        return self._model.create_chat_completion(**kwargs)

    def _create_completion_stream(self, session_id: str, **kwargs) -> Generator[dict, None, None]:
        """Создание stream completion (синхронный генератор)"""
        kwargs['stream'] = True
        return self._model.create_chat_completion(**kwargs)
    
    def _acquire_processing(self) -> bool:
        """Попытка занять модель для обработки"""
        acquired = self._processing_lock.acquire(blocking=False)
        if acquired:
            self._is_processing = True
            logger.info("Model acquired for processing")
        else:
            logger.warning("Model is busy, cannot acquire for processing")
        return acquired
    
    def _release_processing(self) -> None:
        """Освобождение модели после обработки"""
        self._is_processing = False
        self._processing_lock.release()
        logger.info("Model released after processing")

    def generate_response_non_stream(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            frequency_penalty: float,
            presence_penalty: float,
            tools: Optional[List[ToolDefinition]] = None,
            session_id: str = None
    ) -> ChatCompletionResponse:
        """Не-потоковая генерация ответа (синхронная)"""
        if session_id is None:
            session_id = f"non_stream_{uuid.uuid4().hex}"

        # Пытаемся занять модель
        if not self._acquire_processing():
            raise ServiceUnavailableError("Service is busy processing another request")

        try:
            logger.info(f"Starting non-stream generation for session {session_id}")
            return self.non_stream_generator.generate(
                messages, temperature, max_tokens, frequency_penalty, presence_penalty, tools, session_id
            )
        except ServiceUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Non-stream generation error: {str(e)}")
            raise ServiceUnavailableError(f"Generation error: {str(e)}")
        finally:
            # Очищаем кэш
            self.reset_cache()
            # Всегда освобождаем модель
            self._release_processing()

    def generate_response_stream(
            self,
            messages: List[Message],
            temperature: float,
            max_tokens: int,
            frequency_penalty: float,
            presence_penalty: float,
            tools: Optional[List[ToolDefinition]] = None,
            session_id: str = None
    ) -> Generator[str, None, None]:
        """Потоковая генерация ответа (синхронный генератор)"""
        if session_id is None:
            session_id = f"stream_{uuid.uuid4().hex}"

        # Пытаемся занять модель
        if not self._acquire_processing():
            raise ServiceUnavailableError("Service is busy processing another request")

        try:
            logger.info(f"Starting stream generation for session {session_id}")
            
            # Генерируем потоковые чанки
            for chunk in self.stream_generator.generate(
                    messages, temperature, max_tokens, frequency_penalty, presence_penalty, tools, session_id
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}")
            # В потоковом режиме мы не можем вернуть ошибку через исключение,
            # поэтому генерируем error chunk
            yield self.stream_generator._create_error_chunk(
                f"chatcmpl-{uuid.uuid4().hex}", 
                f"Generation error: {str(e)}"
            )
            raise
        finally:
            # Очищаем кэш
            self.reset_cache()
            # Всегда освобождаем модель
            self._release_processing()

    def reset_cache(self) -> None:
        """Сброс KV-кэша модели"""
        if self._model:
            try:
                self._model.reset()
                logger.debug("KV cache reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset KV cache: {e}")

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self._model:
            try:
                del self._model
            except Exception as e:
                logger.error(f"Model cleanup error: {e}")
            finally:
                self._model = None
                self._is_initialized = False
        logger.info("LlamaService cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self._model is not None
    
    @property
    def is_available(self) -> bool:
        """Проверка доступности модели для обработки запросов"""
        return self._is_initialized and not self._is_processing