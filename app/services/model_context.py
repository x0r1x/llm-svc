import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelContext:
    """Контекст управления моделью"""

    def __init__(self):
        self._model: Optional[Llama] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Инициализация модели"""
        async with self._lock:
            if self._model is not None:
                return

            logger.info("Initializing model...")
            try:
                # Создаем модель в executor'е
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    self._create_model
                )
                logger.info("Model initialized successfully")
            except Exception as e:
                logger.error(f"Model initialization failed: {e}")
                self._model = None
                raise

    async def generate(self, **kwargs) -> Any:
        """Генерация текста"""
        if self._model is None:
            raise RuntimeError("Model is not initialized")

        async with self._lock:
            try:
                return await self._run_in_executor(self._model.create_chat_completion, **kwargs)
            except asyncio.TimeoutError:
                logger.error("Generation timeout exceeded")
                raise RuntimeError("Generation timeout")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        async with self._lock:
            if self._model:
                try:
                    del self._model
                except Exception as e:
                    logger.error(f"Model cleanup error: {e}")
                finally:
                    self._model = None

    def _load_chat_handler(self) -> Any:
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

    def _create_model(self) -> Llama:
        """Создание экземпляра модели"""
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

        # Добавляем chat_format если указан
        if hasattr(settings.model, 'chat_format') and settings.model.chat_format:
            model_params["chat_format"] = settings.model.chat_format

        # Загружаем chat handler если указан путь к шаблону
        # chat_handler = self._load_chat_handler()
        # if chat_handler:
        #     model_params["chat_handler"] = chat_handler

        logger.info(f"Creating model with params: { {k: v for k, v in model_params.items() if k != 'model_path'} }")
        return Llama(**model_params)

    async def _run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Запуск в executor'е"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @property
    def is_ready(self) -> bool:
        """Проверка готовности модели"""
        return self._model is not None