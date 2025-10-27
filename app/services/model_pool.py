# app/services/model_pool.py
import asyncio
from typing import List
from app.exceptions import ServiceUnavailableError, ModelNotLoadedError, PoolExhaustedError
from .model_context import ModelContext
import logging

logger = logging.getLogger(__name__)


class ModelPool:
    """Пул моделей для обработки запросов с отслеживанием активных запросов"""

    def __init__(self, pool_size: int = 1):
        self.pool_size = pool_size
        self._contexts: List[ModelContext] = []
        self._available: List[ModelContext] = []
        self._in_use: set[ModelContext] = set()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_failed = False
        self._active_requests = 0  # Счетчик активных запросов
        self._max_active_requests = pool_size  # Максимальное количество одновременных запросов

    async def initialize(self) -> None:
        async with self._lock:
            logger.info(f"Initializing model pool with {self.pool_size} instances")

            try:
                self._contexts = []
                self._available = []

                # Синхронная инициализация всех контекстов
                initialization_tasks = []
                for i in range(self.pool_size):
                    ctx = ModelContext()
                    self._contexts.append(ctx)
                    initialization_tasks.append(ctx.initialize())

                # Ждем инициализации всех контекстов
                results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

                # Проверяем результаты инициализации
                successful_init = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to initialize context {i}: {result}")
                    else:
                        self._available.append(self._contexts[i])
                        successful_init += 1

                if successful_init == 0:
                    self._initialization_failed = True
                    raise ServiceUnavailableError("No models could be initialized")

                self._initialized = True
                self._max_active_requests = successful_init  # Обновляем максимум на основе успешных инициализаций
                logger.info(f"Model pool initialized with {successful_init}/{self.pool_size} instances")

            except Exception as e:
                self._initialization_failed = True
                logger.error(f"Model pool initialization failed: {e}")
                raise ServiceUnavailableError(f"Service initialization failed: {str(e)}")

    async def acquire(self) -> ModelContext:
        """Получение модели из пула с проверкой лимита одновременных запросов"""
        async with self._lock:
            if not self._initialized or self._initialization_failed:
                raise ServiceUnavailableError("Service is not ready")

            # Проверяем не превышен ли лимит одновременных запросов
            if self._active_requests >= self._max_active_requests:
                raise PoolExhaustedError(
                    f"Service is busy. Maximum concurrent requests: {self._max_active_requests}, "
                    f"Current active requests: {self._active_requests}, "
                    f"Total instances: {self.total_count}"
                )

            if not self._available:
                raise PoolExhaustedError("No available models in pool")

            context = self._available.pop()
            self._in_use.add(context)
            self._active_requests += 1
            logger.debug(f"Acquired model context. Active requests: {self._active_requests}")
            return context

    async def release(self, context: ModelContext) -> None:
        """Возврат модели в пул и уменьшение счетчика активных запросов"""
        async with self._lock:
            if context in self._in_use:
                self._in_use.remove(context)
                self._active_requests = max(0, self._active_requests - 1)
                # Проверяем, что контекст все еще валиден перед возвратом в пул
                if context.is_ready:
                    self._available.append(context)
                else:
                    logger.warning("Released context is not ready, it will not be returned to pool")

            logger.debug(f"Released model context. Active requests: {self._active_requests}")

    async def cleanup(self) -> None:
        """Очистка пула"""
        async with self._lock:
            cleanup_tasks = [ctx.cleanup() for ctx in self._contexts]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self._contexts.clear()
            self._available.clear()
            self._in_use.clear()
            self._active_requests = 0
            self._initialized = False
            self._initialization_failed = False

    @property
    def is_ready(self) -> bool:
        """Проверка готовности пула - есть ли доступные инстансы"""
        return self._initialized and not self._initialization_failed and len(self._available) > 0

    @property
    def available_count(self) -> int:
        """Количество доступных моделей"""
        return len(self._available)

    @property
    def total_count(self) -> int:
        """Общее количество моделей"""
        return len(self._contexts)

    @property
    def active_requests_count(self) -> int:
        """Количество активных запросов"""
        return self._active_requests

    @property
    def max_concurrent_requests(self) -> int:
        """Максимальное количество одновременных запросов"""
        return self._max_active_requests