import json
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging.config

from app.core.config import settings
from app.api import router as api_router
from app.dependencies import get_llama_service, cleanup_llama_service
from app.services.nexus_client import download_model_from_nexus_if_needed

from fastapi import Request

# Настройка логирования
def setup_logging():
    """Настройка централизованного логирования для всего приложения"""
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,  # Важно: не отключаем существующие логгеры
        'formatters': {
            'default': {
                'format': settings.logging.format
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': settings.logging.level,
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'handlers': ['console'],
            'level': settings.logging.level
        },
        'loggers': {
            'app': {
                'handlers': ['console'],
                'level': settings.logging.level,
                'propagate': False
            },
            'app.api': {
                'handlers': ['console'],
                'level': settings.logging.level,
                'propagate': False
            },
            'app.services': {
                'handlers': ['console'],
                'level': settings.logging.level,
                'propagate': False
            },
            'app.services.llama_handler': {  # Явно настраиваем логгер для llama_handler
                'handlers': ['console'],
                'level': settings.logging.level,
                'propagate': False
            },
            '__main__': {
                'handlers': ['console'],
                'level': settings.logging.level,
                'propagate': False
            }
        }
    })

# Настраиваем логирование при импорте модуля
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan менеджер для управления жизненным циклом приложения.
    Отделяет инициализацию ресурсов от их использования в зависимостях.
    """
    # Инициализация при запуске
    try:
        logger.info("Starting application initialization...")

        # Загружаем модель из Nexus при необходимости
        if not download_model_from_nexus_if_needed():
            logger.error("Failed to download model from Nexus")
            raise RuntimeError("Failed to download model from Nexus")

        # Предварительная инициализация сервиса LLM
        llama_service = await get_llama_service()
        await llama_service.initialize()

        logger.info("Application started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        # Гарантируем очистку ресурсов при ошибке инициализации
        await cleanup_llama_service()
        raise

    yield  # Приложение работает

    # Очистка при завершении
    try:
        await cleanup_llama_service()
        logger.info("Application shut down gracefully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")


def create_application() -> FastAPI:
    """Создание и настройка FastAPI приложения"""
    application = FastAPI(
        title=settings.app.title,
        description=settings.app.description,
        version=settings.app.version,
        lifespan=lifespan,
        docs_url=settings.server.docs_url,
        redoc_url=settings.server.redoc_url,
    )

    # Настройка CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allowed_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    # Подключение роутеров
    application.include_router(api_router, prefix="/v1")

    return application

app = create_application()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов, ответов и их результатов"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Логирование входящего запроса
    logger.info(f"Request {request_id}: {request.method} {request.url}")

    # Логирование тела запроса для POST, PUT, PATCH методов
    try:
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body.decode())
                    logger.info(f"Request {request_id}: Body: {json.dumps(body_json, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    logger.info(f"Request {request_id}: Body: {body.decode()}")
            # Восстанавливаем тело запроса для дальнейшей обработки
            request._body = body
    except Exception as e:
        logger.error(f"Request {request_id}: Error reading request body: {str(e)}")

    # Обработка запроса
    response = await call_next(request)

    # Логирование ответа
    process_time = time.time() - start_time
    
    # Проверяем, является ли ответ потоковым
    if isinstance(response, StreamingResponse):
        logger.info(f"Request {request_id}: Response: [Streaming response - body not logged]")
    else:
        try:
            # Получаем тело ответа
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Декодируем тело ответа
            try:
                response_body_text = response_body.decode()
                logger.info(f"Request {request_id}: Response body: {response_body_text}")
            except UnicodeDecodeError:
                logger.info(f"Request {request_id}: Response body: [Binary data - {len(response_body)} bytes]")
            
            # Восстанавливаем итератор тела ответа
            # Восстанавливаем итератор тела ответа как асинхронный итератор
            async def new_body_iterator():
                yield response_body
            
            response.body_iterator = new_body_iterator()
            
        except Exception as e:
            logger.error(f"Request {request_id}: Error reading response body: {str(e)}")

    logger.info(f"Request {request_id}: Response status: {response.status_code}")
    logger.info(f"Request {request_id}: Process time: {process_time:.2f}s")

    return response

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
    )