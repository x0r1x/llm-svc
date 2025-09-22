import json
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging.config

from app.core.config import settings, reset_settings
from app.api import router as api_router
from app.dependencies import get_llama_handler, cleanup_llama_handler

from fastapi import Request

# Настройка логирования из конфига
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': settings.logging.format
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': settings.logging.level
        }
    },
    'root': {
        'handlers': ['console'],
        'level': settings.logging.level
    }
})

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при запуске
    try:
        # Инициализируем обработчик LLM
        await get_llama_handler()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    yield

    # Очистка при завершении
    await cleanup_llama_handler()
    logger.info("Application shut down gracefully")


def create_application() -> FastAPI:
    """Создание и настройка FastAPI приложения"""
    # Сбрасываем настройки для применения изменений
    reset_settings()
    from app.core.config import settings as current_settings

    application = FastAPI(
        title=current_settings.app.title,
        description=current_settings.app.description,
        version=current_settings.app.version,
        lifespan=lifespan,
        docs_url=current_settings.server.docs_url,
        redoc_url=current_settings.server.redoc_url,
    )

    # Настройка CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=current_settings.cors.allowed_origins,
        allow_credentials=current_settings.cors.allow_credentials,
        allow_methods=current_settings.cors.allow_methods,
        allow_headers=current_settings.cors.allow_headers,
    )

    # Подключение роутеров
    application.include_router(api_router, prefix="/v1")

    return application


app = create_application()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов и ответов"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Логирование входящего запроса
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    logger.info(f"Request {request_id}: Headers: {dict(request.headers)}")

    try:
        # Чтение тела запроса для POST запросов
        if request.method == "POST":
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body.decode())
                    logger.info(f"Request {request_id}: Body: {json.dumps(body_json, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    logger.info(f"Request {request_id}: Body: {body.decode()}")
    except Exception as e:
        logger.error(f"Request {request_id}: Error reading body: {str(e)}")

    # Обработка запроса
    response = await call_next(request)

    # Логирование ответа
    process_time = time.time() - start_time
    logger.info(f"Request {request_id}: Response status: {response.status_code}")
    logger.info(f"Request {request_id}: Process time: {process_time:.2f}s")

    return response

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings as current_settings

    uvicorn.run(
        app,
        host=current_settings.server.host,
        port=current_settings.server.port,
        log_level=current_settings.server.log_level.lower(),
    )
