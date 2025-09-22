import yaml
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path

# Глобальный экземпляр настроек
_settings = None

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


class CorsConfig(BaseModel):
    allowed_origins: List[str] = ["http://localhost:3080", "http://localhost:8000"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]


class ModelConfig(BaseModel):
    path: str = "/app/models/llama-2-7b-chat.Q4_K_M.gguf"
    name: str = "llama-2-7b-chat"
    ctx_size: int = 4096
    gpu_layers: int = 0
    verbose: bool = False


class GenerationConfig(BaseModel):
    default_temperature: float = 0.7
    default_max_tokens: int = 256
    stream: bool = False


class AppConfig(BaseModel):
    title: str = "Llama CPP API Service"
    description: str = "API service for Llama-based models compatible with OpenAI API"
    version: str = "1.0.0"


class MonitoringConfig(BaseModel):
    enabled: bool = False
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = 10485760
    backup_count: int = 5


class CachingConfig(BaseModel):
    enabled: bool = True
    ttl: int = 300
    max_size: int = 1000


class Settings(BaseModel):
    server: ServerConfig
    cors: CorsConfig
    model: ModelConfig
    generation: GenerationConfig
    app: AppConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig
    caching: CachingConfig

    @classmethod
    def from_yaml(cls, config_path: str = None):
        """Загрузка конфигурации из YAML файла"""
        if config_path is None:
            # Поиск config.yml в различных возможных местах
            possible_paths = [
                "config/config.yml",
                "../config/config.yml",
                "./config.yml",
                Path(__file__).parent.parent.parent / "config" / "config.yml"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                # Если файл не найден, используем значения по умолчанию
                return cls()

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return cls(**config_data)
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {str(e)}")


def get_settings() -> Settings:
    """Получение экземпляра настроек."""
    global _settings
    if _settings is None:
        config_path = os.environ.get("CONFIG_PATH")
        _settings = Settings.from_yaml(config_path)
    return _settings

def reset_settings():
    """Сброс настроек (для тестов)."""
    global _settings
    _settings = None

# Инициализация настроек
settings = get_settings()
