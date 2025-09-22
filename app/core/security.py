"""
Модуль для функций безопасности (аутентификация, авторизация и т.д.)
Пока не используется, но оставлен для будущего расширения.
"""

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Заглушка для проверки токена аутентификации.
    В реальном приложении здесь должна быть реализована проверка токена.
    """
    # В реальном приложении здесь должна быть проверка токена
    # Пока что просто пропускаем все запросы
    return credentials