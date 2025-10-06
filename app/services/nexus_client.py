import requests
import os
from pathlib import Path
from typing import Optional
from app.core.config import get_settings

class NexusClient:
    """Клиент для подключения к Nexus и загрузки артефактов"""
    
    def __init__(self):
        """Инициализация клиента Nexus"""
        self.settings = get_settings().nexus
        self.session = requests.Session()
        
        # Настройка аутентификации
        if self.settings.login and self.settings.password:
            self.session.auth = (self.settings.login, self.settings.password)
        
        # Настройка SSL сертификата, если указан путь
        if self.settings.cert_path and os.path.exists(self.settings.cert_path):
            self.session.verify = self.settings.cert_path
    
    def is_enabled(self) -> bool:
        """Проверка, включена ли загрузка из Nexus"""
        return self.settings.enabled
    
    def download_artifact(self, destination_path: str) -> bool:
        """
        Загрузка артефакта из Nexus
        
        Args:
            destination_path: Путь для сохранения загруженного файла
            
        Returns:
            bool: True если загрузка успешна, False в противном случае
        """
        if not self.is_enabled():
            print("Загрузка из Nexus отключена")
            return False
        
        if not all([self.settings.url, self.settings.repo, self.settings.id, self.settings.version, self.settings.file_name]):
            print("Не все параметры Nexus заданы")
            return False
        
        # Формирование URL для загрузки артефакта
        url = f"{self.settings.url.rstrip('/')}/repository/{self.settings.repo}/{self.settings.id}/{self.settings.version}/{self.settings.file_name}"
        
        try:
            print(f"Загрузка артефакта из Nexus: {url}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Создание директории для сохранения файла, если она не существует
            Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Сохранение файла
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Артефакт успешно загружен в: {destination_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке артефакта из Nexus: {str(e)}")
            return False
    
    def check_artifact_exists(self) -> bool:
        """
        Проверка существования артефакта в Nexus
        
        Returns:
            bool: True если артефакт существует, False в противном случае
        """
        if not self.is_enabled():
            return False
        
        if not all([self.settings.url, self.settings.repo, self.settings.id, self.settings.version, self.settings.file_name]):
            return False
        
        # Формирование URL для проверки существования артефакта
        url = f"{self.settings.url.rstrip('/')}/repository/{self.settings.repo}/{self.settings.id}/{self.settings.version}/{self.settings.file_name}"
        
        try:
            response = self.session.head(url)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка при проверке существования артефакта в Nexus: {str(e)}")
            return False

# Глобальный экземпляр клиента Nexus
_nexus_client: Optional[NexusClient] = None

def get_nexus_client() -> NexusClient:
    """Получение экземпляра клиента Nexus"""
    global _nexus_client
    if _nexus_client is None:
        _nexus_client = NexusClient()
    return _nexus_client