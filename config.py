# config.py
import os
import json
import logging
from typing import Dict, Any, List, Optional

# Настройка логирования для этого модуля
logger = logging.getLogger(__name__)

# Пути к основным папкам проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_TEMPLATES_DIR = os.path.join(BASE_DIR, 'label_templates')
FIELD_PROCESSORS_DIR = os.path.join(BASE_DIR, 'field_processors')  # Эта строка была пропущена
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
LABELS_UPLOAD_DIR = os.path.join(UPLOADS_DIR, 'labels')
FIELDS_UPLOAD_DIR = os.path.join(UPLOADS_DIR, 'fields')
FIELDS_EDITED_DIR = os.path.join(UPLOADS_DIR, 'fields_edited')

# Создаем необходимые директории
os.makedirs(LABEL_TEMPLATES_DIR, exist_ok=True)
os.makedirs(FIELD_PROCESSORS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(LABELS_UPLOAD_DIR, exist_ok=True)
os.makedirs(FIELDS_UPLOAD_DIR, exist_ok=True)
os.makedirs(FIELDS_EDITED_DIR, exist_ok=True)

def get_available_templates() -> List[str]:
    """Получает список доступных шаблонов из папки label_templates."""
    templates = []
    if os.path.exists(LABEL_TEMPLATES_DIR):
        for filename in os.listdir(LABEL_TEMPLATES_DIR):
            if filename.startswith('roi_template_') and filename.endswith('.json'):
                template_name = filename[len('roi_template_'):-len('.json')]
                templates.append(template_name)
    return templates

def load_template_config(template_name: str) -> Optional[Dict[str, Any]]:
    """Загружает конфигурацию шаблона из JSON файла."""
    template_filename = f"roi_template_{template_name}.json"
    template_path = os.path.join(LABEL_TEMPLATES_DIR, template_filename)
    
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке шаблона {template_name}: {e}")
            return None
    else:
        logger.error(f"Шаблон {template_name} не найден по пути {template_path}")
        return None

def get_field_processors_for_template(template_name: str) -> Dict[str, str]:
    """
    Возвращает словарь сопоставления полей с процессорами для шаблона.
    В будущем может быть расширен для поддержки подпапок field_processors.
    """
    # Пока используем общий пул процессоров
    # В дальнейшем можно реализовать логику выбора процессоров по шаблону
    processors = {}
    template_config = load_template_config(template_name)
    
    if template_config and "fields" in template_config:
        for field_name, field_info in template_config["fields"].items():
            # Можно добавить логику определения процессора по типу поля или другим критериям
            # Пока используем имя поля для определения процессора
            processors[field_name] = f"{field_name}_processor"
    
    return processors

# Пример конфигурации для разных режимов обработки страниц
PROCESSING_MODES = {
    'all': 'Весь документ',
    'specific': 'Определенные страницы',
    'range': 'Диапазон страниц'
}

