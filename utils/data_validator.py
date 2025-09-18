# utils/data_validator.py
"""
Модуль для валидации извлеченных данных из этикеток согласно правилам шаблона.
"""
import re
from datetime import datetime
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def validate_extracted_data(
    parsed_fields: Dict[str, str], 
    template_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Валидирует извлеченные данные согласно правилам из шаблона.
    
    Args:
        parsed_fields: Словарь извлеченных полей.
        template_config: Конфигурация шаблона.
        
    Returns:
        Словарь с результатами валидации.
    """
    logger.debug("Начало валидации извлеченных данных...")
    
    validation_results = {
        "fields_status": {},
        "has_errors": False
    }
    
    # Получаем правила валидации из шаблона
    validation_rules = template_config.get("validation", {})
    
    # Проверяем каждое поле
    for field_name, field_value in parsed_fields.items():
        field_rules = validation_rules.get(field_name, {})
        field_status = {
            "value": field_value,
            "valid": True,
            "errors": []
        }
        
        # Проверка обязательности
        if field_rules.get("required", False) and not field_value:
            field_status["valid"] = False
            field_status["errors"].append("Поле обязательно для заполнения")
            
        # Проверка максимальной длины
        max_length = field_rules.get("max_length")
        if max_length and len(field_value) > max_length:
            field_status["valid"] = False
            field_status["errors"].append(f"Превышена максимальная длина ({max_length} символов)")
            
        # Проверка формата даты
        date_format = field_rules.get("format")
        if date_format == "DD.MM.YYYY" and field_value:
            if not is_valid_date(field_value):
                field_status["valid"] = False
                field_status["errors"].append("Неверный формат даты. Ожидается DD.MM.YYYY")
        
        # Добавляем результат проверки поля
        validation_results["fields_status"][field_name] = field_status
        
        # Обновляем общий статус ошибок
        if not field_status["valid"]:
            validation_results["has_errors"] = True
            
        if field_status["errors"]:
            logger.debug(f"Поле '{field_name}' имеет ошибки: {', '.join(field_status['errors'])}")
    
    logger.debug(f"Валидация завершена. Ошибок найдено: {validation_results['has_errors']}")
    return validation_results

def is_valid_date(date_str: str) -> bool:
    """
    Проверяет, является ли строка корректной датой в формате DD.MM.YYYY.
    
    Args:
        date_str: Строка с датой.
        
    Returns:
        True, если дата корректна, иначе False.
    """
    try:
        datetime.strptime(date_str, "%d.%m.%Y")
        return True
    except ValueError:
        return False