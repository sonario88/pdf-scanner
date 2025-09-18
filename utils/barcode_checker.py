# utils/barcode_checker.py
"""
Модуль для проверки наличия и корректности штрихкодов на изображениях этикеток.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Any

# Инициализируем флаги доступности библиотеки
PYZBAR_AVAILABLE = False
pyzbar_module = None

# Попытка импорта pyzbar внутри блока try-except
try:
    import pyzbar.pyzbar as pyzbar_import
    pyzbar_module = pyzbar_import
    PYZBAR_AVAILABLE = True
    logging.info("pyzbar успешно импортирован.")
except ImportError as e:
    PYZBAR_AVAILABLE = False
    logging.warning(f"pyzbar не установлен или не может быть импортирован: {e}. Проверка штрихкодов будет пропущена.")
except Exception as e:
    PYZBAR_AVAILABLE = False
    logging.error(f"Неожиданная ошибка при импорте pyzbar: {e}. Проверка штрихкодов будет пропущена.")

logger = logging.getLogger(__name__)

def check_barcodes(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Проверяет наличие и корректность штрихкодов на изображении этикетки.
    
    Args:
        pil_image: Изображение этикетки PIL.
        
    Returns:
        Словарь с результатами проверки штрихкодов.
    """
    # Двойная проверка доступности модуля
    if not PYZBAR_AVAILABLE or pyzbar_module is None:
        logger.warning("Библиотека pyzbar не доступна. Проверка штрихкодов будет пропущена.")
        return {
            "passed": False,
            "details": "Библиотека pyzbar не доступна или не установлена",
            "found_codes": []
        }
    
    try:
        logger.debug("Начало проверки штрихкодов...")
        
        # Конвертируем PIL Image в OpenCV Mat
        # Убедимся, что изображение в RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Пытаемся декодировать штрихкоды
        decoded_objects = pyzbar_module.decode(open_cv_image)
        
        found_codes = []
        for obj in decoded_objects:
            try:
                code_data = obj.data.decode("utf-8")
                code_type = obj.type
                rect = obj.rect
                
                found_codes.append({
                    "data": code_data,
                    "type": code_type,
                    "x": rect.left,
                    "y": rect.top,
                    "width": rect.width,
                    "height": rect.height
                })
                
                logger.debug(f"Найден штрихкод: {code_type} - {code_data}")
            except Exception as decode_error:
                logger.warning(f"Ошибка декодирования данных штрихкода: {decode_error}")
                # Пропускаем этот объект, но продолжаем обработку других
                
        # Определяем результат проверки
        passed = len(found_codes) > 0
        details = "Штрихкоды найдены" if passed else "Штрихкоды не найдены"
        
        logger.debug(f"Проверка штрихкодов завершена. Найдено: {len(found_codes)}")
        
        return {
            "passed": passed,
            "details": details,
            "found_codes": found_codes
        }
        
    except Exception as e:
        logger.error(f"Ошибка при проверке штрихкодов: {e}", exc_info=True)
        return {
            "passed": False,
            "details": f"Ошибка при проверке штрихкодов: {str(e)}",
            "found_codes": []
        }