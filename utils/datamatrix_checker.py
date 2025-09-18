# utils/datamatrix_checker.py
"""
Модуль для проверки наличия и корректности DataMatrix кодов на изображениях этикеток.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Any

try:
    import pylibdmtx.pylibdmtx as dmtx
    PYLIBDMTX_AVAILABLE = True
except ImportError:
    PYLIBDMTX_AVAILABLE = False
    logging.warning("pylibdmtx не установлен. Проверка DataMatrix будет пропущена.")

logger = logging.getLogger(__name__)

def check_datamatrix(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Проверяет наличие и корректность DataMatrix кодов на изображении этикетки.
    
    Args:
        pil_image: Изображение этикетки PIL.
        
    Returns:
        Словарь с результатами проверки DataMatrix.
    """
    if not PYLIBDMTX_AVAILABLE:
        return {
            "passed": False,
            "details": "Библиотека pylibdmtx не установлена",
            "found_codes": []
        }
    
    try:
        logger.debug("Начало проверки DataMatrix кодов...")
        
        # Конвертируем PIL Image в массив numpy
        numpy_image = np.array(pil_image)
        
        # Пытаемся декодировать DataMatrix коды
        decoded_objects = dmtx.decode(numpy_image)
        
        found_codes = []
        for obj in decoded_objects:
            code_data = obj.data.decode("utf-8")
            # Координаты возвращаются в другом формате
            found_codes.append({
                "data": code_data,
                # Для DataMatrix координаты могут быть сложнее для определения
            })
            
            logger.debug(f"Найден DataMatrix код: {code_data}")
        
        # Определяем результат проверки
        passed = len(found_codes) > 0
        details = "DataMatrix коды найдены" if passed else "DataMatrix коды не найдены"
        
        logger.debug(f"Проверка DataMatrix кодов завершена. Найдено: {len(found_codes)}")
        
        return {
            "passed": passed,
            "details": details,
            "found_codes": found_codes
        }
        
    except Exception as e:
        logger.error(f"Ошибка при проверке DataMatrix кодов: {e}", exc_info=True)
        return {
            "passed": False,
            "details": f"Ошибка при проверке DataMatrix кодов: {str(e)}",
            "found_codes": []
        }