# field_processors/barcode_processor.py
"""
Модуль для распознавания штрих-кодов поля 'barcode'.
Использует Aspose.BarCode для Python via .NET.
"""
import os
import logging
from PIL import Image
import numpy as np
from io import BytesIO

# Импортируем Aspose.BarCode
try:
    import aspose.barcode as barcode
    from aspose.barcode import barcoderecognition
    ASPOSE_BARCODE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Aspose.BarCode успешно импортирован.")
except ImportError as e:
    ASPOSE_BARCODE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Aspose.BarCode не установлен или не может быть импортирован: {e}")

# Глобальная переменная для пути к папке с обработанными изображениями
# Будет установлена utils/ocr_extractor.py перед вызовом функции обработки
FIELD_OUTPUT_DIR = None

def preprocess_and_ocr_barcode_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple[Image.Image, str]:
    """
    Распознает штрихкод на изображении поля 'barcode'.
    Сохраняет обработанное изображение в FIELD_OUTPUT_DIR.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла.
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки.

    Returns:
        tuple: (обработанное_pil_image, распознанный_текст_штрихкода)
    """
    logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Начало распознавания штрихкода...")
    
    barcode_text = ""
    
    if not ASPOSE_BARCODE_AVAILABLE:
        logger.warning(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Aspose.BarCode недоступен. Пропущено.")
        return pil_image, barcode_text

    try:
        # --- Распознавание штрихкода ---
        # Конвертируем PIL Image в байтовый поток
        img_byte_arr = BytesIO()
        # Убедимся, что изображение в режиме, поддерживаемом форматом
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Создаем временный файл для Aspose
        temp_filename = f"temp_{original_filename if original_filename else 'barcode'}.png"
        temp_filepath = os.path.join(os.getcwd(), temp_filename) # Используем текущую рабочую директорию для временного файла
        
        # Сохраняем изображение во временный файл
        with open(temp_filepath, 'wb') as f:
            f.write(img_byte_arr.getvalue())
        
        # Создаем BarCodeReader
        reader = barcoderecognition.BarCodeReader(temp_filepath, barcoderecognition.DecodeType.AllSupportedTypes)
        
        # Читаем штрихкоды
        recognized_results = reader.read_bar_codes()
        
        # Обрабатываем результаты
        if recognized_results:
            # Берем первый найденный штрихкод
            first_barcode = recognized_results[0]
            barcode_text = first_barcode.code_text
            barcode_type = first_barcode.code_type_name
            logger.info(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Найден штрихкод. Тип: {barcode_type}, Текст: '{barcode_text}'")
        else:
            logger.info(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Штрихкоды не найдены.")
            
        # Закрываем reader и удаляем временный файл
        reader.dispose()
        try:
            os.remove(temp_filepath)
            logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Временный файл {temp_filepath} удален.")
        except Exception as e:
            logger.warning(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Не удалось удалить временный файл {temp_filepath}: {e}")
        # --- КОНЕЦ Распознавания штрихкода ---
        
        # --- Сохранение обработанного изображения ---
        if FIELD_OUTPUT_DIR and original_filename:
            try:
                os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)
                
                # Формируем имя файла для сохранения
                name_part, ext_part = os.path.splitext(original_filename)
                if name_part.endswith('_processed'):
                    name_part = name_part[:-10]  # Убираем '_processed'
                if not ext_part:
                    ext_part = '.png'
                    
                processed_filename = f"{name_part}_processed{ext_part}"
                save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
                logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Попытка сохранить обработанное изображение как: {save_path}")
                
                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if pil_image.mode in ("RGBA", "P"):
                    pil_image = pil_image.convert("RGB")
                    
                pil_image.save(save_path)
                logger.info(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Обработанное изображение поля 'barcode' сохранено: {save_path}")
            except Exception as save_error:
                error_msg = f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Ошибка при сохранении обработанного изображения поля 'barcode': {save_error}"
                logger.error(error_msg, exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        # --- КОНЕЦ СОХРАНЕНИЯ ---
        
        # Возвращаем оригинальное изображение и распознанный текст
        return pil_image, barcode_text.strip()
        
    except Exception as e:
        error_msg = f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Ошибка при распознавании штрихкода или сохранении изображения: {e}"
        logger.error(error_msg, exc_info=True)
        # Возвращаем оригинальное изображение и пустую строку в случае ошибки
        return pil_image, ""

# --- Функция для совместимости ---
def preprocess_and_ocr_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple[Image.Image, str]:
    """
    Совместимость: вызывает специфичную функцию для этого процессора.
    """
    return preprocess_and_ocr_barcode_field(pil_image, original_filename, page_num, label_num)