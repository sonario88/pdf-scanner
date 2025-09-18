# field_processors/article_processor.py
"""
Модуль для предобработки и OCR изображений поля 'артикул'.
"""
import os
import logging
from PIL import Image
import cv2
import numpy as np
from pytesseract import image_to_string

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# Глобальная переменная для пути к папке с обработанными изображениями
FIELD_OUTPUT_DIR = None  # Устанавливается в app.py

def preprocess_and_ocr_article_field(
    pil_image: Image.Image, 
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple:
    """
    Предобрабатывает изображение поля 'артикул' и выполняет OCR.
    
    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла.
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки.
        
    Returns:
        tuple: (обработанное_pil_image, распознанный_текст)
    """
    try:
        logger.debug("Начало предобработки и OCR для поля 'артикул'...")
        
        # Конвертируем PIL Image в OpenCV Mat
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Преобразование в градации серого
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Применение бинаризации Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Увеличение изображения
        scale_factor = 4
        height, width = binary.shape
        new_dimensions = (width * scale_factor, height * scale_factor)
        resized = cv2.resize(binary, new_dimensions, interpolation=cv2.INTER_CUBIC)
        
        # Морфологическое закрытие для соединения разорванных символов
        kernel = np.ones((2,2), np.uint8)
        morph_close = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        
        # Конвертируем обратно в PIL Image
        processed_pil_image = Image.fromarray(morph_close)
        
        # Выполняем OCR
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = image_to_string(processed_pil_image, lang='eng', config=custom_config)
        
        logger.debug(f"OCR для поля 'артикул' завершен. Распознанный текст: '{text.strip()}'")
        
        # Сохранение обработанного изображения (если путь задан)
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
                logger.debug(f"Попытка сохранить обработанное изображение как: {save_path}")
                
                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if processed_pil_image.mode in ("RGBA", "P"):
                    processed_pil_image = processed_pil_image.convert("RGB")
                    
                processed_pil_image.save(save_path)
                logger.debug(f"Обработанное изображение поля 'артикул' сохранено: {save_path}")
            except Exception as save_error:
                logger.error(f"Ошибка при сохранении обработанного изображения поля 'артикул': {save_error}", exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        
        # Возвращаем обработанное изображение и распознанный текст
        return processed_pil_image, text.strip()
        
    except Exception as e:
        logger.error(f"Ошибка при предобработке или OCR изображения поля 'артикул': {e}", exc_info=True)
        # Возвращаем оригинальное изображение и пустую строку в случае ошибки
        return pil_image, ""