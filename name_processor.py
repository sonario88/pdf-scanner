# field_processors/name_processor.py
"""
Модуль для предобработки и OCR изображений поля 'наименование'.
Только увеличение и OCR.
"""
import os
import logging
from PIL import Image
import cv2
import numpy as np
from pytesseract import image_to_string
import re

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# Глобальная переменная для пути к папке с обработанными изображениями
FIELD_OUTPUT_DIR = None # Устанавливается в app.py

def preprocess_and_ocr_name_field(pil_image, original_filename=None, page_num=None, label_num=None):
    """
    Увеличивает изображение поля 'наименование' и выполняет OCR.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла.
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки на странице.

    Returns:
        tuple: (увеличенное_pil_image, распознанный_текст)
    """
    ocr_text = "" # Инициализируем переменную заранее
    processed_pil_image = pil_image # На случай ошибки, вернем оригинальное изображение
    try:
        logger.debug("Начало УПРОЩЕННОЙ обработки изображения поля 'наименование' (только увеличение)")

        # --- 1. Преобразование в OpenCV (для увеличения) ---
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # --- 2. Увеличение изображения ---
        scale_percent = 1450  # Увеличение на 1350%
        width = int(img_cv.shape[1] * scale_percent / 100)
        height = int(img_cv.shape[0] * scale_percent / 100)
        dim = (width, height)
        # Используем INTER_LANCZOS4 для лучшего качества увеличения
        img_resized = cv2.resize(img_cv, dim, interpolation=cv2.INTER_LANCZOS4)

        # --- 3. Конвертация обратно в PIL Image ---
        processed_pil_image = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        
        logger.debug("Увеличение изображения поля 'наименование' завершено")

        # --- 4. OCR ---
        # Конфигурация Tesseract: только русский язык, PSM 6 (блок текста), OEM 1 (LSTM)
        custom_config = r'--oem 1 --psm 6 -l rus' 
        
        logger.debug("Начало OCR для поля 'наименование'")
        ocr_text = image_to_string(processed_pil_image, config=custom_config).strip()
        logger.debug(f"OCR для поля 'наименование' завершен. Распознанный текст: '{ocr_text}'")

        # --- 5. Очистка текста ---
        # Заменяем одинарные кавычки на двойные
        intermediate_text = ocr_text.replace("'", '"')
        
        # --- 6. УДАЛЯЕМ ПРОБЕЛ МЕЖДУ ЦИФРАМИ В ДВОЙНЫХ КАВЫЧКАХ ---
        # Ищем паттерн: ""цифра пробел цифра""
        # и заменяем на: ""цифрацифра""
        pattern = r'""(\d)\s+(\d)""'
        final_text = re.sub(pattern, r'""\1\2""', intermediate_text)
        
        # --- 7. Сохранение увеличенного изображения (если путь задан) ---
        # ВАЖНО: Этот блок должен быть в try-except, чтобы ошибка сохранения не прерывала весь процесс
        try:
            if FIELD_OUTPUT_DIR and original_filename:
                os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)
                
                # ИСПРАВЛЕНИЕ ОШИБКИ: Правильное формирование имени файла
                # Убираем "_processed" из оригинального имени, если оно есть
                name_part, ext_part = os.path.splitext(original_filename)
                if name_part.endswith('_processed'):
                    name_part = name_part[:-10] # Убираем '_processed'
                if not ext_part or ext_part == '.png_processed': # На случай, если расширение уже неправильное
                     ext_part = '.png'
                
                # Формируем правильное имя файла для сохранения
                processed_filename = f"{name_part}_processed{ext_part}"
                
                save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
                logger.debug(f"Попытка сохранить увеличенное изображение как: {save_path}")
                if processed_pil_image.mode in ("RGBA", "P"):
                    processed_pil_image = processed_pil_image.convert("RGB")
                processed_pil_image.save(save_path)
                logger.debug(f"Увеличенное изображение поля 'наименование' сохранено: {save_path}")
        except Exception as save_error:
             logger.error(f"Ошибка при сохранении обработанного изображения поля 'наименование': {save_error}", exc_info=True)
             # Не прерываем основной процесс, просто логируем ошибку

        # Возвращаем обработанное изображение и ОЧИЩЕННЫЙ текст
        return processed_pil_image, final_text

    except Exception as e:
        logger.error(f"Ошибка при УПРОЩЕННОЙ обработке или OCR изображения поля 'наименование': {e}", exc_info=True)
        # Возвращаем оригинальное (или увеличенное) изображение и текст, который у нас есть (даже если это пустая строка)
        return processed_pil_image, ocr_text # Возвращаем ocr_text, который был до ошибки
