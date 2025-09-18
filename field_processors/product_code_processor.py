# field_processors/code_processor.py
"""
Модуль для предобработки и OCR изображений поля 'код'.
"""
import os
import logging
from PIL import Image
import cv2
import numpy as np
from pytesseract import image_to_string

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# Путь к папке для сохранения обработанных изображений этого поля
# Будет установлен в app.py
FIELD_OUTPUT_DIR = None

def preprocess_and_ocr_code_field(pil_image, original_filename=None):
    """
    Предварительная обработка изображения поля 'код' и выполнение OCR.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла для генерации имени обработанного.

    Returns:
        tuple: (обработанное_pil_image, распознанный_текст)
    """
    try:
        logger.debug("Начало предобработки изображения поля 'код'")

        # --- 1. Преобразование в OpenCV ---
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # --- 2. Увеличение изображения ---
        # Увеличение в 10 раз (1000%)
        scale_percent = 1000
        width = int(img_cv.shape[1] * scale_percent / 100)
        height = int(img_cv.shape[0] * scale_percent / 100)
        dim = (width, height)
        # Используем INTER_LANCZOS4 для лучшего качества увеличения
        img_resized = cv2.resize(img_cv, dim, interpolation=cv2.INTER_LANCZOS4)

        # --- 3. Конвертация обратно в PIL Image ---
        processed_pil_image = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

        logger.debug("Предобработка изображения поля 'код' завершена")

        # --- 4. OCR ---
        # Конфигурация Tesseract для улучшения распознавания коротких цифровых кодов
        # --oem 1: Используем только нейросетевую модель LSTM (лучше для четких изображений)
        # --psm 8: Предполагаем, что блок текста является отдельным словом/символом (подходит для кодов)
        # -c tessedit_char_whitelist=0123456789: Ограничиваем распознавание только цифрами
        custom_config = r'--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789'

        logger.debug("Начало OCR для поля 'код'")
        text = image_to_string(processed_pil_image, config=custom_config).strip()
        # Дополнительно очищаем результат, оставляя только цифры
        text = ''.join(filter(str.isdigit, text))
        logger.debug(f"OCR для поля 'код' завершен. Распознанный текст: '{text}'")

        # --- 5. Сохранение обработанного изображения (если путь задан) ---
        if FIELD_OUTPUT_DIR and original_filename:
            os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)
            # Генерируем имя файла для сохранения
            name, ext = os.path.splitext(original_filename)
            # Убедимся, что расширение правильное
            if not ext or ext.lower() not in ['.png', '.jpg', '.jpeg']:
                ext = '.png'
            # Удаляем возможное "_processed" из имени, если оно уже есть
            if name.endswith('_processed'):
                name = name[:-10]
            processed_filename = f"{name}_processed{ext}"
            save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
            logger.debug(f"Попытка сохранить обработанное изображение как: {save_path}")
            # Убедимся, что изображение в режиме, поддерживаемом форматом
            if processed_pil_image.mode in ("RGBA", "P"):
                processed_pil_image = processed_pil_image.convert("RGB")
            processed_pil_image.save(save_path)
            logger.debug(f"Обработанное изображение поля 'код' сохранено: {save_path}")

        return processed_pil_image, text

    except Exception as e:
        logger.error(f"Ошибка при предобработке или OCR изображения поля 'код': {e}", exc_info=True)
        # Возвращаем оригинальное изображение и пустую строку в случае ошибки
        return pil_image, ""
