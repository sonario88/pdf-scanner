# field_processors/code_corob_processor.py
"""
Модуль для предобработки и OCR изображений поля 'code_corob' с главной этикетки партии (page1_mane.png).
Сохраняет обработанное изображение в папку fields_edited/mane/code_corob.
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
# Будет установлена utils/ocr_extractor.py перед вызовом функции обработки
FIELD_OUTPUT_DIR = None

def preprocess_and_ocr_code_corob_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None # Для mane label_num будет 0
) -> tuple[Image.Image, str]:
    """
    Предобрабатывает изображение поля 'code_corob' и выполняет OCR.
    Увеличивает изображение в 3 раза.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла (например, page1_mane_field_code_corob.png).
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки (0 для mane).

    Returns:
        tuple: (обработанное_pil_image, распознанный_текст)
    """
    try:
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Начало предобработки и OCR...")

        # --- НОВОЕ: Предобработка изображения ---
        # Конвертируем PIL Image в OpenCV Mat
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Преобразование в градации серого
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Преобразовано в градации серого.")

        # 2. Применение адаптивного порога для улучшения контраста
        # Попробуем несколько методов и выберем лучший по дисперсии
        thresh_mean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        thresh_gaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Выберем лучший (с большей дисперсией пикселей)
        var_mean = np.var(thresh_mean)
        var_gaussian = np.var(thresh_gaussian)
        thresh = thresh_gaussian if var_gaussian > var_mean else thresh_mean
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Применен адаптивный порог (выбран метод: {'GAUSSIAN' if var_gaussian > var_mean else 'MEAN'}).")

        # 3. Увеличение изображения в 3 раза
        scale_factor = 3
        height, width = thresh.shape
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized = cv2.resize(thresh, new_dimensions, interpolation=cv2.INTER_CUBIC)
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Изображение увеличено до {new_dimensions}.")

        # 4. Медианная фильтрация для уменьшения шума
        denoised = cv2.medianBlur(resized, 3)
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Применена медианная фильтрация.")

        # Конвертируем обратно в PIL Image
        processed_pil_image = Image.fromarray(denoised)
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Изображение преобразовано обратно в PIL.")

        # --- КОНЕЦ НОВОЙ ПРЕДОБРАБОТКИ ---

        # --- НОВОЕ: Выполняем OCR ---
        custom_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = image_to_string(processed_pil_image, lang='eng', config=custom_config)
        logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] OCR завершен. Распознанный текст: '{text.strip()}'")
        # --- КОНЕЦ OCR ---

        # --- НОВОЕ: Сохранение обработанного изображения ---
        if FIELD_OUTPUT_DIR and original_filename:
            try:
                os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)

                # Формируем имя файла для сохранения
                # Используем оригинальное имя, но добавляем суффикс _processed
                name_part, ext_part = os.path.splitext(original_filename)
                if name_part.endswith('_processed'):
                    name_part = name_part[:-10]  # Убираем '_processed'
                if not ext_part:
                    ext_part = '.png'
                    
                processed_filename = f"{name_part}_processed{ext_part}"
                save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
                logger.debug(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Попытка сохранить обработанное изображение как: {save_path}")

                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if processed_pil_image.mode in ("RGBA", "P"):
                    processed_pil_image = processed_pil_image.convert("RGB")
                    
                processed_pil_image.save(save_path)
                logger.info(f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Обработанное изображение поля 'code_corob' сохранено: {save_path}")
            except Exception as save_error:
                error_msg = f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при сохранении обработанного изображения поля 'code_corob': {save_error}"
                logger.error(error_msg, exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        # --- КОНЕЦ СОХРАНЕНИЯ ---

        # Возвращаем обработанное изображение и распознанный текст
        return processed_pil_image, text.strip()

    except Exception as e:
        error_msg = f"[Поле 'code_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при предобработке или OCR изображения поля 'code_corob': {e}"
        logger.error(error_msg, exc_info=True)
        # Возвращаем оригинальное изображение и пустую строку в случае ошибки
        return pil_image, ""

# --- Функция для совместимости (если кто-то вызывает общий метод) ---
def preprocess_and_ocr_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple[Image.Image, str]:
    """
    Совместимость: вызывает специфичную функцию для этого процессора.
    """
    return preprocess_and_ocr_code_corob_field(pil_image, original_filename, page_num, label_num)
