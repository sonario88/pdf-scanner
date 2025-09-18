# field_processors/barcode_processor.py
"""
Модуль для предобработки и OCR изображений поля 'barcode'.
Сохраняет обработанное изображение в папку fields_edited/barcode.
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

def preprocess_and_ocr_barcode_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple[Image.Image, str]:
    """
    Предобработывает изображение поля 'barcode' и выполняет OCR.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла (например, page2_label1_field_barcode.png).
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки на странице.

    Returns:
        tuple: (обработанное_pil_image, распознанный_текст)
    """
    try:
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Начало предобработки и OCR...")

        # --- Предобработка изображения ---
        # Конвертируем PIL Image в OpenCV Mat
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Преобразование в градации серого
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Преобразовано в градации серого.")

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
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Применен адаптивный порог (выбран метод: {'GAUSSIAN' if var_gaussian > var_mean else 'MEAN'}).")

        # 3. Увеличение изображения (если нужно)
        scale_factor = 3 # Увеличим в 3 раза для лучшего OCR
        height, width = thresh.shape
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized = cv2.resize(thresh, new_dimensions, interpolation=cv2.INTER_CUBIC)
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Изображение увеличено до {new_dimensions} (3x).")

        # 4. Медианная фильтрация для уменьшения шума
        denoised = cv2.medianBlur(resized, 3)
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Применена медианная фильтрация.")

        # Конвертируем обратно в PIL Image
        processed_pil_image = Image.fromarray(denoised)
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Изображение преобразовано обратно в PIL.")

        # --- КОНЕЦ Предобработки ---

        # --- Выполняем OCR ---
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789' # <-- Whitelist только цифры
        raw_text = image_to_string(processed_pil_image, lang='eng', config=custom_config).strip()
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] OCR завершен. Сырой текст: '{raw_text}'")
        
        # Очищаем текст, оставляя только цифры
        cleaned_text = ''.join(filter(str.isdigit, raw_text)).strip()
        final_text = cleaned_text
        logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Финальный очищенный текст: '{final_text}'")
        # --- КОНЕЦ OCR ---

        # --- Сохранение обработанного изображения ---
        if FIELD_OUTPUT_DIR and original_filename:
            try:
                os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)

                # Формируем имя файла для сохранения
                # Используем оригинальное имя, но добавляем суффикс _processed
                name_part, ext_part = os.path.splitext(original_filename)
                if not ext_part:
                    ext_part = '.png'
                processed_filename = f"{name_part}_processed{ext_part}"
                save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
                logger.debug(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Попытка сохранить обработанное изображение как: {save_path}")

                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if processed_pil_image.mode in ("RGBA", "P"):
                    processed_pil_image = processed_pil_image.convert("RGB")

                processed_pil_image.save(save_path)
                logger.info(f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Обработанное изображение поля 'barcode' сохранено: {save_path}")
            except Exception as save_error:
                error_msg = f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Ошибка при сохранении обработанного изображения поля 'barcode': {save_error}"
                logger.error(error_msg, exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        # --- КОНЕЦ СОХРАНЕНИЯ ---

        # Возвращаем обработанное изображение и ТОЛЬКО очищенный текст
        return processed_pil_image, final_text

    except Exception as e:
        error_msg = f"[Поле 'barcode' на стр. {page_num}, этикетка {label_num}] Ошибка при предобработке или OCR изображения поля 'barcode': {e}"
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
