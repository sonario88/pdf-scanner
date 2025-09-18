# field_processors/adres_corob_processor.py
"""
Модуль для предобработки и OCR изображений поля 'adres_corob' с главной этикетки партии (page1_mane.png).
Сохраняет обработанное изображение в папку fields_edited/mane/adres_corob.
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
# Будет установлена utils/ocr_extractor.py перед вызовом функции обработки
FIELD_OUTPUT_DIR = None

def preprocess_and_ocr_adres_corob_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None # Для mane label_num будет 0
) -> tuple[Image.Image, str]:
    """
    Предобрабатывает изображение поля 'adres_corob' и выполняет OCR.
    Всегда удаляет префикс "Адрес:" из распознанного текста.
    Перед OCR закрашивает белым прямоугольником 190x69 пикселей в левом верхнем углу.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла (например, page1_mane_field_adres_corob.png).
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки (0 для mane).

    Returns:
        tuple: (обработанное_pil_image, распознанный_текст_без_"Адрес:")
               Возвращает только очищенный текст.
    """
    try:
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Начало УПРОЩЕННОЙ обработки (увеличение 3x)...")

        # --- УПРОЩЕННАЯ Предобработка изображения ---
        # 1. Преобразование в OpenCV Mat
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Преобразовано в OpenCV Mat.")

        # --- НОВОЕ: Закрашивание области 190x69 в левом верхнем углу ---
        # Определяем координаты прямоугольника для закрашивания
        rect_x1 = 0
        rect_y1 = 0
        rect_x2 = min(71, open_cv_image.shape[1]) # Убедимся, что не выходим за правую границу
        rect_y2 = min(17, open_cv_image.shape[0])  # Убедимся, что не выходим за нижнюю границу
        
        # Закрашиваем область белым цветом (255, 255, 255 в BGR)
        cv2.rectangle(open_cv_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1) # -1 означает заполнение
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Закрашена область ({rect_x1}, {rect_y1}, {rect_x2}, {rect_y2}) белым прямоугольником 190x69.")
        # --- КОНЕЦ НОВОГО ---

        # 2. Увеличение изображения в 3 раза (300%)
        scale_factor = 3
        height, width = open_cv_image.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized = cv2.resize(open_cv_image, new_dimensions, interpolation=cv2.INTER_LANCZOS4)
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Изображение увеличено до {new_dimensions} (3x).")

        # 3. Медианная фильтрация для уменьшения шума (опционально)
        # denoised = cv2.medianBlur(resized, 3)
        # logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Применена медианная фильтрация.")
        # resized = denoised # Используем denoised вместо resized

        # Конвертируем обратно в PIL Image
        processed_pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Изображение преобразовано обратно в PIL.")
        # --- КОНЕЦ УПРОЩЕННОЙ Предобработки ---

        # --- Выполняем OCR ---
        custom_config = r'--oem 1 --psm 6' # Упрощенная конфигурация, --psm 6 для многострочного текста
        raw_text = image_to_string(processed_pil_image, lang='rus+eng', config=custom_config).strip()
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] OCR завершен. Сырой текст: '{raw_text}'")
        
        # --- Постобработка текста: УДАЛЕНИЕ ПРЕФИКСА ---
        logger.debug("Начало постобработки текста для поля 'adres_corob'")
        
        # 1. Базовая очистка
        cleaned_text_stage1 = raw_text.strip()
        
        # 2. Удаление префикса "Адрес:" с помощью регулярного выражения
        # ^[Аа][дД][рР][еЕ][сС]\s*:?\s* - Адрес (с учетом регистра) и возможное двоеточие
        pattern = r'^[Аа][дД][рР][еЕ][сС]\s*:?\s*'
        cleaned_text_stage2 = re.sub(pattern, '', cleaned_text_stage1)
        
        # Альтернативная, более простая очистка на случай, если регулярка не сработала
        # Убираем известные варианты префиксов "в лоб"
        prefixes_to_remove = [
            "Адрес:", "адрес:",
            "Адрес :", "адрес :",
            "Address:", "address:",
            "Address :", "address :",
            # Можно добавить другие варианты, если они появляются
        ]
        final_text = cleaned_text_stage2
        for prefix in prefixes_to_remove:
            if cleaned_text_stage2.lower().startswith(prefix.lower()):
                # Убираем префикс и возможные пробелы после него
                final_text = cleaned_text_stage2[len(prefix):].lstrip()
                logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Удален АЛЬТЕРНАТИВНЫЙ префикс '{prefix}'. Текст после очистки: '{final_text}'")
                break # Прекращаем перебор после первого совпадения
        
        if final_text == cleaned_text_stage2:
            logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Префикс 'Адрес:' не найден (ни регулярным выражением, ни вручную). Текст остался без изменений.")
        else:
            logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Префикс 'Адрес:' успешно удален. Финальный текст: '{final_text}'")
            
        logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Финальный очищенный текст: '{final_text}'")
        # --- КОНЕЦ Постобработки ---
        
        # --- Сохранение обработанного изображения ---
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
                logger.debug(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Попытка сохранить обработанное изображение как: {save_path}")

                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if processed_pil_image.mode in ("RGBA", "P"):
                    processed_pil_image = processed_pil_image.convert("RGB")

                processed_pil_image.save(save_path)
                logger.info(f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Обработанное изображение поля 'adres_corob' сохранено: {save_path}")
            except Exception as save_error:
                error_msg = f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при сохранении обработанного изображения поля 'adres_corob': {save_error}"
                logger.error(error_msg, exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        # --- КОНЕЦ СОХРАНЕНИЯ ---

        # Возвращаем обработанное изображение и ТОЛЬКО очищенный текст
        return processed_pil_image, final_text

    except Exception as e:
        error_msg = f"[Поле 'adres_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при УПРОЩЕННОЙ обработке или OCR изображения поля 'adres_corob': {e}"
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
    return preprocess_and_ocr_adres_corob_field(pil_image, original_filename, page_num, label_num)
