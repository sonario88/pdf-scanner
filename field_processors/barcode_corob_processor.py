# field_processors/barcode_corob_processor.py
"""
Модуль для предобработки и OCR изображений поля 'barcode_corob' с главной этикетки партии (page1_mane.png).
Сохраняет обработанное изображение в папку fields_edited/mane/barcode_corob.
Использует zxing для чтения штрихкода EAN-13.
"""
import os
import logging
from PIL import Image
import cv2
import numpy as np
import tempfile
import shutil

# --- НОВОЕ: Импорт zxing ---
ZXING_AVAILABLE = False
try:
    import zxing
    ZXING_AVAILABLE = True
    logging.getLogger(__name__).info("zxing успешно импортирован.")
except ImportError as e:
    logging.getLogger(__name__).warning(f"zxing не установлен или не может быть импортирован: {e}. Чтение штрихкодов будет пропущено.")
# --- КОНЕЦ НОВОГО ---

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

# Глобальная переменная для пути к папке с обработанными изображениями
# Будет установлена utils/ocr_extractor.py перед вызовом функции обработки
FIELD_OUTPUT_DIR = None

def preprocess_and_ocr_barcode_corob_field(
    pil_image: Image.Image, 
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None # Для mane label_num будет 0
) -> tuple[str, dict[str, str], dict[str, any]]:
    """
    Предобрабатывает изображение поля 'barcode_corob' и выполняет OCR (чтение штрихкода EAN-13).
    Увеличивает изображение до 636x290 пикселей и пытается прочитать штрихкод с помощью zxing.
    Возвращает данные в формате, ожидаемом utils/ocr_extractor.py.

    Args:
        pil_image (PIL.Image): Исходное изображение поля.
        original_filename (str, optional): Имя оригинального файла (например, page1_mane_field_barcode_corob.png).
        page_num (int, optional): Номер страницы.
        label_num (int, optional): Номер этикетки (0 для mane).

    Returns:
        tuple: (full_text, parsed_fields, field_results)
               Возвращает полный текст, словарь распознанных полей и словарь результатов обработки.
               full_text: Пустая строка (так как это специализированный процессор).
               parsed_fields: {"barcode_corob": "распознанный_штрихкод_EAN_13"}.
               field_results: Словарь с информацией о результатах обработки поля.
    """
    try:
        logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Начало предобработки и OCR (чтение штрихкода EAN-13)...")

        # --- Предобработка изображения ---
        # 1. Преобразование в OpenCV Mat
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Преобразовано в OpenCV Mat.")

        # 2. Изменение размера изображения до 636x290 пикселей с сохранением пропорций (добавляет белые поля)
        target_size = (636, 290)
        logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Изменение размера изображения до {target_size}...")
        
        # Масштабируем изображение, сохраняя пропорции, чтобы оно поместилось в target_size
        img_thumbnail = pil_image.copy()
        img_thumbnail.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Создаем новое изображение с целевым размером и белым фоном
        resized_pil_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Вычисляем координаты для центрирования
        x = (target_size[0] - img_thumbnail.width) // 2
        y = (target_size[1] - img_thumbnail.height) // 2
        
        # Вставляем масштабированное изображение по центру
        resized_pil_image.paste(img_thumbnail, (x, y))
        logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Изображение изменено размер до {resized_pil_image.size}.")

        # --- Сохранение обработанного изображения ---
        if FIELD_OUTPUT_DIR and original_filename:
            try:
                os.makedirs(FIELD_OUTPUT_DIR, exist_ok=True)

                # Формируем имя файла для сохранения
                # Используем оригинальное имя, но добавляем суффикс _resized_636x290
                name_part, ext_part = os.path.splitext(original_filename)
                if not ext_part:
                    ext_part = '.png'
                processed_filename = f"{name_part}_resized_636x290{ext_part}"
                save_path = os.path.join(FIELD_OUTPUT_DIR, processed_filename)
                logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Попытка сохранить обработанное изображение как: {save_path}")

                # Убедимся, что изображение в режиме, поддерживаемом форматом
                if resized_pil_image.mode in ("RGBA", "P"):
                    resized_pil_image = resized_pil_image.convert("RGB")

                resized_pil_image.save(save_path)
                logger.info(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Увеличенное (636x290) изображение поля 'barcode_corob' сохранено: {save_path}")
                
                # --- НОВОЕ: Логирование в стиле примера ---
                logger.info(f"---  1/1: {processed_filename} ---")
                logger.info(f"  ✅ Изображение изменено и сохранено: {save_path}")
                # --- КОНЕЦ НОВОГО ---
                
            except Exception as save_error:
                error_msg = f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при сохранении увеличенного изображения поля 'barcode_corob': {save_error}"
                logger.error(error_msg, exc_info=True)
                # Не прерываем основной процесс, просто логируем ошибку
        # --- КОНЕЦ СОХРАНЕНИЯ ---

        # --- НОВОЕ: Чтение штрихкода с помощью zxing ---
        barcode_text = ""
        if ZXING_AVAILABLE:
            temp_file_path = None
            try:
                logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Попытка чтения штрихкода EAN-13 с помощью zxing...")
                
                # Создаем временный файл для zxing
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                    # Убедимся, что изображение в режиме, поддерживаемом форматом
                    if resized_pil_image.mode in ("RGBA", "P"):
                        resized_pil_image_for_save = resized_pil_image.convert("RGB")
                    else:
                        resized_pil_image_for_save = resized_pil_image
                    resized_pil_image_for_save.save(temp_file_path)
                
                # Создаем reader и декодируем
                reader = zxing.BarCodeReader()
                barcode = reader.decode(temp_file_path, possible_formats=['EAN_13']) # <-- УКАЗЫВАЕМ ФОРМАТ
                
                if barcode and barcode.parsed:
                    barcode_data = barcode.parsed.decode("utf-8") if isinstance(barcode.parsed, bytes) else barcode.parsed
                    barcode_type = barcode.format
                    # rect = barcode.rect # Координаты могут быть доступны
                    
                    barcode_text = barcode_data
                    logger.info(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Найден штрихкод EAN-13 (zxing): {barcode_type} - {barcode_data}")
                    
                    # --- НОВОЕ: Логирование в стиле примера ---
                    logger.info(f"  ✅ Формат: {barcode_type}")
                    logger.info(f"  ✅ Текст: '{barcode_data}'")
                    # --- КОНЕЦ НОВОГО ---
                    
                else:
                    logger.info(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Штрихкоды EAN-13 не найдены (zxing).")
                    # --- НОВОЕ: Логирование в стиле примера ---
                    logger.info(f"  ❌ Штрихкоды EAN_13 не найдены")
                    # --- КОНЕЦ НОВОГО ---
                    
            except Exception as decode_error:
                error_msg = f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при декодировании штрихкода EAN-13 с помощью zxing: {decode_error}"
                logger.error(error_msg, exc_info=True)
                barcode_text = ""
                # --- НОВОЕ: Логирование в стиле примера ---
                logger.info(f"  ❌ Ошибка декодирования: {decode_error}")
                # --- КОНЕЦ НОВОГО ---
            finally:
                # Удаляем временный файл
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Временный файл {temp_file_path} удален.")
                    except Exception as delete_error:
                        logger.warning(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Не удалось удалить временный файл {temp_file_path}: {delete_error}")
                # --- КОНЕЦ НОВОГО ---
        else:
            logger.warning(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] zxing недоступен. Чтение штрихкода EAN-13 пропущено.")
            # --- НОВОЕ: Логирование в стиле примера ---
            logger.info(f"  ⚠️  zxing недоступен. Чтение штрихкода EAN-13 пропущено.")
            # --- КОНЕЦ НОВОГО ---
        # --- КОНЕЦ НОВОГО ---
        
        # --- НОВОЕ: Логирование результата ---
        if barcode_text:
            logger.info(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Результат OCR EAN-13: '{barcode_text}'")
        else:
            logger.info(f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Результат OCR EAN-13: ТЕКСТ НЕ РАСПОЗНАН")
        # --- КОНЕЦ НОВОГО ---

        # --- Формируем возвращаемые значения в формате, ожидаемом ocr_extractor ---
        full_text = "" # Специализированный процессор не возвращает общий текст
        parsed_fields = {"barcode_corob": barcode_text.strip()} # <-- ВАЖНО: Возвращаем распознанный текст в словаре
        field_results = {
            "barcode_corob": {
                "text": barcode_text.strip(),
                "status": "success" if barcode_text else "error",
                "details": "Штрихкод EAN-13 распознан" if barcode_text else "Штрихкод EAN-13 не распознан"
            }
        }
        # --- КОНЕЦ Формирования ---

        # Возвращаем данные в формате, ожидаемом utils/ocr_extractor.py
        return full_text, parsed_fields, field_results

    except Exception as e:
        error_msg = f"[Поле 'barcode_corob' на стр. {page_num}, этикетка 'mane'] Ошибка при предобработке или OCR изображения поля 'barcode_corob': {e}"
        logger.error(error_msg, exc_info=True)
        
        # --- НОВОЕ: Логирование в стиле примера ---
        logger.info(f"  ❌ Критическая ошибка: {e}")
        # --- КОНЕЦ НОВОГО ---
        
        # Возвращаем пустые данные в случае ошибки
        return "", {"barcode_corob": ""}, {
            "barcode_corob": {
                "text": "",
                "status": "error",
                "details": f"Критическая ошибка: {e}",
                "error": str(e)
            }
        }

# --- Функция для совместимости (если кто-то вызывает общий метод) ---
def preprocess_and_ocr_field(
    pil_image: Image.Image,
    original_filename: str = None,
    page_num: int = None,
    label_num: int = None
) -> tuple[str, dict[str, str], dict[str, any]]:
    """
    Совместимость: вызывает специфичную функцию для этого процессора.
    """
    return preprocess_and_ocr_barcode_corob_field(pil_image, original_filename, page_num, label_num)
