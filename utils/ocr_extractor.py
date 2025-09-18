# utils/ocr_extractor.py
"""
Модуль для извлечения текста из изображений этикеток с использованием специализированных процессоров полей.
Этот модуль сначала вызывает специализированные процессоры для каждой области (ROI).
Затем выполняет OCR на изображениях, сохранённых этими процессорами в static/uploads/fields_edited.
"""
import os
import importlib
import logging
from PIL import Image
import cv2
import numpy as np
from pytesseract import image_to_string
import sys

# Добавляем корень проекта в путь для импорта config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем наш конфиг
from config import (
    FIELDS_UPLOAD_DIR, FIELDS_EDITED_DIR,
    load_template_config
)

logger = logging.getLogger(__name__)

def extract_text_with_processors(
    label_image: Image.Image, 
    template_config: dict, 
    page_num: int, 
    label_num: int,
    is_main_batch_label: bool = False # <-- НОВОЕ: Флаг для определения типа этикетки
) -> tuple[str, dict[str, str], dict[str, any]]:
    """
    Извлекает текст из изображения этикетки, используя специализированные процессоры для каждого поля.
    Затем выполняет OCR на изображениях из static/uploads/fields_edited.

    Args:
        label_image (PIL.Image): Исходное изображение этикетки.
        template_config (dict): Конфигурация шаблона.
        page_num (int): Номер страницы (1-based).
        label_num (int): Номер этикетки на странице (1-based).
        is_main_batch_label (bool): Является ли эта этикетка главной этикеткой партии (page1_mane.png).

    Returns:
        tuple: (полный_распознанный_текст, словарь_полей, словарь_результатов_обработки_полей)
    """
    logger.debug(f"Начало OCR для этикетки на странице {page_num}, позиция {label_num}, is_main_batch: {is_main_batch_label}")
    
    full_text = ""
    parsed_fields = {}
    field_results = {}

    # --- НОВОЕ: Определяем, какие поля использовать ---
    if is_main_batch_label:
        # Для page1_mane.png используем специальные поля
        fields_info_source = "page_1_mane_fields"
        # Базовое имя файла для сохранения оригинальных изображений полей
        original_filename_base_prefix = f"page{page_num}_mane"
        # Базовая папка для сохранения обработанных изображений: fields_edited/mane
        base_output_dir_for_this_label = os.path.join(FIELDS_EDITED_DIR, "mane")
        logger.debug("Обработка как главной этикетки партии (mane).")
    else:
        # Для обычных этикеток используем основные поля
        fields_info_source = "fields"
        # Базовое имя файла для сохранения оригинальных изображений полей
        original_filename_base_prefix = f"page{page_num}_label{label_num}"
        # Базовая папка для сохранения обработанных изображений: fields_edited
        base_output_dir_for_this_label = FIELDS_EDITED_DIR
        logger.debug("Обработка как стандартной этикетки.")
        
    fields_info = template_config.get(fields_info_source, {})
    logger.debug(f"Используется источник полей: {fields_info_source}")
    # --- КОНЕЦ НОВОГО ---

    # --- ЭТАП 1: Обработка областей процессорами и сохранение в fields_edited ---
    # Сначала прогоняем все области через соответствующие процессоры
    # и сохраняем результаты в FIELDS_EDITED_DIR
    processed_images_paths = {} # Словарь для хранения путей к обработанным изображениям
    for field_name, field_info in fields_info.items():
        bbox = field_info.get("bbox")
        if not bbox or len(bbox) != 4:
            logger.warning(f"Некорректные координаты bbox для поля '{field_name}' в шаблоне.")
            field_results[field_name] = {
                "text": "",
                "status": "warning",
                "error": "Invalid bbox coordinates"
            }
            continue

        # Извлекаем изображение поля
        field_image = label_image.crop(bbox)

        # Создаем папки для оригинальных изображений полей (FIELDS_UPLOAD_DIR)
        field_original_dir = os.path.join(FIELDS_UPLOAD_DIR, field_name)
        os.makedirs(field_original_dir, exist_ok=True)
        original_filename_base = f"{original_filename_base_prefix}_field_{field_name}"
        original_field_path = os.path.join(field_original_dir, f"{original_filename_base}.png")
        field_image.save(original_field_path)
        logger.debug(f"Оригинальное изображение поля '{field_name}' сохранено как {original_field_path}")

        # --- НОВОЕ: Определяем путь для сохранения обработанных изображений ---
        # Для mane: fields_edited/mane/<field_name>
        # Для обычных: fields_edited/<field_name>
        if is_main_batch_label:
            edited_field_dir = os.path.join(base_output_dir_for_this_label, field_name)
        else:
            edited_field_dir = os.path.join(base_output_dir_for_this_label, field_name)
        os.makedirs(edited_field_dir, exist_ok=True)
        # --- КОНЕЦ НОВОГО ---

        # Определяем имя процессора для поля из шаблона
        processor_name_from_config = field_info.get("processor")
        if not processor_name_from_config:
             logger.warning(f"Процессор не указан для поля '{field_name}' в шаблоне. Пропущено.")
             field_results[field_name] = {
                "text": "",
                "status": "skipped",
                "error": "Processor not specified in template"
            }
             continue

        try:
            # Динамически импортируем процессор
            processor_module_name = f"field_processors.{processor_name_from_config}"
            processor_module = importlib.import_module(processor_module_name)
            logger.debug(f"Процессор {processor_module_name} успешно импортирован для поля '{field_name}'.")

            # --- НОВОЕ: Устанавливаем путь для сохранения обработанных изображений в процессор ---
            # Передаем путь в процессор через глобальную переменную FIELD_OUTPUT_DIR
            # Предполагаем, что процессор использует эту переменную
            if hasattr(processor_module, 'FIELD_OUTPUT_DIR'):
                processor_module.FIELD_OUTPUT_DIR = edited_field_dir
                logger.debug(f"Установлен FIELD_OUTPUT_DIR для процессора {processor_module_name}: {edited_field_dir}")
            else:
                 logger.warning(f"Процессор {processor_module_name} не имеет переменной FIELD_OUTPUT_DIR. Возможна ошибка сохранения.")
            # --- КОНЕЦ НОВОГО ---

            # Определяем имя функции обработки
            # Стандартное имя: preprocess_and_ocr_<field_name>_field
            standard_processor_function_name = f"preprocess_and_ocr_{field_name}_field"
            # Универсальное имя (если специфичная функция не найдена): preprocess_and_ocr_field
            universal_processor_function_name = "preprocess_and_ocr_field" 
            
            processor_function = None
            
            if hasattr(processor_module, standard_processor_function_name):
                processor_function = getattr(processor_module, standard_processor_function_name)
                logger.debug(f"Найдена функция {standard_processor_function_name} в процессоре {processor_module_name}.")
            elif hasattr(processor_module, universal_processor_function_name):
                processor_function = getattr(processor_module, universal_processor_function_name)
                logger.debug(f"Найдена универсальная функция {universal_processor_function_name} в процессоре {processor_module_name}.")
            else:
                logger.error(f"Функция обработки не найдена в процессоре {processor_module_name} (ни {standard_processor_function_name}, ни {universal_processor_function_name}).")
                field_results[field_name] = {
                    "text": "",
                    "status": "error",
                    "error": f"Processing function not found in {processor_module_name}"
                }
                continue # Переходим к следующему полю

            # --- ВЫПОЛНЕНИЕ ОБРАБОТКИ ---
            # Вызываем функцию обработки.
            # Предполагаем, что процессор:
            # 1. Обрабатывает изображение.
            # 2. СОХРАНЯЕТ обработанное изображение в processor_module.FIELD_OUTPUT_DIR.
            # 3. Возвращает обработанное PIL Image и OCR текст (второй аргумент может игнорироваться).

            # Передаем имя оригинального файла, чтобы процессор мог сгенерировать имя для сохранённого файла
            result_image, _ = processor_function(
                field_image, 
                original_filename=f"{original_filename_base}.png", # Передаем имя оригинального файла
                page_num=page_num,
                label_num=label_num # или 0 для mane, если нужно
            )
            
            # --- НОВОЕ: Сохраняем путь к обработанному изображению ---
            # Формируем путь, куда процессор должен был сохранить изображение
            # Обычно это: {edited_field_dir}/{original_filename}_processed.png или подобное
            # Для простоты, предположим, что процессор сохраняет с именем {original_filename}_processed.png
            # НО! Лучше, если процессор возвращает путь к сохраненному файлу.
            # Пока сделаем предположение: файл сохранен как {original_filename}_processed.png в edited_field_dir
            # В реальности, процессор должен управлять именем файла самостоятельно при сохранении.
            # Для надежности, лучше, чтобы процессор возвращал путь.
            
            # Формируем ожидаемый путь к обработанному изображению
            processed_filename_in_edited = f"{original_filename_base}_processed.png"
            processed_image_path = os.path.join(edited_field_dir, processed_filename_in_edited)
            
            # Сохраняем путь к обработанному изображению для последующего OCR
            processed_images_paths[field_name] = processed_image_path
            
            logger.debug(f"Поле '{field_name}' обработано процессором. Обработанное изображение предположительно сохранено как {processed_image_path}")
            # --- КОНЕЦ НОВОГО ---
                
        except ModuleNotFoundError:
            error_msg = f"Процессор {processor_module_name} (указанный для поля '{field_name}') не найден."
            logger.error(error_msg)
            field_results[field_name] = {
                "text": "",
                "status": "error",
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"Ошибка при обработке поля '{field_name}' процессором {processor_module_name}: {e}"
            logger.error(error_msg, exc_info=True)
            field_results[field_name] = {
                "text": "",
                "status": "error",
                "error": error_msg
            }


    # --- ЭТАП 2: OCR на изображениях из fields_edited ---
    # Теперь выполняем OCR на изображениях, которые были сохранены процессорами в fields_edited
    for field_name, field_info in fields_info.items():
         # Проверяем, было ли поле обработано успешно на этапе 1
         # Для простоты, попробуем OCR для всех полей из шаблона

        try:
            # --- НОВОЕ: Формируем путь к папке обработанных изображений ---
            # Для mane: fields_edited/mane/<field_name>
            # Для обычных: fields_edited/<field_name>
            if is_main_batch_label:
                edited_field_dir = os.path.join(base_output_dir_for_this_label, field_name)
            else:
                edited_field_dir = os.path.join(base_output_dir_for_this_label, field_name)
            # --- КОНЕЦ НОВОГО ---

            original_filename_base = f"{original_filename_base_prefix}_field_{field_name}"
            # Предполагаем, что процессор сохранил файл с тем же именем
            processed_image_path = os.path.join(edited_field_dir, f"{original_filename_base}_processed.png")
            
            # Альтернативно, если известно другое имя (например, с _processed)
            # Это сложно без знания внутренностей процессора.
            # Можно попробовать несколько вариантов.
            alternative_paths = [
                os.path.join(edited_field_dir, f"{original_filename_base}.png"), # Может быть сохранен без _processed
                # Добавить другие возможные имена, если они известны
            ]
            
            final_image_path = None
            if os.path.exists(processed_image_path):
                final_image_path = processed_image_path
                logger.debug(f"Найдено обработанное изображение для поля '{field_name}': {final_image_path}")
            else:
                # Проверяем альтернативные пути
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        final_image_path = alt_path
                        logger.debug(f"Найдено АЛЬТЕРНАТИВНОЕ обработанное изображение для поля '{field_name}': {final_image_path}")
                        break
            
            if final_image_path and os.path.exists(final_image_path):
                # Открываем изображение из fields_edited
                processed_img_for_ocr = Image.open(final_image_path)
                
                # Простая предобработка перед OCR (можно усложнить)
                # grayscale
                gray_img = processed_img_for_ocr.convert('L')
                # thresholding (optional, can be part of processor)
                # Для pytesseract часто достаточно grayscale
                
                # Выполняем OCR
                # Можно добавить настройки --psm, --oem, whitelist и т.д. в зависимости от типа поля
                custom_config = r'--oem 3 --psm 6' # Общий конфиг
                # Можно добавить специфичные настройки для разных типов полей позже
                
                extracted_text = image_to_string(gray_img, lang='rus+eng', config=custom_config).strip()
                
                parsed_fields[field_name] = extracted_text
                full_text += f"{field_name}: {extracted_text}\n"
                
                field_results[field_name] = {
                    "text": extracted_text,
                    "status": "success"
                }
                
                logger.debug(f"OCR для поля '{field_name}' из {final_image_path} завершен. Текст: '{extracted_text}'")
            else:
                 warning_msg = f"Обработанное изображение поля '{field_name}' не найдено в fields_edited по ожидаемым путям."
                 logger.warning(warning_msg)
                 field_results[field_name] = {
                    "text": "",
                    "status": "warning",
                    "error": warning_msg
                }
                
        except Exception as e:
            error_msg = f"Ошибка OCR для поля '{field_name}' из fields_edited: {e}"
            logger.error(error_msg, exc_info=True)
            field_results[field_name] = {
                "text": "",
                "status": "error",
                "error": error_msg
            }

    logger.debug(f"OCR для этикетки завершен. Извлечено {len([f for f in parsed_fields.values() if f])} полей с текстом.")
    return full_text.strip(), parsed_fields, field_results

# --- Функция для совместимости с предыдущим кодом (если использовалась) ---
# def extract_text_by_roi(...): 
#     # Можно оставить как обёртку или удалить, если она больше не нужна.
#     pass

# ... (остальные функции, если есть, остаются без изменений) ...