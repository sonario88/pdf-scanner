# utils/pdf_processor.py
"""
Модуль для обработки PDF-файлов с этикетками.
Содержит логику извлечения этикеток, OCR, проверки изображений и валидации данных.
Полностью полагается на конфигурацию из JSON-шаблона.
"""
import os
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import io
import sys

# Добавляем корень проекта в путь для импорта config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем наш конфиг
from config import (
    LABELS_UPLOAD_DIR, FIELDS_UPLOAD_DIR, FIELDS_EDITED_DIR,
    load_template_config
)

# --- НОВОЕ: Импортируем extract_text_with_processors и validate_extracted_data ---
from utils.ocr_extractor import extract_text_with_processors
from utils.data_validator import validate_extracted_data
# --- КОНЕЦ НОВОГО ---

logger = logging.getLogger(__name__)

def process_pdf(
    pdf_path: str, 
    template_config: Dict[str, Any], 
    pages_to_process: Optional[List[int]] = None,
    split_only: bool = False
) -> Dict[str, Any]:
    """
    Обрабатывает PDF-файл с этикетками, полностью полагаясь на template_config.

    Args:
        pdf_path: Путь к PDF-файлу.
        template_config: Конфигурация шаблона.
        pages_to_process: Список номеров страниц для обработки (0-based). 
                         Если None, обрабатываются все страницы.
        split_only: Если True, только разделяет на этикетки без OCR и проверок.
        
    Returns:
        Словарь с отчетом о результатах обработки.
    """
    logger.info(f"Начало обработки PDF: {pdf_path} с шаблоном {template_config.get('template_name')}")

    # Открываем PDF
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    
    # Определяем, какие страницы обрабатывать
    if pages_to_process is None:
        pages_to_process = list(range(total_pages))
    else:
        # Фильтруем страницы, которые находятся в диапазоне документа
        pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]
    
    logger.info(f"Обрабатываются страницы: {[p+1 for p in pages_to_process]} (всего {len(pages_to_process)})")
    
    # Инициализируем отчет
    report = {
        "pdf_file": os.path.basename(pdf_path),
        "template_used": template_config.get("template_name", "unknown"),
        "processing_mode": "split_only" if split_only else "full_processing",
        "pages_processed": len(pages_to_process),
        "total_pages": total_pages,
        "summary": {
            "total_labels": 0,
            "labels_with_errors": 0
        },
        "pages": []
    }
    
    # Только если НЕ split_only, импортируем дополнительные модули
    barcode_checker = None
    datamatrix_checker = None
    if not split_only:
        try:
            from utils.barcode_checker import check_barcodes
            barcode_checker = check_barcodes
            logger.debug("Модуль barcode_checker успешно импортирован.")
        except Exception as e:
            logger.warning(f"Не удалось импортировать barcode_checker: {e}")
            barcode_checker = None
            
        try:
            from utils.datamatrix_checker import check_datamatrix
            datamatrix_checker = check_datamatrix
            logger.debug("Модуль datamatrix_checker успешно импортирован.")
        except Exception as e:
            logger.warning(f"Не удалось импортировать datamatrix_checker: {e}")
            datamatrix_checker = None
    
    # Обрабатываем каждую страницу
    for page_num in pages_to_process:
        page = doc[page_num]
        page_report = {
            "page_number": page_num + 1,  # 1-based для отчета
            "labels": []
        }
        
        # Преобразуем страницу в изображение (рендерим с высоким DPI для лучшего качества)
        zoom = 200 / 72  # 72 - стандартный DPI для PDF
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        page_image = Image.open(io.BytesIO(img_data))
        
        # --- НОВОЕ: Предобработка страницы на основе шаблона ---
        # Определяем тип страницы для конфигурации
        page_type_key = "page_1" if page_num == 0 else "page_n"
        current_page_config = template_config.get("page_processing", {}).get(page_type_key, {})
        
        # Применяем предобработку (например, обрезку) на основе конфигурации
        page_image = _preprocess_page(page_image, current_page_config.get("preprocessing", {}), page_num)
        # --- КОНЕЦ НОВОГО ---
        
        # --- НОВОЕ: Сохраняем полное изображение страницы ---
        try:
            full_page_filename = f"page{page_num + 1}_full.png"
            full_page_filepath = os.path.join(LABELS_UPLOAD_DIR, full_page_filename)
            page_image.save(full_page_filepath)
            logger.info(f"Полное изображение страницы {page_num + 1} сохранено как {full_page_filepath}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении полного изображения страницы {page_num + 1}: {e}")
        # --- КОНЕЦ НОВОГО ---
        
        # --- НОВОЕ: Извлекаем отдельные этикетки из полной страницы на основе шаблона ---
        labels = _process_page_by_config(page_image, current_page_config, page_num)
        # --- КОНЕЦ НОВОГО ---
        
        # Обрабатываем каждую этикетку
        for i, label_img in enumerate(labels):
            # --- ИСПРАВЛЕНО: Правильное определение имени файла ---
            # Определяем, является ли текущая этикетка главной этикеткой партии
            is_main_batch_label = (page_num == 0 and i == 0) # Первая этикетка на первой странице

            if is_main_batch_label:
                label_id = f"page_{page_num + 1}_mane"
                label_filename = f"page{page_num + 1}_mane.png"
            elif page_num == 0 and i >= 1:
                # Для остальных этикеток на первой странице используем корректный порядок
                # i начинается с 1, поэтому label_num начинается с 1
                label_id = f"page_{page_num + 1}_label_{i}"
                label_filename = f"page{page_num + 1}_label{i}.png"
            else:
                # Для страниц 2 и далее
                label_position_on_page = i + 1
                label_id = f"page_{page_num + 1}_label_{label_position_on_page}"
                label_filename = f"page{page_num + 1}_label{label_position_on_page}.png"
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            
            # Сохраняем изображение этикетки
            label_filepath = os.path.join(LABELS_UPLOAD_DIR, label_filename)
            label_img.save(label_filepath)
            logger.debug(f"Этикетка {i+1} на странице {page_num + 1} сохранена как {label_filepath}")
            
            # Если выбран режим "Разъединить этикетки", пропускаем дальнейшую обработку
            if split_only:
                label_report = {
                    "label_id": label_id,
                    "position": i + 1,
                    "text_check": {
                        "passed": True, 
                        "details": "Режим 'Разъединить этикетки'", 
                        "extracted_text": "", 
                        "parsed_fields": {}, 
                        "validated_fields_info": {}
                    },
                    "barcode_check": {
                        "passed": True, 
                        "details": "Режим 'Разъединить этикетки'", 
                        "found_codes": []
                    },
                    "datamatrix_cz_check": None,
                    "image_quality_check": {
                        "passed": True, 
                        "details": "Режим 'Разъединить этикетки'", 
                        "sharpness": 0.0
                    }
                }
                page_report["labels"].append(label_report)
                continue

            # Проверка качества изображения
            sharpness_score = check_image_sharpness(label_img)
            is_sharp = sharpness_score > 100.0  # Порог резкости

            # --- НОВОЕ: Извлечение текста с использованием процессоров полей ---
            # Передаем флаг is_main_batch_label в extract_text_with_processors
            try:
                extracted_text, parsed_fields, field_results = extract_text_with_processors(
                    label_img,
                    template_config,
                    page_num + 1, # 1-based
                    i + 1,       # 1-based
                    is_main_batch_label=is_main_batch_label # <-- ПЕРЕДАЕМ ФЛАГ
                )
                logger.debug(f"OCR для этикетки {label_id} завершен.")
            except Exception as e:
                logger.error(f"Ошибка в ocr_extractor для этикетки {label_id}: {e}", exc_info=True)
                # Создаем отчет об ошибке для этой этикетки
                label_report = {
                    "label_id": label_id,
                    "position": i + 1,
                    "image_quality_check": {
                        "passed": is_sharp,
                        "details": "OK" if is_sharp else "Изображение размыто",
                        "sharpness": sharpness_score
                    },
                    "text_check": {
                        "passed": False,
                        "details": f"Критическая ошибка OCR: {e}",
                        "extracted_text": "",
                        "parsed_fields": {},
                        "validated_fields_info": {}
                    },
                    "barcode_check": {
                        "passed": False,
                        "details": "Модуль проверки штрихкодов недоступен или ошибка",
                        "found_codes": []
                    },
                    "datamatrix_cz_check": None,
                    "image_quality_check": {
                        "passed": is_sharp,
                        "details": "OK" if is_sharp else "Изображение размыто",
                        "sharpness": sharpness_score
                    }
                }
                page_report["labels"].append(label_report)
                continue # Переходим к следующей этикетке
            # --- КОНЕЦ НОВОГО ---

            # Проверка штрихкодов (если модуль доступен)
            if barcode_checker:
                barcode_results = barcode_checker(label_img)
            else:
                barcode_results = {
                    "passed": False,
                    "details": "Модуль проверки штрихкодов недоступен",
                    "found_codes": []
                }

            # Проверка DataMatrix (если модуль доступен)
            if datamatrix_checker:
                datamatrix_results = datamatrix_checker(label_img)
            else:
                datamatrix_results = {
                    "passed": False,
                    "details": "Модуль проверки DataMatrix недоступен",
                    "found_codes": []
                }

            # Валидация извлеченных данных
            # --- ИСПРАВЛЕНО: Используем импортированную функцию ---
            validation_result = validate_extracted_data(parsed_fields, template_config)
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            # Формируем отчет по этикетке
            label_report = {
                "label_id": label_id,
                "position": i + 1,
                "image_quality_check": {
                    "passed": is_sharp,
                    "details": "OK" if is_sharp else "Изображение размыто",
                    "sharpness": sharpness_score
                },
                "text_check": {
                    "passed": bool(extracted_text),
                    "details": "Текст распознан" if extracted_text else "Текст не распознан",
                    "extracted_text": extracted_text,
                    "parsed_fields": parsed_fields,
                    "validated_fields_info": validation_result.get("fields_status", {})
                },
                "barcode_check": barcode_results,
                "datamatrix_cz_check": datamatrix_results
            }

            # Определяем общий статус этикетки
            has_error = not (
                label_report["text_check"]["passed"] and
                label_report["barcode_check"]["passed"] and
                label_report["image_quality_check"]["passed"]
            )

            if label_report["datamatrix_cz_check"] and not label_report["datamatrix_cz_check"]["passed"]:
                has_error = True

            if has_error:
                report["summary"]["labels_with_errors"] += 1

            page_report["labels"].append(label_report)

        report["summary"]["total_labels"] += len(page_report["labels"])
        report["pages"].append(page_report)

    doc.close()
    logger.info("Обработка PDF завершена.")
    return report


def _preprocess_page(page_image: Image.Image, preprocessing_config: Dict[str, Any], page_num: int) -> Image.Image:
    """
    Предварительная обработка изображения страницы на основе конфигурации.

    Args:
        page_image: Исходное изображение страницы PIL.
        preprocessing_config: Конфигурация предобработки из шаблона.
        page_num: Номер страницы (для логирования).

    Returns:
        Обработанное изображение страницы PIL.
    """
    logger.debug(f"Начало предобработки страницы {page_num + 1}...")

    # Обрезка страницы
    crop_config = preprocessing_config.get("crop", {})
    if crop_config:
        target_width = crop_config.get("width")
        target_height = crop_config.get("height")
        if target_width and target_height:
            img_width, img_height = page_image.size
            if img_width >= target_width and img_height >= target_height:
                cropped_page_image = page_image.crop((0, 0, target_width, target_height))
                logger.info(
                    f"Страница {page_num + 1} обрезана до размера {target_width}x{target_height} пикселей (по шаблону).")
                return cropped_page_image
            else:
                logger.warning(
                    f"Страница {page_num + 1} ({img_width}x{img_height}) меньше требуемого размера обрезки ({target_width}x{target_height}). Обрезка не выполнена.")

    logger.debug(f"Предобработка страницы {page_num + 1} завершена (без изменений).")
    return page_image


def _process_page_by_config(page_image: Image.Image, page_config: Dict[str, Any], page_num: int) -> List[Image.Image]:
    """
    Извлекает этикетки со страницы на основе конфигурации.

    Args:
        page_image: Изображение страницы PIL.
        page_config: Конфигурация обработки страницы из шаблона.
        page_num: Номер страницы (для логирования).

    Returns:
        Список изображений этикеток PIL.
    """
    labels = []
    label_configs = page_config.get("labels", [])

    for i, label_config in enumerate(label_configs):
        label_layout_type = label_config.get("type")
        label_name = label_config.get("name", f"label_{i}")

        logger.debug(
            f"Обработка конфигурации '{label_layout_type}' с именем '{label_name}' для страницы {page_num + 1}...")

        if label_layout_type == "grid":
            labels.extend(_extract_labels_grid(page_image, label_config, page_num, label_name))
        elif label_layout_type == "main_batch":
            # --- НОВОЕ: Обработка типа "main_batch" ---
            # Этот тип извлекает одну большую этикетку
            main_labels = _extract_labels_main_batch(page_image, label_config, page_num, label_name)
            labels.extend(main_labels)
            # --- КОНЕЦ НОВОГО ---
        elif label_layout_type == "standard_row":
            labels.extend(_extract_labels_standard_row(page_image, label_config, page_num, label_name))
        elif label_layout_type == "main_batch_with_row":
            # --- НОВОЕ: Обработка типа "main_batch_with_row" ---
            # Этот тип может извлекать несколько групп этикеток
            main_labels = _extract_labels_main_batch(page_image, label_config, page_num, f"{label_name}_main")
            row_labels = _extract_labels_standard_row(page_image, label_config, page_num, f"{label_name}_row")
            labels.extend(main_labels)
            labels.extend(row_labels)
            # --- КОНЕЦ НОВОГО ---
        else:
            logger.warning(
                f"Неизвестный тип размещения этикеток '{label_layout_type}' в конфигурации шаблона для страницы {page_num + 1}")

    logger.info(f"Из страницы {page_num + 1} извлечено {len(labels)} этикеток.")
    return labels


def _extract_labels_grid(page_image: Image.Image, label_config: Dict[str, Any], page_num: int,
                         label_group_name: str) -> List[Image.Image]:
    """Извлекает этикетки по сетке."""
    labels = []
    bbox_template = label_config.get("bbox_template", {})
    label_width_px = bbox_template.get("width", 0)
    label_height_px = bbox_template.get("height", 0)

    layout = label_config.get("layout", {})
    rows = layout.get("rows", 1)
    cols = layout.get("cols", 1)
    base_left_margin = layout.get("left_margin", 0)
    top_margin = layout.get("top_margin", 0)
    horizontal_spacing = layout.get("horizontal_spacing", 0)
    vertical_spacing = layout.get("vertical_spacing", 0)

    # Получаем информацию о специальных отступах
    special_left_margins = layout.get("special_left_margins", {})
    special_indices = special_left_margins.get("indices", [])
    special_value = special_left_margins.get("value", base_left_margin)

    for row in range(rows):
        for col in range(cols):
            # Индекс этикетки в порядке чтения (слева направо, сверху вниз)
            label_index = row * cols + col

            # Определяем отступ слева
            current_left_margin = special_value if label_index in special_indices else base_left_margin

            # Вычисляем координаты области этикетки
            x1 = current_left_margin + col * (label_width_px + horizontal_spacing)
            y1 = top_margin + row * (label_height_px + vertical_spacing)
            x2 = x1 + label_width_px
            y2 = y1 + label_height_px

            # Проверка на выход за границы
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(page_image.width, x2)
            y2 = min(page_image.height, y2)

            if x2 > x1 and y2 > y1:
                label_bbox = (x1, y1, x2, y2)
                label_img = page_image.crop(label_bbox)
                labels.append(label_img)
                logger.debug(
                    f"Извлечена 'grid' этикетка ({row+1},{col+1}) из группы '{label_group_name}' на странице {page_num + 1} с координатами {label_bbox}")
            else:
                logger.warning(
                    f"Некорректные координаты для 'grid' этикетки ({row+1},{col+1}) из группы '{label_group_name}' на странице {page_num + 1}: {label_bbox}")

    return labels


def _extract_labels_main_batch(page_image: Image.Image, label_config: Dict[str, Any], page_num: int,
                               label_name: str) -> List[Image.Image]:
    """Извлекает одну большую этикетку."""
    labels = []
    bbox = label_config.get("bbox", {})
    x1 = bbox.get("x", 0)
    y1 = bbox.get("y", 0)
    x2 = x1 + bbox.get("width", 0)
    y2 = y1 + bbox.get("height", 0)

    # Проверка на выход за границы
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(page_image.width, x2)
    y2 = min(page_image.height, y2)

    if x2 > x1 and y2 > y1:
        label_bbox = (x1, y1, x2, y2)
        label_img = page_image.crop(label_bbox)
        labels.append(label_img)
        logger.debug(
            f"Извлечена 'main_batch' этикетка '{label_name}' со страницы {page_num + 1} с координатами {label_bbox}")
    else:
        logger.warning(
            f"Некорректные координаты для 'main_batch' этикетки '{label_name}' на странице {page_num + 1}: {label_bbox}")

    return labels


def _extract_labels_standard_row(page_image: Image.Image, label_config: Dict[str, Any], page_num: int,
                                 label_group_name: str) -> List[Image.Image]:
    """Извлекает ряд стандартных этикеток."""
    labels = []
    count = label_config.get("count", 0)
    bbox_template = label_config.get("bbox_template", {})
    label_width_px = bbox_template.get("width", 0)
    label_height_px = bbox_template.get("height", 0)

    layout = label_config.get("layout", {})
    first_left_margin = layout.get("first_left_margin", 0)
    top_margin = layout.get("top_margin", 0)
    horizontal_spacing = layout.get("horizontal_spacing", 0) # Используем spacing из конфига

    # --- НОВОЕ: Поддержка стратегии смещения ---
    shift_strategy = layout.get("shift_strategy", {})
    strategy_type = shift_strategy.get("type")
    offset_px = shift_strategy.get("offset_px", 0)
    # --- КОНЕЦ НОВОГО ---

    current_x1 = None # Для отслеживания позиции при стратегии смещения

    for i in range(count):
        if strategy_type == "fixed_offset_from_previous_right_edge" and i > 0 and current_x1 is not None:
            # Используем стратегию смещения
            x1 = current_x1 + label_width_px + offset_px
        else:
            # Стандартный расчет или первая этикетка
            x1 = first_left_margin + i * (label_width_px + horizontal_spacing)

        y1 = top_margin
        x2 = x1 + label_width_px
        y2 = y1 + label_height_px

        current_x1 = x1 # Обновляем для следующей итерации

        # Проверка на выход за границы
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(page_image.width, x2)
        y2 = min(page_image.height, y2)

        if x2 > x1 and y2 > y1:
            label_bbox = (x1, y1, x2, y2)
            label_img = page_image.crop(label_bbox)
            labels.append(label_img)
            logger.debug(
                f"Извлечена 'standard_row' этикетка ({i + 1}/{count}) из группы '{label_group_name}' на странице {page_num + 1} с координатами {label_bbox}")
        else:
            logger.warning(
                f"Некорректные координаты для 'standard_row' этикетки {i + 1} из группы '{label_group_name}' на странице {page_num + 1}: {label_bbox}")

    return labels


def check_image_sharpness(pil_image: Image.Image, threshold: float = 100.0) -> float:
    """
    Проверяет резкость изображения (через variance of Laplacian).

    Args:
        pil_image: Изображение PIL.
        threshold: Порог резкости.

    Returns:
        Значение резкости (чем выше, тем резче).
    """
    # Конвертируем PIL Image в OpenCV Mat
    # Убедимся, что изображение в RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)
