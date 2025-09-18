# app.py
import os
import json
import logging
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import shutil
import zipfile
from datetime import datetime
import sys

# --- НОВОЕ: Импорты для обработки PDF и OCR ---
# Импортируем наш конфиг
from config import (
    BASE_DIR, LABEL_TEMPLATES_DIR, FIELD_PROCESSORS_DIR,
    UPLOADS_DIR, LABELS_UPLOAD_DIR, FIELDS_UPLOAD_DIR, FIELDS_EDITED_DIR,
    get_available_templates, load_template_config, get_field_processors_for_template,
    PROCESSING_MODES
)

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Макс. размер файла 100MB
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Убедимся, что базовые папки существуют (уже созданы в config.py, но на всякий случай)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(LABELS_UPLOAD_DIR, exist_ok=True)
os.makedirs(FIELDS_UPLOAD_DIR, exist_ok=True)
os.makedirs(FIELDS_EDITED_DIR, exist_ok=True)

def clear_folders_internal():
    """Очищает содержимое папок labels и fields."""
    folders_to_clear = [LABELS_UPLOAD_DIR, FIELDS_UPLOAD_DIR, FIELDS_EDITED_DIR]
    errors = []
    
    for folder in folders_to_clear:
        try:
            if os.path.exists(folder):
                # Очищаем содержимое папки, но не удаляем саму папку
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        error_msg = f"Не удалось удалить {file_path}. Причина: {e}"
                        logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)
                logger.info(f"Папка '{folder}' успешно очищена.")
            else:
                logger.info(f"Папка '{folder}' не существует, пропускаем.")
        except Exception as e:
            error_msg = f"Ошибка при очистке папки '{folder}': {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            
    # Не возвращаем JSON, просто логируем ошибки
    if errors:
        logger.warning("Предупреждения при очистке папок: " + "; ".join(errors))

def create_field_folders_from_template(template_config):
    """Создает папки для каждого поля из конфигурации шаблона."""
    # Создаем базовые папки
    os.makedirs(FIELDS_UPLOAD_DIR, exist_ok=True)
    os.makedirs(FIELDS_EDITED_DIR, exist_ok=True)
    
    # Создаем папки для обычных полей
    if "fields" in template_config:
        for field_name in template_config["fields"]:
            field_folder = os.path.join(FIELDS_UPLOAD_DIR, field_name)
            os.makedirs(field_folder, exist_ok=True)
            
            field_edited_folder = os.path.join(FIELDS_EDITED_DIR, field_name)
            os.makedirs(field_edited_folder, exist_ok=True)
    
    # Создаем папку и подпапки для mane полей
    # Папка mane в fields_edited
    mane_edited_base_folder = os.path.join(FIELDS_EDITED_DIR, "mane")
    os.makedirs(mane_edited_base_folder, exist_ok=True)
    
    # Подпапки внутри mane для каждого поля из page_1_mane_fields
    if "page_1_mane_fields" in template_config:
        for field_name in template_config["page_1_mane_fields"]:
            mane_field_edited_folder = os.path.join(mane_edited_base_folder, field_name)
            os.makedirs(mane_field_edited_folder, exist_ok=True)
            
    # Также создаем папки в fields для mane полей (если нужно для оригинальных изображений)
    # Это может быть опционально, если вы хотите хранить оригинальные области отдельно
    # if "page_1_mane_fields" in template_config:
    #     for field_name in template_config["page_1_mane_fields"]:
    #         mane_field_original_folder = os.path.join(FIELDS_UPLOAD_DIR, field_name) # Или другое имя, например, f"mane_{field_name}"
    #         os.makedirs(mane_field_original_folder, exist_ok=True)

@app.route('/')
def index():
    """Главная страница с формой загрузки."""
    templates = get_available_templates()
    # Определяем режимы обработки здесь
    processing_modes = {
        'all': 'Весь документ',
        'specific': 'Определенные страницы',
        'range': 'Диапазон страниц'
    }
    return render_template('index.html', templates=templates, processing_modes=processing_modes)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загруженного PDF-файла."""
    # Очищаем папки перед началом новой обработки (внутренняя функция)
    clear_folders_internal()

    # Получение файла
    if 'file' not in request.files:
        logger.error("Файл не выбран в запросе.")
        return "Файл не выбран", 400
        
    file = request.files['file']
    
    if file.filename == '':
        logger.error("Имя файла пустое.")
        return "Файл не выбран", 400
        
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Файл {filename} успешно загружен и сохранен как {filepath}")
        
        # Получение параметров обработки
        template_name = request.form.get('template', 'ls_single_label4')  # Шаблон по умолчанию
        processing_mode = request.form.get('processing_mode', 'all')  # Режим обработки по умолчанию
        specific_pages = request.form.get('specific_pages', '')  # Список конкретных страниц
        page_range_start = request.form.get('page_range_start', '')
        page_range_end = request.form.get('page_range_end', '')
        split_only = 'split_only' in request.form  # Чекбокс "Разъединить этикетки"
        
        # Валидация и преобразование параметров страниц
        pages_to_process = None
        if processing_mode == 'specific' and specific_pages:
            try:
                pages_to_process = [int(p.strip()) - 1 for p in specific_pages.split(',') if p.strip().isdigit()]  # 0-based индексация
            except ValueError:
                logger.error("Некорректный формат списка страниц.")
                return "Некорректный формат списка страниц.", 400
        elif processing_mode == 'range':
            try:
                start = int(page_range_start) - 1 if page_range_start.isdigit() else None  # 0-based
                end = int(page_range_end) - 1 if page_range_end.isdigit() else None      # 0-based
                if start is not None and end is not None and start <= end:
                    pages_to_process = list(range(start, end + 1))
                else:
                    logger.error("Некорректный диапазон страниц.")
                    return "Некорректный диапазон страниц.", 400
            except ValueError:
                logger.error("Некорректный формат диапазона страниц.")
                return "Некорректный формат диапазона страниц.", 400
        
        try:
            # Загружаем конфигурацию шаблона
            template_config = load_template_config(template_name)
            if not template_config:
                logger.error(f"Не удалось загрузить конфигурацию шаблона {template_name}")
                return f"Не удалось загрузить конфигурацию шаблона {template_name}", 400
            
            # Создаем папки для полей на основе шаблона
            create_field_folders_from_template(template_config)
            
            # Импортируем модуль обработки PDF (отложенная загрузка)
            from utils.pdf_processor import process_pdf
            
            # Обрабатываем PDF
            report_data = process_pdf(
                pdf_path=filepath,
                template_config=template_config,
                pages_to_process=pages_to_process,
                split_only=split_only
            )
            
            # Сохраняем отчет
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.json"
            report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Отчет сохранен как {report_filepath}")
            
            # Если был выбран режим "Разъединить этикетки", перенаправляем на страницу завершения
            if split_only:
                return redirect(url_for('split_complete'))
            
            # Иначе показываем отчет
            return render_template(
                'report.html', 
                report=report_data, 
                report_filename=report_filename,
                split_only_mode=split_only,
                template_config=template_config
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке PDF: {e}", exc_info=True)
            return f"Ошибка при обработке PDF: {str(e)}", 500
    else:
        logger.error("Загруженный файл не является PDF.")
        return "Загруженный файл должен быть в формате PDF.", 400

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание отчета в формате JSON."""
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        if os.path.exists(file_path):
            logger.info(f"Отправка файла {safe_filename} для скачивания.")
            return send_file(file_path, as_attachment=True)
        else:
            logger.error(f"Файл {safe_filename} не найден для скачивания.")
            return "Файл не найден", 404
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла {filename}: {e}")
        return "Ошибка при скачивании файла", 500

@app.route('/split_complete')
def split_complete():
    """Страница, отображаемая после завершения режима 'Разъединить этикетки'."""
    return render_template('split_complete.html')

if __name__ == '__main__':
    logger.info("Запуск Flask-приложения...")
    app.run(host='0.0.0.0', port=5000, debug=True)
