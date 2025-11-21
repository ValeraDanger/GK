# ============= 1. OCR модуль (без изменений) =============
import glob
import sys
import time
from pathlib import Path
from typing import List, Dict
import re

import grpc

from config import *
from services.html_parser import extract_text_from_html_with_ocr


class YandexOCRProcessor:
    def __init__(self, api_key: str):
        """Инициализация процессора OCR"""
        self.api_key = api_key
        sys.path.append('/app/yc-vision-ocr-recognizer/src')
        import async_ocr_client
        self.ocr_client = async_ocr_client

    def clear_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

        text = text.replace('┌', '').replace('┐', '')
        text = text.replace('\\', '')
        
        text = re.sub(r'\s+', ' ', text)

        # \u200B - zero width space, \u200E - left-to-right mark, \u200F - right-to-left mark
        text = re.sub(r'[\u200B\u200E\u200F]', '', text)

        text = text.strip()
        return text

    def wait_for_operation(self, operation_id: str, max_retries: int = MAX_RETRIES,
                          delay: int = RETRY_DELAY):
        """Ожидание завершения операции OCR с периодической проверкой статуса"""
        print(f"⏳ Ожидание завершения операции: {operation_id}")

        for attempt in range(1, max_retries + 1):
            try:
                results = self.ocr_client.get_recognition_results(
                    operation_id,
                    self.api_key
                )
                print(f"✓ Операция завершена (попытка {attempt}/{max_retries})")
                return results

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    if "operation data is not ready" in e.details():
                        print(f"⏳ Попытка {attempt}/{max_retries}: данные не готовы, ждём {delay}с...")
                        time.sleep(delay)
                    else:
                        print(f"✗ Операция не найдена: {e.details()}")
                        return None
                else:
                    print(f"✗ RPC ошибка: {e.code()}, {e.details()}")
                    return None
            except Exception as e:
                print(f"✗ Неожиданная ошибка: {type(e).__name__}: {e}")
                return None

        print(f"✗ Превышено время ожидания после {max_retries} попыток")
        return None

    def process_file(self, file_path: str) -> str:
        """Обработка одного файла через OCR"""
        print(f"\n{'='*60}")
        print(f"📄 Обработка файла: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.html':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                text = extract_text_from_html_with_ocr(html_content, self)  # self — ваш OCRProcessor
                print(f"✓ Распознано текста из HTML (с OCR изображений): {len(text)} символов")
                return text
            except Exception as e:
                print(f"✗ Ошибка при парсинге файла .html: {type(e).__name__}: {e}")
                return None
        else:

            try:
                operation_id = self.ocr_client.recognize_text_async(
                    file_path,
                    self.api_key,
                )
                print(f"✓ Файл отправлен на распознавание, operation_id: {operation_id}")

                results = self.wait_for_operation(operation_id)

                if results is None:
                    return None

                full_text = ""
                for page_idx, page_result in enumerate(results, 1):
                    if hasattr(page_result, 'text_annotation'):
                        page_text = page_result.text_annotation.full_text
                        full_text += page_text + "\n"
                        print(f"  Страница {page_idx}: {len(page_text)} символов")

                print(f"✓ Всего распознано: {len(full_text)} символов")
                return full_text

            except Exception as e:
                print(f"✗ Ошибка при обработке файла: {type(e).__name__}: {e}")
                return None

    def process_folder(self, input_folder: str, output_folder: str) -> List[Dict]:
        """Обработка всех файлов в папке"""
        os.makedirs(output_folder, exist_ok=True)

        supported_formats = ['*.pdf', '*.jpg', '*.html',  '*.jpeg', '*.png', '*.tiff']
        processed_files = []
        all_files = []

        for pattern in supported_formats:
            all_files.extend(glob.glob(os.path.join(input_folder, pattern)))

        print(f"\n📁 Найдено файлов для обработки: {len(all_files)}")

        for idx, file_path in enumerate(all_files, 1):
            print(f"\n[{idx}/{len(all_files)}]")

            try:
                text = self.process_file(file_path)

                if text is None or len(text.strip()) == 0:
                    print(f"⚠️  Пропускаем файл (нет текста)")
                    continue

                try:
                    text = self.clear_text(text)
                except Exception as e:
                    print(f"✗ Ошибка при удалении символов в тексте ocr: {type(e).__name__}: {e}")
                    return None

                filename = Path(file_path).stem
                text_file_path = os.path.join(
                    output_folder,
                    f"{filename}.txt"
                )

                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                processed_files.append({
                    'original_file': file_path,
                    'text_file': text_file_path,
                    'text': text
                })

                print(f"✓ Текст сохранён: {text_file_path}")

            except Exception as e:
                print(f"✗ Ошибка при обработке {file_path}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"✓ Успешно обработано файлов: {len(processed_files)}/{len(all_files)}")
        return processed_files


def process_single_file_formatted(self, file_path: str, output_folder: str) -> List[Dict]:
        """Обработка одного конкретного файла с сохранением результата"""
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📄 Обработка одиночного файла: {file_path}")

        try:
            text = self.process_file(file_path)

            if text is None or len(text.strip()) == 0:
                print(f"⚠️  Файл пуст или не распознан")
                return []

            try:
                text = self.clear_text(text)
            except Exception as e:
                print(f"✗ Ошибка очистки текста: {e}")
                return []

            filename = Path(file_path).stem
            text_file_path = os.path.join(
                output_folder,
                f"{filename}.txt"
            )

            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"✓ Текст сохранён: {text_file_path}")

            # Возвращаем список из одного элемента, чтобы совместимо с create_knowledge_base
            return [{
                'original_file': file_path,
                'text_file': text_file_path,
                'text': text
            }]

        except Exception as e:
            print(f"✗ Критическая ошибка при обработке файла: {e}")
            return []
