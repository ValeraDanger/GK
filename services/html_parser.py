from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
import base64
import os
from io import BytesIO
from typing import Optional

# Предполагаем, что YandexOCRProcessor есть и реализует метод process_image_bytes
# который принимает байты изображения и возвращает распознанный текст

def extract_text_from_html_with_ocr(html_content: str, ocr_processor) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')

    # Извлечение основного текста html
    main_text = soup.get_text(separator='\n', strip=True)

    # Обработка всех изображений с OCR
    ocr_texts = []

    for img_tag in soup.find_all('img'):
        src = img_tag.get('src', '')

        image_bytes = None

        if src.startswith('data:image'):
            # base64 встроенное изображение
            base64_str = src.split(',', 1)[1]
            image_bytes = base64.b64decode(base64_str)

        elif os.path.isfile(src):
            # локальный путь - читаем файл изображения
            with open(src, 'rb') as f:
                image_bytes = f.read()

        if image_bytes:
            # Используем OCR для распознавания текста на изображении
            ocr_result = ocr_processor.process_image_bytes(image_bytes)
            if ocr_result:
                ocr_texts.append(ocr_result)

    full_text = main_text + '\n\n' + '\n\n'.join(ocr_texts) if ocr_texts else main_text
    return full_text


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()
