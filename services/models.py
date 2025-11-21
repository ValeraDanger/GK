# ============= Структуры данных =============

from dataclasses import dataclass
from typing import Dict

import nltk

# spaCy для извлечения сущностей
import spacy

from config import *

# === 1. NLP-модели ===
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
try:
    nlp = spacy.load("ru_core_news_sm")
except:
    import os
    os.system("python -m spacy download ru_core_news_sm")
    nlp = spacy.load("ru_core_news_sm")

print("NLTK и spaCy готовы.")

import os

lock_file = os.path.join(QDRANT_PATH, '.lock')
try:
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print('Удалён lock файл: работа с QdrantDb разблокирована.')
except Exception as e:
    print(f'Ошибка при удалении lock файла: {e}')


@dataclass
class SearchResult:
    """Результат поиска с указанием источника"""
    chunk_id: str
    content: str
    score: float
    source: str  # 'vector' или 'graph'
    metadata: Dict

    def __repr__(self):
        source_icon = "🔍" if self.source == "vector" else "🕸️"
        return f"{source_icon} [{self.source.upper()}] {self.chunk_id} (score: {self.score:.3f})"
