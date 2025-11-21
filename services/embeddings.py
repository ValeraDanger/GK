# ============= 2. Эмбеддинг модуль (Cloud.ru) =============
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class CloudRuEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "Qwen/Qwen3-Embedding-0.6B"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинг списка документов"""
        if not texts:
            raise ValueError("Список текстов для эмбеддинга не может быть пустым")

        clean_texts = [str(t).strip() for t in texts if t and str(t).strip()]

        if not clean_texts:
            raise ValueError("После очистки не осталось валидных текстов")

        response = self.client.embeddings.create(
            model=self.model,
            input=clean_texts
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг одного запроса"""
        if not text or not text.strip():
            raise ValueError("Текст запроса не может быть пустым")

        response = self.client.embeddings.create(
            model=self.model,
            input=[text.strip()]
        )
        return response.data[0].embedding