# ============= 4. Qdrant Vector Manager =============
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document

from services.models import SearchResult


class QdrantVectorManager:
    """Управление векторной базой данных Qdrant"""

    def __init__(self, host: str, port: int, collection_name: str, vector_size: int):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._create_collection()

    def _create_collection(self):
        """Создание коллекции если не существует"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Qdrant коллекция '{self.collection_name}' создана")
        else:
            print(f"✓ Qdrant коллекция '{self.collection_name}' уже существует")

    def add_chunks(self, chunks: List[Document], embeddings: List[List[float]]):
        """Добавление чанков с эмбеддингами в Qdrant"""
        points = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    'chunk_id': chunk.metadata['chunk_id'],
                    'content': chunk.page_content,
                    'source': chunk.metadata.get('source', ''),
                    'chunk_index': chunk.metadata.get('chunk_index', 0)
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✓ Добавлено {len(points)} векторов в Qdrant")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
      """Векторный поиск (Qdrant local/disk mode)"""

      results = self.client.query_points(
          collection_name=self.collection_name,
          query=query_vector,
          limit=top_k
      )

      out = []
      for r in results.points:
          payload = r.payload or {}
          out.append(SearchResult(
              chunk_id=payload.get("chunk_id", ""),
              content=payload.get("content", ""),
              score=r.score,
              source="vector",
              metadata = payload
          ))
      return out


    '''
    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Векторный поиск в Qdrant"""
        results = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        limit=top_k
        )

        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.payload['chunk_id'],
                content=result.payload['content'],
                score=result.score,
                source='vector',
                metadata={
                    'source_file': result.payload['source'],
                    'chunk_index': result.payload['chunk_index']
                }
            ))

       return search_results
'''
    def clear_collection(self):
        """Очистка коллекции"""
        self.client.delete_collection(self.collection_name)
        self._create_collection()
        print(f"✓ Qdrant коллекция '{self.collection_name}' очищена")
