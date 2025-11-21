# ============= 3. Neo4j Graph Manager =============
from typing import List, Dict

from neo4j import GraphDatabase

from services.models import SearchResult, nlp


class Neo4jGraphManager:
    """Управление графом знаний в Neo4j"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        """Создание индексов и ограничений"""
        with self.driver.session() as session:
            # Индексы для быстрого поиска
            session.run("""
                CREATE INDEX chunk_id_index IF NOT EXISTS
                FOR (c:Chunk) ON (c.chunk_id)
            """)

            session.run("""
                CREATE INDEX entity_name_index IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)

            session.run("""
                CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name, e.type]
            """)

            print("✓ Neo4j индексы созданы")

    def add_chunk_with_entities(self, chunk_id: str, content: str,
                                metadata: Dict, entities: List[Dict]):
        """
        Добавление чанка с извлеченными сущностями в граф

        Args:
            chunk_id: Уникальный ID чанка
            content: Текст чанка
            metadata: Метаданные (source, chunk_index и т.д.)
            entities: Список сущностей [{name, type, start, end}, ...]
        """
        with self.driver.session() as session:
            # Создаём узел чанка
            session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.source = $source,
                    c.chunk_index = $chunk_index,
                    c.length = $length
            """, chunk_id=chunk_id, content=content,
               source=metadata.get('source', ''),
               chunk_index=metadata.get('chunk_index', 0),
               length=len(content))

            # Добавляем сущности и связи
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type

                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e)-[r:MENTIONED_IN]->(c)
                    SET r.position = $position
                """, name=entity['name'],
                   type=entity['type'],
                   chunk_id=chunk_id,
                   position=entity.get('start', 0))

    def add_chunk_sequence(self, chunk_ids: List[str]):
        """Создание связей NEXT между последовательными чанками"""
        with self.driver.session() as session:
            for i in range(len(chunk_ids) - 1):
                session.run("""
                    MATCH (c1:Chunk {chunk_id: $id1})
                    MATCH (c2:Chunk {chunk_id: $id2})
                    MERGE (c1)-[:NEXT]->(c2)
                    MERGE (c2)-[:PREV]->(c1)
                """, id1=chunk_ids[i], id2=chunk_ids[i+1])

    def search_by_entities(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Поиск чанков через граф знаний по сущностям

        Алгоритм:
        1. Извлекаем сущности из запроса
        2. Ищем эти сущности в графе (fuzzy match через fulltext)
        3. Находим связанные чанки
        4. Ранжируем по количеству совпадений
        """
        # Извлекаем сущности из запроса
        doc = nlp(query)
        query_entities = [ent.text.lower() for ent in doc.ents]
        query_tokens = [token.lemma_.lower() for token in doc
                       if not token.is_stop and token.is_alpha]

        all_terms = list(set(query_entities + query_tokens[:5]))  # Топ-5 токенов

        if not all_terms:
            return []

        print(f"  🕸️  Поиск в графе по терминам: {all_terms}")

        with self.driver.session() as session:
            # Поиск через fulltext индекс
            result = session.run("""
                CALL db.index.fulltext.queryNodes('entity_fulltext', $search_terms)
                YIELD node AS e, score

                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)

                WITH c, SUM(score) as total_score, COUNT(DISTINCT e) as entity_count
                ORDER BY total_score DESC, entity_count DESC
                LIMIT $top_k

                RETURN c.chunk_id AS chunk_id,
                       c.content AS content,
                       total_score,
                       entity_count,
                       c.source AS source,
                       c.chunk_index AS chunk_index
            """, search_terms=' OR '.join(all_terms), top_k=top_k)

            results = []
            for record in result:
                results.append(SearchResult(
                    chunk_id=record['chunk_id'],
                    content=record['content'],
                    score=float(record['total_score']),
                    source='graph',
                    metadata={
                        'source_file': record['source'],
                        'chunk_index': record['chunk_index'],
                        'entity_count': record['entity_count']
                    }
                ))

            return results

    def clear_all(self):
        """Очистка всей базы данных"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Neo4j база данных очищена")