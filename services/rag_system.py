# ========== УЛУЧШЕННАЯ ГИБРИДНАЯ RAG СИСТЕМА ==========
from typing import Dict, List

from pathlib import Path

# LangChain - используем langchain_core напрямую
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

from services.models import SearchResult
from services.entity_extractor import EntityExtractor
from services.neo4j_manager import Neo4jGraphManager
from services.qdrant_manager import QdrantVectorManager

from config import *


class HybridRAGSystem:
    """
    Гибридная RAG система с умным chunking
    """

    def __init__(self, embeddings, qdrant_path, collection_name,
                 neo4j_uri, neo4j_user, neo4j_password, llm_api_key: str, llm_base_url: str):
        self.embeddings = embeddings
        self.qdrant = QdrantVectorManager(QDRANT_HOST, QDRANT_PORT, collection_name, VECTOR_SIZE)
        self.neo4j = Neo4jGraphManager(neo4j_uri, neo4j_user, neo4j_password)
        self.entity_extractor = EntityExtractor()

        # Инициализация LLM клиента Cloud.ru (GigaChat)
        self.llm_client = OpenAI(
            api_key=llm_api_key,
            base_url=f"{llm_base_url}"
        )

        self.llm_model = "GigaChat/GigaChat-2-Max"

        # Двухуровневая стратегия chunking

        # 1. Предварительный chunker (для больших документов)
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Предварительные большие чанки
            chunk_overlap=500,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )

        # 2. Semantic chunker (для финальной обработки)
        try:
            self.semantic_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.6
            )
            self.use_semantic = True
            print("✓ Semantic chunking включен")
        except Exception as e:
            print(f"⚠️  Semantic chunking недоступен, используется RecursiveCharacterTextSplitter")
            self.use_semantic = False

        # 3. Fallback chunker (если semantic не работает)
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        print("✓ Гибридная RAG система инициализирована")

    def _smart_chunk_text(self, text: str, metadata: Dict) -> List[Document]:
        """
        Умный chunking с автоматическим fallback

        Стратегия:
        1. Если текст < 5000 символов → semantic chunking напрямую
        2. Если текст > 5000 символов → сначала предварительная нарезка,
           потом semantic для каждого куска
        3. При ошибке → fallback на RecursiveCharacterTextSplitter
        """
        text_length = len(text)
        source = metadata.get('source', 'unknown')

        print(f"  📏 Размер текста: {text_length:,} символов")

        # Стратегия 1: Малый текст — semantic напрямую
        if text_length < 5000 and self.use_semantic:
            try:
                print(f"  🎯 Стратегия: Semantic chunking (малый текст)")
                chunks = self.semantic_splitter.create_documents(
                    texts=[text],
                    metadatas=[metadata]
                )
                print(f"  ✓ Создано {len(chunks)} semantic chunks")
                return chunks
            except Exception as e:
                print(f"  ⚠️  Semantic chunking failed: {str(e)[:100]}")
                print(f"  🔄 Fallback на RecursiveCharacterTextSplitter")

        # Стратегия 2: Большой текст — двухэтапная обработка
        if text_length >= 5000:
            print(f"  🎯 Стратегия: Двухэтапный chunking (большой текст)")

            # Этап 1: Предварительная нарезка на большие куски
            pre_chunks = self.pre_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            print(f"  📊 Этап 1: {len(pre_chunks)} предварительных чанков")

            # Этап 2: Semantic chunking каждого большого куска
            all_final_chunks = []

            for i, pre_chunk in enumerate(pre_chunks):
                if self.use_semantic:
                    try:
                        # Попытка semantic chunking
                        semantic_chunks = self.semantic_splitter.create_documents(
                            texts=[pre_chunk.page_content],
                            metadatas=[pre_chunk.metadata]
                        )
                        all_final_chunks.extend(semantic_chunks)
                    except Exception as e:
                        # Fallback для этого конкретного чанка
                        print(f"    ⚠️  Semantic failed для чанка {i + 1}, использую fallback")
                        fallback_chunks = self.fallback_splitter.create_documents(
                            texts=[pre_chunk.page_content],
                            metadatas=[pre_chunk.metadata]
                        )
                        all_final_chunks.extend(fallback_chunks)
                else:
                    # Semantic недоступен — используем fallback сразу
                    fallback_chunks = self.fallback_splitter.create_documents(
                        texts=[pre_chunk.page_content],
                        metadatas=[pre_chunk.metadata]
                    )
                    all_final_chunks.extend(fallback_chunks)

            print(f"  ✓ Этап 2: {len(all_final_chunks)} финальных чанков")
            return all_final_chunks

        # Стратегия 3: Fallback (если всё остальное не сработало)
        print(f"  🔄 Fallback на RecursiveCharacterTextSplitter")
        chunks = self.fallback_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        print(f"  ✓ Создано {len(chunks)} fallback chunks")
        return chunks

    def create_knowledge_base(self, processed_files: List[Dict]):
        """
        Создание базы знаний из обработанных файлов
        """
        print(f"\n{'=' * 60}")
        print(f"🔨 Создание базы знаний из {len(processed_files)} документов...")
        print("=" * 60)

        all_chunks = []

        for idx, file_info in enumerate(processed_files, 1):
            print(f"\n[{idx}/{len(processed_files)}] 📄 {file_info['original_file']}")

            try:
                # Умный chunking с автоматическим fallback
                chunks = self._smart_chunk_text(
                    text=file_info['text'],
                    metadata={
                        'source': file_info['original_file'],
                        'text_file': file_info.get('text_file', '')
                    }
                )

                # Добавляем метаданные и обрабатываем каждый чанк
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{Path(file_info['original_file']).stem}_chunk{i}"
                    chunk.metadata['chunk_id'] = chunk_id
                    chunk.metadata['chunk_index'] = i
                    chunk.metadata['total_chunks'] = len(chunks)

                    # Извлечение сущностей (с ограничением размера)
                    text_for_entities = chunk.page_content[:10000]  # Лимит для spaCy
                    entities = self.entity_extractor.extract_entities(text_for_entities)

                    # Добавление в Neo4j
                    self.neo4j.add_chunk_with_entities(
                        chunk_id=chunk_id,
                        content=chunk.page_content,
                        metadata=chunk.metadata,
                        entities=entities
                    )

                    all_chunks.append(chunk)

                print(f"  🕸️  Добавлено {len(chunks)} чанков в граф")

            except Exception as e:
                print(f"  ❌ Ошибка при обработке файла: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_chunks:
            raise ValueError("Не удалось создать ни одного чанка!")

        # Создание эмбеддингов батчами
        print(f"\n🔍 Создание эмбеддингов для {len(all_chunks)} чанков...")
        chunk_texts = [c.page_content for c in all_chunks]

        batch_size = 32  # Батчи для стабильности
        all_embeddings = []

        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                print(f"  ✓ Обработано: {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)}")
            except Exception as e:
                print(f"  ⚠️  Ошибка в батче {i // batch_size + 1}: {e}")
                # Пытаемся обработать по одному
                for text in batch:
                    try:
                        emb = self.embeddings.embed_query(text)
                        all_embeddings.append(emb)
                    except:
                        # Добавляем нулевой вектор как fallback
                        all_embeddings.append([0.0] * VECTOR_SIZE)

        # Добавление в Qdrant
        print(f"💾 Сохранение в Qdrant...")
        self.qdrant.add_chunks(all_chunks, all_embeddings)

        print(f"\n{'=' * 60}")
        print(f"✅ База знаний создана успешно!")
        print(f"   📚 Всего чанков: {len(all_chunks)}")
        print(f"   📁 Документов: {len(processed_files)}")
        print(f"   🔍 Векторов в Qdrant: {len(all_embeddings)}")
        print("=" * 60)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[SearchResult]:
        """Гибридный поиск (без изменений)"""
        print(f"\n🔍 Гибридный поиск: '{query}'")
        print(f"   Alpha (вектор/граф): {alpha:.2f}/{1 - alpha:.2f}")

        # Векторный поиск
        print(f"  🔍 Векторный поиск...")
        query_vector = self.embeddings.embed_query(query)
        vector_results = self.qdrant.search(query_vector, top_k=top_k)
        print(f"     Найдено: {len(vector_results)}")

        # Графовый поиск
        graph_results = self.neo4j.search_by_entities(query, top_k=top_k)
        print(f"     Найдено: {len(graph_results)}")

        # Объединение
        all_results = {}

        if vector_results:
            max_score = max(r.score for r in vector_results)
            for r in vector_results:
                norm = r.score / max_score if max_score > 0 else 0
                if r.chunk_id not in all_results:
                    all_results[r.chunk_id] = r
                    all_results[r.chunk_id].score = norm * alpha

        if graph_results:
            max_score = max(r.score for r in graph_results)
            for r in graph_results:
                norm = r.score / max_score if max_score > 0 else 0
                if r.chunk_id not in all_results:
                    all_results[r.chunk_id] = r
                    all_results[r.chunk_id].score = norm * (1 - alpha)
                else:
                    all_results[r.chunk_id].score += norm * (1 - alpha)
                    all_results[r.chunk_id].source = 'hybrid'

        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)[:top_k]
        print(f"  ✅ Итого: {len(sorted_results)} результатов\n")

        return sorted_results

    # --- НОВЫЙ МЕТОД: Генерация ответа на основе RAG ---
    def generate_answer(self, query: str, context: str) -> str:
        # Генерация ответа LLM на основе контекста RAG

        prompt = f"""
                Ты — умный помощник. Используй только контекст ниже.

                === КОНТЕКСТ ===
                {context}

                === ВОПРОС ===
                {query}

                Ответь точно по контексту, без домыслов:
                """

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content

    def rag(self, query: str, top_k=5):
        # Полный RAG-процесс: поиск + LLM ответ

        search_results = self.hybrid_search(query, top_k)

        # Собираем контекст
        context = "\n\n".join([
            f"[{r.source.upper()} score={r.score:.3f}] {r.content}"
            for r in search_results
        ])

        # Генерируем ответ LLM
        answer = self.generate_answer(query, context)
        return answer, search_results

    def close(self):
        self.neo4j.close()

