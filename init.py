# init_all.py

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Neo4j
from neo4j import GraphDatabase

from config import *
from services.embeddings import CloudRuEmbeddings
from services.ocr import YandexOCRProcessor
from services.rag_system import HybridRAGSystem

# === 2. Инициализация Qdrant ===
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
collections = [c.name for c in qdrant_client.get_collections().collections]
if QDRANT_COLLECTION not in collections:
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION}' created.")
else:
    print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")

# === 3. Инициализация Neo4j и индексов ===
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with neo4j_driver.session() as session:
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
print("Neo4j indexes created.")

# Пример использования:
if __name__ == "__main__":
    # 1. OCR обработка
    ocr_processor = YandexOCRProcessor(YANDEX_API_KEY)  # <-- Ваш класс
    processed_files = ocr_processor.process_folder(INPUT_FOLDER, TEXT_OUTPUT_FOLDER)
    print(")")
    if not processed_files:
        print("Нет обработанных файлов!")
        exit(1)

    # 2. Embeddings + RAG
    embeddings = CloudRuEmbeddings(api_key=CLOUD_API_KEY, base_url=CLOUD_RU_URL)  # <-- Ваш класс
    rag = HybridRAGSystem(
        embeddings=embeddings,
        qdrant_path=QDRANT_PATH,
        collection_name=QDRANT_COLLECTION,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        llm_api_key=CLOUD_API_KEY,
        llm_base_url=CLOUD_RU_URL
    )
    rag.create_knowledge_base(processed_files)

    print("Проект инициализирован полностью: документы обработаны, знания построены!")

