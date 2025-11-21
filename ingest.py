# ingest.py

import os
from pathlib import Path

# Импорт необходимых классов — либо из init_all.py, либо из отдельных файлов
from init import YandexOCRProcessor, CloudRuEmbeddings, HybridRAGSystem
from config import *

def ingest_files(
    input_folder=INPUT_FOLDER,
    output_folder=TEXT_OUTPUT_FOLDER,
    qdrant_path=QDRANT_PATH,
    qdrant_collection=QDRANT_COLLECTION,
    vector_size=VECTOR_SIZE,
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    cloud_api_key=CLOUD_API_KEY,
    cloud_ru_url=CLOUD_RU_URL,
    yandex_api_key=YANDEX_API_KEY,
):
    print("=" * 60)
    print("📂 Добавление новых документов в базу знаний RAG")
    print("=" * 60)

    # 1. OCR обработка новых файлов
    ocr_processor = YandexOCRProcessor(yandex_api_key)
    processed_files = ocr_processor.process_folder(input_folder, output_folder)
    if not processed_files:
        print("Нет новых файлов для загрузки!")
        return

    print(f"📝 Файлов для дальнейшей загрузки: {len(processed_files)}")

    # 2. Embeddings + RAG
    embeddings = CloudRuEmbeddings(api_key=cloud_api_key, base_url=cloud_ru_url)
    rag = HybridRAGSystem(
        embeddings=embeddings,
        qdrant_path=qdrant_path,
        collection_name=qdrant_collection,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        llm_api_key=cloud_api_key,
        llm_base_url=cloud_ru_url
    )
    rag.create_knowledge_base(processed_files)

    print("✅ Документы успешно добавлены в базу знаний!")

if __name__ == "__main__":
    ingest_files()
