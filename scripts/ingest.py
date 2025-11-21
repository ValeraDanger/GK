from services.ocr import YandexOCRProcessor
from services.embeddings import CloudRuEmbeddings
from services.rag_system import HybridRAGSystem
from utils.config import (
    INPUT_FOLDER, TEXT_OUTPUT_FOLDER, QDRANT_PATH, QDRANT_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, CLOUD_API_KEY, CLOUD_RU_URL,
    YANDEX_API_KEY
)
from utils.logger import get_logger

log = get_logger("[IngestScript]")

def ingest_files():
    log.info("Starting ingestion process...")

    # 1. OCR обработка новых файлов
    ocr_processor = YandexOCRProcessor(YANDEX_API_KEY)
    processed_files = ocr_processor.process_folder(INPUT_FOLDER, TEXT_OUTPUT_FOLDER)
    
    if not processed_files:
        log.info("No new files to ingest.")
        return {"status": "no_files", "count": 0}

    log.info(f"Files to ingest: {len(processed_files)}")

    # 2. Embeddings + RAG
    embeddings = CloudRuEmbeddings(api_key=CLOUD_API_KEY, base_url=CLOUD_RU_URL)
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

    log.info("Documents successfully added to knowledge base!")
    return {"status": "success", "count": len(processed_files)}
