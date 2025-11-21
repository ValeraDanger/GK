# from services.ocr import YandexOCRProcessor
# from services.embeddings import CloudRuEmbeddings
# from services.rag_system import HybridRAGSystem
# from utils.config import (
#     INPUT_FOLDER, TEXT_OUTPUT_FOLDER, QDRANT_PATH, QDRANT_COLLECTION,
#     NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, CLOUD_API_KEY, CLOUD_RU_URL,
#     YANDEX_API_KEY
# )
# from utils.logger import get_logger

# log = get_logger("[IngestScript]")

# def ingest_files():
#     log.info("Starting ingestion process...")

#     # 1. OCR обработка новых файлов
#     ocr_processor = YandexOCRProcessor(YANDEX_API_KEY)
#     processed_files = ocr_processor.process_folder(INPUT_FOLDER, TEXT_OUTPUT_FOLDER)
    
#     if not processed_files:
#         log.info("No new files to ingest.")
#         return {"status": "no_files", "count": 0}

#     log.info(f"Files to ingest: {len(processed_files)}")

#     # 2. Embeddings + RAG
#     embeddings = CloudRuEmbeddings(api_key=CLOUD_API_KEY, base_url=CLOUD_RU_URL)
#     rag = HybridRAGSystem(
#         embeddings=embeddings,
#         qdrant_path=QDRANT_PATH,
#         collection_name=QDRANT_COLLECTION,
#         neo4j_uri=NEO4J_URI,
#         neo4j_user=NEO4J_USER,
#         neo4j_password=NEO4J_PASSWORD,
#         llm_api_key=CLOUD_API_KEY,
#         llm_base_url=CLOUD_RU_URL
#     )
#     rag.create_knowledge_base(processed_files)

#     log.info("Documents successfully added to knowledge base!")
#     return {"status": "success", "count": len(processed_files)}


import glob

from services.ocr import YandexOCRProcessor
from services.embeddings import CloudRuEmbeddings
from services.rag_system import HybridRAGSystem

from utils.logger import get_logger
from utils.config import *

log = get_logger("[IngestScript]")

def ingest_files(files_to_ingest=None):
    log.info("Starting ingestion process...")

    if files_to_ingest is None:
        files_to_ingest = glob.glob(os.path.join(INPUT_FOLDER, '*'))

    if not files_to_ingest:
        log.info("No files provided for ingestion.")
        return {"status": "no_files", "count": 0}

    ocr_processor = YandexOCRProcessor(YANDEX_API_KEY)
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

    processed_files = []
    skipped_files = []
    contradictions_detected = []

    existing_sources = rag.get_all_sources()

    for file_path in files_to_ingest:
        file_name = os.path.basename(file_path)

        if file_path in existing_sources or file_name in existing_sources:
            log.info(f"Skipping duplicate file: {file_path}")
            skipped_files.append(file_path)
            continue

        try:
            text = ocr_processor.process_file(file_path)
            if not text or text.strip() == '':
                log.warning(f"Empty text for file {file_path}, skipping")
                skipped_files.append(file_path)
                continue

            # Векторизация текста
            file_embedding = embeddings.embed_text(text[:2000])

            # Поиск похожих документов в векторной базе
            search_results = rag.search_vector(file_embedding, top_k=5)

            # Формируем контекст для LLM по найденным релевантным фрагментам
            context = "\n\n---\n\n".join([r.content for r in search_results])
            if context is None or context == '':
                llm_response = 'Нет'
            
            else:
                contradiction_query = "Есть ли в приведённых документах текст, противоречащий следующему новому тексту? Ответь 'Да' или 'Нет'."
                contradiction_context = f"НОВЫЙ ТЕКСТ:\n{text}\n\nСУЩЕСТВУЮЩИЕ ТЕКСТЫ:\n{context}"
    
                llm_response = rag.generate_answer(contradiction_query, contradiction_context)

            if 'да' in llm_response.lower():
                contradictions_detected.append(file_path)
                log.warning(f"Противоречия обнаружены в файле: {file_path}")
            else:
                log.info(f"Противоречий не обнаружено для файла: {file_path}")

            # Добавляем файл в базу знаний
            file_info = {'original_file': file_path, 'text': text}
            rag.create_knowledge_base([file_info])

            processed_files.append(file_path)
            log.info(f"File ingested: {file_path}")

        except Exception as e:
            log.error(f"Ошибка при обработке файла {file_path}: {e}")
            skipped_files.append(file_path)

    return {
        "status": "success",
        "processed_count": len(processed_files),
        "skipped_count": len(skipped_files),
        "contradictions": contradictions_detected,
        "processed_files": processed_files,
        "skipped_files": skipped_files
    }
